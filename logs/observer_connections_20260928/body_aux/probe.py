"""CPU-only counterfactual evaluation of four saved-weight corners; no training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / 'source'))
from adamw_spectra import gn_probe as P
from adamw_spectra.data import TokenStream

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ARM = ROOT / 'logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s'
OFFSETS = [2_500_065_536, 2_600_065_536]
N = 8
CORNERS = ['base', 'body', 'aux', 'full']


def write_json(name, obj):
    tmp = HERE / (name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(HERE / name)


def dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def difference(a, b):
    return [x-y for x,y in zip(a,b)]


def gram(vectors):
    return {a: {b: dot(x,y) for b,y in vectors.items()} for a,x in vectors.items()}


def main():
    started = time.time()
    kept = ARM / 'scientific/kept'
    checkpoint = kept / 'step000183.pt'; nxt = kept / 'step000184_weights.pt'
    before = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    after = torch.load(nxt, map_location='cpu', weights_only=False, mmap=True)
    assert before['step'] == 183 and after['step'] == 184
    cfg = before['config']
    model = P.build_model(cfg['model'], before['model'], torch.device('cpu'))
    layers = P.hidden_linears(model)
    body_ids = {id(layer.weight) for layer in layers.values()}
    body_keys = [key for key,p in model.named_parameters() if id(p) in body_ids]
    aux_keys = [key for key,p in model.named_parameters() if id(p) not in body_ids]
    all_keys = set(body_keys + aux_keys)
    assert all_keys == set(before['model']) == set(after['model'])
    assert len(body_keys) == 48
    parameters = dict(model.named_parameters())
    for key,p in parameters.items(): p.requires_grad_(key in body_keys)
    weights = [parameters[key] for key in body_keys]
    body_set = set(body_keys)
    def set_corner(corner):
        with torch.no_grad():
            for key,p in parameters.items():
                use_next = corner == 'full' or (corner == 'body' and key in body_set) or (corner == 'aux' and key not in body_set)
                p.copy_((after if use_next else before)['model'][key])

    stream = TokenStream(str(ROOT / cfg['train_pattern']))
    T = cfg['model']['seq_len']
    x,y = stream.batch(OFFSETS[0], 1, T, torch.device('cpu'))
    # Direct-load comparisons qualify the state partition and forward path.
    qualification = {}
    for corner,state in [('base',before),('full',after)]:
        set_corner(corner)
        assert all(torch.equal(p.detach(), state['model'][key]) for key,p in parameters.items())
        with torch.no_grad():
            corner_logits = model(x)
            direct = P.build_model(cfg['model'], state['model'], torch.device('cpu'))
            direct_logits = direct(x)
            max_error = float((corner_logits-direct_logits).abs().max())
            qualification[corner] = {'max_logit_error':max_error,'loss':float(P.token_losses(corner_logits,y).mean())}
            assert max_error == 0
        del direct, direct_logits, corner_logits
    manifest = json.loads((HERE/'source/manifest.json').read_text())
    for row in manifest.values():
        p = ROOT / row['snapshot']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
    input_records = {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in [checkpoint,nxt]}
    report = {'arm':str(ARM),'base_step':183,'next_step':184,'config':cfg,
              'partition':{'body':body_keys,'aux':aux_keys,'body_numel':sum(parameters[k].numel() for k in body_keys),'aux_numel':sum(parameters[k].numel() for k in aux_keys)},
              'qualification':qualification,'banks':[],'source_manifest':manifest,'checkpoint_files':input_records,
              'precision':'CPU float32 eager; no autocast','sequence_length':T,'status':'running'}
    write_json('status.json',{'status':'qualified','pid':os.getpid(),'seconds':time.time()-started})
    print(json.dumps({'stage':'qualified','seconds':time.time()-started,'qualification':qualification}),flush=True)
    banks = []
    for bank_id, offset in enumerate(OFFSETS):
        sums = {c:[torch.zeros_like(w) for w in weights] for c in CORNERS}
        losses = {c:[] for c in CORNERS}
        for seq in range(N):
            seq_started = time.time()
            x,y = stream.batch(offset+seq*T,1,T,torch.device('cpu'))
            for corner in CORNERS:
                set_corner(corner)
                loss = model(x,y)
                gg = torch.autograd.grad(loss,weights)
                assert all(torch.isfinite(g).all() for g in gg)
                losses[corner].append(float(loss.detach()))
                for acc,g in zip(sums[corner],gg): acc.add_(g.detach(),alpha=1/N)
                del loss,gg
            elapsed = time.time()-seq_started
            if bank_id == 0 and seq == 0:
                forecast = (2*N)*elapsed
                report['cost_qualification']={'seconds_per_four_corners':elapsed,'projected_measurement_seconds':forecast}
                if forecast > 900:
                    report['status']='stopped_cost_gate';write_json('result.json',report)
                    write_json('status.json',{'status':'stopped_cost_gate','seconds':time.time()-started})
                    return
            write_json('status.json',{'status':'running','pid':os.getpid(),'bank':bank_id,'completed_sequences':seq+1,'seconds':time.time()-started})
            print(json.dumps({'bank':bank_id,'sequence':seq+1,'loss':{c:losses[c][-1] for c in CORNERS},'seconds':time.time()-started}),flush=True)
        banks.append(sums)
        g = gram(sums)
        changes = {'body':difference(sums['body'],sums['base']),
                   'aux':difference(sums['aux'],sums['base']),
                   'total':difference(sums['full'],sums['base'])}
        changes['interaction'] = difference(difference(changes['total'],changes['body']),changes['aux'])
        loss_means={c:sum(v)/N for c,v in losses.items()}
        row={'offset':offset,'sequences':N,'losses':losses,'mean_loss':loss_means,
             'gradient_gram':g,'gradient_cosines':{a:{b:v/math.sqrt(g[a][a]*g[b][b]) for b,v in rr.items()} for a,rr in g.items()},
             'change_gram':gram(changes),
             'loss_interaction':loss_means['full']-loss_means['body']-loss_means['aux']+loss_means['base']}
        report['banks'].append(row)
        torch.save({'body_keys':body_keys,'means':sums,'offset':offset,'sequences':N,'step':183},HERE/f'bank{bank_id}_body_gradient_means.pt')
        write_json('result.json',report)
        del changes
    cross={a:{b:dot(banks[0][a],banks[1][b]) for b in CORNERS} for a in CORNERS}
    report['cross_bank_gradient_gram']=cross
    report['cross_bank_cosines_symmetric']={a:{b:(cross[a][b]+cross[b][a])/2/math.sqrt(cross[a][a]*cross[b][b]) if min(cross[a][a],cross[b][b])>0 else None for b in CORNERS} for a in CORNERS}
    report['status']='complete';report['seconds']=time.time()-started
    for raw,record in input_records.items():
        p=ROOT/raw;assert p.stat().st_size==record['bytes'] and p.stat().st_mtime_ns==record['mtime_ns']
    write_json('result.json',report)
    write_json('status.json',{'status':'complete','pid':os.getpid(),'seconds':report['seconds']})
    print(json.dumps({'stage':'complete','seconds':report['seconds'],'cross_bank_cosines':report['cross_bank_cosines_symmetric']}),flush=True)


if __name__ == '__main__':
    main()
