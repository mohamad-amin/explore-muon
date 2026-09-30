"""CPU saved-tensor inventory; imports no project/model/optimizer code."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHUNK = 1_048_576


def dump(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def fhash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def group(key):
    if key == 'head.weight': return 'head'
    if key == 'embed.weight': return 'token_embedding'
    if key == 'position.weight': return 'position_embedding'
    if key == 'norm.weight': return 'final_norm'
    if re.fullmatch(r'blocks\.\d+\.ln[12]\.weight', key): return 'body_norm'
    if re.fullmatch(r'blocks\.\d+\.attn\.[qk]_norm\.weight', key): return 'qk_norm'
    raise ValueError(key)


def finalized(row):
    w2, v2, d2, wd = (row[k] for k in ['old_sqnorm', 'next_sqnorm', 'delta_sqnorm', 'old_dot_delta'])
    assert w2 > 0 and d2 > 0
    row.update(old_norm=math.sqrt(w2), next_norm=math.sqrt(v2), delta_norm=math.sqrt(d2),
               old_rms=math.sqrt(w2/row['numel']), next_rms=math.sqrt(v2/row['numel']),
               delta_rms=math.sqrt(d2/row['numel']), relative_delta=math.sqrt(d2/w2),
               radial_relative=wd/w2, cosine_old_delta=wd/math.sqrt(w2*d2),
               norm_relative_change=math.sqrt(v2/w2)-1,
               ideal_decay_norm=math.sqrt(row['ideal_decay_sqnorm']),
               adaptive_plus_rounding_norm=math.sqrt(row['adaptive_plus_rounding_sqnorm']))
    row['norm_identity_relative_error'] = abs((v2-w2)-(2*wd+d2))/max(w2, v2)
    assert row['norm_identity_relative_error'] < 1e-12
    return row


def measure(a, b, decay):
    assert a.dtype == b.dtype == torch.float32 and a.is_contiguous() and b.is_contiguous()
    ah, bh = hashlib.sha256(), hashlib.sha256()
    out = dict(numel=a.numel(), old_sqnorm=0., next_sqnorm=0., delta_sqnorm=0.,
               old_dot_delta=0., ideal_decay_sqnorm=0., adaptive_plus_rounding_sqnorm=0.)
    af, bf = a.view(-1), b.view(-1)
    for start in range(0, a.numel(), CHUNK):
        ar, br = af[start:start+CHUNK], bf[start:start+CHUNK]
        ah.update(memoryview(ar.numpy())); bh.update(memoryview(br.numpy()))
        w, v = ar.double(), br.double()
        d = v-w
        out['old_sqnorm'] += float(w.square().sum())
        out['next_sqnorm'] += float(v.square().sum())
        out['delta_sqnorm'] += float(d.square().sum())
        out['old_dot_delta'] += float((w*d).sum())
        out['adaptive_plus_rounding_sqnorm'] += float((d+decay*w).square().sum())
    out['ideal_decay_sqnorm'] = decay**2*out['old_sqnorm']
    return finalized(out), ah.hexdigest(), bh.hexdigest()


def main():
    started = time.time()
    prior = json.loads((HERE.parent/'body_aux/result.json').read_text())
    source_paths = ['body_aux/result.json', 'body_aux/probe.py', 'body_aux/source/adamw_spectra/model.py',
                    'value_step_geometry/result.json']
    sources = {str((HERE.parent/p).relative_to(ROOT)): fhash(HERE.parent/p) for p in source_paths}
    records = [dict(method='M4M', step=183,
                    checkpoint=str(Path(prior['arm']).relative_to(ROOT)/'scientific/kept/step000183.pt'),
                    nextweights=str(Path(prior['arm']).relative_to(ROOT)/'scientific/kept/step000184_weights.pt'))]
    for p in json.loads((HERE.parent/'value_step_geometry/result.json').read_text())['provenance']:
        if p['step'] in {200,500,900}:
            records.append(dict(method=Path(p['marginal']).name.split('_')[0], step=p['step'],
                                checkpoint=p['checkpoint'], nextweights=p['nextweights']))
    assert len(records) == 13
    tensor_rows, group_rows, provenance = [], [], []
    sumkeys = ['numel','old_sqnorm','next_sqnorm','delta_sqnorm','old_dot_delta',
               'ideal_decay_sqnorm','adaptive_plus_rounding_sqnorm']
    for record in records:
        cp, np = ROOT/record['checkpoint'], ROOT/record['nextweights']
        stats = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in [cp,np]}
        before = torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
        after = torch.load(np,map_location='cpu',weights_only=False,mmap=True)
        cfg = before['config']; step = record['step']
        assert before['step'] == step and after['step'] == step+1
        assert after['tokens']-before['tokens'] == cfg['batch_tokens']
        assert set(before['model']) == set(after['model']) == set(prior['partition']['body']+prior['partition']['aux'])
        body = [k for k,v in before['model'].items() if k.startswith('blocks.') and v.ndim == 2]
        aux = [k for k in before['model'] if k not in body]
        assert body == prior['partition']['body'] and aux == prior['partition']['aux']
        assert len(body) == 48 and len(aux) == 36
        assert cfg['model'] == prior['config']['model']
        cohort = cp.parents[3]
        manifest = json.loads((cohort/'frozen_manifest.json').read_text())
        for f in ['model.py','muon.py','train.py','distributed.py']:
            source = cohort/'frozen/adamw_spectra'/f
            actual = fhash(source)
            assert actual == manifest['adamw_spectra/'+f]
            sources[str(source.relative_to(ROOT))] = actual
        groups = before['optimizer']['param_groups']
        assert groups[0]['algorithm'] == 'muon' and len(groups[0]['params']) == 48
        assert groups[1]['algorithm'] == 'adamw' and len(groups[1]['params']) == 36
        warmup = min(1., (step+1)/max(1,cfg['warmup_steps']))
        cooldown_start = cfg['total_tokens']*(1-cfg['cooldown_fraction'])
        cooldown = min(1.,(cfg['total_tokens']-before['tokens'])/(cfg['total_tokens']-cooldown_start))
        body_lr = cfg['learning_rate']*warmup*max(0.,cooldown)
        aux_lr = body_lr*groups[1]['lr_scale']
        assert math.isclose(groups[1]['lr_scale'],cfg['aux_learning_rate']/cfg['learning_rate'])
        assert math.isclose(aux_lr,cfg['aux_learning_rate'])  # all selected states are on plateau
        assert groups[1]['weight_decay'] == cfg['weight_decay'] == .01
        decay = aux_lr*cfg['weight_decay']
        hashes = {}; local_rows = []
        for key in aux:
            a,b = before['model'][key],after['model'][key]
            row, ah,bh = measure(a,b,decay)
            row = {**record,'group':group(key),'key':key,'shape':list(a.shape),'dtype':str(a.dtype),
                   'aux_lr_next':aux_lr,'weight_decay':cfg['weight_decay'],'ideal_decay_coefficient':decay,**row}
            tensor_rows.append(row); local_rows.append(row)
            hashes[key] = dict(old_sha256=ah,next_sha256=bh)
        selections = {g:[r for r in local_rows if r['group']==g] for g in sorted({r['group'] for r in local_rows})}
        selections['embeddings_combined'] = [r for r in local_rows if r['group'].endswith('embedding')]
        selections['norms_combined'] = [r for r in local_rows if r['group'].endswith('norm')]
        selections['all_aux'] = local_rows
        for label,rr in selections.items():
            totals = {k:sum(r[k] for r in rr) for k in sumkeys}
            grouped = {**record,'group':label,'keys':[r['key'] for r in rr],'tensors':len(rr),
                       'aux_lr_next':aux_lr,'weight_decay':cfg['weight_decay'],
                       'ideal_decay_coefficient':decay,**finalized(totals)}
            group_rows.append(grouped)
        for p in [cp,np]:
            assert dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) == stats[str(p.relative_to(ROOT))]
        provenance.append({**record,'config':cfg,'before_tokens':before['tokens'],'next_tokens':after['tokens'],
                           'checkpoint_file_stats':stats,'body_keys':body,'aux_keys':aux,'aux_tensor_hashes':hashes,
                           'optimizer_param_groups_metadata':groups,'body_lr_next':body_lr,'aux_lr_next':aux_lr,
                           'body_numel':sum(before['model'][k].numel() for k in body),
                           'aux_numel':sum(before['model'][k].numel() for k in aux)})
        print(json.dumps({'method':record['method'],'step':step,'elapsed_seconds':time.time()-started,
                          'groups':[{k:r[k] for k in ['group','numel','old_norm','delta_norm','relative_delta','radial_relative']}
                                    for r in group_rows if r['method']==record['method'] and r['step']==step]}),flush=True)
        del before,after
    for filename,rows in [('tensors.csv',tensor_rows),('groups.csv',group_rows)]:
        with (HERE/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result = {'status':'complete','seconds':time.time()-started,'torch_version':torch.__version__,
              'resource_scope':'CPU mmap; two intra-op threads; one inter-op; no model, forward/backward, optimizer or GPU',
              'parameter_read_scope':'Only auxiliary tensor values; body tensor shapes/keys only; no optimizer moments',
              'interpretation':'Actual saved displacement includes decay and parameter-write rounding; not an optimizer direction.',
              'tensor_records':tensor_rows,'group_records':group_rows,'provenance':provenance,
              'source_sha256':sources,'inventory_source_sha256':fhash(Path(__file__))}
    dump('result.json',result)
    dump('status.json',{'status':'complete','seconds':result['seconds'],'checkpoint_pairs':len(records),
                        'tensor_records':len(tensor_rows),'group_records':len(group_rows)})
    print(json.dumps({'status':'complete','seconds':result['seconds']}),flush=True)


if __name__ == '__main__': main()
