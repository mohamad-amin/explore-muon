"""Fixed saved-weight angular/radial comparison. No model or optimizer calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'run1'
CAP = 180.
STEPS = (9, 46, 83)
KINDS = ('q', 'k', 'v', 'o', 'up', 'down')


def write(name, data):
    tmp = OUT / (name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(OUT / name)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def geometry(w, next_w, decay):
    w = w.double().flatten()
    next_w = next_w.double().flatten()
    d = next_w - w
    r = float(w.norm())
    rn = float(next_w.norm())
    assert r > 0 and rn > 0
    u = w / r
    radial = float(u @ d)
    tangent = float((d - radial * u).norm())
    dn = float(d.norm())
    chord = float((next_w / rn - u).norm())
    cosine = float(u @ (next_w / rn))
    adapt = d + decay * w
    adapt_norm = float(adapt.norm())
    adapt_radial = float(u @ adapt)
    identity_error = abs(rn * rn - ((r + radial)**2 + tangent**2)) / max(1., rn * rn)
    chord_error = abs(chord**2 - (2 - 2*cosine))
    assert identity_error < 1e-10 and chord_error < 1e-10
    return dict(radius=r, next_radius=rn, delta_norm=dn,
                relative_delta=dn/r, radial=radial, tangent=tangent,
                radial_cosine=radial/dn if dn else 0.,
                radial_cross=2*r*radial, step_square=dn**2,
                radius_square_change=rn**2-r**2,
                chord=chord, cosine=cosine,
                adaptive_plus_rounding_norm=adapt_norm,
                adaptive_plus_rounding_radial=adapt_radial,
                adaptive_plus_rounding_radial_cosine=adapt_radial/adapt_norm if adapt_norm else 0.,
                decay_radial=-decay*r,
                identity_error=identity_error, chord_identity_error=chord_error)


def save_csv(name, rows):
    with (OUT/name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    tree = ast.parse((HERE.parent/'momentum_lr/analyze.py').read_text())
    arms = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'ARMS' for t in n.targets))
    previous = json.loads((HERE.parent/'momentum_lr/result.json').read_text())
    manifest = {}
    def read(path):
        raw = path.read_bytes()
        manifest[str(path.relative_to(ROOT))] = dict(bytes=len(raw), sha256=digest(raw))
        return json.loads(raw)
    metadata = {}
    traces = {}
    source_checks = {}
    for key, arm in arms.items():
        base = ROOT/'logs/muon_spectra'/arm/'scientific'
        meta = read(base/'metadata.json')
        assert meta == previous['metadata'][str(key)]
        assert read(base/'status.json')['status'] == 'complete'
        cfg = meta['config']
        assert cfg['model']['qk_norm'] and not cfg['model']['bias']
        assert cfg['model']['n_embd'] == 512 and cfg['model']['n_head'] == 8
        assert cfg.get('data_norm_decay', 'decoupled') == 'decoupled'
        metadata[key] = meta
        traces[key] = {}
        for step in (1, 10, 25, 47, 50, 75, 84, 92):
            p = base/'steps'/f'step{step:06d}.json'
            traces[key][step] = read(p)
            old = previous['input_manifest'][str(p.relative_to(ROOT))]
            assert manifest[str(p.relative_to(ROOT))] == old
        checked = {}
        for name, expected in meta['source_sha256'].items():
            path = base/'source'/name
            if path.exists():
                actual = digest(path.read_bytes())
                assert actual == expected, (arm, name)
                checked[name] = actual
        # SOAP is implemented in muon.py in these frozen revisions.
        assert {'model.py', 'muon.py', 'data_norm_muon.py',
                'distributed.py', 'train.py', 'data.py'} <= set(checked)
        source_checks[str(key)] = checked
    # Synthetic identities include purely radial and purely tangent writes.
    w = torch.tensor([3., 4.], dtype=torch.float64)
    for d in (torch.zeros(2), torch.tensor([.3, .4]), torch.tensor([.4, -.3])):
        geometry(w, w+d.double(), .01)
    write('prepared.json', dict(arms={str(k):v for k,v in arms.items()}, steps=STEPS,
                               source_checks=source_checks, scalar_manifest=manifest,
                               source_sha256=digest(Path(__file__).read_bytes()),
                               protocol_sha256=digest((HERE/'PROTOCOL.md').read_bytes())))
    matrices = []
    heads = []
    provenance = []
    pair_times = []
    forecast = None
    for step in STEPS:
        for key, arm in arms.items():
            tick = time.monotonic()
            root = ROOT/'logs/muon_spectra'/arm/'scientific/kept'
            paths = [root/f'step{step:06d}.pt', root/f'step{step+1:06d}_weights.pt']
            info = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
            before, after = [torch.load(p, map_location='cpu', weights_only=False, mmap=True) for p in paths]
            assert before['step'] == step and after['step'] == step+1
            assert before['config'] == metadata[key]['config']
            assert before['tokens'] == step*16777216 and after['tokens'] == (step+1)*16777216
            body = sorted(k for k,v in before['model'].items() if k.startswith('blocks.') and v.ndim == 2)
            assert len(body) == 48
            assert set(before['model']) == set(after['model'])
            next_lr = traces[key][step+1]['lr']
            decay = next_lr*metadata[key]['config']['weight_decay']
            hashes = [{}, {}]
            count_heads = 0
            for name in body:
                w, n = before['model'][name], after['model'][name]
                assert w.shape == n.shape and w.dtype == n.dtype == torch.float32
                assert torch.isfinite(w).all() and torch.isfinite(n).all()
                for dest, tensor in zip(hashes, (w,n)):
                    dest[name] = digest(tensor.contiguous().numpy().tobytes())
                layer = int(name.split('.')[1])
                kind = name.split('.')[-2]
                assert kind in KINDS
                common = dict(lr=key[0], beta=key[1], step=step, next_step=step+1,
                              next_lr=next_lr, kind=kind, layer=layer, tensor=name)
                matrices.append({**common, **geometry(w,n,decay)})
                if kind in ('q','k'):
                    assert w.shape == (512,512)
                    for head, (wh,nh) in enumerate(zip(w.reshape(8,64,512),n.reshape(8,64,512))):
                        heads.append({**common, 'head':head, **geometry(wh,nh,decay)})
                        count_heads += 1
            assert count_heads == 128
            for p, initial in zip(paths, info):
                assert (p.stat().st_size,p.stat().st_mtime_ns) == initial
            provenance.append(dict(lr=key[0],beta=key[1],step=step,
                                   files=[dict(path=str(p.relative_to(ROOT)),bytes=i[0],mtime_ns=i[1],
                                               consumed_body_tensor_sha256=h)
                                          for p,i,h in zip(paths,info,hashes)]))
            del before, after
            pair_times.append(time.monotonic()-tick)
            elapsed = time.monotonic()-started
            if len(pair_times) == 1:
                forecast = elapsed + 11*pair_times[0]*1.25 + 10.
                assert forecast < CAP, ('forecast',forecast)
            write('progress.json',dict(status='running',completed_pairs=len(pair_times),seconds=elapsed,
                                       forecast_seconds=forecast))
            print(json.dumps(dict(completed_pairs=len(pair_times),seconds=elapsed)),flush=True)
            assert elapsed < CAP, ('cost boundary',elapsed)
    scalar_radii = []
    for key, trace in traces.items():
        wd = metadata[key]['config']['weight_decay']
        for step in (1,25,50,75,92):
            d = trace[step]
            assert len(d['matrices']) == 24 and wd > 0 and d['lr'] > 0
            for name, row in d['matrices'].items():
                radius = row['weight_decay_step_norm']/(d['lr']*wd)
                scalar_radii.append(dict(lr=key[0],beta=key[1],update=step,incoming_weight_step=step-1,
                                         panel=name,kind=name.split('.')[-1],radius=radius,
                                         nominal_relative_step=row['adaptive_step_norm']/radius))
    summaries = []
    for step in STEPS:
        for key in arms:
            for kind in KINDS:
                for scope,source in [('matrix',matrices),('head',heads)]:
                    rows = [r for r in source if (r['lr'],r['beta'])==key and r['step']==step and r['kind']==kind]
                    if not rows: continue
                    summaries.append(dict(lr=key[0],beta=key[1],step=step,kind=kind,scope=scope,n=len(rows),
                        radius_median=statistics.median(r['radius'] for r in rows),
                        chord_median=statistics.median(r['chord'] for r in rows),
                        chord_rms=math.sqrt(statistics.mean(r['chord']**2 for r in rows)),
                        delta_norm_median=statistics.median(r['delta_norm'] for r in rows),
                        radial_cosine_median=statistics.median(r['radial_cosine'] for r in rows),
                        sum_radial_cross=sum(r['radial_cross'] for r in rows),
                        sum_step_square=sum(r['step_square'] for r in rows),
                        sum_radius_square_change=sum(r['radius_square_change'] for r in rows)))
    index = {(r['lr'],r['beta'],r['step'],r['kind'],r['layer'],r['head']):r for r in heads}
    head_ratios = []
    contrasts = []
    for step in STEPS:
        for kind in ('q','k'):
            for factor, fixeds in [('lr',(.9,.8)),('beta',(.028,.04))]:
                for fixed in fixeds:
                    low,high = ((.028,fixed),(.04,fixed)) if factor=='lr' else ((fixed,.9),(fixed,.8))
                    rows = []
                    for layer in range(8):
                        for head in range(8):
                            a=index[(*low,step,kind,layer,head)]
                            b=index[(*high,step,kind,layer,head)]
                            r=dict(factor=factor,fixed=fixed,step=step,kind=kind,layer=layer,head=head,
                                   chord_ratio=b['chord']/a['chord'],radius_ratio=b['radius']/a['radius'],
                                   next_radius_ratio=b['next_radius']/a['next_radius'],
                                   tangent_ratio=b['tangent']/a['tangent'],
                                   delta_norm_ratio=b['delta_norm']/a['delta_norm'])
                            rows.append(r);head_ratios.append(r)
                    contrast=dict(factor=factor,fixed=fixed,step=step,kind=kind,n=64,
                                  **{k+'_median':statistics.median(r[k] for r in rows)
                                     for k in ('chord_ratio','radius_ratio','next_radius_ratio','tangent_ratio','delta_norm_ratio')},
                                  chord_ratio_min=min(r['chord_ratio'] for r in rows),
                                  chord_ratio_max=max(r['chord_ratio'] for r in rows),
                                  heads_ratio_below_1_2=sum(r['chord_ratio']<1.2 for r in rows),
                                  heads_ratio_above_1_1=sum(r['chord_ratio']>1.1 for r in rows))
                    contrasts.append(contrast)
    selected=lambda factor:[r for r in contrasts if r['factor']==factor and r['step'] in (46,83)]
    predictions=dict(P1_radius_compensation=all(r['chord_ratio_median']<1.20 for r in selected('lr')),
                     P2_short_beta_larger_turn=all(r['chord_ratio_median']>1.10 for r in selected('beta')))
    result=dict(status='complete',seconds=time.monotonic()-started,forecast_seconds=forecast,
                pair_times=pair_times,predictions=predictions,summaries=summaries,contrasts=contrasts,
                provenance=provenance,source_checks=source_checks,scalar_manifest=manifest,
                max_radius_identity_error=max(r['identity_error'] for r in matrices+heads),
                max_chord_identity_error=max(r['chord_identity_error'] for r in matrices+heads),
                source_sha256=digest(Path(__file__).read_bytes()))
    for name,rows in [('matrices.csv',matrices),('heads.csv',heads),('head_ratios.csv',head_ratios),
                      ('scalar_radii.csv',scalar_radii),('summaries.csv',summaries),('contrasts.csv',contrasts)]:
        save_csv(name,rows)
    write('result.json',result)
    write('status.json',dict(status='complete',seconds=result['seconds'],pairs=len(pair_times),
                             model_forwards=0,gpu_calls=0))
    print(json.dumps(dict(predictions=predictions,seconds=result['seconds'],contrasts=contrasts),indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists(): write('failure.json',dict(traceback=traceback.format_exc()))
        raise
