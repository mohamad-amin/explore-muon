"""CPU-only scalar trajectory audit; no model or optimizer imports."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import csv
import hashlib
import json
import time
import numpy as np

PAIRS = {
    'PD_260925': (
        'soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s',
        'soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s'),
    'SPD_260925': (
        'soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada',
        'soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada'),
    'SPD_260926': (
        'soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada',
        'soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada'),
}
LEVELS = [4.25, 4.30, 4.35, 4.40]
WIDTHS = [11, 21, 31]
ANCHORS = [110, 120, 130, 140]


def crossings(xs, ys, level):
    """All crossings of the piecewise-linear description, without extrapolation."""
    events = []
    for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if y0 == y1 == level:
            event = dict(kind='flat', bracket=[float(x0), float(x1)], time=None)
        elif y0 != y1 and min(y0, y1) <= level <= max(y0, y1):
            event = dict(kind='down' if y1 < y0 else 'up',
                         bracket=[float(x0), float(x1)],
                         time=float(x0 + (level-y0)/(y1-y0)*(x1-x0)))
        else:
            continue
        # Adjacent segments can share an exact knot. Keep distinct directions.
        if not any(e['time'] == event['time'] and e['kind'] == event['kind']
                   and event['time'] is not None for e in events):
            events.append(event)
    if not events:
        status = ('below_observed_range' if level < min(ys) else
                  'above_observed_range' if level > max(ys) else 'unbracketed')
    elif len(events) == 1 and events[0]['kind'] == 'down':
        status = 'unique_down'
    else:
        status = 'ambiguous'
    return dict(status=status, events=events)


def paired_level(xw, yw, xr, yr, level):
    w, r = crossings(xw, yw, level), crossings(xr, yr, level)
    out = dict(level=level, warmup=w, constant=r, lead=None,
               monotonicity_conditional_lead_bracket=None)
    if w['status'] == r['status'] == 'unique_down':
        ew, er = w['events'][0], r['events'][0]
        out['lead'] = er['time'] - ew['time']
        out['monotonicity_conditional_lead_bracket'] = [
            er['bracket'][0]-ew['bracket'][1], er['bracket'][1]-ew['bracket'][0]]
    return out


def smooth(rows, width):
    h = width // 2
    output = []
    for record in range(94+h, 166-h+1):
        support = list(range(record-h, record+h+1))
        assert len(support) == width and support[0] >= 94 and support[-1] <= 166
        assert all(rows[t]['batch_tokens'] == 16777216 for t in support)
        output.append(dict(record_step=record, state_step=record-1,
                           support_first_record=support[0], support_last_record=support[-1],
                           loss=float(np.mean([rows[t]['train_nll'] for t in support]))))
    return output


def qualify():
    exact = paired_level([0, 10], [8, 6], [0, 10], [9, 7], 7.5)
    assert abs(exact['lead']-5) < 1e-12
    multi = crossings([0, 1, 2, 3], [3, 1, 3, 1], 2)
    assert multi['status'] == 'ambiguous' and [e['kind'] for e in multi['events']] == ['down','up','down']
    assert crossings([0, 1], [2, 1], .5)['status'] == 'below_observed_range'
    assert crossings([0, 1], [2, 2], 2)['status'] == 'ambiguous'
    assert crossings([0, 1, 2], [3, 2, 1], 2)['status'] == 'unique_down'
    assert crossings([0, 1], [1, 2], 1.5)['status'] == 'ambiguous'
    return dict(linear_translation=True, multiple_crossings=True, censoring=True,
                flat_segment=True, exact_knot_deduplication=True, upward_crossing_retained=True)


def witnesses(vals):
    r100, w100 = vals['constant'][100], vals['warmup'][100]
    r150, w150 = vals['constant'][150], vals['warmup'][150]
    assert r100 > w100 > r150 > w150
    outputs = []
    for name, d0, d1 in [('constant',5,5),('eroding',10,2),('increasing',1,9)]:
        x = np.array([100,100+d0,150,150+d1],dtype=float)
        y = np.array([r100,w100,r150,w150])
        t = np.linspace(100,150,501)
        delta = d0 + (d1-d0)*(t-100)/50
        w = np.interp(t+delta,x,y)
        assert np.all(np.diff(x)>0) and np.all(np.diff(y)<0)
        assert np.all(np.diff(w)<0) and x[-1]<=159
        anchor_error = max(abs(w[0]-w100),abs(w[-1]-w150),
                           abs(np.interp(100,x,y)-r100),abs(np.interp(150,x,y)-r150))
        assert anchor_error < 1e-14
        # Exact inverse of monotone reference for each constructed warmup score.
        inverse = np.interp(w,y[::-1],x[::-1])
        inverse_error = float(np.max(np.abs(inverse-t-delta)))
        assert inverse_error < 1e-11
        outputs.append(dict(name=name, delta_at100=d0, delta_at150=d1,
                            reference_knots=list(zip(x.tolist(),y.tolist())),
                            anchor_error=anchor_error, inverse_identity_error=inverse_error,
                            time_map_derivative=1+(d1-d0)/50))
    return outputs


def write_csv(name, rows):
    with (HERE/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    assert not (HERE/'result.json').exists(), 'Preserve completed reduction; choose new output for rerun.'
    started=time.monotonic()
    qualification=qualify()
    manifest={}
    def blob(p):
        b=p.read_bytes();manifest[str(p.relative_to(ROOT))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
        return b
    def read(p): return json.loads(blob(p))
    for name in ['PROTOCOL.md','DESIGN_REVIEW.md','MAIN_COVERAGE_REVIEW.md','analyze.py']:
        blob(HERE/name)
    blob(HERE.parent/'confidence_calibration/PAIRING_REVIEW.md')
    results={};raw=[];smoothed=[];anchor_csv=[];level_csv=[]
    for pair, paths in PAIRS.items():
        rows={};metadata={};sources={}
        for label,path in zip(['warmup','constant'],paths):
            p=ROOT/'logs/muon_spectra'/path/'scientific'
            m=metadata[label]=read(p/'metadata.json')
            status=read(p/'status.json');side=read(p/'checkpoint.json')
            assert status['status']=='complete' and status['step']==side['step']==184
            assert status['tokens']==side['tokens']==3079741440
            rows[label]={}
            for file in sorted((p/'steps').glob('step*.json')):
                r=read(file);assert r['step'] not in rows[label];rows[label][r['step']]=r
            assert set(rows[label])==set(range(185))
            sources[label]={}
            for name, expected in m['source_sha256'].items():
                actual=hashlib.sha256(blob(p/'source'/name)).hexdigest();assert actual==expected
                sources[label][name]=actual
        wm,rm=metadata['warmup'],metadata['constant']
        for field in ['initial_model_sha256','train_manifest','validation_manifest','parameter_count',
                      'world_size','device_name','effective_precision','effective_compile','parameter_optimizers',
                      'effective_optimizer_impl','ns_coefficients','nesterov']:
            assert wm[field]==rm[field],(pair,field)
        if pair!='SPD_260925': assert sources['warmup']==sources['constant']
        wc,rc=wm['config'],rm['config']
        config_diff={k:[wc.get(k),rc.get(k)] for k in set(wc)|set(rc) if wc.get(k)!=rc.get(k)}
        defaults={'muon_prefilter':'none','head_whitening_alpha':0.,'head_whitening_center':False,
                  'head_whitening_norm':'match','muon_momentum_start':-1.,'muon_momentum_warmup':0.}
        norm_w=defaults|wc;norm_r=defaults|rc
        normalized_diff={k:[norm_w.get(k),norm_r.get(k)] for k in set(norm_w)|set(norm_r) if norm_w.get(k)!=norm_r.get(k)}
        assert normalized_diff=={'muon_momentum_start':[.8,-1.],'muon_momentum_warmup':[.5,0.]}
        betas={};consumed=0;cool=[]
        vals={label:{} for label in rows}
        for step in range(185):
            w,r=rows['warmup'][step],rows['constant'][step]
            for key in ['tokens','batch_tokens','lr','aux_lr','world_size','dummy_sequences']:
                assert w.get(key)==r.get(key),(pair,step,key)
            if step:
                beta=.8+(.9-.8)*min(1.,consumed/(.5*3079741440));betas[step]=beta
                if step>3 and w['lr']<.028: cool.append(step)
                assert w['tokens']==consumed+w['batch_tokens']
                if step<184: assert w['batch_tokens']==16777216
                raw.append(dict(pair=pair,record_step=step,state_step=step-1,
                                tokens=w['tokens'],batch_tokens=w['batch_tokens'],lr=w['lr'],aux_lr=w['aux_lr'],
                                beta_warmup=beta,train_warmup=w['train_nll'],train_constant=r['train_nll']))
                consumed=w['tokens']
            for label,record in [('warmup',w),('constant',r)]:
                if 'validation_nll' in record: vals[label][step]=record['validation_nll']
        assert next(t for t,b in betas.items() if b==.9)==93 and cool[0]==167
        assert list(vals['warmup'])==list(vals['constant'])==[0,50,100,150,184]
        assert all(rows['warmup'][t]['lr']==.028 for t in range(94,167))
        val_readouts={}
        for mode in ['primary_100_150','omit100','omit150']:
            ts=[100,150] if mode=='primary_100_150' else [t for t in vals['warmup'] if t!=int(mode[4:])]
            val_readouts[mode]=[paired_level(ts,[vals['warmup'][t] for t in ts],ts,
                [vals['constant'][t] for t in ts],level) for level in LEVELS]
            for d in val_readouts[mode]:
                level_csv.append(dict(pair=pair,mode=mode,level=d['level'],lead=d['lead'],
                                     warmup_status=d['warmup']['status'],constant_status=d['constant']['status']))
        anchor_controls=[]
        for t, left,right in [(100,50,150),(150,100,184)]:
            a=dict(omitted=t,left=left,right=right,arms={})
            for label in vals:
                v=vals[label]
                interpolated=v[left]+(v[right]-v[left])*(t-left)/(right-left)
                inferred=left+(v[t]-v[left])/(v[right]-v[left])*(right-left)
                a['arms'][label]=dict(recorded_nll=v[t],interpolated_nll=interpolated,
                                      nll_error=interpolated-v[t],inverse_step_error=inferred-t)
            a['paired_nll_error']=a['arms']['warmup']['nll_error']-a['arms']['constant']['nll_error']
            a['paired_inverse_time_error']=a['arms']['constant']['inverse_step_error']-a['arms']['warmup']['inverse_step_error']
            anchor_controls.append(a)
        training={}
        for width in WIDTHS:
            sw,sr=smooth(rows['warmup'],width),smooth(rows['constant'],width)
            x=[d['state_step'] for d in sw];yw=[d['loss'] for d in sw];yr=[d['loss'] for d in sr]
            anchors=[]
            for state in ANCHORS:
                assert state in x
                target=yw[x.index(state)];cr=crossings(x,yr,target)
                lead=cr['events'][0]['time']-state if cr['status']=='unique_down' else None
                record=dict(state_step=state,target=target,lead=lead,constant_crossings=cr)
                anchors.append(record)
                anchor_csv.append(dict(pair=pair,width=width,state_step=state,target=target,lead=lead,status=cr['status']))
            for i,d in enumerate(sw):
                target=yw[i];cr=crossings(x,yr,target)
                smoothed.append(dict(pair=pair,width=width,**{k:v for k,v in d.items() if k!='loss'},
                    warmup_nll=target,constant_nll=yr[i],vertical_gap=target-yr[i],
                    lead=cr['events'][0]['time']-x[i] if cr['status']=='unique_down' else None,status=cr['status']))
            changes=[a['lead'] for a in anchors]
            training[str(width)]=dict(state_range=[x[0],x[-1]],
                up_segments_warmup=int(np.sum(np.diff(yw)>0)),up_segments_constant=int(np.sum(np.diff(yr)>0)),
                anchors=anchors,lead_change_140_minus110=changes[-1]-changes[0] if None not in [changes[0],changes[-1]] else None,
                fixed_levels=[paired_level(x,yw,x,yr,level) for level in LEVELS],
                full_crossings=[dict(state_step=s,**crossings(x,yr,target)) for s,target in zip(x,yw)])
        witness=witnesses(vals)
        endpoint=crossings(list(vals['constant']),list(vals['constant'].values()),vals['warmup'][184])
        assert endpoint['status']=='below_observed_range'
        results[pair]=dict(paths=dict(zip(['warmup','constant'],paths)),config_differences=config_diff,
            source_identical=sources['warmup']==sources['constant'],source_qualified=True,
            first_common_beta_update=93,first_cooldown_update=167,validation=vals,
            validation_readouts=val_readouts,anchor_controls=anchor_controls,
            endpoint_matching=endpoint,training=training,witnesses=witness,
            identification='Unresolved: all three monotone lead trends match the four primary anchors exactly.')
    write_csv('raw_training.csv',raw);write_csv('smoothed_training.csv',smoothed)
    write_csv('training_anchors.csv',anchor_csv);write_csv('validation_leads.csv',level_csv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for col,(pair,d) in enumerate(results.items()):
        ax=axs[0,col]
        for label,color in [('constant','#D55E00'),('warmup','#0072B2')]:
            v=d['validation'][label];ts=[50,100,150,184]
            ax.plot(ts,[v[t] for t in ts],'o--',color=color,label=label)
        ax.axvspan(93,166,color='grey',alpha=.08)
        ax.set(title=pair,xlabel='Model state after update',ylabel='Fixed-bank validation NLL')
        ax.legend(frameon=False);ax.grid(alpha=.15)
        ax=axs[1,col]
        for width,color in zip(WIDTHS,['#0072B2','#D55E00','#009E73']):
            ss=[r for r in smoothed if r['pair']==pair and r['width']==width]
            ax.plot([r['state_step'] for r in ss],[r['lead'] if r['lead'] is not None else np.nan for r in ss],label=f'{width}-row mean',color=color)
        ax.axhline(0,color='grey',lw=.8);ax.set(xlabel='Warmup model state',ylabel='Apparent lead (updates)')
        ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.suptitle('Warmup: fixed-bank anchors above; changing-batch inverse means below\nDashed validation connections are interpolation, not extra observations',fontsize=12)
    fig.tight_layout();fig.savefig(HERE/'progress_lead.png',dpi=170);fig.savefig(HERE/'progress_lead.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for ax,w in zip(axs,results['PD_260925']['witnesses']):
        x,y=np.array(w['reference_knots']).T;t=np.linspace(100,150,501)
        delta=w['delta_at100']+(w['delta_at150']-w['delta_at100'])*(t-100)/50
        ax.plot(x,y,color='#D55E00',label='Possible reference curve')
        ax.plot(t,np.interp(t+delta,x,y),color='#0072B2',label='Possible warmup curve')
        for label,color in [('constant','#D55E00'),('warmup','#0072B2')]:
            v=results['PD_260925']['validation'][label]
            ax.scatter([100,150],[v[100],v[150]],color=color,zorder=3,s=45)
        ax.set(title=f"{w['name']}: lead {w['delta_at100']} → {w['delta_at150']}",xlabel='Model state after update')
        ax.grid(alpha=.15)
    axs[0].set_ylabel('NLL');axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Same four observed validation anchors; three incompatible lead trends\nMathematical monotone completions, not newly measured losses',fontsize=12)
    fig.tight_layout();fig.savefig(HERE/'nonidentification.png',dpi=170);fig.savefig(HERE/'nonidentification.pdf');plt.close(fig)
    # Verify immutable consumed sources after reduction and plotting.
    for name,expected in manifest.items():
        p=ROOT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'],name
    result=dict(qualification=qualification,pairs=results,input_manifest=manifest,seconds=time.monotonic()-started,
                primary_verdict='Fixed-bank progress trend is not identified; warmup endpoint gains remain established.',
                limits='Secondary horizontal training comparisons use different batches; no confidence intervals or causal rate claim.')
    assert result['seconds'] < 120
    (HERE/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(seconds=result['seconds'],input_files=len(manifest),qualification=qualification,
        pairs={k:dict(primary_leads=[r['lead'] for r in v['validation_readouts']['primary_100_150']],
                     omit100_leads=[r['lead'] for r in v['validation_readouts']['omit100']],
                     omit150_leads=[r['lead'] for r in v['validation_readouts']['omit150']],
                     train_anchors={w:[a['lead'] for a in x['anchors']] for w,x in v['training'].items()}) for k,v in results.items()}),indent=2))


if __name__=='__main__': main()
