"""Prospective endpoint interaction and phase-control rules; no model inference."""
import math
import statistics

SEEDS=(20261001,20261002)
BATCHES=(65536,1048576)
METHODS=('muon','spd')
BETAS=(.8,.9)
LRS=(.01,.02)
TOTAL=96242176


def index_base(rows):
    keyed={(r['method'],r['batch_tokens'],r['momentum'],r['lr'],r['seed']):r for r in rows}
    expected={(m,b,beta,lr,s) for m in METHODS for b in BATCHES for beta in BETAS for lr in LRS for s in SEEDS}
    if len(rows)!=32 or set(keyed)!=expected:raise ValueError('Need the complete32-cell balanced base family')
    return keyed


def gain(a,b):
    return a['full_development_nll']-b['full_development_nll'] if a['status']==b['status']=='complete' else None


def choose_rate(keyed,method,batch,betas):
    options=[]
    for lr in LRS:
        rows=[keyed[method,batch,beta,lr,seed] for beta in betas for seed in SEEDS]
        value=statistics.mean(r['full_development_nll'] for r in rows) if all(r['status']=='complete' for r in rows) else None
        options.append(dict(lr=lr,mean_nll=value))
    finite=[r for r in options if r['mean_nll'] is not None and math.isfinite(r['mean_nll'])]
    selected=min(finite,key=lambda r:(r['mean_nll'],r['lr']))['lr'] if finite else None
    return dict(selected_lr=selected,candidates=options,scope='best of two tested rates; no bracketed optimum claim')


def first_crossing(curve,target):
    for i,r in enumerate(curve):
        if r['validation_nll']<=target:
            if i==0:return dict(status='initial',left=0,right=0,step=0.)
            l=curve[i-1];f=(l['validation_nll']-target)/(l['validation_nll']-r['validation_nll'])
            return dict(status='reached',left=l['step'],right=r['step'],step=l['step']+f*(r['step']-l['step']))
    return dict(status='unreached',left=None,right=None,step=None)


def analyse_base(rows):
    k=index_base(rows);contrasts=[];methods={};selections={};control={};crossings=[]
    for method in METHODS:
        member=[]
        for seed in SEEDS:
            for lr in LRS:
                gains=[gain(k[method,b,.9,lr,seed],k[method,b,.8,lr,seed]) for b in BATCHES]
                lo,hi=gains;interaction=hi-lo if None not in gains else None
                record=dict(method=method,seed=seed,lr=lr,small_gain=lo,large_gain=hi,interaction=interaction,
                    tested_preference_reversal=bool(lo is not None and hi is not None and lo<0<hi))
                contrasts.append(record);member.append(record)
                for b in BATCHES:
                    for t in range(40,96):
                        ca=first_crossing(k[method,b,.9,lr,seed]['evaluations'],t/10)
                        cb=first_crossing(k[method,b,.8,lr,seed]['evaluations'],t/10)
                        if k[method,b,.9,lr,seed]['status']!='complete':ca['status']='unstable'
                        if k[method,b,.8,lr,seed]['status']!='complete':cb['status']='unstable'
                        valid=ca['status']==cb['status']=='reached' and min(ca['left'],cb['left'])>0
                        crossings.append(dict(method=method,seed=seed,lr=lr,batch_tokens=b,threshold=t/10,
                            beta09=ca,beta08=cb,step_ratio=ca['step']/cb['step'] if valid else None,
                            lower=ca['left']/cb['right'] if valid else None,upper=ca['right']/cb['left'] if valid else None))
        seed_means=[]
        for seed in SEEDS:
            r=[x for x in member if x['seed']==seed]
            valid=all(x['large_gain'] is not None and x['interaction'] is not None for x in r)
            seed_means.append(dict(seed=seed,large_gain=statistics.mean(x['large_gain'] for x in r) if valid else None,
                interaction=statistics.mean(x['interaction'] for x in r) if valid else None))
        signs=all(x['large_gain'] is not None and x['interaction'] is not None and x['large_gain']>0 and x['interaction']>0 for x in member)
        material=all(x['large_gain'] is not None and x['interaction'] is not None and x['large_gain']>=.005 and x['interaction']>=.005 for x in seed_means)
        methods[method]=dict(all_fixed_rate_signs_pass=signs,each_seed_material_means_pass=material,seed_means=seed_means,passed=signs and material)
        for batch in BATCHES:
            control[f'{method}:{batch}']=choose_rate(k,method,batch,BETAS)
            for beta in BETAS:selections[f'{method}:{batch}:{beta}']=choose_rate(k,method,batch,(beta,))
    secondary=[]
    for method in METHODS:
        for seed in SEEDS:
            gains=[]
            for b in BATCHES:
                lr9=selections[f'{method}:{b}:0.9']['selected_lr'];lr8=selections[f'{method}:{b}:0.8']['selected_lr']
                gains.append(gain(k[method,b,.9,lr9,seed],k[method,b,.8,lr8,seed]) if lr9 is not None and lr8 is not None else None)
            secondary.append(dict(method=method,seed=seed,small_gain=gains[0],large_gain=gains[1],
                interaction=gains[1]-gains[0] if None not in gains else None))
    return dict(primary_by_method=methods,contrasts=contrasts,control_rate_choices=control,secondary_lr_choices=selections,
        secondary_contrasts=secondary,all_fixed_rate_crossings=crossings,
        both_methods_pass=all(r['passed'] for r in methods.values()),ordering_gate_still_failed=True,
        batch_rate_growth_gate_still_failed=True,full_goal_qualified=False,sealed_test_scored=False)


def analyse_controls(base,controls,choices):
    k=index_base(base)
    expected={(kind,m,b,beta,s) for kind in ('matched','extended') for m in METHODS
              for b in (BATCHES if kind=='matched' else (BATCHES[-1],)) for beta in BETAS for s in SEEDS}
    rows={(r['control_kind'],r['method'],r['batch_tokens'],r['momentum'],r['seed']):r for r in controls}
    if len(controls)!=24 or set(rows)!=expected:raise ValueError('Need all24 declared controls')
    records=[]
    for method in METHODS:
        for seed in SEEDS:
            base_g=[];matched_g=[]
            for b in BATCHES:
                lr=choices[f'{method}:{b}']['selected_lr']
                base_g.append(gain(k[method,b,.9,lr,seed],k[method,b,.8,lr,seed]))
                pair=[rows['matched',method,b,beta,seed] for beta in BETAS]
                if any(r['lr']!=lr or r['total_tokens']!=128*b for r in pair):raise ValueError('Control LR/horizon mismatch')
                matched_g.append(gain(pair[1],pair[0]))
            pair=[rows['extended',method,BATCHES[-1],beta,seed] for beta in BETAS]
            if any(r['lr']!=choices[f'{method}:{BATCHES[-1]}']['selected_lr'] or r['total_tokens']!=2*TOTAL for r in pair):
                raise ValueError('Extended LR/horizon mismatch')
            extended=gain(pair[1],pair[0])
            records.append(dict(method=method,seed=seed,base_small_gain=base_g[0],base_large_gain=base_g[1],
                base_interaction=base_g[1]-base_g[0] if None not in base_g else None,
                matched_small_gain=matched_g[0],matched_large_gain=matched_g[1],
                matched_interaction=matched_g[1]-matched_g[0] if None not in matched_g else None,
                extended_large_gain=extended,extended_minus_base_large_gain=extended-base_g[1] if extended is not None and base_g[1] is not None else None,
                retained_fraction=extended/base_g[1] if extended is not None and base_g[1] is not None and base_g[1]>0 else None))
    return dict(records=records,interpretation='Fixed-recipe phase classifiers; no control can rescue a failed base interaction or prior qualification gate',
                independent_confirmation=False,full_goal_qualified=False)
