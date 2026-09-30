"""Plot the fixed paired-head ratios; descriptive quartiles, not uncertainty."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with (HERE/'run1/head_ratios.csv').open() as f:
    rows=list(csv.DictReader(f))
fig, axes=plt.subplots(1,2,figsize=(11.5,4.4))
steps=[9,46,83]
for ax,factor,fixeds,title in zip(axes,['lr','beta'],[[.9,.8],[.028,.04]],
                               ['43% higher LR: normalized Q/K turn',
                                'Shorter momentum: normalized Q/K turn']):
    for fixed,color in zip(fixeds,['#0072B2','#D55E00']):
        for kind,style in [('q','-'),('k','--')]:
            values=[np.array([float(r['chord_ratio']) for r in rows
                     if r['factor']==factor and float(r['fixed'])==fixed
                     and int(r['step'])==step and r['kind']==kind]) for step in steps]
            assert all(len(v)==64 for v in values)
            median=[np.median(v) for v in values]
            low=[np.quantile(v,.25) for v in values]
            high=[np.quantile(v,.75) for v in values]
            label=(f'beta {fixed}' if factor=='lr' else f'LR {fixed}')+f', {kind.upper()}'
            ax.plot(steps,median,style,color=color,marker='o',label=label)
            ax.fill_between(steps,low,high,color=color,alpha=.08)
    ax.axhline(1.,color='gray',linewidth=.8)
    if factor=='lr':
        ax.axhline(.04/.028,color='black',linestyle=':',linewidth=1,label='Nominal LR ratio')
        ax.axhline(1.2,color='#555555',linestyle='-.',linewidth=.8,label='P1 bound')
    else:
        ax.axhline(1.1,color='#555555',linestyle='-.',linewidth=.8,label='P2 bound')
    ax.set_title(title,fontsize=11)
    ax.set_xlabel('Saved incoming state; next actual write scored')
    ax.set_ylabel('Median matched-head chord ratio')
    ax.set_xticks(steps)
    ax.grid(alpha=.12)
    ax.legend(frameon=True,facecolor='white',edgecolor='none',framealpha=.95,fontsize=8,loc='best')
fig.suptitle('One paired seed; all 64 heads per Q/K kind. Shading: head quartiles, not confidence intervals.',fontsize=10)
fig.tight_layout()
fig.savefig(HERE/'run1/angular_ratios.png',dpi=180)
fig.savefig(HERE/'run1/angular_ratios.pdf')
