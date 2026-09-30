import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,statistics,collections,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
r=json.loads((H/'attention_transport.json').read_text())['records']; rr=json.loads((H/'results.json').read_text())['records']
med=statistics.median;steps=[10,50,100,200,500,900,1300,1469];methods=['M','PD','S','SPD'];names={'M':'Muon','PD':'PD','S':'SOAP-Muon','SPD':'SOAP-PD'}
def groups(method,step):return [x for x in r if x['method']==method and x['step']==step]
text=['# Saved attention geometry tables','', 'All values below are medians over eight layers, not independent statistical replicates. All 32 states belong to one seed and four own-optimizer trajectories.','', '## Final step: three locations in attention','', '| Method | Input mean share | Value mean share before attention | Measured mean share after attention | Mean/centered V-map gain | Centered energy after/before attention | cos(post mean, V input mean) |','|---|---:|---:|---:|---:|---:|---:|']
keys=['input_mean_share','pre_value_mean_share','post_attention_mean_share','mean_gain_over_centered_gain','centered_energy_transport','mean_cosine']
for m in methods:
 rows=groups(m,1469);text.append('| '+names[m]+' | '+' | '.join(f'{med(x[k] for x in rows):.4f}' for k in keys)+' |')
text+=['','## Through training: learned V-map mean gain relative to centered gain','','| Method | '+' | '.join(map(str,steps))+' |','|---|'+'---:|'*len(steps)]
for m in methods:text.append('| '+names[m]+' | '+' | '.join(f'{med(x["mean_gain_over_centered_gain"] for x in groups(m,s)):.4f}' for s in steps)+' |')
text+=['','## All checkpoints, decomposition and transport','','| Method | Step | Input mean | Value mean | Output mean | Centered transport | Post mean cosine | Selection mean shift / post mean |','|---|---:|---:|---:|---:|---:|---:|---:|']
for m in methods:
 for s in steps:
  z=groups(m,s); keys=['input_mean_share','pre_value_mean_share','post_attention_mean_share','centered_energy_transport','mean_cosine','selection_mean_shift_relative']
  text.append('| '+names[m]+' | '+str(s)+' | '+' | '.join(f'{med(x[k] for x in z):.4f}' for k in keys)+' |')
text+=['','## Attention covariance and marginal models at final step','','| Method | Between-sequence covariance share of input C | Global mean share of between moment | K exact marginal fit residual | V exact marginal fit residual | V curvature share along input mean |','|---|---:|---:|---:|---:|---:|']
for m in methods:
 k=[x for x in rr if x['method']==m and x['step']==1469 and x['kind']=='k'];v=[x for x in rr if x['method']==m and x['step']==1469 and x['kind']=='v']
 vals=[med(x['between_cov_share'] for x in k),med(x['global_fraction_of_between_trace'] for x in k),med(x['two_fit']['residual'] for x in k),med(x['two_fit']['residual'] for x in v),med(x['mean_exact_trace_share'] for x in v)]
 text.append('| '+names[m]+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
(H/'TABLES.md').write_text('\n'.join(text)+'\n')
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
colors={'M':'#444444','PD':'#008c95','S':'#cb8417','SPD':'#9353a0'}
panels=[('input_mean_share','Input shared-mean energy / total'),('pre_value_mean_share','Projected value shared-mean energy / total'),('post_attention_mean_share','Measured post-attention mean energy / total'),('mean_gain_over_centered_gain','V-map mean gain / centered gain')]
for ax,(key,title) in zip(axs.flat,panels):
 for m in methods:
  y=[med(x[key] for x in groups(m,s)) for s in steps]
  ax.plot(steps,y,'o-',label=names[m],color=colors[m],ms=3)
 ax.set_xscale('log');ax.set_xticks([10,50,200,500,1469]);ax.set_xticklabels(['10','50','200','500','1469']);ax.set_xlabel('Training update (log scale)');ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
axs[0,0].legend(frameon=False,fontsize=9)
fig.suptitle('Attention transports the shared mean differently across optimizer trajectories\nSaved frontier trajectories; one seed, median over eight layers',fontsize=12)
fig.savefig(H/'attention_mean_transport.png',dpi=180);fig.savefig(H/'attention_mean_transport.pdf');plt.close(fig)
