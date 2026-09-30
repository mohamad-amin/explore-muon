"""Read retained one-dimensional tensor archives with stdlib only; no model/torch import."""
import collections, hashlib, io, json, pickle, statistics, struct, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUNS=ROOT/'tiny_spectra/shakespeare_screen_20260928/runs'
# ROOT is logs; permit only the exact tensor constructor and storage tags used here.
def rebuild(storage, offset, size, stride, requires_grad, hooks):
    assert len(size)==1 and stride==(1,) and not requires_grad
    return storage[offset:offset+size[0]]
class Reader(pickle.Unpickler):
    def __init__(self, z, prefix):
        super().__init__(io.BytesIO(z.read(prefix+'data.pkl'))); self.z=z; self.prefix=prefix
    def find_class(self,module,name):
        if (module,name)==('torch._utils','_rebuild_tensor_v2'): return rebuild
        if module=='torch' and name in ['LongStorage','FloatStorage']: return name
        if (module,name)==('collections','OrderedDict'): return collections.OrderedDict
        raise ValueError((module,name))
    def persistent_load(self,pid):
        typ,dtype,key,device,count=pid
        assert typ=='storage' and device=='cpu'
        return list(struct.unpack('<'+str(count)+{'LongStorage':'q','FloatStorage':'f'}[dtype],self.z.read(self.prefix+'data/'+key)))
def load(path):
    with zipfile.ZipFile(path) as z:
        return Reader(z,z.namelist()[0].split('/')[0]+'/').load()
def stats(v):
    return dict(n=len(v),mean=statistics.mean(v),median=statistics.median(v),sd=statistics.stdev(v),negative=sum(x<0 for x in v),positive=sum(x>0 for x in v))
files=[]; objs={}
for method in ['pd','ts']:
    run=RUNS/f'{method}_b8192_lr0.016_m0.9_s20260928'
    f=run/'final_validation.pt'; w=run/'windows.pt'
    objs[method]=(load(f),load(w),json.loads((run/'summary.json').read_text()))
    files.extend([f,w,run/'summary.json'])
p,pw,ps=objs['pd'];t,tw,ts=objs['ts']
assert p['starts']==t['starts'] and pw['validation']==tw['validation']
assert pw['train']==tw['train']
bank=set(pw['validation']); ids=[i for i,s in enumerate(p['starts']) if s in bank]; rest=[i for i in range(len(p['starts'])) if i not in ids]
diff=[b-a for a,b in zip(p['sequence_nll'],t['sequence_nll'])]
means={}
for method,(d,w,s) in objs.items():
    means[method]=dict(full=statistics.mean(d['sequence_nll']),bank=statistics.mean(d['sequence_nll'][i] for i in ids),complement=statistics.mean(d['sequence_nll'][i] for i in rest),stored_full=s['full_validation_nll'],stored_bank=s['final_validation_nll'])
    assert abs(means[method]['full']-s['full_validation_nll'])<1e-6
    means[method]['bank_reconstruction_error']=means[method]['bank']-s['final_validation_nll']
result=dict(snapshot='2026-09-28 14:25 CDT',contrast='TS minus PD; LR=.016, momentum=.9, seed20260928, batch8192',means=means,full=stats(diff),bank=stats([diff[i] for i in ids]),complement=stats([diff[i] for i in rest]),ordered_10_blocks=[stats(diff[len(diff)*j//10:len(diff)*(j+1)//10]) for j in range(10)],sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
(OUT/'window_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
