"""Metadata inventory only; no model import, tensor arithmetic or GPU."""
import hashlib
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
AUDIT=ROOT/'logs/muon_spectra/second_order_audit_20260926'


def tensors(x):
    if isinstance(x,torch.Tensor):return [x]
    if isinstance(x,dict):return [t for v in x.values() for t in tensors(v)]
    if isinstance(x,(list,tuple)):return [t for v in x for t in tensors(v)]
    return []


def main():
    files=[]
    for directory in ['gn','gn2','gn3','gnmix','gnwarm','gnp','valley/gn_floor','transport']:
        files.extend(sorted((AUDIT/directory).glob('*.pt')))
    rows=[]
    for f in files:
        d=torch.load(f,map_location='cpu',weights_only=False,mmap=True)
        row={'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,
             'mtime_ns':f.stat().st_mtime_ns,'keys':list(d)}
        if 'directions' in d:row['directions_by_input']={k:list(v) for k,v in d['directions'].items()}
        if 'frames' in d:
            first=next(iter(d['frames'].values()))
            row['frames']={'matrices':len(d['frames']),'fields':list(first),'dtypes':{k:str(v.dtype) for k,v in first.items()}}
        if 'Mstar' in d:row['transport_groups']={k:{'matrices':len(v),'dtypes':sorted({str(t.dtype) for t in tensors(v)})} for k,v in d.items() if isinstance(v,dict)}
        row['tensor_count']=len(tensors(d));row['tensor_dtypes']=sorted({str(t.dtype) for t in tensors(d)})
        rows.append(row);del d
    sources={}
    for name in ['one_step_gn.py','transport_test.py','step_profile_probe.py']:
        f=AUDIT/name;sources[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    result={'scope':'Selected retained directions/transport tensor metadata only. No claim of exhaustive project tensor coverage.',
            'source_hashes':sources,'files':rows}
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:print(row['path'],row.get('directions_by_input',list(row.get('transport_groups',{}))))


if __name__=='__main__':main()
