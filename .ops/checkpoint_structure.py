import torch,sys,json
from pathlib import Path
p=Path(sys.argv[1])
def structure(v,depth=0):
 if torch.is_tensor(v):return {'tensor':list(v.shape),'dtype':str(v.dtype)}
 if isinstance(v,dict):return {str(k):structure(w,depth+1) for k,w in v.items()} if depth<3 else {'keys':list(map(str,v.keys()))[:20]}
 if isinstance(v,(list,tuple)):return [structure(x,depth+1) for x in v[:3]]+([f'...len={len(v)}'] if len(v)>3 else [])
 if isinstance(v,(int,float,str,type(None))):return v
 return str(type(v))
for f in sorted(p.glob('iter_*/mp_rank*/model_optim_rng.pt')):
 s=torch.load(f,map_location='cpu');print(json.dumps({'file':str(f),'structure':structure(s),'metadata':{k:v for k,v in s.items() if k.startswith('greyhound_')}},indent=2))
