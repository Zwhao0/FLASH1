import json,re,math,statistics
from pathlib import Path
root=Path('results/multinode'); decoder=json.JSONDecoder()
rx=re.compile(r'iteration\s+(\d+)/\s*(\d+).*?consumed samples:\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+)')
def loadrun(n,name):
 p=root/f'node{n}'/name;text=(p/'console.log').read_text(errors='replace') if (p/'console.log').exists() else ''
 rows={}
 for line in text.splitlines():
  m=rx.search(line)
  if m:rows[int(m[1])]={'iteration':int(m[1]),'target':int(m[2]),'consumed':int(m[3]),'ms':float(m[4])}
 restores=[]
 for m in re.finditer('GREYHOUND_VERIFY_RESTORE ',text):
  try:obj,_=decoder.raw_decode(text[m.end():]); restores.append(obj)
  except ValueError:pass
 stages=[list(map(int,m)) for m in re.findall(r'pp_rank=(\d+), num_layers=(\d+), offset=(\d+)',text)]
 loss=[float(x) for x in re.findall(r'lm loss:\s*([\d.eE+-]+)',text)]
 return {'exit_code':int((p/'exit-code.txt').read_text()) if (p/'exit-code.txt').exists() else None,'last_iteration':max(rows) if rows else max([int(x) for x in re.findall(r'successfully saved checkpoint at iteration\s+(\d+)',text)],default=None),'last_row':rows[max(rows)] if rows else None,'loss_count':len(loss),'all_logged_losses_finite':all(math.isfinite(x) for x in loss),'stages':stages,'restores':restores,'dp_pass_lines':re.findall(r'GREYHOUND_VERIFY_DP PASS[^\n]*',text),'rows':list(rows.values())}
names=['s3-socket','s3-resumed','s4-interrupted-cleanup','s4-resumed','s4-reference-exact','s4-restore-audit']
runs={f'node{n}':{name:loadrun(n,name) for name in names} for n in [0,1]}
rows=[dict(r,phase=phase) for name,phase in [('s3-socket','source'),('s3-resumed','resumed')] for r in runs['node1'][name]['rows']]
(root/'s3-step-records.json').write_text(json.dumps(rows,indent=2))
phases={}
for label,phase,start,end in [('baseline','source',50,158),('fault_before_plan','source',250,400),('fault_after_repartition','resumed',550,950)]:
 raw=[r['ms'] for r in rows if r['phase']==phase and start<=r['iteration']<=end];vs=sorted(v for v in raw if v<2000)
 if vs:
  pos=(len(vs)-1)*.9;lo=int(pos);p90=vs[lo]+(vs[min(lo+1,len(vs)-1)]-vs[lo])*(pos-lo)
  phases[label]={'iterations':[start,end],'n':len(vs),'median_ms':statistics.median(vs),'p90_ms':p90,'excluded_pauses':len(raw)-len(vs)}
def jsonfile(name):
 p=root/name;return json.loads(p.read_text()) if p.exists() else None
summary={'runs':runs,'s3_phases':phases,'s3_repartition_audit':jsonfile('s3-reconfigured-audit.json'),'s4_boundary_exact':jsonfile('s4-boundary-compare.json')['overall_equal'],'s4_final_exact':jsonfile('s4-final-numerical.json')['overall_equal']}
for n in runs:
 for r in runs[n].values():r.pop('rows')
if len(phases)==3:summary['s3_observed_step_reduction_percent']=(1-phases['fault_after_repartition']['median_ms']/phases['fault_before_plan']['median_ms'])*100
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 双机实验摘要','','| 节点 | 实验 | 最后迭代 | 退出码 |','|---|---|---:|---:|']
for n,group in runs.items():
 for name,r in group.items():lines.append(f'| {n} | {name} | {r["last_iteration"]} | {r["exit_code"]} |')
lines+=['','## S3 单轮稳态耗时','','| 阶段 | 区间 | n | 中位数ms | p90ms |','|---|---|---:|---:|---:|']
for name,r in phases.items():lines.append(f'| {name} | {r["iterations"]} | {r["n"]} | {r["median_ms"]:.2f} | {r["p90_ms"]:.2f} |')
(root/'summary.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
