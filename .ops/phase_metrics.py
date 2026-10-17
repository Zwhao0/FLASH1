#!/usr/bin/env python3
import json,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); run=root/'dp-failslow'
rows={}
rx=re.compile(r'iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+)')
for p in (run/'trainlog').glob('*/megatron_output*.log'):
 for line in p.read_text(errors='replace').splitlines():
  m=rx.search(line)
  if m: rows[int(m[1])]=float(m[2])
events=[json.loads(s) for s in (run/'injection.jsonl').read_text().splitlines()]
get=lambda label:next(e for e in events if e.get('event')==label)
injection,plan,restore=(get(label) for label in ('inject','plan_observed','restore'))
i,p,r=(e['iteration'] for e in (injection,plan,restore))
bounds={'baseline':(50,i-5),'fault_unmitigated':(i+5,p-5),'fault_mitigated':(p+5,r-5),'clock_restored':(r+5,max(rows))}
base=statistics.median(v for k,v in rows.items() if 50<=k<=i-5)
threshold=max(2000,4*base)
phases={}
for name,(start,end) in bounds.items():
 selected={k:v for k,v in rows.items() if start<=k<=end}
 steady={k:v for k,v in selected.items() if v<=threshold}
 values=sorted(steady.values())
 pos=(len(values)-1)*0.9; lo=int(pos); hi=min(lo+1,len(values)-1)
 p90=values[lo]+(values[hi]-values[lo])*(pos-lo)
 phases[name]={'iterations':[start,end],'n':len(values),'median_ms':statistics.median(values),'p90_ms':p90,'excluded_long_intervals':{k:v for k,v in selected.items() if v>threshold}}
metrics={'method':'Iteration windows exclude 5 boundary iterations around inject/plan/restore; baseline starts at iteration 50. Steady-step medians exclude long intervals above max(4*baseline median,2000ms); original logs and full-range plot retain them. Single run, not a multi-run benchmark.','pause_threshold_ms':threshold,'events':{e['event']:e for e in (injection,plan,restore)},'phases':phases}
metrics['fault_increase_percent']=(phases['fault_unmitigated']['median_ms']/phases['baseline']['median_ms']-1)*100
metrics['mitigation_reduction_percent']=(1-phases['fault_mitigated']['median_ms']/phases['fault_unmitigated']['median_ms'])*100
(run/'phase-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
