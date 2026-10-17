#!/usr/bin/env python3
import json,re,sys,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]); run=root/'dp-failslow'
pattern=re.compile(r'iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([\d.]+)')
values={}
for p in sorted((run/'trainlog').glob('*/megatron_output*.log')):
 for line in p.read_text(errors='replace').splitlines():
  m=pattern.search(line)
  if m: values[int(m[1])]=float(m[2])
xs=sorted(values); ys=[values[x] for x in xs]
rolling=[statistics.median(ys[max(0,i-4):i+1]) for i in range(len(ys))]
events=[json.loads(s) for s in (run/'injection.jsonl').read_text().splitlines()]
markers=[e for e in events if e.get('event') in ('inject','plan_observed','restore') and 'iteration' in e]
colors={'inject':'#d45444','plan_observed':'#1672b6','restore':'#218568','additional_plan':'#77579c'}
global_log='\n'.join(p.read_text(errors='replace') for p in (run/'trainlog').glob('*/global_controller*.log'))
plan_match=re.search(r'new DP plan is (\[[^\]]+\])',global_log,re.I)
plan_label=plan_match[1] if plan_match else 'new plan'
labels={'inject':'Clock limit','plan_observed':'DP '+plan_label+' observed','restore':'Clock reset'}
plan_lists=re.findall(r'new DP plan is (\[[^\]]+\])',global_log,re.I)
training_text='\n'.join(p.read_text(errors='replace') for p in (run/'trainlog').glob('*/megatron_output*.log'))
plan_iterations=[int(v) for v in re.findall(r'MicrobatchCalculator\] rank 0 DP Changed \(iter=(\d+)\)',training_text)]
for j,iteration in enumerate(plan_iterations[1:],1):
 markers.append({'event':'additional_plan','iteration':iteration,'label':'DP '+(plan_lists[j] if j<len(plan_lists) else 'updated')})
fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[1,2]})
axs[0].plot(xs,ys,color='#68798c',linewidth=0.8)
axs[0].set_ylabel('Full range (ms)'); axs[0].set_title('Greyhound on 4 H800 GPUs: computation fail-slow and DP mitigation',loc='left',pad=15,fontweight='bold')
axs[1].plot(xs,ys,color='#a7b4bf',alpha=0.65,linewidth=0.8,label='Per-iteration timing')
axs[1].plot(xs,rolling,color='#203c60',linewidth=1.8,label='Rolling median (5 iterations)')
ordinary=sorted(y for y in ys if y<2000)
axs[1].set_ylim(max(0,min(ordinary)*0.8),max(ordinary)*1.12)
axs[1].set_ylabel('Detail (ms)'); axs[1].set_xlabel('Training iteration')
for e in markers:
 for ax in axs:
  ax.axvline(e['iteration'],color=colors[e['event']],linestyle='--',linewidth=1.2)
 axs[1].plot([],[],color=colors[e['event']],linestyle='--',label=e.get('label',labels.get(e['event'],'DP update')))
for ax in axs:
 ax.grid(axis='y',alpha=0.2); ax.spines[['top','right']].set_visible(False)
axs[1].legend(loc='best',fontsize=8,frameon=True)
fig.text(0.08,0.015,'H=2048, layers=32, sequence=128; micro-batch=4; sum of DP micro-batches=32 (global batch=128).\nSingle-run artifact reproduction with mock data; full-range panel retains setup/validation pauses.',fontsize=8,color='#445469')
fig.tight_layout(rect=[0,0.065,1,1])
fig.savefig(run/'iteration-timeline.png',dpi=160)
fig.savefig(run/'iteration-timeline.svg')
