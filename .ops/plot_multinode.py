import json,sys,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
records=json.loads(pathlib.Path(sys.argv[1]).read_text())
out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[1,1.25]})
for label,color in [('source','#2a6789'),('resumed','#ce713e')]:
 rows=[r for r in records if r['phase']==label]
 x=np.array([r['iteration'] for r in rows]);y=np.array([r['ms'] for r in rows])
 axes[0].plot(x,y,color=color,lw=.9,label=label)
 ok=y<2000;axes[1].plot(x[ok],y[ok],color=color,lw=.9,label=label)
for ax in axes:
 ax.axvline(163,color='#a43138',ls='--',lw=1)
 ax.axvline(411,color='#624c86',ls='--',lw=1)
 ax.grid(alpha=.2);ax.set_ylabel('Iteration time (ms)')
 ax.spines[['top','right']].set_visible(False)
axes[0].set_yscale('log');axes[0].set_title('Two-node S3: link delay, measured PP plan, full-state restart',loc='left',fontweight='bold')
axes[0].legend(loc='upper right',frameon=False)
axes[1].set_ylim(150,650);axes[1].set_xlabel('Training iteration')
axes[1].text(169,620,'10 ms delay on rank 1 <-> 2',color='#a43138',fontsize=9)
axes[1].text(425,580,'PP layers [4,4,4,4] -> [5,3,3,5]\ncheckpoint at 411; resume with delay active',color='#624c86',fontsize=9)
axes[1].set_title('Steady iterations (values >= 2000 ms omitted from this panel)',loc='left',fontsize=10)
fig.text(.07,.015,'H800; 2 GPUs per node; TP1/DP1/PP4; mock data. Single-run mechanism test; restart wall time is not a step-time sample.',fontsize=8,color='#555')
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'s3-timeline.png',dpi=180);fig.savefig(out/'s3-timeline.svg');plt.close(fig)
