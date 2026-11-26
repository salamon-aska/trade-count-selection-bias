"""Publication rendering of existing summary.csv; no new simulation.
Run: python make_submission_figure.py summary.csv output_directory
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

summary=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(summary); q=d[(d.tp==0.5)&(d.sl==10)].sort_values('m')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(7.5,3.3),sharey=True,layout='constrained')
for ax,suf,title in zip(axs,['conditional','overall'],['(a) Among eligible windows','(b) Across all windows']):
    ax.plot(q.m,q['A_'+suf],color='#173F66',marker='o',linestyle='-',lw=1.9,ms=5,label='A: completed trades')
    ax.plot(q.m,q['E_'+suf],color='#A24C21',marker='s',linestyle='--',lw=1.9,ms=5,label='E: count-matched control')
    ax.axhline(.05,color='#555555',ls=':',lw=1.1,label='Nominal 5% reference')
    ax.set_title(title,loc='left',fontweight='bold'); ax.set_xlabel('Minimum completed trades'); ax.set_xticks(q.m); ax.set_ylim(0,.55);ax.set_yticks([0,.1,.2,.3,.4,.5]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(axis='y',color='#e0e0e0',linewidth=.5);ax.set_axisbelow(True)
axs[0].set_ylabel('Rejection frequency')
axs[0].legend(loc='upper left',frameon=False,fontsize=8,handlelength=2.5)
for ext,dpi in [('tiff',800),('png',300),('svg',None)]:
    kw={} if dpi is None else {'dpi':dpi}
    if ext=='tiff': kw['pil_kwargs']={'compression':'tiff_lzw'}
    fig.savefig(out/f'Figure_1.{ext}',**kw)
plt.close(fig)
