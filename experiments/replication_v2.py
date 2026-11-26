"""New, independently specified simulation; not the recovered historical implementation.
Run: python experiments/replication_v2.py --out data/recomputed
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def cycles(rng, n, tp, sl):
    # Independent signed Gaussian increments have the same law as Gaussian increments.
    x=np.zeros(n); length=np.zeros(n,dtype=int); active=np.arange(n)
    while active.size:
        x[active]+=rng.standard_normal(active.size)
        length[active]+=1
        active=active[(x[active]<tp)&(x[active]>-sl)]
    return x, length+1

def windows(rng,n,tp,sl,T=250):
    increments=rng.standard_normal((n,T))
    pos=np.zeros(n); flat=np.zeros(n,dtype=bool)
    counts=np.zeros(n,dtype=int); rewards=np.zeros((n,T))
    for j in range(T):
        held=~flat
        pos[held]+=increments[held,j]
        exit_=held&((pos>=tp)|(pos<=-sl))
        ix=np.flatnonzero(exit_)
        rewards[ix,counts[ix]]=pos[ix]
        counts[ix]+=1
        pos[ix]=0
        flat=exit_
    return rewards,counts

def stats(rewards,counts):
    den=np.maximum(counts,1)
    sums=rewards.sum(axis=1)
    variance=np.maximum(0,(np.square(rewards).sum(axis=1)-sums*sums/den)/np.maximum(counts-1,1))
    valid=(counts>=2)&(variance>1e-10)
    statistic=np.divide(sums,np.sqrt(den*variance),out=np.zeros_like(sums),where=valid)
    return valid,valid&(statistic>t.ppf(.95,np.maximum(counts-1,1)))

def run(out,n=10000,seed=20261126):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]; paths=[]
    for (tp,sl),ss in zip([(0.5,10),(3,3),(10,0.5)],np.random.SeedSequence(seed).spawn(3)):
        a,b,c=map(np.random.default_rng,ss.spawn(3))
        ra,k=windows(a,n,tp,sl)
        # Exact independent draws, not resampling a finite estimated pool.
        rx,_=cycles(b,int(k.sum()),tp,sl)
        re=np.zeros_like(ra)
        re[np.arange(ra.shape[1])[None,:]<k[:,None]]=rx
        va,ja=stats(ra,k);ve,je=stats(re,k)
        paths.append(pd.DataFrame(dict(tp=tp,sl=sl,window=np.arange(n),count=k,valid_A=va,valid_E=ve,reject_A=ja,reject_E=je,mean_A=ra.sum(axis=1)/np.maximum(k,1))))
        for m in [2,20,30]:
            eligible=k>=m; aa=eligible&va; ee=eligible&ve; joint=aa&ee
            d=ja[joint].astype(int)-je[joint].astype(int)
            # Paired bootstrap, resampling the empirical three-category distribution.
            if len(d):
                p=np.bincount(d+1,minlength=3)/len(d)
                bs=c.multinomial(len(d),p,size=20000)
                lo,hi=np.quantile((bs[:,2]-bs[:,0])/len(d),[.025,.975])
            else: lo=hi=np.nan
            rows.append(dict(tp=tp,sl=sl,m=m,total=n,eligible=int(eligible.sum()),valid_A=int(aa.sum()),valid_E=int(ee.sum()),joint_valid=int(joint.sum()),reject_A=int((aa&ja).sum()),reject_E=int((ee&je).sum()),A_only=int((joint&ja&~je).sum()),E_only=int((joint&~ja&je).sum()),delta=float(d.mean()) if len(d) else np.nan,delta_lo=lo,delta_hi=hi))
    df=pd.DataFrame(rows)
    for g in ["A","E"]:
        df[g+"_conditional"]=df["reject_"+g]/df["valid_"+g]
        df[g+"_overall"]=df["reject_"+g]/df.total
    df.to_csv(out/"summary.csv",index=False)
    pd.concat(paths,ignore_index=True).to_csv(out/"paths.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    q=df[(df.tp==.5)&(df.sl==10)]
    ax=q.plot(x="m",y=["A_conditional","E_conditional","A_overall","E_overall"],marker="o")
    ax.axhline(.05,linestyle=":",color="gray");ax.set(ylabel="Rejection frequency",xlabel="Minimum completed trades")
    ax.figure.tight_layout();ax.figure.savefig(out/"rejection_rates.png",dpi=180);plt.close(ax.figure)
    return df
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",default="data/recomputed");p.add_argument("--n",type=int,default=10000);p.add_argument("--seed",type=int,default=20261126)
    args=p.parse_args();print(run(args.out,args.n,args.seed).to_string(index=False))
