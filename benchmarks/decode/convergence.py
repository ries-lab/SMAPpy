"""The four reference trainings, the original's two and the port's two: test loss, Jaccard and RMSE per epoch
and against hours, and the epochs each takes to come within 10, 5 and 2 % of its final test loss.

    python benchmarks/decode/convergence.py      # from the repository's root
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="benchmarks/decode/results/"
def orig(n):
    rows=[json.loads(l) for l in open(R+f"original_run{n}.jsonl")]
    t0=rows[0]["walltime"]
    return {"epoch":np.array([r["epoch"] for r in rows]),"loss":np.array([r.get("learning/test_ep",np.nan) for r in rows]),
            "jac":np.array([r.get("eval/jac",np.nan) for r in rows]),"rmse":np.array([r.get("eval/rmse_lat",np.nan) for r in rows]),
            "hours":(np.array([r["walltime"] for r in rows])-t0)/3600}
def port(n):
    rows=[json.loads(l) for l in open(R+f"port_run{n}_progress.jsonl")]; rows=[r for r in rows if r["kind"]=="epoch"]
    return {"epoch":np.array([r["epoch"] for r in rows]),"loss":np.array([r["loss_test"] for r in rows]),
            "jac":np.array([r["metrics"]["jaccard"] for r in rows]),"rmse":np.array([r["metrics"]["rmse_lat_nm"] for r in rows]),
            "hours":np.cumsum([r["seconds"] for r in rows])/3600}
runs={"original 1":orig(1),"original 2":orig(2),"port 0":port(0),"port 1":port(1)}
sty={"original 1":("C0","-"),"original 2":("C0","--"),"port 0":("C3","-"),"port 1":("C3","--")}
def smooth(y,k=9):
    y=np.array([np.nan if v is None else v for v in y],float); y=np.where(np.isfinite(y),y,np.nanmax(y[np.isfinite(y)]) if np.isfinite(y).any() else 0)
    return np.convolve(np.pad(y,(k//2,k//2),mode="edge"),np.ones(k)/k,mode="valid")
fig,ax=plt.subplots(2,2,figsize=(11,7.5))
for name,d in runs.items():
    c,ls=sty[name]
    ax[0,0].plot(d["epoch"],smooth(d["loss"]),c=c,ls=ls,label=name); ax[0,1].plot(d["epoch"],smooth(d["jac"]),c=c,ls=ls,label=name)
    ax[1,0].plot(d["hours"],smooth(d["loss"]),c=c,ls=ls,label=name); ax[1,1].plot(d["epoch"],smooth(d["rmse"]),c=c,ls=ls,label=name)
ax[0,0].set(xscale="log",yscale="log",xlabel="epoch",ylabel="test loss",title="test loss per epoch")
ax[0,1].set(xscale="log",xlabel="epoch",ylabel="Jaccard",title="Jaccard per epoch")
ax[1,0].set(xscale="log",yscale="log",xlabel="hours of training",ylabel="test loss",title="test loss against time (original: 4090, two at once; port: H100)")
ax[1,1].set(xscale="log",xlabel="epoch",ylabel="nm",title="lateral RMSE per epoch",ylim=(30,70))
for a in ax.flat: a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("benchmarks/decode/results/convergence.png",dpi=110)
# epochs to reach within 5% / 2% of the final (last-20 mean) test loss
print("epochs to reach a test loss within 10 % / 5 % / 2 % of the run's final, and the hours it took")
for name,d in runs.items():
    fin=np.nanmean(d["loss"][-20:]); s=smooth(d["loss"]); out=[]
    for f in (1.10,1.05,1.02):
        i=int(np.argmax(s<=fin*f)); out.append(f"{d['epoch'][i]:4d} ep {d['hours'][i]:5.2f} h")
    print(f"{name:11s} final {fin:6.1f}  "+"  |  ".join(out))
