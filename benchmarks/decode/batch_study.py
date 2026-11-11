"""(f2): batch 256 against batch 64 over 200 epochs -- the table, the epochs batch 64 needs for batch 256's loss, the figure.

    python benchmarks/decode/batch_study.py      # from the repository's root
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="benchmarks/decode/results/"
R="benchmarks/decode/results/"
def load(path):
    rows=[json.loads(l) for l in open(path)]
    restarts=sum(r["kind"]=="restart" for r in rows)
    ep={}
    for r in rows:
        if r["kind"]=="epoch": ep[r["epoch"]]=r          # after a restart, the last run's epoch wins
    e=sorted(ep)[:200]
    f=lambda k: np.array([np.nan if (ep[i]["metrics"].get(k) if k not in ("loss_test","seconds","lr") else ep[i][k]) is None
                          else (ep[i]["metrics"][k] if k not in ("loss_test","seconds","lr") else ep[i][k]) for i in e],float)
    return {"epoch":np.array(e),"loss":f("loss_test"),"jac":f("jaccard"),"rmse":f("rmse_lat_nm"),"rmse_ax":f("rmse_ax_nm"),
            "sec":f("seconds"),"lr":f("lr"),"restarts":restarts}
runs={"batch 64, seed 0":load(R+"port_run0_progress.jsonl"),"batch 64, seed 1":load(R+"port_run1_progress.jsonl"),
      "256, sqrt rate, no warm-up":load(S+"b256_sqrt_s0_progress.jsonl"),"256, linear rate, warm-up":load(S+"b256_linwarm_s0_progress.jsonl"),
      "256, sqrt rate, warm-up":load(S+"b256_sqrtwarm_s0_progress.jsonl"),
      "32, same rate (0.0006)":load(S+"b32_lr6_s0_progress.jsonl"),
      "32, sqrt rate (0.00042)":load(S+"b32_lr42_s0_progress.jsonl")}
def m(a,lo,hi): return float(np.nanmean(a[lo:hi]))
print(f"{'run':28s} {'restarts':>8s} {'loss@50':>8s} {'loss@100':>8s} {'loss 180-200':>12s} {'jac 180-200':>11s} {'rmse_lat':>8s} {'rmse_ax':>8s} {'s/epoch':>7s}")
for n,d in runs.items():
    print(f"{n:28s} {d['restarts']:8d} {m(d['loss'],45,55):8.1f} {m(d['loss'],95,105):8.1f} {m(d['loss'],180,200):12.1f} {m(d['jac'],180,200):11.3f} {m(d['rmse'],180,200):8.1f} {m(d['rmse_ax'],180,200):8.1f} {np.nanmedian(d['sec'][2:]):7.1f}")
ref=np.mean([m(runs[k]['loss'],180,200) for k in ("batch 64, seed 0","batch 64, seed 1")])
def sm(y,k=5): y=np.where(np.isfinite(y),y,np.nanmax(y)); return np.convolve(np.pad(y,(k//2,k//2),mode="edge"),np.ones(k)/k,mode="valid")
print("epochs to a test loss within 10 / 5 / 2 %% of batch 64's at 180-200 (%.1f)" % ref)
for n,d in runs.items():
    s=sm(d["loss"]); out=[]
    for f in (1.10,1.05,1.02):
        i=np.argmax(s<=ref*f); out.append(str(d["epoch"][i]) if s[i]<=ref*f else ">200")
    print(f"  {n:28s} "+" / ".join(out))
print("epoch at which each batch-64 run first reaches each batch-256 run's test loss at 180-200")
for n in [k for k in runs if not k.startswith("batch 64")]:
    target=m(runs[n]["loss"],180,200)
    hits=[int(runs[k]["epoch"][np.argmax(sm(runs[k]["loss"])<=target)]) for k in ("batch 64, seed 0","batch 64, seed 1")]
    print(f"  {n:28s} {target:6.1f}: epochs {hits}")
fig,ax=plt.subplots(1,3,figsize=(15,4.5))
for n,d in runs.items():
    ls="--" if n.startswith("batch 64") else ":" if n.startswith("32") else "-"
    ax[0].plot(d["epoch"],sm(d["loss"]),ls=ls,label=n); ax[1].plot(d["epoch"],sm(d["jac"]),ls=ls,label=n); ax[2].plot(d["epoch"],d["lr"],ls=ls,label=n)
ax[0].set(yscale="log",ylim=(250,2000),xlabel="epoch",title="test loss"); ax[1].set(xlabel="epoch",title="Jaccard")
ax[2].set(yscale="log",xlabel="epoch",title="learning rate trained with")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig("benchmarks/decode/results/batch_study.png",dpi=110)

print("the other way: the epoch at which each run first reaches batch 64's test loss at epochs 180-200")
for k in ("batch 64, seed 0", "batch 64, seed 1"):
    target = m(runs[k]["loss"], 180, 200)
    for n in [x for x in runs if not x.startswith("batch 64")]:
        s_ = sm(runs[n]["loss"]); i = int(np.argmax(s_ <= target))
        print(f"  {k} ({target:.1f}) <- {n:28s} " + (str(runs[n]["epoch"][i]) if s_[i] <= target else ">%d" % len(s_)))
