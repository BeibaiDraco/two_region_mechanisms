#!/usr/bin/env python
"""Task-specific mechanism profiling -- v2.

Fixes: strict temporal precedence, unique D/M contributions,
robust onset detection, task-aligned questions for context/state-setting.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mechanism_classification import (
    estimate_realized_effect, estimate_direct_forcing, estimate_modulation)
from src.analysis.eval_cache import load_or_create_eval_batch
from src.analysis.readouts import default_readout
from src.analysis.rollout import load_run_artifacts
from src.analysis.task_profiles import (
    epoch_indices, build_epoch_masks, extract_all_trials, trace_band,
    unique_contribution, strict_lag_prediction, condition_traces,
    median_split_within, windowed_mean, robust_divergence_time,
    context_selectivity_index, matched_input_context_comparison,
    mode_conditioned_traces, difficulty_conditioned_effect)
from src.utils.io import save_json

EC = {"fixation":"#bdbdbd","cue":"#c7b8ea","stimulus":"#aec7e8",
      "signal":"#aec7e8","delay":"#ffbb78","response":"#98df8a"}

def collect_data(run_dir, n_trials=256, eval_cache=None, refresh_eval_cache=False):
    cfg,task,model,_ = load_run_artifacts(run_dir,checkpoint_name="best.pt",map_location="cpu")
    batch = load_or_create_eval_batch(
        task,
        n_trials=n_trials,
        device="cpu",
        split="val",
        cache_path=eval_cache,
        refresh=refresh_eval_cache,
    )
    model.eval()
    with torch.no_grad():
        coupled = model(batch.inputs,intervention=None,add_noise=False)
        decoupled = model(batch.inputs,intervention={"type":"decouple"},add_noise=False)
    effect = estimate_realized_effect(coupled,decoupled)
    direct = estimate_direct_forcing(model,batch,coupled)
    modulation = estimate_modulation(model,batch,coupled,decoupled)
    qc = default_readout(coupled).cpu().numpy()
    qd = default_readout(decoupled).cpu().numpy()
    return {"cfg":cfg,"task":task,"batch":batch,
            "q":qc.T,"q_dec":qd.T,"d":direct["trace"].T,
            "m":modulation["trace"].T,"r":effect["trace"].T,
            "abs_q":np.abs(qc.T),"epochs":task.epochs,
            "meta":{k:v.cpu().numpy() for k,v in batch.meta.items()}}

def _shade(ax, ep_idx):
    for n,idx in ep_idx.items():
        if len(idx)>0: ax.axvspan(idx[0],idx[-1]+1,color=EC.get(n,"#eee"),alpha=.12)

def trace_fig(data, ep_idx, path, title, masks=None, labels=None):
    time=np.arange(data["q"].shape[1])
    fig,axes=plt.subplots(4,1,figsize=(8,9),sharex=True)
    for ax,(arr,yl) in zip(axes,[(data["abs_q"],"|q|"),(data["d"],"|D|"),
                                  (data["m"],"|M|"),(data["r"],"|dq|")]):
        _shade(ax,ep_idx)
        if masks:
            cs=["#d62728","#1f77b4","#2ca02c","#ff7f0e"]
            for i,(mk,lb) in enumerate(zip(masks,labels)):
                if mk.sum()==0: continue
                mn,p25,p75=trace_band(arr[mk])
                ax.fill_between(time,p25,p75,color=cs[i%4],alpha=.1)
                ax.plot(time,mn,color=cs[i%4],lw=1.6,label=lb)
        else:
            mn,p25,p75=trace_band(arr)
            ax.fill_between(time,p25,p75,color="#ccc",alpha=.4)
            ax.plot(time,mn,color="k",lw=1.8)
        ax.set_ylabel(yl,fontsize=10)
        if ax==axes[0] and labels: ax.legend(fontsize=7,loc="upper left")
    axes[-1].set_xlabel("timestep"); fig.suptitle(title,fontsize=11)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)

# === COHERENCE TASKS (binary, delayed, redundant) ===

def analyze_coherence(data, out_dir, name, primary_ep, outcome_ep):
    meta=data["meta"]; coh=meta["coherence"]; diff=np.abs(coh)
    N,T=data["q"].shape; epochs=data["epochs"]; ep=epoch_indices(epochs)
    fd=out_dir/"figures"; fd.mkdir(parents=True,exist_ok=True)
    bl=ep[list(epochs.keys())[0]]
    oi=ep[outcome_ep]; pi=ep[primary_ep]
    faq=np.abs(data["q"][:,oi].mean(axis=1))
    feat=extract_all_trials(data["d"],data["m"],data["abs_q"],ep,list(epochs.keys()),
                             {**meta,"difficulty":diff},faq,bl)
    pd.DataFrame(feat).to_csv(out_dir/f"{name}_trial_features.csv",index=False)
    trace_fig(data,ep,fd/f"{name}.png",name)
    med=np.median(faq); hi=faq>=med; lo=~hi
    trace_fig(data,ep,fd/f"{name}_hilo.png",f"{name}: hi vs lo |q_f|",
              [hi,lo],["high |q_f|","low |q_f|"])
    strict=epochs[primary_ep][1]<=epochs[outcome_ep][0]
    dp=windowed_mean(data["d"],pi); mp=windowed_mean(data["m"],pi)
    qo=windowed_mean(data["abs_q"],oi)
    uc=unique_contribution(dp,mp,qo,diff)
    vd=np.array([r["d_onset"] for r in feat]); vd=vd[vd>=0]
    vm=np.array([r["m_onset"] for r in feat]); vm=vm[vm>=0]
    vq=np.array([r["q_onset"] for r in feat]); vq=vq[vq>=0]
    s={"experiment":name,"task":data["cfg"]["task"]["name"],"n":N,
       "primary":primary_ep,"outcome":outcome_ep,"strict":strict,
       "d_onset":round(float(vd.mean()),1) if len(vd) else -1,
       "m_onset":round(float(vm.mean()),1) if len(vm) else -1,
       "q_onset":round(float(vq.mean()),1) if len(vq) else -1,
       **{f"uc_{k}":v for k,v in uc.items()}}
    save_json(s,out_dir/"metrics"/f"{name}_profile.json")
    tag="STRICT" if strict else "OVERLAP"
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  Onset: D={s['d_onset']}  M={s['m_onset']}  q={s['q_onset']}")
    print(f"  {primary_ep} -> {outcome_ep} ({tag})")
    print(f"  D unique (ctrl M+diff): {uc['d_unique_of_m']}")
    print(f"  M unique (ctrl D+diff): {uc['m_unique_of_d']}")
    print(f"  Raw: D={uc['raw_d']}  M={uc['raw_m']}")
    return uc

def analyze_binary(data, out_dir):
    analyze_coherence(data,out_dir,out_dir.name,"stimulus","response")

def analyze_delayed(data, out_dir):
    analyze_coherence(data,out_dir,out_dir.name,"stimulus","response")
    ep=epoch_indices(data["epochs"]); diff=np.abs(data["meta"]["coherence"])
    # delay->response (strict)
    dd=windowed_mean(data["d"],ep["delay"]); md=windowed_mean(data["m"],ep["delay"])
    qr=windowed_mean(data["abs_q"],ep["response"])
    uc2=unique_contribution(dd,md,qr,diff)
    print(f"  --- delay->response (STRICT) ---")
    print(f"  D unique: {uc2['d_unique_of_m']}  M unique: {uc2['m_unique_of_d']}")
    # stimulus->late_delay (strict)
    di=ep["delay"]; late_d=di[len(di)//2:]
    ds=windowed_mean(data["d"],ep["stimulus"]); ms=windowed_mean(data["m"],ep["stimulus"])
    qld=windowed_mean(data["abs_q"],late_d)
    uc3=unique_contribution(ds,ms,qld,diff)
    print(f"  --- stimulus->late_delay (STRICT) ---")
    print(f"  D unique: {uc3['d_unique_of_m']}  M unique: {uc3['m_unique_of_d']}")

def analyze_redundant(data, out_dir):
    analyze_coherence(data,out_dir,out_dir.name,"stimulus","response")
    ep=epoch_indices(data["epochs"]); diff=np.abs(data["meta"]["coherence"])
    ct=difficulty_conditioned_effect(data["q"],data["q_dec"],diff,ep["response"])
    pd.DataFrame(ct).to_csv(out_dir/"redundant_lowrank_decoupling_cost.csv",index=False)
    print(f"  --- Decoupling cost by difficulty ---")
    for r in ct:
        print(f"    coh={r['difficulty']:.1f}: cost={r['decoupling_cost']:.4f} "
              f"(rel={r['relative_cost']:.4f}, n={r['n']})")

# === CONTEXT TASK ===

def analyze_context(data, out_dir):
    name=out_dir.name; meta=data["meta"]
    ctx=meta["context"]; c1=meta["coh1"]; c2=meta["coh2"]
    rel=np.where(ctx==0,c1,c2); irrel=np.where(ctx==0,c2,c1); diff=np.abs(rel)
    N,T=data["q"].shape; epochs=data["epochs"]; ep=epoch_indices(epochs)
    fd=out_dir/"figures"; fd.mkdir(parents=True,exist_ok=True)
    bl=ep.get("cue",ep[list(epochs.keys())[0]])[:3]

    trace_fig(data,ep,fd/f"{name}.png",name)
    trace_fig(data,ep,fd/f"{name}_ctx.png",f"{name}: by context",
              [ctx==0,ctx==1],["ctx=0","ctx=1"])

    print(f"\n{'='*60}\n  {name} -- Context Routing\n{'='*60}")
    print(f"  Epoch selectivity (|mean_ctx0 - mean_ctx1| / sum):")
    for en in epochs:
        idx=ep[en]
        if len(idx)==0: continue
        sq=context_selectivity_index(data["abs_q"],ctx,idx)
        sm=context_selectivity_index(data["m"],ctx,idx)
        sd=context_selectivity_index(data["d"],ctx,idx)
        print(f"    {en:12s}  q={sq:.4f}  M={sm:.4f}  D={sd:.4f}")

    print(f"\n  Matched relevant-coherence, different context (M traces):")
    mc=matched_input_context_comparison(data["m"],ctx,rel,n_bins=4)
    for bi,info in mc.items():
        lo,hi=info["coh_range"]
        if "stimulus" in ep:
            c0s=info["ctx0_mean"][ep["stimulus"]].mean()
            c1s=info["ctx1_mean"][ep["stimulus"]].mean()
            print(f"    |rel_coh|[{lo:.2f},{hi:.2f}]: ctx0_stim_M={c0s:.4f} ctx1_stim_M={c1s:.4f}")

    c0m=data["m"][ctx==0].mean(axis=0); c1m=data["m"][ctx==1].mean(axis=0)
    c0q=data["abs_q"][ctx==0].mean(axis=0); c1q=data["abs_q"][ctx==1].mean(axis=0)
    mdiv=robust_divergence_time(c0m,c1m,bl); qdiv=robust_divergence_time(c0q,c1q,bl)
    print(f"\n  Context divergence (robust): M={mdiv}, q={qdiv}")
    if mdiv>=0 and qdiv>=0:
        if mdiv<qdiv: print(f"  M leads q by {qdiv-mdiv} steps")
        else: print(f"  q leads M by {mdiv-qdiv} steps")

    if "response" in ep:
        sc=context_selectivity_index(data["abs_q"],ctx,ep["response"])
        sd=context_selectivity_index(np.abs(data["q_dec"]),ctx,ep["response"])
        print(f"\n  Response context selectivity: coupled={sc:.4f} decoupled={sd:.4f}")
        print(f"  Decoupling reduces selectivity by {sc-sd:.4f}")

    save_json({"experiment":name,"n":N,"m_div":mdiv,"q_div":qdiv},
              out_dir/"metrics"/f"{name}_profile.json")

# === STATE-SETTING TASK ===

def analyze_state_setting(data, out_dir):
    name=out_dir.name; meta=data["meta"]
    mode=meta["mode"]; mn={0:"integrate",1:"memory",2:"transient"}
    N,T=data["q"].shape; epochs=data["epochs"]; ep=epoch_indices(epochs)
    fd=out_dir/"figures"; fd.mkdir(parents=True,exist_ok=True)
    bl=ep.get("cue",ep[list(epochs.keys())[0]])[:3]

    trace_fig(data,ep,fd/f"{name}.png",name)
    trace_fig(data,ep,fd/f"{name}_mode.png",f"{name}: by mode",
              [mode==i for i in range(3)],[mn[i] for i in range(3)])

    print(f"\n{'='*60}\n  {name} -- Regime Setting\n{'='*60}")

    print(f"  Pairwise mode divergence (robust):")
    for i in range(3):
        for j in range(i+1,3):
            mi=data["m"][mode==i].mean(axis=0); mj=data["m"][mode==j].mean(axis=0)
            qi=data["abs_q"][mode==i].mean(axis=0); qj=data["abs_q"][mode==j].mean(axis=0)
            md=robust_divergence_time(mi,mj,bl); qd=robust_divergence_time(qi,qj,bl)
            lead=""
            if md>=0 and qd>=0 and md<qd: lead=f" (M leads by {qd-md})"
            print(f"    {mn[i]} vs {mn[j]}: M_div={md}, q_div={qd}{lead}")

    print(f"\n  Mode selectivity per epoch:")
    for en in epochs:
        idx=ep[en]
        if len(idx)==0: continue
        mm=[data["m"][mode==i][:,idx].mean() for i in range(3)]
        mq=[data["abs_q"][mode==i][:,idx].mean() for i in range(3)]
        print(f"    {en:12s}: M_var={np.var(mm):.6f}  q_var={np.var(mq):.6f}")

    if "response" in ep:
        ri=ep["response"]
        cb=[np.abs(data["q"][mode==i][:,ri]).mean() for i in range(3)]
        db=[np.abs(data["q_dec"][mode==i][:,ri]).mean() for i in range(3)]
        print(f"\n  Response |q| by mode:  {'mode':>12s} {'coupled':>8s} {'decoupled':>8s} {'cost':>8s}")
        for i in range(3):
            print(f"    {mn[i]:>12s} {cb[i]:8.4f} {db[i]:8.4f} {cb[i]-db[i]:8.4f}")
        sc=np.std(cb); sd=np.std(db)
        print(f"  Mode spread: coupled={sc:.4f} decoupled={sd:.4f} reduction={sc-sd:.4f}")

    save_json({"experiment":name,"n":N},out_dir/"metrics"/f"{name}_profile.json")

# === MAIN ===

DISPATCH={"BinaryCategorizationTask":analyze_binary,
          "DelayedCategorizationTask":analyze_delayed,
          "ContextDependentDecisionTask":analyze_context,
          "StateSettingTask":analyze_state_setting,
          "RedundantInputControlTask":analyze_redundant}
ALL=["outputs/experiments/binary_additive","outputs/experiments/delayed_gated",
     "outputs/experiments/context_gated","outputs/experiments/state_setting_reciprocal",
     "outputs/experiments/redundant_lowrank"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=str,default=None)
    p.add_argument("--all",action="store_true")
    p.add_argument("--n-trials",type=int,default=256)
    p.add_argument("--eval-cache",type=str,default=None)
    p.add_argument("--refresh-eval-cache",action="store_true")
    a=p.parse_args()
    dirs=ALL if a.all else ([a.run_dir] if a.run_dir else [])
    for rd in dirs:
        d=collect_data(rd,a.n_trials,a.eval_cache,a.refresh_eval_cache)
        fn=DISPATCH.get(d["cfg"]["task"]["name"])
        if fn: fn(d,Path(rd))

if __name__=="__main__": main()
