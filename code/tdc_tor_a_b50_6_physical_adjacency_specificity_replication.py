#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDC TOR A / B50.6 — Physical Adjacency Specificity Replication

Confirmatory matched-control test after B50.5
REPRESENTATION_STABLE_BUT_ORDER_NULL_COMPATIBLE.

Question:
For fixed source state s_i and fixed prior physical context s_{i-1}->s_i,
is the actual next state s_{i+1} special relative to non-adjacent states s_j
matched on full-operator pair distance?

No B59/B59.1 data are read or used.
"""

from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "B50.6_physical_adjacency_specificity_replication_v1"

CHANNELS = [
    "speed", "acceleration", "turn_change",
    "density_scale_change", "diffusion_step",
]

TH = {
    "match_median_max": 0.10,
    "match_q90_max": 0.25,
    "primary_p_max": 0.01,
    "primary_abs_effect_min": 0.05,
    "config_direction_agreement_min": 0.80,
    "third_p_max": 0.05,
    "third_abs_effect_min": 0.04,
    "third_same_direction_required": 3,
    "third_nominal_required": 2,
}


def now_s():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, obj: Any):
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True))


def atomic_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def safe_spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) != len(b) or len(a) < 3 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def validate_state(df):
    req = {"range_name", "range_index"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"state dataset missing {sorted(miss)}")
    idx = df["range_index"].astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(len(df))):
        raise ValueError("range_index is not exactly 0..n-1")
    if df["range_name"].duplicated().any():
        raise ValueError("duplicate range_name")


def import_b504(path: Path):
    spec = importlib.util.spec_from_file_location("b504_frozen_b506", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("full_pca", "build_knn_graph", "diffusion_modes"):
        if not hasattr(mod, name):
            raise AttributeError(f"B50.4 missing {name}")
    front = None
    for name in ("transition_front_features", "front_features"):
        if hasattr(mod, name):
            front = getattr(mod, name)
            break
    if front is None:
        raise AttributeError("B50.4 has no front_features-compatible function")
    return mod, front


def unpack_graph(x):
    if not isinstance(x, tuple):
        raise TypeError("build_knn_graph did not return tuple")
    if len(x) == 4:
        W, _, _, radius = x
    elif len(x) == 3:
        W, _, radius = x
    else:
        raise RuntimeError(f"unsupported build_knn_graph return length {len(x)}")
    return W, np.asarray(radius, float)


def call_front(fn, X, radius, diff):
    attempts = [
        lambda: fn(X, radius, diff),
        lambda: fn(X=X, local_radius=radius, diffusion_coords=diff),
    ]
    errs = []
    for a in attempts:
        try:
            z = a()
            if isinstance(z, pd.DataFrame):
                return z
        except TypeError as e:
            errs.append(str(e))
    raise RuntimeError("cannot call frozen B50.4 front function: " + " | ".join(errs))


def reconstruct_configs(mod, front_fn, P, dims, graph_k, n_modes):
    scores, _, _ = mod.full_pca(P)
    out = {}
    for d in dims:
        X = np.asarray(scores[:, :d], float)
        D = squareform(pdist(X))
        for gk in graph_k:
            cid = f"d{d}_knn{gk}"
            W, radius = unpack_graph(mod.build_knn_graph(D, gk))
            eig, modes = mod.diffusion_modes(W, n_modes)
            eig = np.asarray(eig, float); modes = np.asarray(modes, float)
            diff = modes * eig[None, :]
            fr = call_front(front_fn, X, radius, diff)
            out[cid] = dict(
                dimension=d, graph_k=gk, X=X,
                radius=radius, diff=diff, fronts=fr
            )
    return out


def reproduction_gate(configs, saved):
    rows = []
    for cid, c in sorted(configs.items()):
        a = c["fronts"].sort_values("transition_index")["front_score"].to_numpy(float)
        b = (saved[saved["config_id"].astype(str) == cid]
             .sort_values("transition_index")["front_score"].to_numpy(float))
        if len(a) != len(b):
            rows.append(dict(config_id=cid, max_abs_error=float("inf"), spearman=float("nan"), pass_=False))
            continue
        err = np.abs(a-b)
        rho = safe_spearman(a,b)
        ok = bool(np.max(err) <= 1e-8 and np.isfinite(rho) and rho >= .999999)
        rows.append(dict(config_id=cid, max_abs_error=float(np.max(err)),
                         mean_abs_error=float(np.mean(err)), spearman=rho, pass_=ok))
    df = pd.DataFrame(rows)
    return bool(df["pass_"].all()), df


def frozen_scale(fr):
    out = {}
    for c in CHANNELS:
        x = fr[c].to_numpy(float)
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x-med)))
        s = 1.4826 * mad
        if not np.isfinite(s) or s < 1e-12:
            s = float(np.nanstd(x))
        if not np.isfinite(s) or s < 1e-12:
            s = 0.0
        out[c] = (med, s)
    return out


def score_raw(raw, scale):
    Z = []
    for c in CHANNELS:
        x = np.asarray(raw[c], float)
        med, s = scale[c]
        if s <= 1e-15:
            z = np.zeros_like(x)
        else:
            z = (x-med)/s
            z[~np.isfinite(z)] = 0
            z = np.maximum(z, 0)
        Z.append(z)
    Z = np.column_stack(Z)
    return np.sqrt(np.mean(Z*Z, axis=1))


def candidate_raw(X, radius, diff, i, dest):
    dest = np.asarray(dest, int)
    prev = X[i] - X[i-1]
    v = X[dest] - X[i]
    speed = np.linalg.norm(v, axis=1)
    acceleration = np.linalg.norm(v-prev[None,:], axis=1)
    den = np.linalg.norm(prev) * speed
    cos = np.ones(len(dest))
    good = den > 1e-15
    if np.any(good):
        cos[good] = (v[good] @ prev) / den[good]
    turn = 1 - np.clip(cos, -1, 1)
    density = np.abs(np.log(np.maximum(radius[dest],1e-15)) - math.log(max(float(radius[i]),1e-15)))
    diffusion = np.linalg.norm(diff[dest]-diff[i][None,:], axis=1) if diff.shape[1] else np.zeros(len(dest))
    return dict(speed=speed, acceleration=acceleration, turn_change=turn,
                density_scale_change=density, diffusion_step=diffusion)


def parity_audit(front_fn, configs, pairs):
    rows = []
    for cid, c in sorted(configs.items()):
        for i,j in pairs:
            raw = candidate_raw(c["X"], c["radius"], c["diff"], i, [j])
            idx = np.asarray([i-1,i,j])
            exact = call_front(front_fn, c["X"][idx], c["radius"][idx], c["diff"][idx]).sort_values("transition_index").iloc[-1]
            errs = {ch: abs(float(raw[ch][0])-float(exact[ch])) for ch in CHANNELS}
            mx = max(errs.values())
            rows.append(dict(config_id=cid, source_i=i, dest_j=j, max_abs_error=mx,
                             **{f"{k}_abs_error":v for k,v in errs.items()}, pass_=mx<=1e-10))
    df = pd.DataFrame(rows)
    return bool(df["pass_"].all()), df


def build_controls(P, m, min_sep):
    D = squareform(pdist(np.asarray(P,float)))
    n = len(P)
    cmap, rows = {}, []
    for i in range(1,n-1):
        phys = i+1
        d0 = float(D[i,phys])
        cand = np.arange(n)
        cand = cand[(np.abs(cand-i) >= min_sep) & (cand != phys)]
        if len(cand) < m:
            raise RuntimeError(f"not enough controls for transition {i}")
        dc = D[i,cand]
        err = np.abs(np.log(np.maximum(dc,1e-15))-math.log(max(d0,1e-15)))
        ord_ = np.lexsort((cand,err))[:m]
        chosen = cand[ord_]
        cmap[i] = chosen.astype(int)
        for r,p in enumerate(ord_,1):
            rows.append(dict(
                transition_index=i, source_index=i, physical_dest_index=phys,
                control_rank=r, control_dest_index=int(cand[p]),
                source_control_index_separation=int(abs(int(cand[p])-i)),
                physical_distance_full_operator=d0,
                control_distance_full_operator=float(dc[p]),
                abs_log_distance_error=float(err[p]),
            ))
    return cmap, pd.DataFrame(rows)


def rank_percentiles(x):
    x = np.asarray(x,float); n=len(x)
    out = np.empty(n)
    for i in range(n):
        less = np.sum(x < x[i]); ties = np.sum(x == x[i]) - 1
        out[i] = (less + .5*ties) / max(n-1,1)
    return out


def build_cube(configs, saved, cmap):
    tis = sorted(cmap); cids = sorted(configs)
    m = len(next(iter(cmap.values())))
    pct = np.empty((len(tis),len(cids),m+1))
    scores = np.empty_like(pct)
    cfgrows = []

    for ci,cid in enumerate(cids):
        c = configs[cid]
        fr = saved[saved["config_id"].astype(str)==cid].sort_values("transition_index")
        scale = frozen_scale(fr)
        phys_lookup = dict(zip(fr["transition_index"].astype(int), fr["front_score"].astype(float)))
        pp = []

        for ti,i in enumerate(tis):
            dest = np.concatenate([[i+1], cmap[i]])
            raw = candidate_raw(c["X"],c["radius"],c["diff"],i,dest)
            sc = score_raw(raw,scale)
            if abs(float(sc[0])-float(phys_lookup[i])) > 1e-8:
                raise RuntimeError(f"physical score mismatch {cid} transition {i}")
            rp = rank_percentiles(sc)
            pct[ti,ci,:] = rp; scores[ti,ci,:] = sc; pp.append(float(rp[0]))

        cfgrows.append(dict(
            config_id=cid, dimension=c["dimension"], graph_k=c["graph_k"],
            mean_physical_percentile=float(np.mean(pp)),
            median_physical_percentile=float(np.median(pp)),
            mean_percentile_effect=float(np.mean(pp)-.5),
        ))
    return tis,cids,pct,scores,pd.DataFrame(cfgrows)


def matched_null(consensus_pct, reps, seed, mask=None):
    X = np.asarray(consensus_pct,float)
    if mask is not None:
        X = X[np.asarray(mask,bool)]
    obs = X[:,0]
    effect = float(np.mean(obs)-.5)
    rng = np.random.default_rng(seed)
    rows = np.arange(len(X))
    null = np.empty(reps)
    for r in range(reps):
        choice = rng.integers(0,X.shape[1],size=len(X))
        null[r] = float(np.mean(X[rows,choice])-.5)
    p = float((1+np.sum(np.abs(null)>=abs(effect)))/(reps+1))
    return dict(
        n_transitions=len(X),
        observed_mean_percentile=float(np.mean(obs)),
        observed_median_percentile=float(np.median(obs)),
        observed_mean_effect=effect,
        null_mean_effect=float(np.mean(null)),
        null_sd_effect=float(np.std(null)),
        p_two_sided=p,
    ), null


def run(args):
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    paths = {
        "operator_stack":Path(args.operator_stack),
        "b50_2_summary":Path(args.b50_2_summary),
        "b50_4_protocol":Path(args.b50_4_protocol),
        "b50_4_summary":Path(args.b50_4_summary),
        "b50_4_front_features":Path(args.b50_4_front_features),
        "b50_5_summary":Path(args.b50_5_summary),
        "state_dataset":Path(args.state_dataset),
        "b50_4_script":Path(args.b50_4_script),
    }
    for k,p in paths.items():
        if not p.exists(): raise FileNotFoundError(f"{k}: {p}")

    b502=json.loads(paths["b50_2_summary"].read_text(encoding="utf-8"))
    p4=json.loads(paths["b50_4_protocol"].read_text(encoding="utf-8"))
    s4=json.loads(paths["b50_4_summary"].read_text(encoding="utf-8"))
    s5=json.loads(paths["b50_5_summary"].read_text(encoding="utf-8"))
    saved=pd.read_csv(paths["b50_4_front_features"])
    state=pd.read_csv(paths["state_dataset"]); validate_state(state)
    with np.load(paths["operator_stack"],allow_pickle=False) as z:
        P=np.asarray(z["P_stack"],float); labels=[str(x) for x in z["labels"]]

    labels_ok = labels == state["range_name"].astype(str).tolist()
    b502_ok = bool(b502.get("reproduction",{}).get("reproduction_ok",False))
    b504_ok = bool(s4.get("integrity",{}).get("labels_exact",False)
                   and s4.get("integrity",{}).get("shape_exact",False)
                   and s4.get("integrity",{}).get("b50_2_reproduction_ok",False))
    b505_ok = bool(s5.get("reproduction_pass",False))
    dims=[int(x) for x in p4["dimensions"]]
    graph_k=[int(x) for x in p4["graph_k_values"]]
    n_modes=int(p4["diffusion_modes"])

    print("=== TDC TOR A / B50.6: PHYSICAL ADJACENCY SPECIFICITY ===")
    print("operator stack       :",P.shape)
    print("labels exact         :",labels_ok)
    print("B50.2 reproduction   :",b502_ok)
    print("B50.4 integrity      :",b504_ok)
    print("B50.5 reproduction   :",b505_ok)
    print("B50.5 verdict        :",s5.get("final_verdict"))
    print("dimensions           :",dims)
    print("graph k              :",graph_k)

    if not (labels_ok and b502_ok and b504_ok and b505_ok):
        summary={"version":VERSION,"final_verdict":"INVALID_REPRODUCTION","reason":"Input lineage gate failed."}
        atomic_json(out/"b50_6_summary.json",summary); atomic_text(out/"b50_6_verdict.txt","FINAL VERDICT: INVALID_REPRODUCTION\n")
        return summary

    atomic_json(out/"b50_6_protocol_manifest.json",{
        "version":VERSION,"created":now_s(),"outcome_blind_to_b59":True,
        "inputs":{k:{"path":str(v),"sha256":sha256_file(v)} for k,v in paths.items()},
        "frozen_b50_4":{"dimensions":dims,"graph_k_values":graph_k,"diffusion_modes":n_modes},
        "matching":{"geometry":"full 4096D B50 operator Euclidean",
                    "controls_per_transition":args.controls_per_transition,
                    "min_index_separation":args.min_index_separation},
        "randomization_replicates":args.randomization_replicates,
        "thresholds":TH,"seed":args.seed,
    })

    print("[0/6] Frozen B50.4 reconstruction",flush=True)
    mod,front_fn=import_b504(paths["b50_4_script"])
    configs=reconstruct_configs(mod,front_fn,P,dims,graph_k,n_modes)
    repro_ok,repro_df=reproduction_gate(configs,saved)
    atomic_csv(out/"b50_6_reproduction_audit.csv",repro_df)
    print("      reproduction PASS:",repro_ok)
    if not repro_ok:
        summary={"version":VERSION,"final_verdict":"INVALID_REPRODUCTION","reason":"Frozen B50.4 front reproduction failed."}
        atomic_json(out/"b50_6_summary.json",summary); atomic_text(out/"b50_6_verdict.txt","FINAL VERDICT: INVALID_REPRODUCTION\n")
        return summary

    print("[1/6] Full-operator distance matching",flush=True)
    cmap,matchdf=build_controls(P,args.controls_per_transition,args.min_index_separation)
    atomic_csv(out/"b50_6_matching_audit.csv",matchdf)
    med=float(matchdf["abs_log_distance_error"].median())
    q90=float(matchdf["abs_log_distance_error"].quantile(.90))
    match_ok=bool(med<=TH["match_median_max"] and q90<=TH["match_q90_max"])
    print(f"      match median={med:.6f} q90={q90:.6f} PASS={match_ok}")
    if not match_ok:
        summary={"version":VERSION,"matching":{"median":med,"q90":q90,"pass":False},
                 "final_verdict":"INVALID_MATCHING_GEOMETRY","reason":"Matched controls fail frozen distance-quality gates."}
        atomic_json(out/"b50_6_summary.json",summary); atomic_text(out/"b50_6_verdict.txt","FINAL VERDICT: INVALID_MATCHING_GEOMETRY\n")
        return summary

    print("[2/6] Arbitrary-pair raw-feature parity",flush=True)
    tis=sorted(cmap)
    positions=np.linspace(0,len(tis)-1,min(5,len(tis)),dtype=int)
    pairs=[(tis[int(p)],int(cmap[tis[int(p)]][0])) for p in positions]
    parity_ok,parity_df=parity_audit(front_fn,configs,pairs)
    atomic_csv(out/"b50_6_raw_feature_parity.csv",parity_df)
    print("      parity PASS:",parity_ok)
    if not parity_ok:
        summary={"version":VERSION,"final_verdict":"INVALID_REPRODUCTION","reason":"Arbitrary-pair raw-feature parity failed."}
        atomic_json(out/"b50_6_summary.json",summary); atomic_text(out/"b50_6_verdict.txt","FINAL VERDICT: INVALID_REPRODUCTION\n")
        return summary

    print("[3/6] Physical vs matched-control scoring",flush=True)
    tis,cids,pct,score,cfgdf=build_cube(configs,saved,cmap)
    consensus=np.median(pct,axis=1)
    phys=consensus[:,0]
    trows=[]
    for ti,i in enumerate(tis):
        trows.append(dict(
            transition_index=i,
            from_range=str(state.iloc[i]["range_name"]),
            to_range=str(state.iloc[i+1]["range_name"]),
            consensus_physical_percentile=float(phys[ti]),
            consensus_percentile_effect=float(phys[ti]-.5),
            median_physical_front_score=float(np.median(score[ti,:,0])),
            median_control_front_score=float(np.median(score[ti,:,1:])),
            control_dest_indices=",".join(str(int(x)) for x in cmap[i]),
        ))
    tdf=pd.DataFrame(trows); atomic_csv(out/"b50_6_transition_specificity.csv",tdf)

    full_effect=float(np.mean(phys)-.5)
    direction=0 if abs(full_effect)<1e-15 else int(math.copysign(1,full_effect))
    cfgsign=np.sign(cfgdf["mean_percentile_effect"].to_numpy(float))
    agree=float(np.mean(cfgsign==direction)) if direction else 0.0
    cfg_pass=bool(agree>=TH["config_direction_agreement_min"])
    cfgdf["same_direction_as_consensus"]=cfgsign==direction
    atomic_csv(out/"b50_6_config_specificity.csv",cfgdf)

    print("[4/6] Matched-set identity randomization",flush=True)
    primary,null=matched_null(consensus,args.randomization_replicates,args.seed+1000)
    primary_pass=bool(primary["p_two_sided"]<=TH["primary_p_max"]
                      and abs(primary["observed_mean_effect"])>=TH["primary_abs_effect_min"])
    primary["pass"]=primary_pass
    atomic_csv(out/"b50_6_primary_null.csv",pd.DataFrame({
        "replicate":np.arange(len(null)),"mean_percentile_effect":null
    }))

    print("[5/6] Three contiguous internal replication strata",flush=True)
    n=len(tis); edges=np.linspace(0,n,4,dtype=int)
    rows=[]; same=0; nominal=0
    for third in range(3):
        mask=np.zeros(n,bool); mask[edges[third]:edges[third+1]]=True
        res,_=matched_null(consensus,args.randomization_replicates,args.seed+2000+third,mask)
        d=0 if abs(res["observed_mean_effect"])<1e-15 else int(math.copysign(1,res["observed_mean_effect"]))
        same_dir=bool(direction and d==direction)
        nom=bool(res["p_two_sided"]<=TH["third_p_max"]
                 and abs(res["observed_mean_effect"])>=TH["third_abs_effect_min"])
        same+=int(same_dir); nominal+=int(nom)
        idx=np.asarray(tis)[mask]
        rows.append(dict(third=third+1,transition_count=int(mask.sum()),
                         first_transition_index=int(idx[0]),last_transition_index=int(idx[-1]),
                         **res,same_direction_as_full=same_dir,nominal_replication_pass=nom))
    thirds=pd.DataFrame(rows); atomic_csv(out/"b50_6_replication_thirds.csv",thirds)
    thirds_pass=bool(same>=TH["third_same_direction_required"] and nominal>=TH["third_nominal_required"])

    print("[6/6] Frozen verdict logic",flush=True)
    if match_ok and primary_pass and cfg_pass and thirds_pass:
        verdict="PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED"
        reason="Matched physical adjacency is specific beyond distance-matched non-adjacent controls, converges across frozen configurations, and internally replicates across all three contiguous strata."
    elif match_ok and primary_pass and cfg_pass and not thirds_pass:
        verdict="PHYSICAL_ADJACENCY_SPECIFICITY_DETECTED_NOT_REPLICATED"
        reason="Full matched sample and configuration convergence pass, but the frozen three-strata replication gate fails."
    elif (not primary_pass) and abs(primary["observed_mean_effect"])<TH["primary_abs_effect_min"]:
        verdict="PHYSICAL_ADJACENCY_NULL_COMPATIBLE"
        reason="Physical next-state percentile is not sufficiently displaced from matched non-adjacent controls."
    else:
        verdict="WEAK_OR_INCONSISTENT_ADJACENCY_SPECIFICITY"
        reason="Adjacency-specificity evidence is mixed under frozen effect-size, randomization, configuration and replication gates."

    summary={
        "version":VERSION,"finished":now_s(),"outcome_blind_to_b59":True,
        "integrity":{"labels_exact":labels_ok,"b50_2_reproduction_ok":b502_ok,
                     "b50_4_integrity_ok":b504_ok,"b50_5_reproduction_ok":b505_ok,
                     "frozen_b50_4_reproduction_ok":repro_ok,"raw_feature_parity_ok":parity_ok},
        "matching":{"controls_per_transition":args.controls_per_transition,
                    "min_index_separation":args.min_index_separation,
                    "median_abs_log_distance_error":med,"q90_abs_log_distance_error":q90,"pass":match_ok},
        "primary_specificity":primary,
        "config_convergence":{"full_effect_direction":direction,
                              "direction_agreement_fraction":agree,"pass":cfg_pass},
        "thirds_replication":{"same_direction_count":same,"nominal_count":nominal,
                              "pass":thirds_pass,"rows":thirds.to_dict(orient="records")},
        "thresholds":TH,"final_verdict":verdict,"final_reason":reason,
        "scientific_boundary":"Internal matched adjacency-specificity test on the existing 666-state sequence; not external replication and not B59 validation."
    }
    atomic_json(out/"b50_6_summary.json",summary)

    plt.figure(figsize=(8,5))
    plt.hist(null,bins=45); plt.axvline(primary["observed_mean_effect"],linewidth=2)
    plt.xlabel("mean percentile effect under matched-set identity null"); plt.ylabel("count")
    plt.title("B50.6 matched adjacency identity randomization"); plt.tight_layout()
    plt.savefig(out/"b50_6_primary_null.png",dpi=160); plt.close()

    plt.figure(figsize=(11,5))
    plt.plot(tdf["transition_index"],tdf["consensus_physical_percentile"],linewidth=1)
    plt.axhline(.5,linestyle="--",linewidth=.8); plt.ylim(-.02,1.02)
    plt.xlabel("physical transition index"); plt.ylabel("physical destination percentile")
    plt.title("B50.6 adjacency specificity by transition"); plt.grid(True); plt.tight_layout()
    plt.savefig(out/"b50_6_specificity_by_transition.png",dpi=160); plt.close()

    plt.figure(figsize=(9,5))
    x=np.arange(len(cfgdf)); plt.plot(x,cfgdf["mean_physical_percentile"],marker="o")
    plt.axhline(.5,linestyle="--",linewidth=.8)
    plt.xticks(x,cfgdf["config_id"],rotation=60,ha="right")
    plt.ylabel("mean physical percentile"); plt.title("B50.6 configuration convergence")
    plt.grid(True); plt.tight_layout(); plt.savefig(out/"b50_6_config_convergence.png",dpi=160); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(thirds["third"],thirds["observed_mean_percentile"],marker="o")
    plt.axhline(.5,linestyle="--",linewidth=.8); plt.xticks([1,2,3])
    plt.xlabel("contiguous third"); plt.ylabel("mean physical percentile")
    plt.title("B50.6 internal replication"); plt.grid(True); plt.tight_layout()
    plt.savefig(out/"b50_6_thirds_replication.png",dpi=160); plt.close()

    report=f"""B50.6 — Physical Adjacency Specificity Replication
=======================================================

Integrity
---------
operator stack shape                 = {P.shape}
labels exact                         = {labels_ok}
B50.2 reproduction OK                = {b502_ok}
B50.4 integrity OK                   = {b504_ok}
B50.5 reproduction OK                = {b505_ok}
frozen B50.4 reproduction PASS       = {repro_ok}
arbitrary-pair raw feature parity    = {parity_ok}

Matched-control protocol
------------------------
physical source transitions used     = {len(tis)}
controls per transition              = {args.controls_per_transition}
minimum index separation             = {args.min_index_separation}
matching geometry                    = full 4096D B50 operator Euclidean
median |log-distance match error|    = {med}
q90 |log-distance match error|       = {q90}
MATCHING PASS                        = {match_ok}

Primary matched adjacency specificity
-------------------------------------
mean physical percentile             = {primary['observed_mean_percentile']}
median physical percentile           = {primary['observed_median_percentile']}
mean percentile effect vs 0.5        = {primary['observed_mean_effect']}
matched identity p_two_sided         = {primary['p_two_sided']}
PRIMARY PASS                         = {primary_pass}

Configuration convergence
-------------------------
direction agreement fraction         = {agree}
CONFIG CONVERGENCE PASS              = {cfg_pass}

Internal replication across thirds
----------------------------------
same-direction thirds                = {same}/3
nominally replicated thirds          = {nominal}/3
THIRDS REPLICATION PASS              = {thirds_pass}

{thirds.to_string(index=False)}

FINAL VERDICT: {verdict}

REASON:
{reason}

Interpretation boundary
-----------------------
This is an internal matched adjacency-specificity replication on the existing
666-state sequence. It is not an external replication on new prime windows.
No B59/B59.1 selector or predictive outcome is used.
No individual-prime prediction, RH, or theorem claim is made.
"""
    atomic_text(out/"b50_6_verdict.txt",report)

    print()
    print("=== B50.6 RESULT ===")
    print("matching PASS :",match_ok)
    print("primary PASS  :",primary_pass)
    print("config PASS   :",cfg_pass)
    print("thirds PASS   :",thirds_pass)
    print("mean percentile:",f"{primary['observed_mean_percentile']:.6f}")
    print("effect vs 0.5 :",f"{primary['observed_mean_effect']:.6f}")
    print("p_two_sided   :",f"{primary['p_two_sided']:.6g}")
    print("FINAL VERDICT :",verdict)
    print("verdict       :",out/"b50_6_verdict.txt")
    return summary


def parse_args():
    p=argparse.ArgumentParser(description="B50.6 Physical Adjacency Specificity Replication")
    p.add_argument("--operator-stack",required=True)
    p.add_argument("--b50-2-summary",required=True)
    p.add_argument("--b50-4-protocol",required=True)
    p.add_argument("--b50-4-summary",required=True)
    p.add_argument("--b50-4-front-features",required=True)
    p.add_argument("--b50-5-summary",required=True)
    p.add_argument("--state-dataset",required=True)
    p.add_argument("--b50-4-script",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--controls-per-transition",type=int,default=20)
    p.add_argument("--min-index-separation",type=int,default=10)
    p.add_argument("--randomization-replicates",type=int,default=5000)
    p.add_argument("--seed",type=int,default=50606)
    a=p.parse_args()
    if a.controls_per_transition<5: p.error("--controls-per-transition must be >=5")
    if a.min_index_separation<3: p.error("--min-index-separation must be >=3")
    if a.randomization_replicates<999: p.error("--randomization-replicates must be >=999")
    return a


def main():
    return 0 if run(parse_args()) is not None else 1


if __name__=="__main__":
    raise SystemExit(main())
