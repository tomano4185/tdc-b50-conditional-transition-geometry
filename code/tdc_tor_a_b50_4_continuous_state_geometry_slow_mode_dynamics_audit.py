#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDC TOR A / B50.4 — Continuous State Geometry & Slow-Mode Dynamics Audit

Outcome-blind audit following:
  B50.1 PERSISTENT_DEGENERACY
  B50.2 SEVERE_LOW_DIMENSION_INFORMATION_LOSS
  B50.3 NO_STABLE_REGIME_TOPOLOGY

B50.4 drops the discrete-basin assumption and tests whether the physical
sequence of Markov-operator states has reproducible continuous geometry,
slow diffusion modes, and a stable continuous transition-front endpoint.

No B59/B59.1 score, cohort, selector, or predictive outcome is read or used.

Pre-declared gates
------------------
TEMPORAL_CONTINUITY_PASS:
  median physical-step percentile <= 0.25
  median next-state rank fraction <= 0.25

SLOW_MODE_PASS:
  median(best 3 nontrivial-mode roughness ratio) <= 0.75
  median(best 3 nontrivial-mode permutation p) <= 0.05

FRONT_STABILITY_PASS:
  median pairwise front-score Spearman >= 0.75
  median top-q Jaccard >= 0.50

Final verdicts
--------------
STABLE_CONTINUOUS_FRONT_ENDPOINT
CONTINUOUS_SLOW_MODE_STRUCTURE
TEMPORAL_CONTINUITY_WITHOUT_STABLE_SLOW_MODE
NO_STABLE_CONTINUOUS_DYNAMICS
INVALID_INPUT_REPRODUCTION

Scientific boundary: representation/dynamics audit only. No forecasting claim,
individual-prime prediction claim, RH claim, or theorem claim is made.
"""
from __future__ import annotations

import argparse, hashlib, json, math, time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "B50.4_continuous_state_geometry_slow_mode_dynamics_v1"
DEFAULT_GRAPH_K = [10, 20, 30]
THRESHOLDS = {
    "max_median_physical_step_percentile": 0.25,
    "max_median_next_state_rank_fraction": 0.25,
    "max_median_best3_roughness_ratio": 0.75,
    "max_median_best3_permutation_p": 0.05,
    "min_median_front_score_spearman": 0.75,
    "min_median_front_topq_jaccard": 0.50,
}
FRONT_CHANNELS = ["speed", "acceleration", "turn_change", "density_scale_change", "diffusion_step"]


def now_s(): return time.strftime("%Y-%m-%d %H:%M:%S")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def atomic_json(path: Path, obj: Any): atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True))

def atomic_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def parse_int_list(s: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
    if not vals or any(v <= 0 for v in vals): raise ValueError("Expected positive integer list.")
    return vals

def validate_state_dataset(df: pd.DataFrame):
    for c in ("range_name", "range_index"):
        if c not in df.columns: raise ValueError(f"Missing state column: {c}")
    idx = pd.to_numeric(df["range_index"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(len(df))): raise ValueError("range_index is not exactly 0..n-1")
    if df["range_name"].duplicated().any(): raise ValueError("Duplicate range_name")

def full_pca(X):
    X = np.asarray(X, np.float64); Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S[None, :]
    eig = S * S; explained = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    return scores, explained, np.cumsum(explained)

def derive_auto_dimensions(summary: Dict[str, Any], rank: int) -> List[int]:
    vd = summary.get("variance_dimensions", {})
    raw = [vd.get("d75"), summary.get("smallest_tested_dimension_meeting_all_adequacy_metrics"),
           vd.get("d90"), int(round(float(summary.get("effective_rank_shannon", 0.0)))), vd.get("d95")]
    vals = sorted(set(int(x) for x in raw if x is not None and 2 <= int(x) <= rank))
    if not vals: raise RuntimeError("Cannot derive B50.4 dimensions from B50.2 summary")
    return vals

def safe_spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15: return float("nan")
    return float(spearmanr(a, b).statistic)

def lag1_corr(x):
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x[:-1]) < 1e-15 or np.std(x[1:]) < 1e-15: return float("nan")
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])

def robust_positive_z(x):
    x = np.asarray(x, float); med = float(np.nanmedian(x)); mad = float(np.nanmedian(np.abs(x - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12: scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale < 1e-12: return np.zeros_like(x)
    z = (x - med) / scale; z[~np.isfinite(z)] = 0.0
    return np.maximum(z, 0.0)

def top_mask(x, q): return np.asarray(x) >= float(np.quantile(np.asarray(x), q))

def jaccard_bool(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); u = int(np.sum(a | b))
    return float(np.sum(a & b) / u) if u else float("nan")


def twonn_dimension(D):
    order = np.sort(np.asarray(D, float), axis=1); r1, r2 = order[:,1], order[:,2]
    m = (r1 > 1e-15) & (r2 > r1); logs = np.log(r2[m] / r1[m]); logs = logs[np.isfinite(logs) & (logs > 0)]
    return {"estimate": float(len(logs) / np.sum(logs)) if len(logs) >= 10 else float("nan"), "n_used": int(len(logs))}

def levina_bickel_dimension(D, k):
    D = np.asarray(D, float); order = np.sort(D, axis=1); vals = []
    for i in range(len(D)):
        neigh = order[i,1:k+1]; Tk = float(neigh[-1]); inner = neigh[:-1]
        if Tk <= 1e-15 or np.any(inner <= 1e-15): continue
        den = float(np.sum(np.log(Tk / inner)))
        if den > 1e-15 and np.isfinite(den): vals.append((k - 2) / den)
    return {"estimate": float(np.mean(vals)) if vals else float("nan"), "median_local": float(np.median(vals)) if vals else float("nan"), "n_used": len(vals), "k": k}

def trajectory_continuity(D, graph_k):
    D = np.asarray(D, float); n = len(D)
    pair = np.sort(D[np.triu_indices(n,1)])
    step = D[np.arange(n-1), np.arange(1,n)]
    pct = np.searchsorted(pair, step, side="right") / max(len(pair),1)
    order = np.argsort(D, axis=1); ranks = np.empty(n-1, int)
    for i in range(n-1): ranks[i] = int(np.where(order[i] == i+1)[0][0])
    rf = ranks / max(n-1,1)
    rk_idx = order[:,graph_k]; rk = D[np.arange(n), rk_idx]
    return {
        "median_physical_step_percentile": float(np.median(pct)),
        "median_next_state_rank": float(np.median(ranks)),
        "median_next_state_rank_fraction": float(np.median(rf)),
        "median_step_over_local_radius": float(np.median(step / np.maximum(rk[:-1],1e-15))),
        "fraction_next_state_inside_knn": float(np.mean(ranks <= graph_k)),
        "local_radius": rk,
    }


def build_knn_graph(D, k):
    D = np.asarray(D, float); n = len(D); order = np.argsort(D, axis=1); nbr = order[:,1:k+1]
    scale = np.maximum(D[np.arange(n), nbr[:,-1]], 1e-15)
    Wd = np.zeros((n,n), float); Gd = np.zeros((n,n), float)
    for i in range(n):
        for j in nbr[i]:
            j = int(j); dij = float(D[i,j]); wij = math.exp(-(dij*dij)/max(scale[i]*scale[j],1e-30))
            Wd[i,j] = max(Wd[i,j], wij); Wd[j,i] = max(Wd[j,i], wij)
            if Gd[i,j] == 0 or dij < Gd[i,j]: Gd[i,j] = Gd[j,i] = dij
    return csr_matrix(Wd), csr_matrix(Gd), scale


def geodesic_metrics(D, G, anchors):
    ncomp, comp = connected_components(G, directed=False); counts = np.bincount(comp)
    ai = np.linspace(0, len(D)-1, min(max(2,anchors),len(D)), dtype=int)
    geo = dijkstra(G, directed=False, indices=ai); eu = D[ai]
    m = np.isfinite(geo) & (eu > 1e-15); stretch = geo[m] / eu[m]; stretch = stretch[np.isfinite(stretch)]
    return {"n_components": int(ncomp), "largest_component_fraction": float(counts.max()/len(D)),
            "geodesic_stretch_median": float(np.median(stretch)) if len(stretch) else float("nan"),
            "geodesic_stretch_q90": float(np.quantile(stretch,.90)) if len(stretch) else float("nan")}


def diffusion_modes(W, n_modes):
    degree = np.maximum(np.asarray(W.sum(axis=1)).reshape(-1), 1e-15)
    S = diags(1.0/np.sqrt(degree)) @ W @ diags(1.0/np.sqrt(degree))
    nev = min(max(2,n_modes+1), W.shape[0]-1)
    vals, vecs = eigsh(S, k=nev, which="LA"); o = np.argsort(vals)[::-1]; vals, vecs = vals[o], vecs[:,o]
    phi = (1.0/np.sqrt(degree))[:,None] * vecs
    vals, phi = vals[1:n_modes+1], phi[:,1:n_modes+1]
    for j in range(phi.shape[1]):
        x = phi[:,j] - phi[:,j].mean(); s = x.std(); phi[:,j] = x/s if s > 1e-15 else x
    return np.asarray(vals,float), np.asarray(phi,float)


def slow_mode_audit(modes, eigvals, permutations, seed):
    rng = np.random.default_rng(seed); rows = []
    for j in range(modes.shape[1]):
        x = modes[:,j]; obs = float(np.mean(np.diff(x)**2)); null = np.empty(permutations,float)
        for r in range(permutations): null[r] = float(np.mean(np.diff(rng.permutation(x))**2))
        nm = float(null.mean()); ratio = obs/max(nm,1e-15); p = float((1 + np.sum(null <= obs))/(permutations+1))
        rows.append({"mode_index":j+1, "eigenvalue":float(eigvals[j]), "lag1_autocorrelation":lag1_corr(x),
                     "temporal_roughness":obs, "null_mean_roughness":nm, "roughness_ratio":ratio, "permutation_p_lower":p})
    return pd.DataFrame(rows)


def front_features(X, local_radius, diffcoords):
    V = np.diff(X,axis=0); speed = np.linalg.norm(V,axis=1)
    accel = np.zeros(len(V)); turn = np.zeros(len(V))
    if len(V)>1:
        accel[1:] = np.linalg.norm(V[1:] - V[:-1],axis=1)
        a,b = V[:-1],V[1:]; den = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1); cos = np.ones(len(a)); good = den>1e-15
        cos[good] = np.sum(a[good]*b[good],axis=1)/den[good]; turn[1:] = 1.0 - np.clip(cos,-1,1)
    dens = np.abs(np.diff(np.log(np.maximum(local_radius,1e-15))))
    dstep = np.linalg.norm(np.diff(diffcoords,axis=0),axis=1) if diffcoords.shape[1] else np.zeros(len(V))
    raw = {"speed":speed,"acceleration":accel,"turn_change":turn,"density_scale_change":dens,"diffusion_step":dstep}
    z = {k:robust_positive_z(v) for k,v in raw.items()}; Z = np.column_stack([z[k] for k in FRONT_CHANNELS])
    score = np.sqrt(np.mean(Z*Z,axis=1))
    return pd.DataFrame({"transition_index":np.arange(len(V)), **raw, **{f"z_{k}":z[k] for k in FRONT_CHANNELS}, "front_score":score})


def front_stability(score_map, q):
    ids = sorted(score_map); rows = []
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            a,b = ids[i],ids[j]
            rows.append({"config_a":a,"config_b":b,"score_spearman":safe_spearman(score_map[a],score_map[b]),
                         "topq_jaccard":jaccard_bool(top_mask(score_map[a],q),top_mask(score_map[b],q))})
    df = pd.DataFrame(rows)
    return df, {"config_count":len(ids),"pair_count":len(df),
                "median_score_spearman":float(df["score_spearman"].median()) if len(df) else float("nan"),
                "median_topq_jaccard":float(df["topq_jaccard"].median()) if len(df) else float("nan"),"top_quantile":q}

def run_audit(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stack_path, b502_path, state_path = Path(args.operator_stack), Path(args.b50_2_summary), Path(args.state_dataset)
    for p in (stack_path,b502_path,state_path):
        if not p.exists(): raise FileNotFoundError(p)
    b502 = json.loads(b502_path.read_text(encoding="utf-8")); state = pd.read_csv(state_path); validate_state_dataset(state)
    with np.load(stack_path,allow_pickle=False) as z:
        P = np.asarray(z["P_stack"],np.float64); labels = [str(x) for x in z["labels"]]
    expected = state["range_name"].astype(str).tolist(); labels_exact = labels == expected
    shape_exact = list(P.shape) == list(b502.get("operator_shape", list(P.shape)))
    repro_ok = bool(b502.get("reproduction",{}).get("reproduction_ok",False))
    integrity = labels_exact and shape_exact and repro_ok
    print("=== TDC TOR A / B50.4: CONTINUOUS STATE GEOMETRY AUDIT ===")
    print(f"operator stack       : {P.shape}")
    print(f"labels exact         : {labels_exact}")
    print(f"shape matches B50.2  : {shape_exact}")
    print(f"B50.2 reproduction   : {repro_ok}")
    if not integrity:
        s={"version":VERSION,"final_verdict":"INVALID_INPUT_REPRODUCTION","integrity":{"labels_exact":labels_exact,"shape_exact":shape_exact,"b50_2_reproduction_ok":repro_ok}}
        atomic_json(out/"b50_4_summary.json",s); atomic_text(out/"b50_4_verdict.txt",f"FINAL VERDICT: INVALID_INPUT_REPRODUCTION\n{s['integrity']}\n"); return s

    scores,expl,cum = full_pca(P); rank = scores.shape[1]
    dims = derive_auto_dimensions(b502,rank) if args.dimensions.lower()=="auto" else [d for d in parse_int_list(args.dimensions) if d<=rank]
    graph_k = [k for k in args.graph_k if k < len(state)]
    if not dims or not graph_k: raise RuntimeError("No valid dimensions / graph k")
    protocol={"version":VERSION,"created":now_s(),"outcome_blind_to_b59":True,
              "inputs":{"operator_stack":str(stack_path),"operator_stack_sha256":sha256_file(stack_path),
                        "b50_2_summary":str(b502_path),"b50_2_summary_sha256":sha256_file(b502_path),
                        "state_dataset":str(state_path),"state_dataset_sha256":sha256_file(state_path)},
              "dimensions":dims,"graph_k_values":graph_k,"diffusion_modes":args.diffusion_modes,
              "permutations":args.permutations,"geodesic_anchors":args.geodesic_anchors,
              "front_top_quantile":args.front_top_quantile,"seed":args.seed,"thresholds":THRESHOLDS}
    atomic_json(out/"b50_4_protocol_manifest.json",protocol)
    print(f"dimensions           : {dims}"); print(f"graph k              : {graph_k}")

    # Intrinsic dimension at highest audited representation.
    Xi = scores[:,:max(dims)]; Di = squareform(pdist(Xi))
    intrinsic={"reference_dimension":max(dims),"twonn":twonn_dimension(Di),
               "levina_bickel_k10":levina_bickel_dimension(Di,10),"levina_bickel_k20":levina_bickel_dimension(Di,20)}
    atomic_json(out/"b50_4_intrinsic_dimension.json",intrinsic)

    cfg_rows=[]; slow_parts=[]; front_parts=[]; score_map={}; step_pcts=[]; rank_fracs=[]; ratios=[]; ps=[]
    total=len(dims)*len(graph_k); job=0; t_all=time.time()
    for d in dims:
        X=scores[:,:d]; D=squareform(pdist(X))
        for gk in graph_k:
            job+=1; cfg=f"d{d}_knn{gk}"; print(f"[{job:02d}/{total:02d}] {cfg}",flush=True); t0=time.time()
            cont=trajectory_continuity(D,gk); W,G,local_radius=build_knn_graph(D,gk); geo=geodesic_metrics(D,G,args.geodesic_anchors)
            eig,modes=diffusion_modes(W,args.diffusion_modes); slow=slow_mode_audit(modes,eig,args.permutations,args.seed+100000*d+1000*gk)
            slow.insert(0,"config_id",cfg); slow.insert(1,"dimension",d); slow.insert(2,"graph_k",gk); slow_parts.append(slow)
            best3=slow.sort_values(["roughness_ratio","permutation_p_lower"]).head(min(3,len(slow)))
            ratio=float(best3["roughness_ratio"].median()); pp=float(best3["permutation_p_lower"].median())
            diffcoords=modes*eig[None,:]; fronts=front_features(X,local_radius,diffcoords)
            fronts.insert(0,"config_id",cfg); fronts.insert(1,"dimension",d); fronts.insert(2,"graph_k",gk); front_parts.append(fronts)
            score_map[cfg]=fronts["front_score"].to_numpy(float)
            step_pcts.append(cont["median_physical_step_percentile"]); rank_fracs.append(cont["median_next_state_rank_fraction"]); ratios.append(ratio); ps.append(pp)
            cfg_rows.append({"config_id":cfg,"dimension":d,"cumulative_variance":float(cum[d-1]),"graph_k":gk,
                             "median_physical_step_percentile":cont["median_physical_step_percentile"],
                             "median_next_state_rank":cont["median_next_state_rank"],
                             "median_next_state_rank_fraction":cont["median_next_state_rank_fraction"],
                             "median_step_over_local_radius":cont["median_step_over_local_radius"],
                             "fraction_next_state_inside_knn":cont["fraction_next_state_inside_knn"],**geo,
                             "best3_median_roughness_ratio":ratio,"best3_median_permutation_p":pp,
                             "leading_nontrivial_eigenvalue":float(eig[0]),"elapsed_seconds":time.time()-t0})

    cfg_df=pd.DataFrame(cfg_rows); slow_df=pd.concat(slow_parts,ignore_index=True); front_long=pd.concat(front_parts,ignore_index=True)
    atomic_csv(out/"b50_4_configuration_metrics.csv",cfg_df); atomic_csv(out/"b50_4_slow_modes.csv",slow_df); atomic_csv(out/"b50_4_front_features_long.csv",front_long)
    stab_df,stab=front_stability(score_map,args.front_top_quantile); atomic_csv(out/"b50_4_front_stability_pairs.csv",stab_df)

    ids=sorted(score_map); score_mat=np.column_stack([score_map[c] for c in ids]); top_mat=np.column_stack([top_mask(score_map[c],args.front_top_quantile) for c in ids])
    consensus_df=pd.DataFrame({"transition_index":np.arange(len(state)-1),
                               "from_range":state["range_name"].astype(str).iloc[:-1].to_numpy(),
                               "to_range":state["range_name"].astype(str).iloc[1:].to_numpy(),
                               "front_score_median":np.median(score_mat,axis=1),"front_score_mean":np.mean(score_mat,axis=1),
                               "topq_consensus_fraction":np.mean(top_mat.astype(float),axis=1)})
    consensus_df["consensus_rank"]=consensus_df["front_score_median"].rank(method="first",ascending=False).astype(int)
    atomic_csv(out/"b50_4_front_consensus_ranking.csv",consensus_df.sort_values(["consensus_rank","transition_index"]))

    med_step=float(np.median(step_pcts)); med_rank=float(np.median(rank_fracs)); med_ratio=float(np.median(ratios)); med_p=float(np.median(ps))
    continuity_pass = med_step<=THRESHOLDS["max_median_physical_step_percentile"] and med_rank<=THRESHOLDS["max_median_next_state_rank_fraction"]
    slow_pass = med_ratio<=THRESHOLDS["max_median_best3_roughness_ratio"] and med_p<=THRESHOLDS["max_median_best3_permutation_p"]
    front_pass = stab["median_score_spearman"]>=THRESHOLDS["min_median_front_score_spearman"] and stab["median_topq_jaccard"]>=THRESHOLDS["min_median_front_topq_jaccard"]
    if continuity_pass and slow_pass and front_pass:
        verdict="STABLE_CONTINUOUS_FRONT_ENDPOINT"; reason="Physical continuity, diffusion slow-mode smoothness, and front stability all pass the frozen criteria."
    elif slow_pass:
        verdict="CONTINUOUS_SLOW_MODE_STRUCTURE"; reason="A reproducible slow-mode structure is present, but the complete continuous-front endpoint does not pass every criterion."
    elif continuity_pass:
        verdict="TEMPORAL_CONTINUITY_WITHOUT_STABLE_SLOW_MODE"; reason="The physical sequence is locally continuous, but the frozen diffusion slow-mode criterion fails."
    else:
        verdict="NO_STABLE_CONTINUOUS_DYNAMICS"; reason="The audited representations do not jointly support the frozen temporal-continuity / slow-mode / stable-front criteria."

    summary={"version":VERSION,"finished":now_s(),"outcome_blind_to_b59":True,
             "integrity":{"labels_exact":labels_exact,"shape_exact":shape_exact,"b50_2_reproduction_ok":repro_ok},
             "operator_shape":list(P.shape),"pca_rank":rank,"dimensions":dims,"graph_k_values":graph_k,
             "intrinsic_dimension":intrinsic,
             "aggregate_continuity":{"median_physical_step_percentile":med_step,"median_next_state_rank_fraction":med_rank,"pass":bool(continuity_pass)},
             "aggregate_slow_mode":{"median_best3_roughness_ratio":med_ratio,"median_best3_permutation_p":med_p,"pass":bool(slow_pass)},
             "front_stability":{**stab,"pass":bool(front_pass)},"thresholds":THRESHOLDS,
             "final_verdict":verdict,"final_reason":reason,"elapsed_seconds":time.time()-t_all,
             "scientific_boundary":"Continuous representation/dynamics audit only; no B59/B59.1 selector or predictive outcome used."}
    atomic_json(out/"b50_4_summary.json",summary)

    # Plots
    eig_plot=slow_df.groupby("mode_index",as_index=False)["eigenvalue"].median().sort_values("mode_index")
    plt.figure(figsize=(8,5)); plt.plot(eig_plot["mode_index"],eig_plot["eigenvalue"],marker="o"); plt.xlabel("nontrivial diffusion mode"); plt.ylabel("median eigenvalue"); plt.title("B50.4 diffusion slow spectrum"); plt.grid(True); plt.tight_layout(); plt.savefig(out/"b50_4_diffusion_spectrum.png",dpi=160); plt.close()
    plt.figure(figsize=(9,5))
    for gk in graph_k:
        g=cfg_df[cfg_df["graph_k"]==gk].sort_values("dimension"); plt.plot(g["dimension"],g["best3_median_roughness_ratio"],marker="o",label=f"kNN {gk}")
    plt.axhline(THRESHOLDS["max_median_best3_roughness_ratio"],linestyle="--",linewidth=.8); plt.xlabel("PCA dimension"); plt.ylabel("best-3 median roughness ratio"); plt.title("B50.4 temporal slow-mode stability"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(out/"b50_4_slow_mode_roughness.png",dpi=160); plt.close()
    plt.figure(figsize=(9,5))
    for gk in graph_k:
        g=cfg_df[cfg_df["graph_k"]==gk].sort_values("dimension"); plt.plot(g["dimension"],g["median_physical_step_percentile"],marker="o",label=f"kNN {gk}")
    plt.axhline(THRESHOLDS["max_median_physical_step_percentile"],linestyle="--",linewidth=.8); plt.xlabel("PCA dimension"); plt.ylabel("median physical-step percentile"); plt.title("B50.4 physical trajectory continuity"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(out/"b50_4_trajectory_continuity.png",dpi=160); plt.close()
    by=consensus_df.sort_values("transition_index"); plt.figure(figsize=(12,5)); plt.plot(by["transition_index"],by["front_score_median"],linewidth=1); plt.xlabel("physical transition index"); plt.ylabel("median front score"); plt.title("B50.4 continuous front score"); plt.grid(True); plt.tight_layout(); plt.savefig(out/"b50_4_front_score_by_transition.png",dpi=160); plt.close()
    plt.figure(figsize=(12,5)); plt.plot(by["transition_index"],by["topq_consensus_fraction"],linewidth=1); plt.xlabel("physical transition index"); plt.ylabel("top-q consensus fraction"); plt.ylim(-.02,1.02); plt.title("B50.4 front consensus"); plt.grid(True); plt.tight_layout(); plt.savefig(out/"b50_4_front_consensus.png",dpi=160); plt.close()

    report=f"""B50.4 — Continuous State Geometry & Slow-Mode Dynamics Audit
================================================================

Integrity
---------
operator stack shape             = {P.shape}
labels exact                     = {labels_exact}
shape matches B50.2              = {shape_exact}
B50.2 reproduction OK            = {repro_ok}

Audit protocol
--------------
dimensions                       = {dims}
graph k                          = {graph_k}
diffusion modes                  = {args.diffusion_modes}
permutations per mode            = {args.permutations}
front top quantile               = {args.front_top_quantile}

Intrinsic dimension
-------------------
TwoNN                            = {intrinsic['twonn']['estimate']}
Levina-Bickel k=10               = {intrinsic['levina_bickel_k10']['estimate']}
Levina-Bickel k=20               = {intrinsic['levina_bickel_k20']['estimate']}

Physical trajectory continuity
------------------------------
median physical-step percentile  = {med_step}
median next-state rank fraction  = {med_rank}
PASS                             = {continuity_pass}

Diffusion slow modes
--------------------
median best-3 roughness ratio    = {med_ratio}
median best-3 permutation p      = {med_p}
PASS                             = {slow_pass}

Continuous-front stability
--------------------------
median score Spearman            = {stab['median_score_spearman']}
median top-q Jaccard             = {stab['median_topq_jaccard']}
PASS                             = {front_pass}

FINAL VERDICT: {verdict}

REASON:
{reason}

Interpretation boundary
-----------------------
Outcome-blind continuous representation/dynamics audit only.
No B59/B59.1 selector or predictive outcome is used.
No individual-prime prediction, RH, or theorem claim is made.
"""
    atomic_text(out/"b50_4_verdict.txt",report)
    print("\n=== B50.4 RESULT ==="); print(f"continuity PASS : {continuity_pass}"); print(f"slow mode PASS  : {slow_pass}"); print(f"front PASS      : {front_pass}"); print(f"FINAL VERDICT   : {verdict}"); print(f"verdict         : {out/'b50_4_verdict.txt'}")
    return summary


def parse_args():
    p=argparse.ArgumentParser(description="B50.4 Continuous State Geometry & Slow-Mode Dynamics Audit")
    p.add_argument("--operator-stack",required=True); p.add_argument("--b50-2-summary",required=True); p.add_argument("--state-dataset",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--dimensions",default="auto"); p.add_argument("--graph-k",default="10,20,30"); p.add_argument("--diffusion-modes",type=int,default=8)
    p.add_argument("--permutations",type=int,default=500); p.add_argument("--geodesic-anchors",type=int,default=48); p.add_argument("--front-top-quantile",type=float,default=.95); p.add_argument("--seed",type=int,default=50404)
    a=p.parse_args(); a.graph_k=parse_int_list(a.graph_k)
    if a.diffusion_modes<3: p.error("--diffusion-modes must be >=3")
    if a.permutations<99: p.error("--permutations must be >=99")
    if a.geodesic_anchors<2: p.error("--geodesic-anchors must be >=2")
    if not (.80<=a.front_top_quantile<1.0): p.error("--front-top-quantile must be in [0.80,1.0)")
    return a


def main(): return 0 if run_audit(parse_args()) is not None else 1

if __name__=="__main__": raise SystemExit(main())
