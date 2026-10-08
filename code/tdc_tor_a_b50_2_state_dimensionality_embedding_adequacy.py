#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TDC TOR A / B50.2 — State Dimensionality & Embedding Adequacy Audit

Outcome-blind audit of whether B50 loses too much geometry when the 64x64
Markov operator is compressed to c1,c2 (basins) / c1,c2,c3 (state dynamics).

Modes:
  preflight : verify original B50 import, exact range/state order, and one range
  extract   : reconstruct exact B50 operator vectors with resumable per-range cache
  audit     : full PCA spectrum + distance / neighborhood / physical-step preservation
  all       : extract + audit

B50.2 never reads B59/B59.1 selectors or outcomes and never tunes dimensions
against basin-jump labels.
"""

from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, platform, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "B50.2_state_dimensionality_embedding_adequacy_v1"
DEFAULT_DIMS = [2,3,5,8,10,16,24,32,48,64,96,128,192,256]
DEFAULT_KS = [5,10,20]

ADEQUACY = {
    "min_cumulative_variance": 0.50,
    "min_distance_spearman": 0.90,
    "min_knn10_overlap": 0.70,
    "min_step_spearman": 0.90,
}
SEVERE = {
    "max_cumulative_variance": 0.25,
    "max_distance_spearman": 0.75,
    "max_knn10_overlap": 0.50,
}


def now_s():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
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


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def parse_int_list(s: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("Expected positive integer list.")
    return vals


def safe_corr(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) != len(b) or len(a) < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0,1])


def safe_spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def import_b50(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("tdc_b50_frozen_engine_b502", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import B50 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    required = ["load_full_range_primes","primes_to_gaps","build_markov_operator","pca_decompose","safe_corr"]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise AttributeError(f"B50 missing required functions: {missing}")
    return mod


def validate_state_dataset(df: pd.DataFrame):
    required = ["range_name","range_index","mean_log_gap","gap_std","n_primes","n_gaps","c1","c2","c3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"State dataset missing columns: {missing}")
    idx = pd.to_numeric(df["range_index"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(len(df))):
        raise ValueError("range_index is not exactly 0..n-1.")
    if df["range_name"].duplicated().any():
        raise ValueError("Duplicate range_name in state dataset.")


def ordered_range_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("range_")], key=lambda p:p.name)
    if not dirs:
        raise RuntimeError(f"No range_* directories in {root}")
    return dirs


def verify_range_order(range_dirs: List[Path], state_df: pd.DataFrame):
    actual = [p.name for p in range_dirs]
    expected = state_df["range_name"].astype(str).tolist()
    if actual != expected:
        first = None
        for i,(a,b) in enumerate(zip(actual, expected)):
            if a != b:
                first = {"index":i,"actual":a,"expected":b}; break
        raise RuntimeError(f"Range/state order mismatch: n_dirs={len(actual)} n_states={len(expected)} first={first}")


def recompute_one_range(b50, rd: Path, gap_bins: int) -> Dict[str,Any]:
    primes = b50.load_full_range_primes(rd)
    gaps = b50.primes_to_gaps(primes).astype(float)
    if len(gaps) < 100:
        raise RuntimeError(f"Too few gaps in {rd}: {len(gaps)}")
    P = b50.build_markov_operator(gaps, n_bins=gap_bins)
    return {
        "range_name": rd.name,
        "operator_vector": np.asarray(P, np.float64).reshape(-1),
        "mean_log_gap": float(np.mean(np.log(gaps))),
        "gap_mean": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "n_primes": int(len(primes)),
        "n_gaps": int(len(gaps)),
    }


def compare_meta(rec: Dict[str,Any], saved: pd.Series) -> Dict[str,Any]:
    e1 = abs(float(rec["mean_log_gap"]) - float(saved["mean_log_gap"]))
    e2 = abs(float(rec["gap_std"]) - float(saved["gap_std"]))
    npm = int(rec["n_primes"]) == int(saved["n_primes"])
    ngm = int(rec["n_gaps"]) == int(saved["n_gaps"])
    ok = e1 <= 1e-10 and e2 <= 1e-8 and npm and ngm
    return {
        "meta_ok": bool(ok),
        "mean_log_gap_abs_error": e1,
        "gap_std_abs_error": e2,
        "n_primes_match": bool(npm),
        "n_gaps_match": bool(ngm),
    }


def cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}_operator.npz"


def save_cache(path: Path, rec: Dict[str,Any], gap_bins: int):
    atomic_npz(
        path,
        range_name=np.asarray([rec["range_name"]]),
        operator_vector=np.asarray(rec["operator_vector"], np.float64),
        mean_log_gap=np.asarray([rec["mean_log_gap"]], np.float64),
        gap_mean=np.asarray([rec["gap_mean"]], np.float64),
        gap_std=np.asarray([rec["gap_std"]], np.float64),
        n_primes=np.asarray([rec["n_primes"]], np.int64),
        n_gaps=np.asarray([rec["n_gaps"]], np.int64),
        gap_bins=np.asarray([gap_bins], np.int64),
    )


def load_cache(path: Path, expected_name: str, gap_bins: int) -> Dict[str,Any]:
    with np.load(path, allow_pickle=False) as z:
        name = str(z["range_name"][0]); gb = int(z["gap_bins"][0])
        if name != expected_name or gb != gap_bins:
            raise RuntimeError(f"Invalid cache {path}: name={name}, gap_bins={gb}")
        return {
            "range_name": name,
            "operator_vector": np.asarray(z["operator_vector"], np.float64),
            "mean_log_gap": float(z["mean_log_gap"][0]),
            "gap_mean": float(z["gap_mean"][0]),
            "gap_std": float(z["gap_std"][0]),
            "n_primes": int(z["n_primes"][0]),
            "n_gaps": int(z["n_gaps"][0]),
        }


def preflight(args):
    b50 = import_b50(Path(args.b50_script))
    state = pd.read_csv(args.state_dataset)
    validate_state_dataset(state)
    dirs = ordered_range_dirs(Path(args.ranges_root))
    verify_range_order(dirs, state)
    print("=== B50.2 PREFLIGHT ===")
    print(f"ranges       : {len(dirs)}")
    print(f"first range  : {dirs[0].name}")
    t0 = time.time()
    rec = recompute_one_range(b50, dirs[0], args.gap_bins)
    audit = compare_meta(rec, state.iloc[0])
    print(f"operator dim : {len(rec['operator_vector'])}")
    print(f"metadata OK  : {audit['meta_ok']}")
    print(f"elapsed      : {time.time()-t0:.2f}s")
    if not audit["meta_ok"]:
        raise RuntimeError(f"Preflight metadata mismatch: {audit}")


def extract(args) -> Path:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    b50_path = Path(args.b50_script)
    state_path = Path(args.state_dataset)
    ranges_root = Path(args.ranges_root)
    b50 = import_b50(b50_path)
    state = pd.read_csv(state_path); validate_state_dataset(state)
    dirs = ordered_range_dirs(ranges_root); verify_range_order(dirs, state)

    cdir = out / "operator_cache"; cdir.mkdir(parents=True, exist_ok=True)
    status, meta, vectors = [], [], []
    t_all = time.time()

    print("=== B50.2 OPERATOR EXTRACTION ===")
    print(f"ranges      : {len(dirs)}")
    print(f"reuse cache : {args.reuse_existing}")

    for i, rd in enumerate(dirs):
        cp = cache_path(cdir, rd.name); t0 = time.time()
        if args.reuse_existing and cp.exists():
            rec = load_cache(cp, rd.name, args.gap_bins); source = "cache"
        else:
            rec = recompute_one_range(b50, rd, args.gap_bins)
            save_cache(cp, rec, args.gap_bins); source = "recomputed"

        ma = compare_meta(rec, state.iloc[i])
        row = {
            "range_index":i,"range_name":rd.name,"source":source,
            "operator_dim":len(rec["operator_vector"]),
            "n_primes":rec["n_primes"],"n_gaps":rec["n_gaps"],
            **ma,"elapsed_seconds":time.time()-t0,"cache_path":str(cp),
        }
        status.append(row)
        if not ma["meta_ok"]:
            atomic_csv(out/"b50_2_extraction_status.csv", pd.DataFrame(status))
            raise RuntimeError(f"Metadata reproduction failed at {rd.name}: {ma}")

        vectors.append(rec["operator_vector"])
        meta.append({
            "range_index":i,"range_name":rd.name,
            "mean_log_gap":rec["mean_log_gap"],"gap_mean":rec["gap_mean"],
            "gap_std":rec["gap_std"],"n_primes":rec["n_primes"],"n_gaps":rec["n_gaps"],
        })

        print(f"[{i+1:03d}/{len(dirs):03d}] {rd.name} {source} time={row['elapsed_seconds']:.1f}s", flush=True)

        if args.checkpoint_every and (i+1) % args.checkpoint_every == 0:
            atomic_csv(out/"b50_2_extraction_status.csv", pd.DataFrame(status))
            atomic_json(out/"b50_2_extraction_checkpoint.json", {
                "version":VERSION,"updated":now_s(),"completed":i+1,"total":len(dirs),
                "last_range":rd.name,"elapsed_seconds":time.time()-t_all
            })

    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise RuntimeError(f"Inconsistent operator dims: {sorted(dims)}")
    P_stack = np.vstack(vectors)
    labels = np.asarray([r["range_name"] for r in meta])
    stack_path = out/"b50_2_operator_stack.npz"
    atomic_npz(stack_path, P_stack=P_stack, labels=labels, gap_bins=np.asarray([args.gap_bins],np.int64))
    atomic_csv(out/"b50_2_operator_metadata.csv", pd.DataFrame(meta))
    atomic_csv(out/"b50_2_extraction_status.csv", pd.DataFrame(status))
    atomic_json(out/"b50_2_extraction_manifest.json", {
        "version":VERSION,"created":now_s(),"ranges_root":str(ranges_root),
        "state_dataset":str(state_path),"state_dataset_sha256":sha256_file(state_path),
        "b50_script":str(b50_path),"b50_script_sha256":sha256_file(b50_path),
        "gap_bins":args.gap_bins,"range_count":P_stack.shape[0],
        "operator_dim":P_stack.shape[1],"stack_shape":list(P_stack.shape),
        "all_metadata_reproduced":all(r["meta_ok"] for r in status),
        "elapsed_seconds":time.time()-t_all,
    })
    print(f"[B50.2] stack saved: {stack_path} shape={P_stack.shape}")
    return stack_path


def full_pca(X):
    X = np.asarray(X,np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U*S[None,:]
    eig = S*S
    explained = eig/eig.sum() if eig.sum()>0 else np.zeros_like(eig)
    return scores,S,explained,np.cumsum(explained)


def dimension_for_fraction(cum, target):
    idx = np.where(cum >= target)[0]
    return int(idx[0]+1) if len(idx) else None


def effective_rank(explained):
    p=np.asarray(explained,float); p=p[p>0]
    return float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.0


def participation_ratio(explained):
    p=np.asarray(explained,float); den=float(np.sum(p*p))
    return float(1.0/den) if den>0 else 0.0


def knn_idx(D,k):
    return np.argsort(D,axis=1)[:,1:k+1]


def knn_overlap(ref, emb):
    k=ref.shape[1]
    vals=np.empty(len(ref),float)
    for i in range(len(ref)):
        vals[i]=len(set(ref[i].tolist()) & set(emb[i].tolist()))/k
    return float(vals.mean()),float(np.median(vals))


def median_relative_error(approx, ref):
    approx=np.asarray(approx,float); ref=np.asarray(ref,float)
    m=np.abs(ref)>1e-15
    return float(np.median(np.abs(approx[m]-ref[m])/np.abs(ref[m]))) if np.any(m) else float("nan")


def embedding_metrics(scores,cumulative,dims,ks):
    n,rank=scores.shape
    dims_eff=sorted(set(min(d,rank) for d in dims if d>0))
    if rank not in dims_eff: dims_eff.append(rank)
    full_vec=pdist(scores); full_D=squareform(full_vec)
    full_knn={k:knn_idx(full_D,k) for k in ks if k<n}
    full_steps=np.linalg.norm(np.diff(scores,axis=0),axis=1)
    rows=[]
    for d in dims_eff:
        Sd=scores[:,:d]
        vec=pdist(Sd); D=squareform(vec)
        steps=np.linalg.norm(np.diff(Sd,axis=0),axis=1)
        row={
            "dimension":d,
            "cumulative_variance":float(cumulative[d-1]),
            "pairwise_distance_pearson":safe_corr(vec,full_vec),
            "pairwise_distance_spearman":safe_spearman(vec,full_vec),
            "physical_step_spearman":safe_spearman(steps,full_steps),
            "physical_step_median_relative_error":median_relative_error(steps,full_steps),
            "is_full_rank_reference":bool(d==rank),
        }
        for k,ref in full_knn.items():
            mean,med=knn_overlap(ref,knn_idx(D,k))
            row[f"knn{k}_overlap_mean"]=mean
            row[f"knn{k}_overlap_median"]=med
        rows.append(row)
    return pd.DataFrame(rows)


def reproduce_b50_pca(b50,P_stack,state,verdict):
    ret=b50.pca_decompose(P_stack)
    if not isinstance(ret,tuple) or len(ret)!=5:
        raise RuntimeError("B50 pca_decompose no longer returns expected 5-tuple.")
    _,scores,_,expl,cum=ret
    scores=np.asarray(scores,float); expl=np.asarray(expl,float); cum=np.asarray(cum,float)
    if scores.shape[1]<3: raise RuntimeError("B50 PCA returned <3 components.")
    work=scores.copy()
    mean_log=state["mean_log_gap"].to_numpy(float); gap_std=state["gap_std"].to_numpy(float)
    if b50.safe_corr(work[:,0],mean_log)<0: work[:,0]*=-1
    if b50.safe_corr(work[:,1],gap_std)<0: work[:,1]*=-1
    saved=state[["c1","c2","c3"]].to_numpy(float)
    c1=safe_corr(work[:,0],saved[:,0]); c2=safe_corr(work[:,1],saved[:,1]); c3=safe_corr(work[:,2],saved[:,2])
    e1=float(expl[0]); e2=float(cum[1]); e3=float(cum[2])
    t1=float(verdict["rank1_explained"]); t2=float(verdict["rank2_cumulative"]); t3=float(verdict["rank3_cumulative"])
    errors=[abs(e1-t1),abs(e2-t2),abs(e3-t3)]
    ok=(
        np.isfinite(c1) and c1>=0.999999 and
        np.isfinite(c2) and c2>=0.999999 and
        np.isfinite(c3) and abs(c3)>=0.999999 and
        max(errors)<=1e-10
    )
    return {
        "c1_corr":c1,"c2_corr":c2,"c3_corr_raw":c3,"c3_abs_corr":abs(c3),
        "rank1_recomputed":e1,"rank2_cumulative_recomputed":e2,"rank3_cumulative_recomputed":e3,
        "rank1_abs_error":errors[0],"rank2_abs_error":errors[1],"rank3_abs_error":errors[2],
        "reproduction_ok":bool(ok),
    }


def row_dim(metrics,d):
    r=metrics[metrics["dimension"]==d]
    if len(r)!=1: raise RuntimeError(f"Missing/duplicate d={d}")
    return r.iloc[0]


def adequate(r):
    k=float(r.get("knn10_overlap_mean",float("nan")))
    return bool(
        float(r["cumulative_variance"])>=ADEQUACY["min_cumulative_variance"] and
        float(r["pairwise_distance_spearman"])>=ADEQUACY["min_distance_spearman"] and
        np.isfinite(k) and k>=ADEQUACY["min_knn10_overlap"] and
        float(r["physical_step_spearman"])>=ADEQUACY["min_step_spearman"]
    )


def classify(metrics,repro_ok):
    if not repro_ok:
        return "INVALID_REPRODUCTION","Operator/PCA reconstruction failed integrity gates.",{}
    r2=row_dim(metrics,2); r3=row_dim(metrics,3)
    ok2,ok3=adequate(r2),adequate(r3)
    core={"d2_adequate":ok2,"d3_adequate":ok3,"d2":r2.to_dict(),"d3":r3.to_dict()}
    if ok2 and ok3:
        return "LOW_DIMENSION_ADEQUATE","Both B50 2D basin and 3D state embeddings pass all pre-declared preservation criteria.",core
    if (not ok2) and ok3:
        return "BASIN_2D_INADEQUATE_STATE_3D_ADEQUATE","The c1,c2 basin embedding fails, while c1,c2,c3 passes.",core
    k3=float(r3.get("knn10_overlap_mean",float("nan")))
    severe=(
        float(r3["cumulative_variance"])<SEVERE["max_cumulative_variance"] or
        float(r3["pairwise_distance_spearman"])<SEVERE["max_distance_spearman"] or
        (np.isfinite(k3) and k3<SEVERE["max_knn10_overlap"])
    )
    if severe:
        return "SEVERE_LOW_DIMENSION_INFORMATION_LOSS","The 3-PC state fails adequacy and crosses a pre-declared severe information-loss threshold.",core
    return "MODERATE_LOW_DIMENSION_INFORMATION_LOSS","The 3-PC state fails adequacy but not the severe thresholds.",core


def make_plots(out,expl,cum,metrics):
    rank=len(expl); dshow=min(rank,100)
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(1,dshow+1),expl[:dshow],marker=".",linewidth=1)
    plt.yscale("log"); plt.xlabel("principal component"); plt.ylabel("explained variance fraction")
    plt.title("B50.2 full PCA spectrum"); plt.grid(True); plt.tight_layout()
    plt.savefig(out/"b50_2_pca_spectrum.png",dpi=160); plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(np.arange(1,rank+1),cum,linewidth=1.5)
    for y in (0.50,0.75,0.90,0.95): plt.axhline(y,linestyle="--",linewidth=.8)
    plt.xlabel("retained PCs"); plt.ylabel("cumulative explained variance"); plt.ylim(0,1.01)
    plt.title("B50.2 cumulative operator variance"); plt.grid(True); plt.tight_layout()
    plt.savefig(out/"b50_2_cumulative_variance.png",dpi=160); plt.close()

    m=metrics[~metrics["is_full_rank_reference"]]
    if not m.empty:
        plt.figure(figsize=(9,5))
        plt.plot(m["dimension"],m["pairwise_distance_spearman"],marker="o",label="pairwise distance")
        plt.plot(m["dimension"],m["physical_step_spearman"],marker="o",label="physical step")
        plt.axhline(.90,linestyle="--",linewidth=.8)
        plt.xlabel("retained PCs"); plt.ylabel("Spearman vs full geometry"); plt.ylim(-.05,1.02)
        plt.title("B50.2 geometry preservation"); plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(out/"b50_2_distance_step_preservation.png",dpi=160); plt.close()

        cols=[c for c in m.columns if c.startswith("knn") and c.endswith("_overlap_mean")]
        if cols:
            plt.figure(figsize=(9,5))
            for c in cols: plt.plot(m["dimension"],m[c],marker="o",label=c.replace("_overlap_mean",""))
            plt.axhline(.70,linestyle="--",linewidth=.8)
            plt.xlabel("retained PCs"); plt.ylabel("mean kNN overlap"); plt.ylim(-.05,1.02)
            plt.title("B50.2 neighborhood preservation"); plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(out/"b50_2_knn_preservation.png",dpi=160); plt.close()


def audit(args):
    out=Path(args.output_dir); stack_path=out/"b50_2_operator_stack.npz"
    if not stack_path.exists(): raise FileNotFoundError(f"{stack_path}; run --mode extract first.")
    state=pd.read_csv(args.state_dataset); validate_state_dataset(state)
    verdict=json.loads(Path(args.verdict_json).read_text(encoding="utf-8"))
    b50=import_b50(Path(args.b50_script))
    with np.load(stack_path,allow_pickle=False) as z:
        P=np.asarray(z["P_stack"],np.float64); labels=[str(x) for x in z["labels"]]; gb=int(z["gap_bins"][0])
    labels_exact=labels==state["range_name"].astype(str).tolist()
    if gb!=args.gap_bins: raise RuntimeError(f"gap_bins mismatch: stack={gb}, cli={args.gap_bins}")
    if P.shape[0]!=len(state): raise RuntimeError(f"rows mismatch: stack={P.shape[0]}, states={len(state)}")

    print("=== B50.2 DIMENSIONALITY AUDIT ===")
    print(f"operator stack : {P.shape}")
    repro=reproduce_b50_pca(b50,P,state,verdict)
    repro["labels_exact"]=labels_exact
    repro_ok=bool(repro["reproduction_ok"] and labels_exact)
    print(f"reproduction OK: {repro_ok}")

    t0=time.time()
    scores,S,expl,cum=full_pca(P)
    rank=scores.shape[1]
    spectrum=pd.DataFrame({"pc":np.arange(1,rank+1),"singular_value":S,"explained_variance_fraction":expl,"cumulative_variance":cum})
    atomic_csv(out/"b50_2_pca_spectrum.csv",spectrum)

    targets={f"d{int(t*100)}":dimension_for_fraction(cum,t) for t in (.50,.75,.90,.95,.99)}
    dims=sorted(set([d for d in args.dimensions if d<=rank]+[2,3,rank]))
    metrics=embedding_metrics(scores,cum,dims,args.knn_k)
    atomic_csv(out/"b50_2_embedding_metrics.csv",metrics)

    label,reason,core=classify(metrics,repro_ok)
    tested=[int(r["dimension"]) for _,r in metrics.iterrows() if adequate(r)]
    smallest=min(tested) if tested else None

    summary={
        "version":VERSION,"finished":now_s(),"outcome_blind_to_b59":True,
        "operator_shape":list(P.shape),"full_pca_rank":rank,
        "reproduction":repro,
        "effective_rank_shannon":effective_rank(expl),
        "participation_ratio":participation_ratio(expl),
        "variance_dimensions":targets,
        "smallest_tested_dimension_meeting_all_adequacy_metrics":smallest,
        "adequacy_thresholds":ADEQUACY,"severe_thresholds":SEVERE,
        "core_d2_d3":core,"final_verdict":label,"final_reason":reason,
        "scientific_boundary":"Representation adequacy only; no B59/B59.1 labels or predictive outcomes used.",
        "elapsed_audit_seconds":time.time()-t0,
    }
    atomic_json(out/"b50_2_summary.json",summary)
    make_plots(out,expl,cum,metrics)

    def line(d):
        r=row_dim(metrics,d); k=float(r.get("knn10_overlap_mean",float("nan")))
        return (f"d={d}: variance={float(r['cumulative_variance']):.6f}, "
                f"dist_spearman={float(r['pairwise_distance_spearman']):.6f}, "
                f"knn10={k:.6f}, step_spearman={float(r['physical_step_spearman']):.6f}")

    report=f"""B50.2 — State Dimensionality & Embedding Adequacy Audit
=========================================================

Integrity
---------
operator stack shape     = {P.shape}
labels exact             = {labels_exact}
B50/PCA reproduction OK  = {repro_ok}
c1 corr                  = {repro.get('c1_corr')}
c2 corr                  = {repro.get('c2_corr')}
|c3 corr|                = {repro.get('c3_abs_corr')}

Full operator dimensionality
----------------------------
PCA rank                 = {rank}
Shannon effective rank   = {summary['effective_rank_shannon']}
participation ratio      = {summary['participation_ratio']}
d50                      = {targets['d50']}
d75                      = {targets['d75']}
d90                      = {targets['d90']}
d95                      = {targets['d95']}
d99                      = {targets['d99']}

Critical B50 truncations
------------------------
{line(2)}
{line(3)}

Smallest TESTED dimension satisfying all adequacy criteria
----------------------------------------------------------
{smallest}

FINAL VERDICT: {label}

REASON:
{reason}

Interpretation boundary
-----------------------
This is an outcome-blind representation audit. It does not validate B59,
forecasting skill, individual-prime prediction, or a theorem.
"""
    atomic_text(out/"b50_2_verdict.txt",report)
    print(line(2)); print(line(3))
    print(f"effective rank : {summary['effective_rank_shannon']:.3f}")
    print(f"d50/d75/d90    : {targets['d50']}/{targets['d75']}/{targets['d90']}")
    print(f"FINAL VERDICT  : {label}")
    print(f"verdict        : {out/'b50_2_verdict.txt'}")
    return summary


def parse_args():
    p=argparse.ArgumentParser(description="B50.2 State Dimensionality & Embedding Adequacy Audit")
    p.add_argument("--mode",choices=["preflight","extract","audit","all"],default="preflight")
    p.add_argument("--ranges-root",default=None)
    p.add_argument("--state-dataset",required=True)
    p.add_argument("--verdict-json",required=True)
    p.add_argument("--b50-script",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--gap-bins",type=int,default=64)
    p.add_argument("--checkpoint-every",type=int,default=10)
    p.add_argument("--reuse-existing",action="store_true")
    p.add_argument("--dimensions",default="2,3,5,8,10,16,24,32,48,64,96,128,192,256")
    p.add_argument("--knn-k",default="5,10,20")
    a=p.parse_args()
    a.dimensions=parse_int_list(a.dimensions); a.knn_k=parse_int_list(a.knn_k)
    if a.mode in ("preflight","extract","all") and not a.ranges_root:
        p.error("--ranges-root is required for preflight/extract/all")
    return a


def main():
    a=parse_args()
    if a.mode=="preflight":
        preflight(a); return 0
    if a.mode in ("extract","all"):
        extract(a)
    if a.mode in ("audit","all"):
        audit(a)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
