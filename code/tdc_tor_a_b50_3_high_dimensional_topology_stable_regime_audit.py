#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDC TOR A / B50.3 — High-Dimensional State Topology & Stable Regime Audit
==========================================================================

Outcome-blind audit of stable regime topology in the higher-dimensional PCA
geometry recovered by B50.2.

Inputs
------
- b50_2_operator_stack.npz
- b50_2_summary.json
- original b50_state_dataset.csv (ordering / integrity only)

Default audited dimensions are derived from B50.2:
  d50, d75, first tested adequate dimension, d90,
  rounded Shannon effective rank, d95.

Independent routes
------------------
- k-means (multiple deterministic restarts)
- Ward hierarchical clustering
- kNN graph topology diagnostics

Stability diagnostics
---------------------
For k=2..8 and every audited dimension:
- silhouette
- occupancy / smallest cluster fraction
- physical adjacent-label crossing rate (diagnostic only)
- subsample projection stability (ARI)
- k-means restart stability
- Ward vs k-means ARI
- adjacent-dimension ARI

Pre-declared stable regime criteria for candidate k
---------------------------------------------------
ALL must hold:
- median silhouette                 >= 0.10
- median cross-method ARI           >= 0.75
- median adjacent-dimension ARI     >= 0.75
- median subsample projection ARI   >= 0.70
- balanced config fraction          >= 0.80
where balanced means minimum cluster fraction >= 0.05.

Final verdicts
--------------
STABLE_REGIME_TOPOLOGY
MULTISCALE_STABLE_REGIME_TOPOLOGY
NO_STABLE_REGIME_TOPOLOGY
INVALID_INPUT_REPRODUCTION

Scientific boundary
-------------------
This is a topology / representation audit. It does not use B59/B59.1 scores or
selected/anti cohorts to choose dimensions or k. It is not forecasting evidence,
individual-prime prediction, or a theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "B50.3_high_dimensional_topology_stable_regime_audit_v1"

THRESH = {
    "min_median_silhouette": 0.10,
    "min_median_cross_method_ari": 0.75,
    "min_median_adjacent_dimension_ari": 0.75,
    "min_median_subsample_ari": 0.70,
    "min_balanced_config_fraction": 0.80,
    "min_cluster_fraction": 0.05,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True))


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def parse_int_list(s: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("Expected a non-empty positive integer list.")
    return vals


def full_pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S[None, :]
    eig = S * S
    explained = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    cumulative = np.cumsum(explained)
    return scores, explained, cumulative


def derive_auto_dimensions(summary: Dict[str, Any], rank: int) -> List[int]:
    vd = summary.get("variance_dimensions", {})
    vals = [
        vd.get("d50"),
        vd.get("d75"),
        summary.get("smallest_tested_dimension_meeting_all_adequacy_metrics"),
        vd.get("d90"),
        round(float(summary.get("effective_rank_shannon", 0.0))) if summary.get("effective_rank_shannon") is not None else None,
        vd.get("d95"),
    ]
    dims = sorted(set(int(x) for x in vals if x is not None and 2 <= int(x) <= rank))
    if not dims:
        raise RuntimeError("Could not derive dimensions from B50.2 summary.")
    return dims


def validate_state(df: pd.DataFrame) -> None:
    if "range_name" not in df.columns or "range_index" not in df.columns:
        raise ValueError("State dataset must contain range_name and range_index.")
    idx = pd.to_numeric(df["range_index"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(len(df))):
        raise ValueError("range_index is not exactly 0..n-1.")
    if df["range_name"].duplicated().any():
        raise ValueError("Duplicate range_name in state dataset.")


def occupancy(labels: np.ndarray) -> Dict[str, Any]:
    _, counts = np.unique(labels, return_counts=True)
    frac = counts.astype(float) / counts.sum()
    return {
        "min_cluster_fraction": float(frac.min()),
        "max_cluster_fraction": float(frac.max()),
        "singleton_clusters": int(np.sum(counts == 1)),
        "balanced": bool(frac.min() >= THRESH["min_cluster_fraction"]),
    }


def physical_crossing_rate(labels: np.ndarray) -> Tuple[int, float]:
    j = labels[:-1] != labels[1:]
    return int(j.sum()), float(j.mean())


def centers_from_labels(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    C = []
    for c in range(k):
        pts = X[labels == c]
        if len(pts) == 0:
            raise RuntimeError(f"Empty cluster {c}")
        C.append(pts.mean(axis=0))
    return np.vstack(C)


def nearest_centroid(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d2 = (
        np.sum(X * X, axis=1)[:, None]
        + np.sum(C * C, axis=1)[None, :]
        - 2.0 * (X @ C.T)
    )
    return np.argmin(d2, axis=1).astype(int)


def fit_kmeans(X: np.ndarray, k: int, seed: int, restarts: int):
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=restarts,
        max_iter=500,
        random_state=seed,
        algorithm="lloyd",
    )
    labels = model.fit_predict(X).astype(int)
    return labels, np.asarray(model.cluster_centers_, float), float(model.inertia_)


def fit_ward(X: np.ndarray, k: int):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X).astype(int)
    return labels, centers_from_labels(X, labels, k)


def kmeans_restart_stability(X: np.ndarray, k: int, seed: int, repeats: int) -> float:
    sols = []
    for r in range(repeats):
        m = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=1,
            max_iter=500,
            random_state=seed + 7919 * r,
            algorithm="lloyd",
        )
        sols.append(m.fit_predict(X).astype(int))
    ref = sols[0]
    return float(np.median([adjusted_rand_score(ref, s) for s in sols]))


def subsample_stability(
    X: np.ndarray,
    full_labels: np.ndarray,
    method: str,
    k: int,
    seed: int,
    repeats: int,
    fraction: float,
    kmeans_restarts: int,
) -> Tuple[float, float]:
    n = len(X)
    m = max(k + 2, int(round(n * fraction)))
    if m >= n:
        m = n - 1
    rng = np.random.default_rng(seed)
    vals = []

    for r in range(repeats):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        Xs = X[idx]
        if method == "kmeans":
            labs_s, C, _ = fit_kmeans(
                Xs, k, seed + 104729 * (r + 1), max(3, min(kmeans_restarts, 5))
            )
        elif method == "ward":
            labs_s, C = fit_ward(Xs, k)
        else:
            raise ValueError(method)

        projected = nearest_centroid(X, C)
        vals.append(adjusted_rand_score(full_labels, projected))

    return float(np.median(vals)), float(np.mean(vals))


def knn_graph_metrics(X: np.ndarray, graph_k: List[int]) -> pd.DataFrame:
    D = squareform(pdist(X, metric="euclidean"))
    order = np.argsort(D, axis=1)
    n = len(X)
    rows = []

    for k in graph_k:
        if k >= n:
            continue
        nbr = order[:, 1:k+1]
        directed = np.zeros((n, n), dtype=bool)
        directed[np.repeat(np.arange(n), k), nbr.reshape(-1)] = True
        sym = directed | directed.T
        mutual = directed & directed.T
        n_comp, comp = connected_components(csr_matrix(sym.astype(np.int8)), directed=False)
        counts = np.bincount(comp)
        deg = sym.sum(axis=1)
        rows.append({
            "knn_k": k,
            "n_components": int(n_comp),
            "largest_component_fraction": float(counts.max() / n),
            "mutual_neighbor_fraction": float(mutual.sum() / max(directed.sum(), 1)),
            "degree_mean": float(deg.mean()),
            "degree_std": float(deg.std()),
            "degree_min": int(deg.min()),
            "degree_max": int(deg.max()),
        })
    return pd.DataFrame(rows)


def run(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stack_path = Path(args.operator_stack)
    summary_path = Path(args.b50_2_summary)
    state_path = Path(args.state_dataset)

    b502 = json.loads(summary_path.read_text(encoding="utf-8"))
    state = pd.read_csv(state_path)
    validate_state(state)

    with np.load(stack_path, allow_pickle=False) as z:
        P = np.asarray(z["P_stack"], dtype=np.float64)
        labels = [str(x) for x in z["labels"]]

    labels_exact = labels == state["range_name"].astype(str).tolist()
    if P.shape[0] != len(state):
        labels_exact = False

    if not labels_exact:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_INPUT_REPRODUCTION",
            "final_reason": "Operator-stack rows/labels do not exactly reproduce B50 state order.",
        }
        atomic_json(out / "b50_3_summary.json", summary)
        atomic_text(out / "b50_3_verdict.txt", "FINAL VERDICT: INVALID_INPUT_REPRODUCTION\n")
        print("FINAL VERDICT: INVALID_INPUT_REPRODUCTION")
        return summary

    print("=== B50.3 HIGH-DIMENSIONAL TOPOLOGY AUDIT ===")
    print(f"operator stack : {P.shape}")
    print(f"labels exact   : {labels_exact}")

    scores, explained, cumulative = full_pca(P)
    rank = scores.shape[1]
    dims = derive_auto_dimensions(b502, rank) if args.dimensions.lower() == "auto" else [d for d in parse_int_list(args.dimensions) if d <= rank]
    k_values = [k for k in args.k_values if k < len(state)]

    print(f"dimensions     : {dims}")
    print(f"k values       : {k_values}")
    print(f"subsamples     : {args.subsample_repeats} x {args.subsample_fraction:.2f}")

    protocol = {
        "version": VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outcome_blind_to_b59": True,
        "inputs": {
            "operator_stack": str(stack_path),
            "operator_stack_sha256": sha256_file(stack_path),
            "b50_2_summary": str(summary_path),
            "b50_2_summary_sha256": sha256_file(summary_path),
            "state_dataset": str(state_path),
            "state_dataset_sha256": sha256_file(state_path),
        },
        "dimensions": dims,
        "k_values": k_values,
        "graph_k": args.graph_k,
        "kmeans_restarts": args.kmeans_restarts,
        "subsample_repeats": args.subsample_repeats,
        "subsample_fraction": args.subsample_fraction,
        "seed": args.seed,
        "stable_thresholds": THRESH,
    }
    atomic_json(out / "b50_3_protocol_manifest.json", protocol)

    partition_rows = []
    assignments = {}
    graph_rows = []
    total = len(dims) * len(k_values) * 2
    job = 0
    t_all = time.time()

    for d in dims:
        X = scores[:, :d]
        D = squareform(pdist(X, metric="euclidean"))
        g = knn_graph_metrics(X, args.graph_k)
        g.insert(0, "dimension", d)
        graph_rows.append(g)

        for k in k_values:
            job += 1
            print(f"[{job:03d}/{total:03d}] d={d} k={k} kmeans", flush=True)
            t0 = time.time()
            km_labels, km_centers, inertia = fit_kmeans(X, k, args.seed + 100003*d + 1009*k, args.kmeans_restarts)
            km_restart = kmeans_restart_stability(X, k, args.seed + 300007*d + 2003*k, args.restart_stability_repeats)
            sub_med, sub_mean = subsample_stability(
                X, km_labels, "kmeans", k,
                args.seed + 500009*d + 3001*k,
                args.subsample_repeats, args.subsample_fraction, args.kmeans_restarts
            )
            occ = occupancy(km_labels)
            n_cross, cross_rate = physical_crossing_rate(km_labels)
            partition_rows.append({
                "dimension": d, "k": k, "method": "kmeans",
                "silhouette": float(silhouette_score(D, km_labels, metric="precomputed")),
                "inertia": inertia,
                "restart_stability_ari_median": km_restart,
                "subsample_ari_median": sub_med,
                "subsample_ari_mean": sub_mean,
                "physical_crossings_diagnostic": n_cross,
                "physical_crossing_rate_diagnostic": cross_rate,
                **occ,
                "elapsed_seconds": time.time() - t0,
            })
            assignments[("kmeans", d, k)] = km_labels

            job += 1
            print(f"[{job:03d}/{total:03d}] d={d} k={k} ward", flush=True)
            t0 = time.time()
            w_labels, _ = fit_ward(X, k)
            sub_med, sub_mean = subsample_stability(
                X, w_labels, "ward", k,
                args.seed + 700001*d + 4001*k,
                args.subsample_repeats, args.subsample_fraction, args.kmeans_restarts
            )
            occ = occupancy(w_labels)
            n_cross, cross_rate = physical_crossing_rate(w_labels)
            partition_rows.append({
                "dimension": d, "k": k, "method": "ward",
                "silhouette": float(silhouette_score(D, w_labels, metric="precomputed")),
                "inertia": float("nan"),
                "restart_stability_ari_median": float("nan"),
                "subsample_ari_median": sub_med,
                "subsample_ari_mean": sub_mean,
                "physical_crossings_diagnostic": n_cross,
                "physical_crossing_rate_diagnostic": cross_rate,
                **occ,
                "elapsed_seconds": time.time() - t0,
            })
            assignments[("ward", d, k)] = w_labels

    partitions = pd.DataFrame(partition_rows)
    graph_df = pd.concat(graph_rows, ignore_index=True)
    atomic_csv(out / "b50_3_partition_metrics.csv", partitions)
    atomic_csv(out / "b50_3_knn_graph_metrics.csv", graph_df)

    cross_rows = []
    for d in dims:
        for k in k_values:
            cross_rows.append({
                "dimension": d,
                "k": k,
                "ari_kmeans_vs_ward": float(adjusted_rand_score(assignments[("kmeans", d, k)], assignments[("ward", d, k)])),
            })
    cross = pd.DataFrame(cross_rows)
    atomic_csv(out / "b50_3_cross_method_ari.csv", cross)

    adj_rows = []
    for method in ("kmeans", "ward"):
        for k in k_values:
            for d0, d1 in zip(dims[:-1], dims[1:]):
                adj_rows.append({
                    "method": method,
                    "k": k,
                    "dimension_from": d0,
                    "dimension_to": d1,
                    "ari": float(adjusted_rand_score(assignments[(method, d0, k)], assignments[(method, d1, k)])),
                })
    adjacent = pd.DataFrame(adj_rows)
    atomic_csv(out / "b50_3_adjacent_dimension_ari.csv", adjacent)

    k_rows = []
    for k in k_values:
        p = partitions[partitions["k"] == k]
        cm = cross[cross["k"] == k]
        ad = adjacent[adjacent["k"] == k]
        med_sil = float(p["silhouette"].median())
        med_cm = float(cm["ari_kmeans_vs_ward"].median())
        med_ad = float(ad["ari"].median())
        med_sub = float(p["subsample_ari_median"].median())
        bal_frac = float(p["balanced"].astype(float).mean())
        stable = bool(
            med_sil >= THRESH["min_median_silhouette"] and
            med_cm >= THRESH["min_median_cross_method_ari"] and
            med_ad >= THRESH["min_median_adjacent_dimension_ari"] and
            med_sub >= THRESH["min_median_subsample_ari"] and
            bal_frac >= THRESH["min_balanced_config_fraction"]
        )
        k_rows.append({
            "k": k,
            "median_silhouette": med_sil,
            "median_cross_method_ari": med_cm,
            "median_adjacent_dimension_ari": med_ad,
            "median_subsample_ari": med_sub,
            "balanced_config_fraction": bal_frac,
            "median_physical_crossing_rate_diagnostic": float(p["physical_crossing_rate_diagnostic"].median()),
            "qualifies_stable_regime": stable,
        })

    k_summary = pd.DataFrame(k_rows)
    atomic_csv(out / "b50_3_k_stability_summary.csv", k_summary)
    stable_k = k_summary.loc[k_summary["qualifies_stable_regime"], "k"].astype(int).tolist()

    if len(stable_k) == 0:
        verdict = "NO_STABLE_REGIME_TOPOLOGY"
        reason = "No k satisfies all pre-declared cross-method, cross-dimension, subsample, silhouette and occupancy criteria."
    elif len(stable_k) == 1:
        verdict = "STABLE_REGIME_TOPOLOGY"
        reason = f"Exactly one candidate regime count k={stable_k[0]} satisfies all pre-declared criteria."
    else:
        verdict = "MULTISCALE_STABLE_REGIME_TOPOLOGY"
        reason = f"Multiple candidate regime counts {stable_k} satisfy all pre-declared criteria; no single k is promoted here."

    long_rows = []
    for (method, d, k), labs in assignments.items():
        for i, lab in enumerate(labs):
            long_rows.append({
                "method": method, "dimension": d, "k": k,
                "range_index": int(state.iloc[i]["range_index"]),
                "range_name": str(state.iloc[i]["range_name"]),
                "cluster": int(lab),
            })
    atomic_csv(out / "b50_3_assignments_long.csv", pd.DataFrame(long_rows))

    summary = {
        "version": VERSION,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outcome_blind_to_b59": True,
        "operator_shape": list(P.shape),
        "pca_rank": int(rank),
        "dimensions": dims,
        "k_values": k_values,
        "stable_k_values": stable_k,
        "stable_thresholds": THRESH,
        "final_verdict": verdict,
        "final_reason": reason,
        "elapsed_seconds": float(time.time() - t_all),
        "scientific_boundary": "Topology/representation audit only; no B59/B59.1 labels or predictive outcomes used for model selection.",
    }
    atomic_json(out / "b50_3_summary.json", summary)

    # Plots
    plt.figure(figsize=(9,5))
    plt.plot(k_summary["k"], k_summary["median_silhouette"], marker="o")
    plt.axhline(THRESH["min_median_silhouette"], linestyle="--", linewidth=.8)
    plt.xlabel("candidate k"); plt.ylabel("median silhouette")
    plt.title("B50.3 regime separation by k"); plt.grid(True); plt.tight_layout()
    plt.savefig(out / "b50_3_silhouette_by_k.png", dpi=160); plt.close()

    plt.figure(figsize=(9,5))
    plt.plot(k_summary["k"], k_summary["median_cross_method_ari"], marker="o", label="cross-method ARI")
    plt.plot(k_summary["k"], k_summary["median_adjacent_dimension_ari"], marker="o", label="adjacent-dim ARI")
    plt.plot(k_summary["k"], k_summary["median_subsample_ari"], marker="o", label="subsample ARI")
    plt.axhline(.75, linestyle="--", linewidth=.8)
    plt.xlabel("candidate k"); plt.ylabel("agreement / stability"); plt.ylim(-.05,1.02)
    plt.title("B50.3 convergent stability"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out / "b50_3_stability_by_k.png", dpi=160); plt.close()

    pivot = cross.pivot(index="dimension", columns="k", values="ari_kmeans_vs_ward")
    plt.figure(figsize=(9,6))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="ARI")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("k"); plt.ylabel("PCA dimension")
    plt.title("B50.3 Ward vs k-means agreement"); plt.tight_layout()
    plt.savefig(out / "b50_3_cross_method_ari_heatmap.png", dpi=160); plt.close()

    plt.figure(figsize=(9,5))
    for gk in sorted(graph_df["knn_k"].unique()):
        g = graph_df[graph_df["knn_k"] == gk].sort_values("dimension")
        plt.plot(g["dimension"], g["largest_component_fraction"], marker="o", label=f"kNN {gk}")
    plt.xlabel("PCA dimension"); plt.ylabel("largest graph component fraction"); plt.ylim(0,1.02)
    plt.title("B50.3 local kNN graph connectivity"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out / "b50_3_knn_graph_connectivity.png", dpi=160); plt.close()

    lines = [
        "B50.3 — High-Dimensional State Topology & Stable Regime Audit",
        "=============================================================",
        "",
        "Integrity",
        "---------",
        f"operator stack shape      = {P.shape}",
        f"labels exact              = {labels_exact}",
        f"PCA rank                  = {rank}",
        f"dimensions                = {dims}",
        f"k values                  = {k_values}",
        "",
        "Candidate k summary",
        "-------------------",
    ]
    for _, r in k_summary.iterrows():
        lines.append(
            f"k={int(r['k'])}: sil={r['median_silhouette']:.6f}, "
            f"cross_method_ARI={r['median_cross_method_ari']:.6f}, "
            f"adj_dim_ARI={r['median_adjacent_dimension_ari']:.6f}, "
            f"subsample_ARI={r['median_subsample_ari']:.6f}, "
            f"balanced={r['balanced_config_fraction']:.6f}, "
            f"stable={bool(r['qualifies_stable_regime'])}"
        )
    lines += [
        "",
        f"stable k values           = {stable_k}",
        "",
        f"FINAL VERDICT: {verdict}",
        "",
        "REASON:",
        reason,
        "",
        "Interpretation boundary",
        "-----------------------",
        "Outcome-blind topology/representation audit only.",
        "No B59/B59.1 selector or predictive outcome is used for choosing k.",
    ]
    atomic_text(out / "b50_3_verdict.txt", "\n".join(lines) + "\n")

    print("=== B50.3 RESULT ===")
    print(f"stable k values : {stable_k}")
    print(f"FINAL VERDICT   : {verdict}")
    print(f"elapsed         : {time.time() - t_all:.1f}s")
    print(f"verdict         : {out / 'b50_3_verdict.txt'}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="B50.3 High-Dimensional State Topology & Stable Regime Audit")
    p.add_argument("--operator-stack", required=True)
    p.add_argument("--b50-2-summary", required=True)
    p.add_argument("--state-dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dimensions", default="auto")
    p.add_argument("--k-values", default="2,3,4,5,6,7,8")
    p.add_argument("--graph-k", default="5,10,20")
    p.add_argument("--kmeans-restarts", type=int, default=12)
    p.add_argument("--restart-stability-repeats", type=int, default=12)
    p.add_argument("--subsample-repeats", type=int, default=20)
    p.add_argument("--subsample-fraction", type=float, default=.80)
    p.add_argument("--seed", type=int, default=50303)
    a = p.parse_args()
    a.k_values = parse_int_list(a.k_values)
    a.graph_k = parse_int_list(a.graph_k)
    if a.kmeans_restarts < 2 or a.restart_stability_repeats < 2:
        p.error("restart counts must be >=2")
    if a.subsample_repeats < 5:
        p.error("--subsample-repeats must be >=5")
    if not (0.5 <= a.subsample_fraction < 1.0):
        p.error("--subsample-fraction must be in [0.5,1.0)")
    return a


def main():
    return 0 if run(parse_args()) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
