#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDC TOR A / B50.5 — Continuous Front Null & Prospective Stability Validation

Confirmatory, outcome-blind validation of the frozen B50.4 continuous-front
endpoint. No B59/B59.1 selector or predictive outcome is used.

Mandatory gates:
1) exact B50.4 front reproduction,
2) circular-shift transition-alignment null,
3) global physical-order permutation null,
4) representation holdout,
5) rolling-origin prospective physical holdout.

Secondary diagnostic: block-preserving order null for block sizes 10,25,50.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "B50.5_continuous_front_null_prospective_validation_v1"

MANDATORY = {
    "alignment_min_spearman": 0.75,
    "alignment_min_topq_jaccard": 0.50,
    "alignment_max_p_upper": 0.01,
    "order_max_p_upper_concentration": 0.05,
    "representation_holdout_min_spearman": 0.75,
    "representation_holdout_min_topq_jaccard": 0.50,
    "prospective_min_spearman": 0.70,
    "prospective_min_topq_jaccard": 0.40,
    "prospective_min_passing_folds": 2,
}

DEFAULT_BLOCK_SIZES = [10, 25, 50]
DEFAULT_PROSPECTIVE_CUTS = [(0.50, 0.67), (0.67, 0.83), (0.83, 1.00)]
FRONT_CHANNELS = [
    "speed",
    "acceleration",
    "turn_change",
    "density_scale_change",
    "diffusion_step",
]


def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def top_quantile_mask(x: np.ndarray, q: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x >= float(np.quantile(x, q))


def jaccard_bool(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.sum(a | b))
    return float(np.sum(a & b) / union) if union else float("nan")


def p_upper(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def p_lower(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return float("nan")
    return float((1 + np.sum(null <= observed)) / (len(null) + 1))


def validate_state_dataset(df: pd.DataFrame) -> None:
    for c in ("range_name", "range_index"):
        if c not in df.columns:
            raise ValueError(f"State dataset missing {c}")
    idx = pd.to_numeric(df["range_index"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(len(df))):
        raise ValueError("range_index is not exactly 0..n-1")
    if df["range_name"].duplicated().any():
        raise ValueError("Duplicate range_name in state dataset")


def import_b504(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("tdc_b504_frozen_for_b505", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import B50.4 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    required = ["full_pca", "build_knn_graph", "diffusion_modes", "transition_front_features"]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise AttributeError(f"B50.4 missing required functions: {missing}")
    return mod


# ---------------------------------------------------------------------------
# Frozen B50.4 reconstruction
# ---------------------------------------------------------------------------

def reconstruct_configs(
    b504,
    P_stack: np.ndarray,
    dimensions: List[int],
    graph_k_values: List[int],
    diffusion_modes_n: int,
) -> Dict[str, Dict[str, Any]]:
    scores, _, _ = b504.full_pca(P_stack)
    configs: Dict[str, Dict[str, Any]] = {}
    total = len(dimensions) * len(graph_k_values)
    job = 0
    for d in dimensions:
        X = np.asarray(scores[:, :d], dtype=float)
        D = squareform(pdist(X, metric="euclidean"))
        for gk in graph_k_values:
            job += 1
            config_id = f"d{d}_knn{gk}"
            print(f"    reconstruct {job}/{total}: {config_id}", flush=True)
            W, _, _, local_radius = b504.build_knn_graph(D, gk)
            eigvals, modes = b504.diffusion_modes(W, diffusion_modes_n)
            diff_coords = modes * eigvals[None, :]
            fronts = b504.transition_front_features(X, local_radius, diff_coords)
            configs[config_id] = {
                "dimension": int(d),
                "graph_k": int(gk),
                "X": X,
                "local_radius": np.asarray(local_radius, dtype=float),
                "diffusion_coords": np.asarray(diff_coords, dtype=float),
                "fronts": fronts.copy(),
                "front_score": fronts["front_score"].to_numpy(float),
            }
    return configs


def reproduction_gate(configs: Dict[str, Dict[str, Any]], saved_long: pd.DataFrame):
    required = {"config_id", "transition_index", "front_score"}
    if not required.issubset(saved_long.columns):
        raise ValueError(f"B50.4 front CSV missing {sorted(required - set(saved_long.columns))}")
    rows = []
    for cid, c in sorted(configs.items()):
        saved = saved_long[saved_long["config_id"].astype(str) == cid].sort_values("transition_index")
        rec = np.asarray(c["front_score"], dtype=float)
        if len(saved) != len(rec):
            rows.append({"config_id": cid, "n_saved": len(saved), "n_recomputed": len(rec), "max_abs_error": float("inf"), "spearman": float("nan"), "pass": False})
            continue
        y = saved["front_score"].to_numpy(float)
        err = np.abs(y - rec)
        rho = safe_spearman(y, rec)
        ok = bool(np.max(err) <= 1e-6 and np.isfinite(rho) and rho >= 0.999999)
        rows.append({
            "config_id": cid,
            "n_saved": len(saved),
            "n_recomputed": len(rec),
            "max_abs_error": float(np.max(err)),
            "mean_abs_error": float(np.mean(err)),
            "spearman": rho,
            "pass": ok,
        })
    df = pd.DataFrame(rows)
    return bool(df["pass"].all()), df


# ---------------------------------------------------------------------------
# Stability statistics
# ---------------------------------------------------------------------------

def pairwise_stability(score_map: Dict[str, np.ndarray], q: float):
    ids = sorted(score_map)
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = score_map[ids[i]]
            b = score_map[ids[j]]
            rows.append({
                "config_a": ids[i],
                "config_b": ids[j],
                "spearman": safe_spearman(a, b),
                "topq_jaccard": jaccard_bool(top_quantile_mask(a, q), top_quantile_mask(b, q)),
            })
    df = pd.DataFrame(rows)
    return df, {
        "median_spearman": float(df["spearman"].median()),
        "median_topq_jaccard": float(df["topq_jaccard"].median()),
    }


def front_energy_concentration(score: np.ndarray, q: float) -> float:
    score = np.asarray(score, dtype=float)
    e = score * score
    den = float(np.sum(e))
    if den <= 1e-30:
        return 0.0
    m = top_quantile_mask(score, q)
    return float(np.sum(e[m]) / den)


def aggregate_order_stats(score_map: Dict[str, np.ndarray], q: float) -> Dict[str, float]:
    return {
        "median_energy_concentration": float(np.median([front_energy_concentration(x, q) for x in score_map.values()])),
        "median_of_config_medians": float(np.median([np.median(x) for x in score_map.values()])),
        "median_q95_front_score": float(np.median([np.quantile(x, q) for x in score_map.values()])),
    }


# ---------------------------------------------------------------------------
# Nulls
# ---------------------------------------------------------------------------

def circular_shift_alignment_null(observed: Dict[str, np.ndarray], q: float, permutations: int, seed: int):
    ids = sorted(observed)
    n = len(next(iter(observed.values())))
    _, obs = pairwise_stability(observed, q)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(permutations):
        shifted = {cid: np.roll(observed[cid], int(rng.integers(1, n))) for cid in ids}
        _, st = pairwise_stability(shifted, q)
        rows.append({"replicate": r, "median_spearman": st["median_spearman"], "median_topq_jaccard": st["median_topq_jaccard"]})
    df = pd.DataFrame(rows)
    summary = {
        "observed_median_spearman": obs["median_spearman"],
        "observed_median_topq_jaccard": obs["median_topq_jaccard"],
        "null_mean_spearman": float(df["median_spearman"].mean()),
        "null_mean_topq_jaccard": float(df["median_topq_jaccard"].mean()),
        "p_upper_spearman": p_upper(obs["median_spearman"], df["median_spearman"].to_numpy(float)),
        "p_upper_topq_jaccard": p_upper(obs["median_topq_jaccard"], df["median_topq_jaccard"].to_numpy(float)),
    }
    return df, summary


def front_score_for_order(b504, c: Dict[str, Any], order: np.ndarray) -> np.ndarray:
    df = b504.transition_front_features(
        c["X"][order],
        c["local_radius"][order],
        c["diffusion_coords"][order],
    )
    return df["front_score"].to_numpy(float)


def order_score_map(b504, configs: Dict[str, Dict[str, Any]], order: np.ndarray):
    return {cid: front_score_for_order(b504, c, order) for cid, c in configs.items()}


def global_order_null(b504, configs, q: float, permutations: int, seed: int):
    n = len(next(iter(configs.values()))["X"])
    identity = np.arange(n, dtype=int)
    observed = aggregate_order_stats(order_score_map(b504, configs, identity), q)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(permutations):
        st = aggregate_order_stats(order_score_map(b504, configs, rng.permutation(n)), q)
        rows.append({"replicate": r, **st})
        if (r + 1) % 25 == 0 or r + 1 == permutations:
            print(f"    global-order null {r+1}/{permutations}", flush=True)
    df = pd.DataFrame(rows)
    summary = {
        "observed": observed,
        "null_mean_energy_concentration": float(df["median_energy_concentration"].mean()),
        "null_mean_median_score": float(df["median_of_config_medians"].mean()),
        "p_upper_energy_concentration": p_upper(observed["median_energy_concentration"], df["median_energy_concentration"].to_numpy(float)),
        "p_lower_median_score": p_lower(observed["median_of_config_medians"], df["median_of_config_medians"].to_numpy(float)),
    }
    return df, summary


def shuffled_block_order(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    blocks = [np.arange(i, min(i + block_size, n), dtype=int) for i in range(0, n, block_size)]
    p = rng.permutation(len(blocks))
    return np.concatenate([blocks[int(i)] for i in p])


def block_order_null(b504, configs, q: float, block_sizes: List[int], permutations: int, seed: int):
    n = len(next(iter(configs.values()))["X"])
    identity = np.arange(n, dtype=int)
    observed = aggregate_order_stats(order_score_map(b504, configs, identity), q)
    rng = np.random.default_rng(seed)
    rows = []
    for bs in block_sizes:
        for r in range(permutations):
            st = aggregate_order_stats(order_score_map(b504, configs, shuffled_block_order(n, bs, rng)), q)
            rows.append({"block_size": bs, "replicate": r, **st})
        print(f"    block-order null block={bs}: {permutations}", flush=True)
    df = pd.DataFrame(rows)
    summary = []
    for bs in block_sizes:
        g = df[df["block_size"] == bs]
        summary.append({
            "block_size": bs,
            "observed_energy_concentration": observed["median_energy_concentration"],
            "null_mean_energy_concentration": float(g["median_energy_concentration"].mean()),
            "p_upper_energy_concentration": p_upper(observed["median_energy_concentration"], g["median_energy_concentration"].to_numpy(float)),
            "observed_median_score": observed["median_of_config_medians"],
            "null_mean_median_score": float(g["median_of_config_medians"].mean()),
            "p_lower_median_score": p_lower(observed["median_of_config_medians"], g["median_of_config_medians"].to_numpy(float)),
        })
    return df, pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# Representation holdout
# ---------------------------------------------------------------------------

def consensus(score_map: Dict[str, np.ndarray], ids: List[str]) -> np.ndarray:
    return np.median(np.column_stack([score_map[c] for c in ids]), axis=1)


def representation_holdout(configs, score_map, dimensions, graph_k_values, q: float):
    ids = sorted(score_map)
    rows = []
    for d in dimensions:
        test = [c for c in ids if configs[c]["dimension"] == d]
        train = [c for c in ids if c not in test]
        a, b = consensus(score_map, train), consensus(score_map, test)
        rows.append({
            "holdout_type": "dimension", "holdout_value": d,
            "n_train_configs": len(train), "n_test_configs": len(test),
            "spearman": safe_spearman(a, b),
            "topq_jaccard": jaccard_bool(top_quantile_mask(a, q), top_quantile_mask(b, q)),
        })
    for gk in graph_k_values:
        test = [c for c in ids if configs[c]["graph_k"] == gk]
        train = [c for c in ids if c not in test]
        a, b = consensus(score_map, train), consensus(score_map, test)
        rows.append({
            "holdout_type": "graph_k", "holdout_value": gk,
            "n_train_configs": len(train), "n_test_configs": len(test),
            "spearman": safe_spearman(a, b),
            "topq_jaccard": jaccard_bool(top_quantile_mask(a, q), top_quantile_mask(b, q)),
        })
    df = pd.DataFrame(rows)
    return df, {
        "holdout_count": len(df),
        "median_spearman": float(df["spearman"].median()),
        "median_topq_jaccard": float(df["topq_jaccard"].median()),
        "min_spearman": float(df["spearman"].min()),
        "min_topq_jaccard": float(df["topq_jaccard"].min()),
    }


# ---------------------------------------------------------------------------
# Prospective rolling-origin scoring
# ---------------------------------------------------------------------------

def positive_z_fit_on_train(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    train = np.asarray(train, dtype=float)
    values = np.asarray(values, dtype=float)
    med = float(np.nanmedian(train))
    mad = float(np.nanmedian(np.abs(train - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.nanstd(train))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values)
    z = (values - med) / scale
    z[~np.isfinite(z)] = 0.0
    return np.maximum(z, 0.0)


def prospective_score(raw_df: pd.DataFrame, train_end: int) -> np.ndarray:
    cols = []
    for c in FRONT_CHANNELS:
        vals = raw_df[c].to_numpy(float)
        cols.append(positive_z_fit_on_train(vals[:train_end], vals))
    Z = np.column_stack(cols)
    return np.sqrt(np.mean(Z * Z, axis=1))


def prospective_rolling_origin(configs, q: float):
    n = len(next(iter(configs.values()))["front_score"])
    rows = []
    for fold, (train_frac, test_end_frac) in enumerate(DEFAULT_PROSPECTIVE_CUTS, start=1):
        train_end = int(round(n * train_frac))
        test_end = int(round(n * test_end_frac))
        train_end = max(20, min(train_end, n - 2))
        test_end = max(train_end + 2, min(test_end, n))
        smap = {cid: prospective_score(c["fronts"], train_end)[train_end:test_end] for cid, c in configs.items()}
        _, st = pairwise_stability(smap, q)
        fold_pass = bool(
            st["median_spearman"] >= MANDATORY["prospective_min_spearman"]
            and st["median_topq_jaccard"] >= MANDATORY["prospective_min_topq_jaccard"]
        )
        rows.append({
            "fold": fold,
            "train_end_exclusive": train_end,
            "test_start": train_end,
            "test_end_exclusive": test_end,
            "test_n": test_end - train_end,
            "median_pairwise_spearman": st["median_spearman"],
            "median_pairwise_topq_jaccard": st["median_topq_jaccard"],
            "fold_pass": fold_pass,
        })
    df = pd.DataFrame(rows)
    return df, {
        "fold_count": len(df),
        "passing_folds": int(df["fold_pass"].sum()),
        "median_spearman": float(df["median_pairwise_spearman"].median()),
        "median_topq_jaccard": float(df["median_pairwise_topq_jaccard"].median()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "operator_stack": Path(args.operator_stack),
        "b50_2_summary": Path(args.b50_2_summary),
        "b50_4_protocol": Path(args.b50_4_protocol),
        "b50_4_summary": Path(args.b50_4_summary),
        "b50_4_front_features": Path(args.b50_4_front_features),
        "state_dataset": Path(args.state_dataset),
        "b50_4_script": Path(args.b50_4_script),
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{name}: {p}")

    b502 = json.loads(paths["b50_2_summary"].read_text(encoding="utf-8"))
    prot = json.loads(paths["b50_4_protocol"].read_text(encoding="utf-8"))
    s504 = json.loads(paths["b50_4_summary"].read_text(encoding="utf-8"))
    saved_long = pd.read_csv(paths["b50_4_front_features"])
    state = pd.read_csv(paths["state_dataset"])
    validate_state_dataset(state)

    with np.load(paths["operator_stack"], allow_pickle=False) as z:
        P_stack = np.asarray(z["P_stack"], dtype=np.float64)
        labels = [str(x) for x in z["labels"]]

    labels_exact = labels == state["range_name"].astype(str).tolist()
    shape_exact = list(P_stack.shape) == list(b502.get("operator_shape", []))
    b502_ok = bool(b502.get("reproduction", {}).get("reproduction_ok", False))
    b504_ok = bool(
        s504.get("integrity", {}).get("labels_exact", False)
        and s504.get("integrity", {}).get("shape_exact", False)
        and s504.get("integrity", {}).get("b50_2_reproduction_ok", False)
    )

    dimensions = [int(x) for x in prot["dimensions"]]
    graph_k_values = [int(x) for x in prot["graph_k_values"]]
    diffusion_modes_n = int(prot["diffusion_modes"])
    q = float(prot["front_top_quantile"])

    print("=== TDC TOR A / B50.5: CONTINUOUS FRONT VALIDATION ===")
    print(f"operator stack       : {P_stack.shape}")
    print(f"labels exact         : {labels_exact}")
    print(f"B50.2 reproduction   : {b502_ok}")
    print(f"B50.4 integrity      : {b504_ok}")
    print(f"dimensions           : {dimensions}")
    print(f"graph k              : {graph_k_values}")
    print(f"front top quantile   : {q}")

    if not (labels_exact and shape_exact and b502_ok and b504_ok):
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_REPRODUCTION",
            "reason": "Input lineage/integrity gate failed before B50.4 front reconstruction.",
            "integrity": {"labels_exact": labels_exact, "shape_exact": shape_exact, "b50_2_reproduction_ok": b502_ok, "b50_4_integrity_ok": b504_ok},
        }
        atomic_json(out / "b50_5_summary.json", summary)
        atomic_text(out / "b50_5_verdict.txt", "FINAL VERDICT: INVALID_REPRODUCTION\n" + summary["reason"] + "\n")
        return summary

    manifest = {
        "version": VERSION,
        "created": now_s(),
        "outcome_blind_to_b59": True,
        "inputs": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in paths.items()},
        "frozen_b50_4_protocol": {"dimensions": dimensions, "graph_k_values": graph_k_values, "diffusion_modes": diffusion_modes_n, "front_top_quantile": q},
        "alignment_permutations": args.alignment_permutations,
        "global_order_permutations": args.order_permutations,
        "block_permutations_per_size": args.block_permutations,
        "block_sizes": args.block_sizes,
        "prospective_cuts": DEFAULT_PROSPECTIVE_CUTS,
        "mandatory_thresholds": MANDATORY,
        "seed": args.seed,
    }
    atomic_json(out / "b50_5_protocol_manifest.json", manifest)

    print("\n[0/5] Frozen B50.4 reproduction", flush=True)
    b504 = import_b504(paths["b50_4_script"])
    configs = reconstruct_configs(b504, P_stack, dimensions, graph_k_values, diffusion_modes_n)
    reproduction_ok, reproduction_df = reproduction_gate(configs, saved_long)
    atomic_csv(out / "b50_5_reproduction_audit.csv", reproduction_df)
    print(f"      PASS: {reproduction_ok}")
    if not reproduction_ok:
        summary = {"version": VERSION, "final_verdict": "INVALID_REPRODUCTION", "reason": "Frozen B50.4 front scores could not be reproduced within tolerance.", "reproduction_pass": False}
        atomic_json(out / "b50_5_summary.json", summary)
        atomic_text(out / "b50_5_verdict.txt", "FINAL VERDICT: INVALID_REPRODUCTION\n" + summary["reason"] + "\n")
        return summary

    observed = {cid: c["front_score"] for cid, c in configs.items()}

    print("[1/5] Circular-shift transition-alignment null", flush=True)
    align_df, alignment = circular_shift_alignment_null(observed, q, args.alignment_permutations, args.seed + 1000)
    atomic_csv(out / "b50_5_alignment_null.csv", align_df)
    alignment_pass = bool(
        alignment["observed_median_spearman"] >= MANDATORY["alignment_min_spearman"]
        and alignment["observed_median_topq_jaccard"] >= MANDATORY["alignment_min_topq_jaccard"]
        and alignment["p_upper_spearman"] <= MANDATORY["alignment_max_p_upper"]
        and alignment["p_upper_topq_jaccard"] <= MANDATORY["alignment_max_p_upper"]
    )
    alignment["pass"] = alignment_pass

    print("[2/5] Global physical-order null", flush=True)
    order_df, order_null = global_order_null(b504, configs, q, args.order_permutations, args.seed + 2000)
    atomic_csv(out / "b50_5_global_order_null.csv", order_df)
    order_pass = bool(order_null["p_upper_energy_concentration"] <= MANDATORY["order_max_p_upper_concentration"])
    order_null["pass"] = order_pass

    print("      Secondary block-preserving null", flush=True)
    block_df, block_summary = block_order_null(b504, configs, q, args.block_sizes, args.block_permutations, args.seed + 3000)
    atomic_csv(out / "b50_5_block_order_null.csv", block_df)
    atomic_csv(out / "b50_5_block_order_null_summary.csv", block_summary)

    print("[3/5] Representation holdout", flush=True)
    rep_df, rep = representation_holdout(configs, observed, dimensions, graph_k_values, q)
    atomic_csv(out / "b50_5_representation_holdout.csv", rep_df)
    rep_pass = bool(
        rep["median_spearman"] >= MANDATORY["representation_holdout_min_spearman"]
        and rep["median_topq_jaccard"] >= MANDATORY["representation_holdout_min_topq_jaccard"]
    )
    rep["pass"] = rep_pass

    print("[4/5] Rolling-origin prospective physical holdout", flush=True)
    pros_df, prospective = prospective_rolling_origin(configs, q)
    atomic_csv(out / "b50_5_prospective_holdout.csv", pros_df)
    prospective_pass = bool(
        prospective["median_spearman"] >= MANDATORY["prospective_min_spearman"]
        and prospective["median_topq_jaccard"] >= MANDATORY["prospective_min_topq_jaccard"]
        and prospective["passing_folds"] >= MANDATORY["prospective_min_passing_folds"]
    )
    prospective["pass"] = prospective_pass

    print("[5/5] Frozen verdict", flush=True)
    if alignment_pass and order_pass and rep_pass and prospective_pass:
        verdict = "VALIDATED_CONTINUOUS_FRONT_ENDPOINT"
        reason = "The frozen B50.4 front is transition-specific beyond circular-shift null, order-specific beyond global adjacency permutation, representation-holdout stable, and prospectively stable on future physical blocks."
    elif alignment_pass and rep_pass and prospective_pass and not order_pass:
        verdict = "REPRESENTATION_STABLE_BUT_ORDER_NULL_COMPATIBLE"
        reason = "Front identity is representation- and future-block stable, but physical-order front-energy concentration is compatible with random global adjacency."
    elif alignment_pass and order_pass and not prospective_pass:
        verdict = "ORDER_SPECIFIC_BUT_PROSPECTIVE_UNSTABLE"
        reason = "The front is transition-aligned and order-specific, but does not generalize sufficiently to frozen future physical blocks."
    elif not alignment_pass:
        verdict = "FRONT_ALIGNMENT_NOT_ABOVE_NULL"
        reason = "Observed cross-configuration front identity does not pass the circular-shift transition-alignment null."
    else:
        verdict = "PARTIAL_CONTINUOUS_FRONT_VALIDATION"
        reason = "The frozen front passes some but not all mandatory null/holdout gates."

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "outcome_blind_to_b59": True,
        "reproduction_pass": reproduction_ok,
        "alignment_null": alignment,
        "global_order_null": order_null,
        "block_order_null_secondary": block_summary.to_dict(orient="records"),
        "representation_holdout": rep,
        "prospective_physical_holdout": prospective,
        "mandatory_thresholds": MANDATORY,
        "final_verdict": verdict,
        "final_reason": reason,
        "scientific_boundary": "Validation of the frozen B50.4 continuous-front representation endpoint only; no B59/B59.1 selector or predictive outcome is used.",
    }
    atomic_json(out / "b50_5_summary.json", summary)

    # Plots
    plt.figure(figsize=(8, 5))
    plt.hist(align_df["median_spearman"].to_numpy(float), bins=40)
    plt.axvline(alignment["observed_median_spearman"], linewidth=2)
    plt.xlabel("median pairwise Spearman under circular-shift null")
    plt.ylabel("count")
    plt.title("B50.5 transition-alignment null")
    plt.tight_layout()
    plt.savefig(out / "b50_5_alignment_null_spearman.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(order_df["median_energy_concentration"].to_numpy(float), bins=35)
    plt.axvline(order_null["observed"]["median_energy_concentration"], linewidth=2)
    plt.xlabel("median top-q front-energy concentration")
    plt.ylabel("count")
    plt.title("B50.5 global time-order null")
    plt.tight_layout()
    plt.savefig(out / "b50_5_global_order_null_concentration.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    x = np.arange(len(rep_df))
    plt.plot(x, rep_df["spearman"], marker="o", label="Spearman")
    plt.plot(x, rep_df["topq_jaccard"], marker="o", label="top-q Jaccard")
    plt.xticks(x, [f"{r.holdout_type}:{int(r.holdout_value)}" for r in rep_df.itertuples()], rotation=45, ha="right")
    plt.ylim(-0.05, 1.02)
    plt.ylabel("holdout agreement")
    plt.title("B50.5 representation holdout")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "b50_5_representation_holdout.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(pros_df["fold"], pros_df["median_pairwise_spearman"], marker="o", label="Spearman")
    plt.plot(pros_df["fold"], pros_df["median_pairwise_topq_jaccard"], marker="o", label="top-q Jaccard")
    plt.xticks(pros_df["fold"])
    plt.ylim(-0.05, 1.02)
    plt.xlabel("rolling-origin fold")
    plt.ylabel("future-block stability")
    plt.title("B50.5 prospective physical holdout")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "b50_5_prospective_holdout.png", dpi=160)
    plt.close()

    report = f"""B50.5 — Continuous Front Null & Prospective Stability Validation
======================================================================

Integrity / reproduction
------------------------
operator stack shape                 = {P_stack.shape}
labels exact                         = {labels_exact}
B50.2 reproduction OK                = {b502_ok}
B50.4 integrity OK                   = {b504_ok}
frozen B50.4 front reproduction PASS = {reproduction_ok}

Frozen B50.4 protocol
---------------------
dimensions                           = {dimensions}
graph k                              = {graph_k_values}
diffusion modes                      = {diffusion_modes_n}
front top quantile                   = {q}

1. Transition-alignment null
----------------------------
observed median Spearman             = {alignment['observed_median_spearman']}
observed median top-q Jaccard        = {alignment['observed_median_topq_jaccard']}
circular-shift p_upper Spearman      = {alignment['p_upper_spearman']}
circular-shift p_upper Jaccard       = {alignment['p_upper_topq_jaccard']}
PASS                                 = {alignment_pass}

2. Global time-order null
-------------------------
observed energy concentration        = {order_null['observed']['median_energy_concentration']}
null mean energy concentration       = {order_null['null_mean_energy_concentration']}
p_upper energy concentration         = {order_null['p_upper_energy_concentration']}
secondary p_lower median score       = {order_null['p_lower_median_score']}
PASS                                 = {order_pass}

3. Representation holdout
-------------------------
median holdout Spearman              = {rep['median_spearman']}
median holdout top-q Jaccard         = {rep['median_topq_jaccard']}
minimum holdout Spearman             = {rep['min_spearman']}
minimum holdout top-q Jaccard        = {rep['min_topq_jaccard']}
PASS                                 = {rep_pass}

4. Rolling-origin prospective holdout
-------------------------------------
passing folds                        = {prospective['passing_folds']}/{prospective['fold_count']}
median future-block Spearman         = {prospective['median_spearman']}
median future-block top-q Jaccard    = {prospective['median_topq_jaccard']}
PASS                                 = {prospective_pass}

Secondary block-order null
--------------------------
{block_summary.to_string(index=False)}

FINAL VERDICT: {verdict}

REASON:
{reason}

Interpretation boundary
-----------------------
This validates only the frozen B50.4 continuous-front representation endpoint.
No B59/B59.1 selector or predictive outcome is used.
No individual-prime prediction, RH, or theorem claim is made.
"""
    atomic_text(out / "b50_5_verdict.txt", report)

    print("\n=== B50.5 RESULT ===")
    print(f"reproduction PASS   : {reproduction_ok}")
    print(f"alignment-null PASS : {alignment_pass}")
    print(f"order-null PASS     : {order_pass}")
    print(f"repr-holdout PASS   : {rep_pass}")
    print(f"prospective PASS    : {prospective_pass}")
    print(f"FINAL VERDICT       : {verdict}")
    print(f"verdict             : {out / 'b50_5_verdict.txt'}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="B50.5 Continuous Front Null & Prospective Stability Validation")
    p.add_argument("--operator-stack", required=True)
    p.add_argument("--b50-2-summary", required=True)
    p.add_argument("--b50-4-protocol", required=True)
    p.add_argument("--b50-4-summary", required=True)
    p.add_argument("--b50-4-front-features", required=True)
    p.add_argument("--state-dataset", required=True)
    p.add_argument("--b50-4-script", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--alignment-permutations", type=int, default=1000)
    p.add_argument("--order-permutations", type=int, default=200)
    p.add_argument("--block-permutations", type=int, default=200)
    p.add_argument("--block-sizes", default="10,25,50")
    p.add_argument("--seed", type=int, default=50505)
    a = p.parse_args()
    a.block_sizes = parse_int_list(a.block_sizes)
    if a.alignment_permutations < 199:
        p.error("--alignment-permutations must be >=199")
    if a.order_permutations < 99:
        p.error("--order-permutations must be >=99")
    if a.block_permutations < 99:
        p.error("--block-permutations must be >=99")
    return a


def main():
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
