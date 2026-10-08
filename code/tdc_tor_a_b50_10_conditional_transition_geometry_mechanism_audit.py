#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDC TOR A / B50.10
Conditional Transition Geometry Mechanism Audit
================================================

Purpose
-------
B50.6-B50.8 established a replicated negative physical-adjacency specificity
effect under full-operator distance-matched controls. B50.10 does NOT retune
that effect and does NOT introduce new controls. It asks which frozen
continuous-front channels carry the effect:

    speed
    acceleration
    turn_change
    density_scale_change
    diffusion_step

It also performs:
1) exact reproduction of the frozen B50.6 full composite endpoint;
2) channel-wise matched-identity percentile tests;
3) leave-one-channel-out composite score decomposition;
4) explicit directional-cosine diagnostic;
5) contiguous-thirds channel stability diagnostics.

Scientific boundary
-------------------
This is a mechanism/decomposition audit on the same B50.6 source sequence and
the exact B50.6 matched controls. It is not independent-corpus replication,
not B59 validation, not individual-prime prediction, and not an RH/theorem
claim.

Operational properties
----------------------
- PLAN -> AUDIT split.
- Frozen SHA-256 input manifest.
- Exact reuse of B50.6 control identities.
- Atomic output writes.
- Resumable mechanism-cube cache.
- No post-hoc threshold changes.

Recommended:
    --mode plan
    inspect b50_10_plan.txt
    --mode audit --reuse-existing

Authoring target: Windows / PowerShell, Python 3.10+
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "B50.10_conditional_transition_geometry_mechanism_audit_v1"

CHANNELS = [
    "speed",
    "acceleration",
    "turn_change",
    "density_scale_change",
    "diffusion_step",
]

# Frozen before the B50.10 mechanism result.
THRESHOLDS = {
    "full_score_max_abs_error": 1e-8,
    "full_percentile_max_abs_error": 1e-12,
    "full_effect_max_abs_error": 1e-12,

    # Confirmed channel association requires BOTH multiplicity-corrected
    # significance and a nontrivial percentile effect.
    "channel_holm_alpha": 0.05,
    "channel_abs_effect_min": 0.03,

    # A channel is called score-localizing only if removing it attenuates
    # |full effect| by at least 25%.
    "loo_material_attenuation_min": 0.25,

    # Thirds are descriptive robustness gates for each channel.
    "third_nominal_p_max": 0.05,
    "third_abs_effect_min": 0.025,
    "third_same_direction_required": 3,
    "third_nominal_required": 2,

    # Directional cosine diagnostic: positive percentile effect means
    # physical next-step cosine is higher than matched controls.
    "directional_abs_effect_min": 0.03,
    "directional_p_max": 0.01,
}

FROZEN_EXPECTED_LINEAGE = {
    "b50_6": "PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED",
    "b50_7": "EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED",
    "b50_8": "MULTI_OFFSET_PHASE_ROBUST_EFFECT_INVARIANCE",
}

DEFAULT_SEED = 501010


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(
        path,
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True),
    )


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y
    except Exception:
        return default


def validate_state(df: pd.DataFrame) -> None:
    req = {"range_name", "range_index"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"state dataset missing columns: {sorted(miss)}")
    idx = pd.to_numeric(df["range_index"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(len(df))):
        raise ValueError("state range_index is not exactly 0..n-1")
    if df["range_name"].astype(str).duplicated().any():
        raise ValueError("state dataset has duplicate range_name")


def rank_percentiles(x: Sequence[float]) -> np.ndarray:
    """
    Exact B50.6 ranking semantics:
      percentile(i) = (# lower + 0.5 * # tied others) / (n-1)
    """
    x = np.asarray(x, float)
    n = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        less = np.sum(x < x[i])
        ties = np.sum(x == x[i]) - 1
        out[i] = (less + 0.5 * ties) / max(n - 1, 1)
    return out


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """
    Holm-Bonferroni adjusted p-values, monotone in sorted order.
    """
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * p[idx]
        running = max(running, val)
        adj_sorted[k] = min(1.0, running)
    out = np.empty(m, dtype=float)
    for k, idx in enumerate(order):
        out[idx] = adj_sorted[k]
    return out


def matched_identity_null(
    consensus_pct: np.ndarray,
    reps: int,
    seed: int,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Same identity-randomization logic as B50.6, but report both tails.
    consensus_pct: transitions x candidates, candidate 0 = physical.
    """
    X = np.asarray(consensus_pct, float)
    if mask is not None:
        X = X[np.asarray(mask, bool)]
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("consensus_pct must be transitions x candidates")

    obs = X[:, 0]
    effect = float(np.mean(obs) - 0.5)

    rng = np.random.default_rng(seed)
    rows = np.arange(len(X))
    null = np.empty(reps, dtype=float)

    for r in range(reps):
        choice = rng.integers(0, X.shape[1], size=len(X))
        null[r] = float(np.mean(X[rows, choice]) - 0.5)

    p_lower = float((1 + np.sum(null <= effect)) / (reps + 1))
    p_upper = float((1 + np.sum(null >= effect)) / (reps + 1))
    p_two = float((1 + np.sum(np.abs(null) >= abs(effect))) / (reps + 1))

    return {
        "n_transitions": int(len(X)),
        "observed_mean_percentile": float(np.mean(obs)),
        "observed_median_percentile": float(np.median(obs)),
        "observed_mean_effect": effect,
        "null_mean_effect": float(np.mean(null)),
        "null_sd_effect": float(np.std(null)),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_two_sided": p_two,
    }, null


# ---------------------------------------------------------------------------
# Frozen B50.4 compatibility layer
# ---------------------------------------------------------------------------

def import_b504(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("b504_frozen_b5010", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import B50.4 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for name in ("full_pca", "build_knn_graph", "diffusion_modes"):
        if not hasattr(mod, name):
            raise AttributeError(f"B50.4 missing required API: {name}")

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
        raise RuntimeError(
            f"unsupported B50.4 build_knn_graph return length {len(x)}"
        )
    return W, np.asarray(radius, float)


def call_front(fn, X, radius, diff):
    attempts = [
        lambda: fn(X, radius, diff),
        lambda: fn(X=X, local_radius=radius, diffusion_coords=diff),
    ]
    errors = []
    for attempt in attempts:
        try:
            z = attempt()
            if isinstance(z, pd.DataFrame):
                return z
        except TypeError as e:
            errors.append(str(e))
    raise RuntimeError(
        "cannot call frozen B50.4 front function: " + " | ".join(errors)
    )


def reconstruct_configs(
    mod,
    front_fn,
    P: np.ndarray,
    dims: Sequence[int],
    graph_k: Sequence[int],
    n_modes: int,
) -> Dict[str, Dict[str, Any]]:
    scores, _, _ = mod.full_pca(P)
    out: Dict[str, Dict[str, Any]] = {}

    for d in dims:
        X = np.asarray(scores[:, :d], float)
        D = squareform(pdist(X))
        for gk in graph_k:
            cid = f"d{d}_knn{gk}"
            W, radius = unpack_graph(mod.build_knn_graph(D, gk))
            eig, modes = mod.diffusion_modes(W, n_modes)
            eig = np.asarray(eig, float)
            modes = np.asarray(modes, float)
            diff = modes * eig[None, :]
            fr = call_front(front_fn, X, radius, diff)
            out[cid] = {
                "dimension": int(d),
                "graph_k": int(gk),
                "X": X,
                "radius": radius,
                "diff": diff,
                "fronts": fr,
            }
    return out


def frozen_scale(fr: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Exact B50.6 scale semantics.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for c in CHANNELS:
        x = pd.to_numeric(fr[c], errors="coerce").to_numpy(float)
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))
        s = 1.4826 * mad
        if not np.isfinite(s) or s < 1e-12:
            s = float(np.nanstd(x))
        if not np.isfinite(s) or s < 1e-12:
            s = 0.0
        out[c] = (med, s)
    return out


def candidate_raw(
    X: np.ndarray,
    radius: np.ndarray,
    diff: np.ndarray,
    i: int,
    dest: Sequence[int],
) -> Dict[str, np.ndarray]:
    """
    Exact B50.6 arbitrary-destination channel semantics.
    """
    dest = np.asarray(dest, int)
    prev = X[i] - X[i - 1]
    v = X[dest] - X[i]

    speed = np.linalg.norm(v, axis=1)
    acceleration = np.linalg.norm(v - prev[None, :], axis=1)

    den = np.linalg.norm(prev) * speed
    cos = np.ones(len(dest), dtype=float)
    good = den > 1e-15
    if np.any(good):
        cos[good] = (v[good] @ prev) / den[good]
    cos = np.clip(cos, -1.0, 1.0)

    turn = 1.0 - cos
    density = np.abs(
        np.log(np.maximum(radius[dest], 1e-15))
        - math.log(max(float(radius[i]), 1e-15))
    )
    diffusion = (
        np.linalg.norm(diff[dest] - diff[i][None, :], axis=1)
        if diff.shape[1]
        else np.zeros(len(dest), dtype=float)
    )

    return {
        "speed": speed,
        "acceleration": acceleration,
        "turn_change": turn,
        "density_scale_change": density,
        "diffusion_step": diffusion,
    }


def score_raw_subset(
    raw_matrix: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
    include_mask: np.ndarray,
) -> np.ndarray:
    """
    B50.6 composite: positive robust z channels, RMS aggregation.
    raw_matrix shape: candidates x channels.
    """
    raw_matrix = np.asarray(raw_matrix, float)
    include_mask = np.asarray(include_mask, bool)
    if include_mask.sum() < 1:
        raise ValueError("score_raw_subset requires at least one channel")

    Z = []
    for ch in np.where(include_mask)[0]:
        x = raw_matrix[:, ch]
        med = float(medians[ch])
        s = float(scales[ch])
        if s <= 1e-15:
            z = np.zeros_like(x)
        else:
            z = (x - med) / s
            z[~np.isfinite(z)] = 0.0
            z = np.maximum(z, 0.0)
        Z.append(z)

    Zm = np.column_stack(Z)
    return np.sqrt(np.mean(Zm * Zm, axis=1))


# ---------------------------------------------------------------------------
# Inputs, lineage, frozen controls
# ---------------------------------------------------------------------------

def input_paths(args) -> Dict[str, Path]:
    b6 = Path(args.b50_6_dir)
    return {
        "operator_stack": Path(args.operator_stack),
        "state_dataset": Path(args.state_dataset),
        "b50_4_script": Path(args.b50_4_script),
        "b50_4_protocol": Path(args.b50_4_protocol),
        "b50_4_front_features": Path(args.b50_4_front_features),

        "b50_6_summary": b6 / "b50_6_summary.json",
        "b50_6_protocol": b6 / "b50_6_protocol_manifest.json",
        "b50_6_transition_specificity": b6 / "b50_6_transition_specificity.csv",
        "b50_6_matching_audit": b6 / "b50_6_matching_audit.csv",
        "b50_6_config_specificity": b6 / "b50_6_config_specificity.csv",

        "b50_7_summary": Path(args.b50_7_summary),
        "b50_8_summary": Path(args.b50_8_summary),
    }


def require_paths(paths: Dict[str, Path]) -> None:
    missing = [(k, p) for k, p in paths.items() if not p.exists()]
    if missing:
        msg = "\n".join(f"{k}: {p}" for k, p in missing)
        raise FileNotFoundError("Missing B50.10 input(s):\n" + msg)


def load_operator_stack(path: Path) -> Tuple[np.ndarray, List[str]]:
    with np.load(path, allow_pickle=False) as z:
        if "P_stack" not in z.files or "labels" not in z.files:
            raise ValueError(
                f"operator stack {path} must contain P_stack and labels"
            )
        P = np.asarray(z["P_stack"], float)
        labels = [str(x) for x in z["labels"]]
    return P, labels


def lineage_gate(paths: Dict[str, Path]) -> Dict[str, Any]:
    s6 = json.loads(paths["b50_6_summary"].read_text(encoding="utf-8"))
    s7 = json.loads(paths["b50_7_summary"].read_text(encoding="utf-8"))
    s8 = json.loads(paths["b50_8_summary"].read_text(encoding="utf-8"))

    v6 = str(s6.get("final_verdict", ""))
    v7 = str(s7.get("final_verdict", ""))
    v8 = str(s8.get("final_verdict", ""))

    ok6 = v6 == FROZEN_EXPECTED_LINEAGE["b50_6"]
    ok7 = v7 == FROZEN_EXPECTED_LINEAGE["b50_7"]
    ok8 = v8 == FROZEN_EXPECTED_LINEAGE["b50_8"]

    b6_effect = safe_float(
        s6.get("primary_specificity", {}).get("observed_mean_effect")
    )
    b7_effect = safe_float(
        s7.get("primary_replication", {}).get("observed_mean_effect")
    )
    b8_mean = safe_float(
        s8.get("effect_invariance", {}).get("new_offsets_mean_effect")
    )

    negative_reference = bool(
        np.isfinite(b6_effect)
        and np.isfinite(b7_effect)
        and np.isfinite(b8_mean)
        and b6_effect < 0
        and b7_effect < 0
        and b8_mean < 0
    )

    return {
        "b50_6_verdict": v6,
        "b50_7_verdict": v7,
        "b50_8_verdict": v8,
        "b50_6_ok": ok6,
        "b50_7_ok": ok7,
        "b50_8_ok": ok8,
        "b50_6_effect": b6_effect,
        "b50_7_effect": b7_effect,
        "b50_8_mean_effect": b8_mean,
        "negative_reference_direction_ok": negative_reference,
        "pass": bool(ok6 and ok7 and ok8 and negative_reference),
    }


def parse_control_indices(s: Any) -> List[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty control_dest_indices: {s!r}")
    return vals


def frozen_control_map(
    transition_df: pd.DataFrame,
    matching_df: pd.DataFrame,
) -> Tuple[List[int], Dict[int, np.ndarray], Dict[str, Any], pd.DataFrame]:
    req_t = {
        "transition_index",
        "from_range",
        "to_range",
        "consensus_physical_percentile",
        "control_dest_indices",
    }
    miss = req_t - set(transition_df.columns)
    if miss:
        raise ValueError(
            f"B50.6 transition_specificity missing {sorted(miss)}"
        )

    req_m = {
        "transition_index",
        "control_rank",
        "control_dest_index",
        "abs_log_distance_error",
    }
    miss_m = req_m - set(matching_df.columns)
    if miss_m:
        raise ValueError(f"B50.6 matching audit missing {sorted(miss_m)}")

    tdf = transition_df.copy()
    tdf["transition_index"] = pd.to_numeric(
        tdf["transition_index"], errors="raise"
    ).astype(int)
    tdf = tdf.sort_values("transition_index", kind="stable").reset_index(drop=True)

    if tdf["transition_index"].duplicated().any():
        raise ValueError("duplicate B50.6 transition_index")

    tis = tdf["transition_index"].tolist()
    if tis != list(range(1, len(tis) + 1)):
        raise ValueError(
            "B50.6 transition sequence is not exactly 1..N; "
            f"first={tis[:5]}, last={tis[-5:]}"
        )

    cmap: Dict[int, np.ndarray] = {}
    rows = []
    n_controls_set = set()

    mdf = matching_df.copy()
    for c in ("transition_index", "control_rank", "control_dest_index"):
        mdf[c] = pd.to_numeric(mdf[c], errors="raise").astype(int)

    for _, r in tdf.iterrows():
        i = int(r["transition_index"])
        from_csv = parse_control_indices(r["control_dest_indices"])

        sub = (
            mdf[mdf["transition_index"] == i]
            .sort_values("control_rank", kind="stable")
        )
        from_match = sub["control_dest_index"].astype(int).tolist()

        exact = from_csv == from_match
        if not exact:
            raise RuntimeError(
                f"Frozen control identity mismatch at transition {i}"
            )

        arr = np.asarray(from_csv, int)
        if len(np.unique(arr)) != len(arr):
            raise RuntimeError(f"duplicate controls at transition {i}")

        cmap[i] = arr
        n_controls_set.add(len(arr))

        rows.append({
            "transition_index": i,
            "from_range": str(r["from_range"]),
            "to_range": str(r["to_range"]),
            "physical_dest_index": i + 1,
            "control_count": len(arr),
            "control_dest_indices": ",".join(str(x) for x in arr),
            "control_identity_exact_between_b50_6_outputs": exact,
            "matching_median_abs_log_error": float(
                pd.to_numeric(
                    sub["abs_log_distance_error"], errors="coerce"
                ).median()
            ),
            "matching_q90_abs_log_error": float(
                pd.to_numeric(
                    sub["abs_log_distance_error"], errors="coerce"
                ).quantile(0.90)
            ),
        })

    if len(n_controls_set) != 1:
        raise RuntimeError(
            f"control count varies across transitions: {sorted(n_controls_set)}"
        )

    audit = pd.DataFrame(rows)
    info = {
        "transition_count": len(tis),
        "controls_per_transition": int(next(iter(n_controls_set))),
        "transition_first": int(tis[0]),
        "transition_last": int(tis[-1]),
        "control_identity_exact": bool(
            audit["control_identity_exact_between_b50_6_outputs"].all()
        ),
    }
    return tis, cmap, info, audit


def extract_frozen_b504_protocol(path: Path) -> Tuple[List[int], List[int], int]:
    p4 = json.loads(path.read_text(encoding="utf-8"))
    dims = [int(x) for x in p4["dimensions"]]
    graph_k = [int(x) for x in p4["graph_k_values"]]
    n_modes = int(p4["diffusion_modes"])
    return dims, graph_k, n_modes


def protocol_modified(args) -> bool:
    return bool(
        args.channel_abs_effect_min != THRESHOLDS["channel_abs_effect_min"]
        or args.loo_material_attenuation_min
        != THRESHOLDS["loo_material_attenuation_min"]
    )


# ---------------------------------------------------------------------------
# PLAN
# ---------------------------------------------------------------------------

def plan_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = input_paths(args)
    require_paths(paths)

    lineage = lineage_gate(paths)

    state = pd.read_csv(paths["state_dataset"])
    validate_state(state)
    P, labels = load_operator_stack(paths["operator_stack"])
    labels_ok = labels == state["range_name"].astype(str).tolist()
    shape_ok = bool(P.ndim == 2 and P.shape[0] == len(state))

    tdf = pd.read_csv(paths["b50_6_transition_specificity"])
    mdf = pd.read_csv(paths["b50_6_matching_audit"])
    tis, cmap, control_info, control_audit = frozen_control_map(tdf, mdf)

    expected_transition_count = len(state) - 2
    transition_count_ok = len(tis) == expected_transition_count

    # Semantic range mapping gate.
    range_mapping_ok = True
    range_rows = []
    for _, r in tdf.sort_values("transition_index").iterrows():
        i = int(r["transition_index"])
        exp_from = str(state.iloc[i]["range_name"])
        exp_to = str(state.iloc[i + 1]["range_name"])
        ok = (
            str(r["from_range"]) == exp_from
            and str(r["to_range"]) == exp_to
        )
        range_mapping_ok &= ok
        range_rows.append({
            "transition_index": i,
            "from_range_b50_6": str(r["from_range"]),
            "from_range_state": exp_from,
            "to_range_b50_6": str(r["to_range"]),
            "to_range_state": exp_to,
            "mapping_exact": ok,
        })

    atomic_csv(out / "b50_10_frozen_control_map.csv", control_audit)
    atomic_csv(
        out / "b50_10_range_mapping_audit.csv",
        pd.DataFrame(range_rows),
    )

    dims, graph_k, n_modes = extract_frozen_b504_protocol(
        paths["b50_4_protocol"]
    )

    hashes = {
        k: {"path": str(v), "sha256": sha256_file(v)}
        for k, v in paths.items()
    }

    pass_ = bool(
        lineage["pass"]
        and labels_ok
        and shape_ok
        and transition_count_ok
        and range_mapping_ok
        and control_info["control_identity_exact"]
        and not protocol_modified(args)
    )

    manifest = {
        "version": VERSION,
        "created": now_s(),
        "scientific_role": "mechanism/decomposition audit of replicated B50 adjacency specificity",
        "lineage": lineage,
        "inputs": hashes,
        "state_integrity": {
            "operator_stack_shape": list(P.shape),
            "state_rows": int(len(state)),
            "labels_exact": labels_ok,
            "shape_exact": shape_ok,
            "transition_count_expected": expected_transition_count,
            "transition_count_frozen": len(tis),
            "transition_count_exact": transition_count_ok,
            "range_mapping_exact": range_mapping_ok,
        },
        "frozen_controls": control_info,
        "frozen_b50_4": {
            "dimensions": dims,
            "graph_k_values": graph_k,
            "diffusion_modes": n_modes,
            "channels": CHANNELS,
        },
        "primary_channel_protocol": {
            "statistic": "mean consensus physical percentile minus 0.5, separately for each raw B50.4 channel",
            "null": "matched-set identity randomization using the exact 20 B50.6 control destinations per transition",
            "multiplicity": "Holm correction across the five channels",
            "association_gate": {
                "holm_p_max": THRESHOLDS["channel_holm_alpha"],
                "abs_effect_min": args.channel_abs_effect_min,
            },
        },
        "leave_one_out_protocol": {
            "score": "exact B50.6 positive robust-z RMS composite with one channel removed",
            "material_attenuation_min": args.loo_material_attenuation_min,
            "role": "score-localization diagnostic; not causal attribution",
        },
        "directional_protocol": {
            "cosine": "cos(previous physical vector, candidate vector)",
            "physical_positive_direction": "higher cosine percentile than matched controls",
            "role": "secondary algebraically related diagnostic to turn_change=1-cosine",
        },
        "thirds_protocol": {
            "split": "three contiguous transition strata",
            "role": "secondary channel stability diagnostic",
        },
        "randomization_replicates": args.randomization_replicates,
        "seed": args.seed,
        "thresholds": THRESHOLDS,
        "protocol_modified": protocol_modified(args),
        "plan_pass": pass_,
        "scientific_boundary": (
            "Same-source mechanism decomposition using exact B50.6 controls. "
            "Not independent-corpus replication, not B59 validation, not "
            "individual-prime prediction, and not theorem/RH evidence."
        ),
    }

    atomic_json(out / "b50_10_protocol_manifest.json", manifest)

    report = f"""B50.10 — Conditional Transition Geometry Mechanism Audit / PLAN
=================================================================

Frozen lineage
--------------
B50.6 verdict                         = {lineage['b50_6_verdict']}
B50.7 verdict                         = {lineage['b50_7_verdict']}
B50.8 verdict                         = {lineage['b50_8_verdict']}
lineage PASS                          = {lineage['pass']}

State / transition integrity
----------------------------
operator stack shape                  = {P.shape}
state rows                            = {len(state)}
labels exact                          = {labels_ok}
shape exact                           = {shape_ok}
frozen transitions                    = {len(tis)}
expected transitions                  = {expected_transition_count}
transition count exact                = {transition_count_ok}
range mapping exact                   = {range_mapping_ok}

Frozen B50.6 controls
---------------------
controls per transition               = {control_info['controls_per_transition']}
control identity exact                = {control_info['control_identity_exact']}
first / last transition               = {control_info['transition_first']} / {control_info['transition_last']}

Frozen B50.4 representation
---------------------------
dimensions                            = {dims}
graph k                               = {graph_k}
diffusion modes                       = {n_modes}
channels                              = {CHANNELS}

Mechanism gates
---------------
Holm alpha across five channels       = {THRESHOLDS['channel_holm_alpha']}
channel |effect| floor                = {args.channel_abs_effect_min}
LOO material attenuation floor        = {args.loo_material_attenuation_min}
randomization replicates              = {args.randomization_replicates}
protocol modified                     = {protocol_modified(args)}

PLAN PASS                             = {pass_}

Interpretation
--------------
B50.10 does not search for new controls and does not retune the B50.6 effect.
It decomposes the frozen replicated effect into raw channels, leave-one-out
composites, and directional geometry.
"""
    atomic_text(out / "b50_10_plan.txt", report)

    print(report)
    return manifest


def ensure_frozen_plan(args) -> Dict[str, Any]:
    path = Path(args.output_dir) / "b50_10_protocol_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen PLAN: {path}. Run --mode plan first."
        )
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not plan.get("plan_pass", False):
        raise RuntimeError("B50.10 PLAN did not pass.")
    if plan.get("protocol_modified", False):
        raise RuntimeError("B50.10 frozen plan reports protocol_modified=True.")
    return plan


def verify_frozen_hashes(args, plan: Dict[str, Any]) -> Dict[str, Any]:
    current_paths = input_paths(args)
    rows = []
    all_ok = True

    for key, frozen in plan["inputs"].items():
        p = current_paths.get(key)
        if p is None or not p.exists():
            rows.append({
                "input": key,
                "exists": False,
                "frozen_sha256": frozen.get("sha256"),
                "current_sha256": None,
                "exact": False,
            })
            all_ok = False
            continue

        h = sha256_file(p)
        ok = h == frozen.get("sha256")
        all_ok &= ok
        rows.append({
            "input": key,
            "exists": True,
            "frozen_sha256": frozen.get("sha256"),
            "current_sha256": h,
            "exact": ok,
        })

    df = pd.DataFrame(rows)
    atomic_csv(Path(args.output_dir) / "b50_10_frozen_input_audit.csv", df)
    return {"pass": bool(all_ok), "rows": rows}


# ---------------------------------------------------------------------------
# Mechanism cube
# ---------------------------------------------------------------------------

def build_mechanism_cube(
    args,
    plan: Dict[str, Any],
    paths: Dict[str, Path],
    tis: List[int],
    cmap: Dict[int, np.ndarray],
) -> Dict[str, Any]:
    out = Path(args.output_dir)
    cache_path = out / "b50_10_mechanism_cube.npz"
    cache_manifest_path = out / "b50_10_cache_manifest.json"

    frozen_hashes = {
        k: v["sha256"] for k, v in plan["inputs"].items()
    }

    if args.reuse_existing and cache_path.exists() and cache_manifest_path.exists():
        cm = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if (
            cm.get("version") == VERSION
            and cm.get("input_hashes") == frozen_hashes
            and cm.get("transition_indices") == tis
            and cm.get("channels") == CHANNELS
        ):
            print("[cache] reusing frozen B50.10 mechanism cube", flush=True)
            with np.load(cache_path, allow_pickle=False) as z:
                return {
                    "transition_indices": z["transition_indices"].astype(int),
                    "config_ids": [str(x) for x in z["config_ids"]],
                    "dest_indices": z["dest_indices"].astype(int),
                    "raw_cube": np.asarray(z["raw_cube"], float),
                    "cosine_cube": np.asarray(z["cosine_cube"], float),
                    "medians": np.asarray(z["medians"], float),
                    "scales": np.asarray(z["scales"], float),
                    "saved_physical_front_scores": np.asarray(
                        z["saved_physical_front_scores"], float
                    ),
                }

    P, labels = load_operator_stack(paths["operator_stack"])
    saved = pd.read_csv(paths["b50_4_front_features"])
    dims = [int(x) for x in plan["frozen_b50_4"]["dimensions"]]
    graph_k = [int(x) for x in plan["frozen_b50_4"]["graph_k_values"]]
    n_modes = int(plan["frozen_b50_4"]["diffusion_modes"])

    print("[1/5] reconstructing frozen B50.4 configurations", flush=True)
    mod, front_fn = import_b504(paths["b50_4_script"])
    configs = reconstruct_configs(
        mod, front_fn, P, dims, graph_k, n_modes
    )
    cids = sorted(configs)
    if set(cids) != set(saved["config_id"].astype(str).unique()):
        raise RuntimeError(
            "Frozen configuration IDs differ from B50.4 front-feature table."
        )

    n_t = len(tis)
    n_c = len(cids)
    n_k = 1 + len(next(iter(cmap.values())))
    n_ch = len(CHANNELS)

    dest_indices = np.empty((n_t, n_k), dtype=int)
    raw_cube = np.empty((n_t, n_c, n_k, n_ch), dtype=np.float64)
    cosine_cube = np.empty((n_t, n_c, n_k), dtype=np.float64)
    medians = np.empty((n_c, n_ch), dtype=np.float64)
    scales = np.empty((n_c, n_ch), dtype=np.float64)
    saved_phys = np.empty((n_t, n_c), dtype=np.float64)

    print(
        f"[2/5] building raw channel cube "
        f"T={n_t} C={n_c} K={n_k} channels={n_ch}",
        flush=True,
    )

    for ci, cid in enumerate(cids):
        c = configs[cid]
        fr = (
            saved[saved["config_id"].astype(str) == cid]
            .sort_values("transition_index", kind="stable")
        )
        scale_map = frozen_scale(fr)
        lookup = dict(
            zip(
                pd.to_numeric(
                    fr["transition_index"], errors="raise"
                ).astype(int),
                pd.to_numeric(fr["front_score"], errors="raise").astype(float),
            )
        )

        for ch_i, ch in enumerate(CHANNELS):
            medians[ci, ch_i] = scale_map[ch][0]
            scales[ci, ch_i] = scale_map[ch][1]

        for ti, i in enumerate(tis):
            dest = np.concatenate(
                [np.asarray([i + 1], dtype=int), cmap[i].astype(int)]
            )
            if ci == 0:
                dest_indices[ti] = dest
            elif not np.array_equal(dest_indices[ti], dest):
                raise RuntimeError(
                    "destination identity changed across configurations"
                )

            raw = candidate_raw(
                c["X"], c["radius"], c["diff"], i, dest
            )
            for ch_i, ch in enumerate(CHANNELS):
                raw_cube[ti, ci, :, ch_i] = raw[ch]

            # cosine = 1 - turn_change exactly under frozen semantics
            cosine_cube[ti, ci, :] = 1.0 - raw["turn_change"]

            if i not in lookup:
                raise RuntimeError(
                    f"saved B50.4 front score missing {cid}, transition {i}"
                )
            saved_phys[ti, ci] = float(lookup[i])

    atomic_npz(
        cache_path,
        transition_indices=np.asarray(tis, dtype=int),
        config_ids=np.asarray(cids, dtype="U64"),
        dest_indices=dest_indices,
        raw_cube=raw_cube,
        cosine_cube=cosine_cube,
        medians=medians,
        scales=scales,
        saved_physical_front_scores=saved_phys,
    )

    cache_manifest = {
        "version": VERSION,
        "created": now_s(),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "input_hashes": frozen_hashes,
        "transition_indices": tis,
        "config_ids": cids,
        "channels": CHANNELS,
        "shape_raw_cube": list(raw_cube.shape),
        "shape_cosine_cube": list(cosine_cube.shape),
    }
    atomic_json(cache_manifest_path, cache_manifest)

    return {
        "transition_indices": np.asarray(tis, int),
        "config_ids": cids,
        "dest_indices": dest_indices,
        "raw_cube": raw_cube,
        "cosine_cube": cosine_cube,
        "medians": medians,
        "scales": scales,
        "saved_physical_front_scores": saved_phys,
    }


# ---------------------------------------------------------------------------
# Full B50.6 reproduction
# ---------------------------------------------------------------------------

def full_composite_from_cube(cube: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    raw = cube["raw_cube"]
    meds = cube["medians"]
    scales = cube["scales"]

    n_t, n_c, n_k, n_ch = raw.shape
    scores = np.empty((n_t, n_c, n_k), dtype=float)
    pct = np.empty_like(scores)
    include = np.ones(n_ch, dtype=bool)

    for ti in range(n_t):
        for ci in range(n_c):
            sc = score_raw_subset(
                raw[ti, ci], meds[ci], scales[ci], include
            )
            scores[ti, ci] = sc
            pct[ti, ci] = rank_percentiles(sc)

    consensus = np.median(pct, axis=1)
    return scores, consensus


def reproduction_audit(
    args,
    paths: Dict[str, Path],
    cube: Dict[str, Any],
) -> Dict[str, Any]:
    out = Path(args.output_dir)
    tdf = pd.read_csv(paths["b50_6_transition_specificity"])
    s6 = json.loads(paths["b50_6_summary"].read_text(encoding="utf-8"))

    scores, consensus = full_composite_from_cube(cube)
    saved_phys_score = cube["saved_physical_front_scores"]

    score_err = np.abs(scores[:, :, 0] - saved_phys_score)
    max_score_err = float(np.max(score_err))
    mean_score_err = float(np.mean(score_err))

    tdf = tdf.sort_values("transition_index", kind="stable")
    saved_pct = pd.to_numeric(
        tdf["consensus_physical_percentile"], errors="raise"
    ).to_numpy(float)
    if len(saved_pct) != consensus.shape[0]:
        raise RuntimeError("B50.6 transition count mismatch during reproduction")

    pct_err = np.abs(consensus[:, 0] - saved_pct)
    max_pct_err = float(np.max(pct_err))
    mean_pct_err = float(np.mean(pct_err))

    observed_effect = float(np.mean(consensus[:, 0]) - 0.5)
    saved_effect = float(
        s6["primary_specificity"]["observed_mean_effect"]
    )
    effect_err = abs(observed_effect - saved_effect)

    pass_ = bool(
        max_score_err <= THRESHOLDS["full_score_max_abs_error"]
        and max_pct_err <= THRESHOLDS["full_percentile_max_abs_error"]
        and effect_err <= THRESHOLDS["full_effect_max_abs_error"]
    )

    # Per-config score reproduction table.
    rows = []
    cids = cube["config_ids"]
    for ci, cid in enumerate(cids):
        e = score_err[:, ci]
        rows.append({
            "config_id": cid,
            "max_abs_physical_front_score_error": float(np.max(e)),
            "mean_abs_physical_front_score_error": float(np.mean(e)),
            "pass": bool(
                np.max(e) <= THRESHOLDS["full_score_max_abs_error"]
            ),
        })
    atomic_csv(
        out / "b50_10_full_score_reproduction_by_config.csv",
        pd.DataFrame(rows),
    )

    audit = {
        "pass": pass_,
        "max_abs_physical_front_score_error": max_score_err,
        "mean_abs_physical_front_score_error": mean_score_err,
        "max_abs_consensus_physical_percentile_error": max_pct_err,
        "mean_abs_consensus_physical_percentile_error": mean_pct_err,
        "reconstructed_full_effect": observed_effect,
        "frozen_b50_6_full_effect": saved_effect,
        "full_effect_abs_error": effect_err,
        "full_consensus_percentiles": consensus,
        "full_scores": scores,
    }

    atomic_json(
        out / "b50_10_reproduction_audit.json",
        {k: v for k, v in audit.items()
         if k not in ("full_consensus_percentiles", "full_scores")},
    )
    return audit


# ---------------------------------------------------------------------------
# Channel decomposition
# ---------------------------------------------------------------------------

def channel_percentile_cubes(cube: Dict[str, Any]) -> Dict[str, np.ndarray]:
    raw = cube["raw_cube"]
    n_t, n_c, n_k, n_ch = raw.shape
    out: Dict[str, np.ndarray] = {}

    for ch_i, ch in enumerate(CHANNELS):
        pct = np.empty((n_t, n_c, n_k), dtype=float)
        for ti in range(n_t):
            for ci in range(n_c):
                pct[ti, ci] = rank_percentiles(raw[ti, ci, :, ch_i])
        out[ch] = np.median(pct, axis=1)

    return out


def channel_analysis(
    args,
    cube: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    out = Path(args.output_dir)
    cpct = channel_percentile_cubes(cube)

    rows = []
    null_rows = []
    null_map: Dict[str, np.ndarray] = {}

    for ch_i, ch in enumerate(CHANNELS):
        res, null = matched_identity_null(
            cpct[ch],
            args.randomization_replicates,
            args.seed + 1000 + 137 * ch_i,
        )
        null_map[ch] = null

        rows.append({
            "channel": ch,
            **res,
            "effect_direction": (
                "negative" if res["observed_mean_effect"] < 0
                else "positive" if res["observed_mean_effect"] > 0
                else "zero"
            ),
        })

        for r, val in enumerate(null):
            null_rows.append({
                "channel": ch,
                "replicate": r,
                "null_mean_percentile_effect": float(val),
            })

    df = pd.DataFrame(rows)
    df["holm_p_two_sided"] = holm_adjust(df["p_two_sided"].to_numpy(float))
    df["effect_size_gate"] = (
        np.abs(df["observed_mean_effect"].to_numpy(float))
        >= args.channel_abs_effect_min
    )
    df["holm_significance_gate"] = (
        df["holm_p_two_sided"].to_numpy(float)
        <= THRESHOLDS["channel_holm_alpha"]
    )
    df["channel_confirmed"] = (
        df["effect_size_gate"] & df["holm_significance_gate"]
    )

    null_df = pd.DataFrame(null_rows)
    atomic_csv(out / "b50_10_channel_effects.csv", df)
    atomic_csv(out / "b50_10_channel_null.csv", null_df)

    return df, null_df, cpct


def channel_thirds_analysis(
    args,
    cpct: Dict[str, np.ndarray],
) -> pd.DataFrame:
    out = Path(args.output_dir)
    n = len(next(iter(cpct.values())))
    edges = np.linspace(0, n, 4, dtype=int)

    rows = []
    for ch_i, ch in enumerate(CHANNELS):
        full_effect = float(np.mean(cpct[ch][:, 0]) - 0.5)
        full_dir = np.sign(full_effect)

        for third in range(3):
            mask = np.zeros(n, dtype=bool)
            mask[edges[third]:edges[third + 1]] = True
            res, _ = matched_identity_null(
                cpct[ch],
                args.randomization_replicates,
                args.seed + 3000 + ch_i * 101 + third,
                mask=mask,
            )
            d = np.sign(res["observed_mean_effect"])
            same_dir = bool(full_dir != 0 and d == full_dir)
            nominal = bool(
                abs(res["observed_mean_effect"])
                >= THRESHOLDS["third_abs_effect_min"]
                and res["p_two_sided"]
                <= THRESHOLDS["third_nominal_p_max"]
            )

            idx = np.asarray(cube_transition_indices_global(cpct), int)[mask]
            rows.append({
                "channel": ch,
                "third": third + 1,
                "n": int(mask.sum()),
                "first_position": int(edges[third]),
                "last_position": int(edges[third + 1] - 1),
                "observed_mean_percentile": res["observed_mean_percentile"],
                "observed_mean_effect": res["observed_mean_effect"],
                "p_two_sided": res["p_two_sided"],
                "p_lower": res["p_lower"],
                "p_upper": res["p_upper"],
                "same_direction_as_full_channel": same_dir,
                "nominal_replication_pass": nominal,
            })

    df = pd.DataFrame(rows)
    atomic_csv(out / "b50_10_channel_thirds.csv", df)
    return df


def cube_transition_indices_global(cpct: Dict[str, np.ndarray]) -> np.ndarray:
    # Positions are sufficient for thirds output; physical transition IDs are
    # added later by audit_mode to avoid global mutable state.
    n = len(next(iter(cpct.values())))
    return np.arange(n)


# ---------------------------------------------------------------------------
# Leave-one-channel-out decomposition
# ---------------------------------------------------------------------------

def leave_one_out_analysis(
    args,
    cube: Dict[str, Any],
    full_consensus: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    out = Path(args.output_dir)
    raw = cube["raw_cube"]
    meds = cube["medians"]
    scales = cube["scales"]

    n_t, n_c, n_k, n_ch = raw.shape
    full_effect = float(np.mean(full_consensus[:, 0]) - 0.5)

    variants = [("full", None)] + [
        (f"minus_{ch}", ch_i) for ch_i, ch in enumerate(CHANNELS)
    ]

    summary_rows = []
    transition_rows = []
    consensus_map: Dict[str, np.ndarray] = {}

    for v_i, (name, removed) in enumerate(variants):
        if removed is None:
            consensus = np.asarray(full_consensus, float)
        else:
            pct = np.empty((n_t, n_c, n_k), dtype=float)
            include = np.ones(n_ch, dtype=bool)
            include[removed] = False

            for ti in range(n_t):
                for ci in range(n_c):
                    sc = score_raw_subset(
                        raw[ti, ci], meds[ci], scales[ci], include
                    )
                    pct[ti, ci] = rank_percentiles(sc)
            consensus = np.median(pct, axis=1)

        consensus_map[name] = consensus
        res, _ = matched_identity_null(
            consensus,
            args.randomization_replicates,
            args.seed + 5000 + v_i * 149,
        )
        effect = float(res["observed_mean_effect"])

        if removed is None or abs(full_effect) <= 1e-15:
            attenuation = 0.0
            retention = 1.0
            material = False
            removed_channel = ""
        else:
            attenuation = 1.0 - abs(effect) / abs(full_effect)
            retention = abs(effect) / abs(full_effect)
            material = attenuation >= args.loo_material_attenuation_min
            removed_channel = CHANNELS[removed]

        summary_rows.append({
            "variant": name,
            "removed_channel": removed_channel,
            **res,
            "full_effect_reference": full_effect,
            "abs_effect_retention_fraction": float(retention),
            "abs_effect_attenuation_fraction": float(attenuation),
            "material_attenuation": bool(material),
        })

        for ti in range(n_t):
            transition_rows.append({
                "transition_index": int(cube["transition_indices"][ti]),
                "variant": name,
                "removed_channel": removed_channel,
                "physical_percentile": float(consensus[ti, 0]),
                "physical_percentile_effect": float(
                    consensus[ti, 0] - 0.5
                ),
            })

    sdf = pd.DataFrame(summary_rows)
    tdf = pd.DataFrame(transition_rows)
    atomic_csv(out / "b50_10_leave_one_out.csv", sdf)
    atomic_csv(out / "b50_10_leave_one_out_by_transition.csv", tdf)
    return sdf, tdf, consensus_map


# ---------------------------------------------------------------------------
# Directional cosine diagnostic
# ---------------------------------------------------------------------------

def directional_analysis(
    args,
    cube: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
    out = Path(args.output_dir)
    cos_cube = cube["cosine_cube"]
    n_t, n_c, n_k = cos_cube.shape

    pct = np.empty_like(cos_cube)
    for ti in range(n_t):
        for ci in range(n_c):
            pct[ti, ci] = rank_percentiles(cos_cube[ti, ci])

    consensus_pct = np.median(pct, axis=1)
    res, null = matched_identity_null(
        consensus_pct,
        args.randomization_replicates,
        args.seed + 7000,
    )

    physical_cos = np.median(cos_cube[:, :, 0], axis=1)
    control_cos = np.median(
        np.median(cos_cube[:, :, 1:], axis=2), axis=1
    )
    advantage = physical_cos - control_cos

    # For cosine, the frozen directional hypothesis is positive:
    # physical next step more aligned with previous vector.
    pass_ = bool(
        res["observed_mean_effect"]
        >= THRESHOLDS["directional_abs_effect_min"]
        and res["p_upper"] <= THRESHOLDS["directional_p_max"]
    )

    summary = {
        **res,
        "median_physical_cosine": float(np.median(physical_cos)),
        "median_control_cosine": float(np.median(control_cos)),
        "mean_physical_minus_control_cosine": float(np.mean(advantage)),
        "median_physical_minus_control_cosine": float(np.median(advantage)),
        "directional_positive_gate": pass_,
        "role": (
            "secondary diagnostic; algebraically related to "
            "turn_change = 1 - cosine"
        ),
    }

    rows = []
    for ti in range(n_t):
        rows.append({
            "transition_index": int(cube["transition_indices"][ti]),
            "consensus_physical_cosine_percentile": float(
                consensus_pct[ti, 0]
            ),
            "median_physical_cosine": float(physical_cos[ti]),
            "median_control_cosine": float(control_cos[ti]),
            "physical_minus_control_cosine": float(advantage[ti]),
        })

    df = pd.DataFrame(rows)
    atomic_csv(out / "b50_10_directional_geometry.csv", df)
    atomic_csv(
        out / "b50_10_directional_null.csv",
        pd.DataFrame({
            "replicate": np.arange(len(null)),
            "null_mean_cosine_percentile_effect": null,
        }),
    )
    atomic_json(out / "b50_10_directional_summary.json", summary)
    return summary, df, null


# ---------------------------------------------------------------------------
# Verdict and plots
# ---------------------------------------------------------------------------

def mechanism_verdict(
    channel_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    directional: Dict[str, Any],
) -> Tuple[str, str, List[str]]:
    confirmed = channel_df[channel_df["channel_confirmed"] == True].copy()

    localized: List[str] = []
    for _, r in confirmed.iterrows():
        ch = str(r["channel"])
        loo = loo_df[loo_df["removed_channel"] == ch]
        if len(loo) == 1 and bool(loo.iloc[0]["material_attenuation"]):
            localized.append(ch)

    # Directional label has the strongest semantic meaning only when all
    # three predeclared pieces align: turn raw channel, turn LOO attenuation,
    # and explicit cosine-positive diagnostic.
    directional_localized = (
        "turn_change" in localized
        and bool(directional.get("directional_positive_gate", False))
    )

    if directional_localized and len(localized) == 1:
        return (
            "DIRECTIONAL_TURN_GEOMETRY_SCORE_LOCALIZED",
            "The frozen B50.6 effect localizes primarily to directional turning geometry: "
            "turn_change is Holm-confirmed with nontrivial effect size, removing it materially "
            "attenuates the composite effect, and the equivalent cosine diagnostic independently "
            "shows higher physical directional alignment under the matched-control test.",
            localized,
        )

    if len(localized) == 1:
        ch = localized[0]
        return (
            f"{ch.upper()}_GEOMETRY_SCORE_LOCALIZED",
            f"The mechanism audit score-localizes the replicated B50 adjacency-specificity "
            f"effect to {ch}: the raw channel is Holm-confirmed and removing that channel "
            f"materially attenuates the frozen composite effect.",
            localized,
        )

    if len(localized) >= 2:
        return (
            "DISTRIBUTED_MULTICHANNEL_SCORE_LOCALIZATION",
            "Multiple raw channels are Holm-confirmed and each materially contributes under "
            "leave-one-channel-out attenuation. The replicated B50 effect is therefore best "
            "described as distributed multichannel geometry rather than a single-channel mechanism.",
            localized,
        )

    if len(confirmed) >= 1:
        return (
            "CHANNEL_ASSOCIATION_WITHOUT_SCORE_LOCALIZATION",
            "At least one raw channel is Holm-confirmed with a nontrivial percentile effect, "
            "but no single channel satisfies the frozen leave-one-out material-attenuation "
            "criterion. The channel structure is associated with the effect without clean "
            "single-channel score localization.",
            localized,
        )

    return (
        "MECHANISM_UNRESOLVED_UNDER_FROZEN_CHANNEL_DECOMPOSITION",
        "None of the five frozen raw channels simultaneously clears the predeclared Holm "
        "significance and effect-size gates. B50.10 therefore does not resolve a channel-level "
        "mechanism for the already replicated composite adjacency-specificity effect.",
        localized,
    )


def make_plots(
    out: Path,
    channel_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    directional_df: pd.DataFrame,
) -> None:
    # Channel effects.
    x = np.arange(len(channel_df))
    plt.figure(figsize=(9, 5))
    plt.bar(x, channel_df["observed_mean_effect"].to_numpy(float))
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xticks(x, channel_df["channel"].astype(str), rotation=35, ha="right")
    plt.ylabel("mean physical percentile effect vs 0.5")
    plt.title("B50.10 raw-channel adjacency specificity")
    plt.tight_layout()
    plt.savefig(out / "b50_10_channel_effects.png", dpi=170)
    plt.close()

    # Leave-one-out effect.
    x = np.arange(len(loo_df))
    plt.figure(figsize=(10, 5))
    plt.bar(x, loo_df["observed_mean_effect"].to_numpy(float))
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xticks(x, loo_df["variant"].astype(str), rotation=40, ha="right")
    plt.ylabel("mean physical percentile effect vs 0.5")
    plt.title("B50.10 leave-one-channel-out composite")
    plt.tight_layout()
    plt.savefig(out / "b50_10_leave_one_out.png", dpi=170)
    plt.close()

    # LOO attenuation.
    loo_only = loo_df[loo_df["removed_channel"].astype(str) != ""].copy()
    x = np.arange(len(loo_only))
    plt.figure(figsize=(9, 5))
    plt.bar(x, loo_only["abs_effect_attenuation_fraction"].to_numpy(float))
    plt.axhline(
        THRESHOLDS["loo_material_attenuation_min"],
        linestyle="--",
        linewidth=0.8,
    )
    plt.xticks(
        x, loo_only["removed_channel"].astype(str), rotation=35, ha="right"
    )
    plt.ylabel("fractional attenuation of |full effect|")
    plt.title("B50.10 score localization by channel removal")
    plt.tight_layout()
    plt.savefig(out / "b50_10_leave_one_out_attenuation.png", dpi=170)
    plt.close()

    # Directional advantage.
    plt.figure(figsize=(10, 5))
    plt.plot(
        directional_df["transition_index"],
        directional_df["physical_minus_control_cosine"],
        linewidth=1.0,
    )
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xlabel("physical transition index")
    plt.ylabel("physical cosine - median matched-control cosine")
    plt.title("B50.10 directional continuation advantage")
    plt.tight_layout()
    plt.savefig(out / "b50_10_directional_advantage.png", dpi=170)
    plt.close()


# ---------------------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------------------

def audit_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plan = ensure_frozen_plan(args)
    paths = input_paths(args)
    require_paths(paths)

    print("=== B50.10 CONDITIONAL TRANSITION GEOMETRY MECHANISM AUDIT ===")

    hash_audit = verify_frozen_hashes(args, plan)
    print("frozen input hashes exact :", hash_audit["pass"])
    if not hash_audit["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_FROZEN_INPUTS",
            "reason": "One or more B50.10 frozen input hashes changed after PLAN.",
        }
        atomic_json(out / "b50_10_summary.json", summary)
        atomic_text(
            out / "b50_10_verdict.txt",
            "FINAL VERDICT: INVALID_FROZEN_INPUTS\n",
        )
        return summary

    lineage = lineage_gate(paths)
    if not lineage["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_LINEAGE",
            "reason": "B50.6/B50.7/B50.8 lineage qualification failed.",
        }
        atomic_json(out / "b50_10_summary.json", summary)
        atomic_text(
            out / "b50_10_verdict.txt",
            "FINAL VERDICT: INVALID_LINEAGE\n",
        )
        return summary

    state = pd.read_csv(paths["state_dataset"])
    validate_state(state)
    tdf = pd.read_csv(paths["b50_6_transition_specificity"])
    mdf = pd.read_csv(paths["b50_6_matching_audit"])
    tis, cmap, control_info, _ = frozen_control_map(tdf, mdf)

    print("frozen transitions        :", len(tis))
    print("controls / transition     :", control_info["controls_per_transition"])

    cube = build_mechanism_cube(args, plan, paths, tis, cmap)

    print("[3/5] exact full B50.6 endpoint reproduction", flush=True)
    repro = reproduction_audit(args, paths, cube)
    print("full reproduction PASS    :", repro["pass"])
    print(
        "reconstructed full effect :",
        repro["reconstructed_full_effect"],
    )
    if not repro["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_B50_6_REPRODUCTION",
            "reason": (
                "B50.10 failed to reproduce the frozen B50.6 composite "
                "front score / physical percentile endpoint exactly."
            ),
            "reproduction": {
                k: v for k, v in repro.items()
                if k not in ("full_consensus_percentiles", "full_scores")
            },
        }
        atomic_json(out / "b50_10_summary.json", summary)
        atomic_text(
            out / "b50_10_verdict.txt",
            "FINAL VERDICT: INVALID_B50_6_REPRODUCTION\n",
        )
        return summary

    print("[4/5] channel decomposition / LOO / directional tests", flush=True)
    channel_df, channel_null_df, cpct = channel_analysis(args, cube)

    thirds_df = channel_thirds_analysis(args, cpct)
    # Replace positional fields with physical transition IDs where useful.
    edges = np.linspace(0, len(tis), 4, dtype=int)
    for idx, row in thirds_df.iterrows():
        th = int(row["third"]) - 1
        thirds_df.at[idx, "first_transition_index"] = int(tis[edges[th]])
        thirds_df.at[idx, "last_transition_index"] = int(tis[edges[th + 1] - 1])
    atomic_csv(out / "b50_10_channel_thirds.csv", thirds_df)

    loo_df, loo_transition_df, loo_map = leave_one_out_analysis(
        args,
        cube,
        repro["full_consensus_percentiles"],
    )

    directional, directional_df, directional_null = directional_analysis(
        args, cube
    )

    # Add channel thirds summaries.
    stability_rows = []
    for ch in CHANNELS:
        sub = thirds_df[thirds_df["channel"] == ch]
        full_effect = float(
            channel_df.loc[
                channel_df["channel"] == ch, "observed_mean_effect"
            ].iloc[0]
        )
        same = int(sub["same_direction_as_full_channel"].sum())
        nominal = int(sub["nominal_replication_pass"].sum())
        stability_rows.append({
            "channel": ch,
            "full_effect": full_effect,
            "same_direction_thirds": same,
            "nominal_thirds": nominal,
            "thirds_stability_pass": bool(
                same >= THRESHOLDS["third_same_direction_required"]
                and nominal >= THRESHOLDS["third_nominal_required"]
            ),
        })
    stability_df = pd.DataFrame(stability_rows)
    atomic_csv(out / "b50_10_channel_stability_summary.csv", stability_df)

    print("[5/5] frozen mechanism verdict", flush=True)
    verdict, reason, localized = mechanism_verdict(
        channel_df, loo_df, directional
    )

    confirmed_channels = (
        channel_df.loc[channel_df["channel_confirmed"] == True, "channel"]
        .astype(str)
        .tolist()
    )

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "lineage": lineage,
        "frozen_input_hashes_exact": hash_audit["pass"],
        "control_identity": control_info,
        "reproduction": {
            k: v for k, v in repro.items()
            if k not in ("full_consensus_percentiles", "full_scores")
        },
        "channel_decomposition": {
            "confirmed_channels": confirmed_channels,
            "rows": channel_df.to_dict(orient="records"),
            "holm_alpha": THRESHOLDS["channel_holm_alpha"],
            "effect_size_floor": args.channel_abs_effect_min,
        },
        "leave_one_out": {
            "material_attenuation_floor": args.loo_material_attenuation_min,
            "localized_channels": localized,
            "rows": loo_df.to_dict(orient="records"),
        },
        "directional_geometry": directional,
        "channel_thirds_stability": stability_df.to_dict(orient="records"),
        "final_verdict": verdict,
        "final_reason": reason,
        "scientific_boundary": (
            "Mechanism/decomposition audit of the already replicated B50.6 "
            "same-corpus matched adjacency-specificity effect. Score localization "
            "is not causal attribution. No independent corpus, B59, individual-prime, "
            "RH, or theorem claim."
        ),
    }
    atomic_json(out / "b50_10_summary.json", summary)

    make_plots(out, channel_df, loo_df, directional_df)

    report = f"""B50.10 — Conditional Transition Geometry Mechanism Audit
===========================================================

Interpretation scope
--------------------
Mechanism/decomposition of frozen B50.6 effect: YES
Exact B50.6 matched controls reused: YES
B50.6-B50.8 lineage frozen: YES
Independent-corpus replication: NO
B59/B59.1 used: NO

Frozen lineage
--------------
B50.6 = {lineage['b50_6_verdict']}
B50.7 = {lineage['b50_7_verdict']}
B50.8 = {lineage['b50_8_verdict']}
frozen input hashes exact = {hash_audit['pass']}

B50.6 endpoint reproduction
---------------------------
max physical front-score error       = {repro['max_abs_physical_front_score_error']}
max consensus percentile error       = {repro['max_abs_consensus_physical_percentile_error']}
reconstructed full effect            = {repro['reconstructed_full_effect']}
frozen B50.6 full effect             = {repro['frozen_b50_6_full_effect']}
effect absolute error                = {repro['full_effect_abs_error']}
REPRODUCTION PASS                    = {repro['pass']}

Raw-channel decomposition
-------------------------
{channel_df.to_string(index=False)}

Confirmed channels after Holm + effect-size gate
------------------------------------------------
{confirmed_channels}

Leave-one-channel-out composite
-------------------------------
{loo_df.to_string(index=False)}

Score-localized channels
------------------------
{localized}

Directional cosine diagnostic
-----------------------------
mean physical cosine percentile      = {directional['observed_mean_percentile']}
mean cosine percentile effect        = {directional['observed_mean_effect']}
p_upper                              = {directional['p_upper']}
median physical cosine               = {directional['median_physical_cosine']}
median control cosine                = {directional['median_control_cosine']}
mean physical-control cosine         = {directional['mean_physical_minus_control_cosine']}
directional positive gate            = {directional['directional_positive_gate']}

Channel thirds stability
------------------------
{stability_df.to_string(index=False)}

FINAL VERDICT: {verdict}

REASON:
{reason}

Interpretation boundary
-----------------------
B50.10 decomposes the already replicated B50.6 matched adjacency-specificity
endpoint. A leave-one-channel-out attenuation identifies score dependence,
not causal mechanism. This is not independent-corpus replication, does not
validate B59, does not predict individual primes, and is not RH/theorem evidence.

Key outputs
-----------
b50_10_protocol_manifest.json
b50_10_plan.txt
b50_10_frozen_control_map.csv
b50_10_range_mapping_audit.csv
b50_10_frozen_input_audit.csv
b50_10_mechanism_cube.npz
b50_10_cache_manifest.json
b50_10_reproduction_audit.json
b50_10_full_score_reproduction_by_config.csv
b50_10_channel_effects.csv
b50_10_channel_null.csv
b50_10_channel_thirds.csv
b50_10_channel_stability_summary.csv
b50_10_leave_one_out.csv
b50_10_leave_one_out_by_transition.csv
b50_10_directional_geometry.csv
b50_10_directional_null.csv
b50_10_directional_summary.json
b50_10_summary.json
b50_10_verdict.txt
b50_10_channel_effects.png
b50_10_leave_one_out.png
b50_10_leave_one_out_attenuation.png
b50_10_directional_advantage.png
"""
    atomic_text(out / "b50_10_verdict.txt", report)

    print()
    print("=== B50.10 RESULT ===")
    print("confirmed channels :", confirmed_channels)
    print("localized channels :", localized)
    print(
        "directional gate   :",
        directional["directional_positive_gate"],
    )
    print("FINAL VERDICT      :", verdict)
    print("verdict            :", out / "b50_10_verdict.txt")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "B50.10 Conditional Transition Geometry Mechanism Audit"
        )
    )
    p.add_argument(
        "--mode",
        choices=["plan", "audit", "all"],
        default="plan",
    )

    p.add_argument("--operator-stack", required=True)
    p.add_argument("--state-dataset", required=True)
    p.add_argument("--b50-4-script", required=True)
    p.add_argument("--b50-4-protocol", required=True)
    p.add_argument("--b50-4-front-features", required=True)
    p.add_argument("--b50-6-dir", required=True)
    p.add_argument("--b50-7-summary", required=True)
    p.add_argument("--b50-8-summary", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument(
        "--randomization-replicates",
        type=int,
        default=5000,
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--reuse-existing", action="store_true")

    # These are exposed for transparent protocol inspection, but changing
    # either value marks the protocol modified and causes PLAN PASS=False.
    p.add_argument(
        "--channel-abs-effect-min",
        type=float,
        default=THRESHOLDS["channel_abs_effect_min"],
    )
    p.add_argument(
        "--loo-material-attenuation-min",
        type=float,
        default=THRESHOLDS["loo_material_attenuation_min"],
    )

    a = p.parse_args()

    if a.randomization_replicates < 999:
        p.error("--randomization-replicates must be >=999")
    if a.channel_abs_effect_min < 0:
        p.error("--channel-abs-effect-min must be >=0")
    if not (0 <= a.loo_material_attenuation_min <= 1):
        p.error("--loo-material-attenuation-min must be in [0,1]")

    return a


def main():
    args = parse_args()

    if args.mode == "plan":
        plan_mode(args)
    elif args.mode == "audit":
        audit_mode(args)
    elif args.mode == "all":
        plan = plan_mode(args)
        if not plan.get("plan_pass", False):
            raise RuntimeError("B50.10 PLAN failed; AUDIT not started.")
        audit_mode(args)
    else:
        raise ValueError(args.mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
