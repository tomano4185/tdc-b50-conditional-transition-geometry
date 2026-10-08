#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDC TOR A / B50.7 — External Offset-Window Adjacency Replication
================================================================

Purpose
-------
B50.6 found internally replicated physical-adjacency specificity on the
original 666-state sequence. B50.7 performs a preregistered replication on a
NEW offset-window state sequence built from the same prime corpus.

Important boundary
------------------
"External" here means external to the original 666 WINDOW STATES and frozen
B50.1–B50.6 state sample. The new windows are offset and are not identical to
any original window. They are, however, constructed from the SAME underlying
prime corpus and overlap it. This is therefore an out-of-sample window
replication, not an independent-corpus replication.

Frozen replication target from B50.6
-------------------------------------
B50.6 established a NEGATIVE matched-adjacency effect:
actual next states had lower frozen continuous-front percentile than
full-operator-distance-matched non-adjacent controls.

B50.7 freezes that direction prospectively:
    expected effect < 0

No B59/B59.1 signal is read or used.

Window construction
-------------------
The original physical windows are parsed from the original B50 state labels,
expected to have the form:
    range_000001-000200

The nominal original start step is inferred from the original sequence.
Default offset is half that step. For the current sequence:
    nominal step = 150
    default offset = +75 segments

Each valid offset window keeps the original window length. Any terminal window
that would exceed --total-segments is dropped. The plan asserts that no offset
window is identical to any original window.

Frozen representation protocol
------------------------------
- The ORIGINAL B50.2 operator stack is the TRAINING representation.
- PCA mean and basis are fit ONLY on the original operator stack.
- Offset-window operators are projected into that frozen PCA basis.
- B50.4 dimensions and graph-k scales are read from the frozen B50.4 protocol.
- B50.4 raw front-channel scaling (median/MAD with std fallback) is frozen from
  the ORIGINAL B50.4 physical transitions and is applied unchanged to the new
  offset-window sequence and its matched controls.
- Diffusion coordinates are reconstructed on the NEW sequence using the exact
  frozen B50.4 graph/diffusion implementation and frozen graph-k/mode counts.

Matched adjacency replication
-----------------------------
For each new physical source i (i>=1), compare actual next state i+1 with M
non-adjacent destinations j matched on full 4096D operator distance from i.
The previous physical state i-1 is held fixed, matching B50.6 semantics.

Primary endpoint
----------------
For every frozen B50.4 representation, compute the physical destination's
percentile among:
    1 physical destination + M matched non-adjacent controls.

Take the median percentile across frozen configurations for every transition,
then average across transitions.

Null expectation: 0.5.
Replication direction frozen from B50.6: LOWER than 0.5.

Primary null
------------
Matched-set identity randomization. For each transition choose one member of
its matched set uniformly as pseudo-physical. One-sided lower-tail p-value.

Pre-declared gates
------------------
OFFSET_NOVELTY_PASS:
    no offset window is identical to an original window
    offset != 0
    all offset windows have original nominal length

EXTRACTION_INTEGRITY_PASS:
    exact B50 operator dim equals training operator dim
    exact offset-window label/order equals frozen plan

MATCHING_QUALITY_PASS:
    median |log-distance error| <= 0.10
    q90    |log-distance error| <= 0.25

PRIMARY_REPLICATION_PASS:
    mean physical percentile effect <= -0.05
    one-sided matched identity p_lower <= 0.01

CONFIG_CONVERGENCE_PASS:
    >=80% frozen B50.4 configurations have negative mean effect

THIRDS_ROBUSTNESS_PASS:
    all 3 contiguous thirds have negative effect
    and >=2/3 thirds have:
        effect <= -0.04
        p_lower <= 0.05

Final verdicts
--------------
EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED
OFFSET_ADJACENCY_SPECIFICITY_REPLICATED_WITH_WEAK_THIRDS
OFFSET_REPLICATION_NULL_COMPATIBLE
OFFSET_REPLICATION_DIRECTION_REVERSED
WEAK_OR_INCONSISTENT_OFFSET_REPLICATION
INVALID_OFFSET_PLAN
INVALID_EXTRACTION
INVALID_MATCHING_GEOMETRY
INVALID_REPRODUCTION

Modes
-----
plan
    Build/freeze the offset-window plan only. No prime-data read.

extract
    Reconstruct operators for the frozen offset plan. Resumable per-window
    cache. Uses hardlinks in a temporary staging folder and never copies the
    prime corpus.

audit
    Run frozen external adjacency replication from the extracted stack.

all
    plan + extract + audit.

Scientific boundary
-------------------
A positive result is an out-of-sample OFFSET-WINDOW replication on new window
states from the same underlying prime corpus. It is not an independent-corpus
replication, does not validate B59, does not predict individual primes, and is
not a number-theoretic theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "B50.7_external_offset_window_adjacency_replication_v1"

CHANNELS = [
    "speed",
    "acceleration",
    "turn_change",
    "density_scale_change",
    "diffusion_step",
]

THRESHOLDS = {
    "matching_median_abs_log_distance_error_max": 0.10,
    "matching_q90_abs_log_distance_error_max": 0.25,
    "primary_effect_max": -0.05,
    "primary_p_lower_max": 0.01,
    "config_negative_fraction_min": 0.80,
    "third_effect_max": -0.04,
    "third_p_lower_max": 0.05,
    "third_negative_required": 3,
    "third_nominal_required": 2,
}

RANGE_RE = re.compile(r"^range_(\d+)-(\d+)$", re.IGNORECASE)
SEGMENT_RE = re.compile(r"primes_batch_(\d+)_all_with_1\.txt$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


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


def atomic_npz(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **kwargs)
    tmp.replace(path)


def safe_spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def validate_state_dataset(df: pd.DataFrame) -> None:
    required = {"range_name", "range_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"State dataset missing columns: {sorted(missing)}")
    idx = pd.to_numeric(df["range_index"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(len(df))):
        raise ValueError("range_index is not exactly 0..n-1")
    if df["range_name"].duplicated().any():
        raise ValueError("Duplicate range_name in state dataset")


def import_module_from_path(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_b50(path: Path):
    mod = import_module_from_path(path, "tdc_b50_frozen_b507")
    required = ["load_full_range_primes", "primes_to_gaps", "build_markov_operator"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise AttributeError(f"B50 missing required functions: {missing}")
    return mod


def import_b504(path: Path):
    mod = import_module_from_path(path, "tdc_b504_frozen_b507")
    required = ["build_knn_graph", "diffusion_modes"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise AttributeError(f"B50.4 missing required functions: {missing}")

    front_fn = None
    for name in ("transition_front_features", "front_features"):
        if hasattr(mod, name):
            front_fn = getattr(mod, name)
            break
    if front_fn is None:
        raise AttributeError("B50.4 exposes neither transition_front_features nor front_features")
    return mod, front_fn


def unpack_graph_result(result):
    if not isinstance(result, tuple):
        raise TypeError("B50.4 build_knn_graph did not return a tuple")
    if len(result) == 4:
        W, _G, _nbr, radius = result
    elif len(result) == 3:
        W, _nbr, radius = result
    else:
        raise RuntimeError(f"Unsupported B50.4 build_knn_graph return length: {len(result)}")
    return W, np.asarray(radius, dtype=float)


def call_front_fn(front_fn, X, radius, diff) -> pd.DataFrame:
    attempts = [
        lambda: front_fn(X, radius, diff),
        lambda: front_fn(X=X, local_radius=radius, diffusion_coords=diff),
    ]
    errs = []
    for fn in attempts:
        try:
            out = fn()
            if isinstance(out, pd.DataFrame):
                return out
        except TypeError as exc:
            errs.append(str(exc))
    raise RuntimeError("Cannot call frozen B50.4 front function: " + " | ".join(errs))


# ---------------------------------------------------------------------------
# Offset plan
# ---------------------------------------------------------------------------


def parse_range_name(name: str) -> Tuple[int, int]:
    m = RANGE_RE.match(str(name))
    if not m:
        raise ValueError(f"Cannot parse physical range label: {name}")
    return int(m.group(1)), int(m.group(2))


def mode_int(values: Sequence[int]) -> Tuple[int, float]:
    vals = np.asarray(values, dtype=int)
    uniq, counts = np.unique(vals, return_counts=True)
    j = int(np.argmax(counts))
    return int(uniq[j]), float(counts[j] / len(vals))


def build_offset_plan(
    state: pd.DataFrame,
    total_segments: int,
    requested_offset: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    original = [parse_range_name(x) for x in state["range_name"].astype(str)]
    starts = np.asarray([a for a, _ in original], dtype=int)
    ends = np.asarray([b for _, b in original], dtype=int)
    lengths = ends - starts + 1

    if len(original) < 3:
        raise ValueError("Need at least 3 original windows")

    step_diffs = np.diff(starts)
    nominal_step, step_mode_fraction = mode_int(step_diffs)
    nominal_length, length_mode_fraction = mode_int(lengths)

    if step_mode_fraction < 0.95:
        raise RuntimeError(
            f"Original start-step is not sufficiently uniform: mode={nominal_step}, fraction={step_mode_fraction}"
        )
    if length_mode_fraction < 0.95:
        raise RuntimeError(
            f"Original window length is not sufficiently uniform: mode={nominal_length}, fraction={length_mode_fraction}"
        )

    auto_offset = nominal_step // 2
    offset = int(auto_offset if requested_offset is None else requested_offset)
    if offset == 0:
        raise ValueError("Offset must be non-zero")

    original_set = set(original)
    rows = []
    dropped = []

    for original_index, (a, b) in enumerate(original):
        na = a + offset
        nb = na + nominal_length - 1
        if na < 1 or nb > total_segments:
            dropped.append({
                "original_index": original_index,
                "original_start": a,
                "original_end": b,
                "offset_start": na,
                "offset_end": nb,
                "reason": "outside_total_segments",
            })
            continue

        label = f"range_{na:06d}-{nb:06d}"
        rows.append({
            "offset_range_index": len(rows),
            "source_original_index": original_index,
            "range_name": label,
            "start_segment": na,
            "end_segment": nb,
            "window_length": nominal_length,
            "offset_segments": offset,
            "identical_to_original_window": (na, nb) in original_set,
        })

    plan = pd.DataFrame(rows)
    if len(plan) < 6:
        raise RuntimeError("Offset plan has fewer than 6 valid windows")

    exact_duplicates = int(plan["identical_to_original_window"].sum())
    labels_unique = not plan["range_name"].duplicated().any()
    starts_increasing = bool(np.all(np.diff(plan["start_segment"].to_numpy(int)) > 0))
    lengths_exact = bool(np.all(plan["window_length"].to_numpy(int) == nominal_length))

    expected_default_offset = int(nominal_step // 2)
    protocol_modified = offset != expected_default_offset

    summary = {
        "version": VERSION,
        "created": now_s(),
        "original_window_count": int(len(original)),
        "offset_window_count": int(len(plan)),
        "dropped_window_count": int(len(dropped)),
        "original_nominal_step": nominal_step,
        "original_step_mode_fraction": step_mode_fraction,
        "original_nominal_length": nominal_length,
        "original_length_mode_fraction": length_mode_fraction,
        "default_half_step_offset": expected_default_offset,
        "applied_offset": offset,
        "protocol_modified_from_default": protocol_modified,
        "total_segments": int(total_segments),
        "exact_duplicate_with_original_count": exact_duplicates,
        "labels_unique": labels_unique,
        "starts_strictly_increasing": starts_increasing,
        "window_lengths_exact": lengths_exact,
        "offset_novelty_pass": bool(
            offset != 0
            and exact_duplicates == 0
            and labels_unique
            and starts_increasing
            and lengths_exact
        ),
        "dropped_windows": dropped,
        "scientific_boundary": (
            "New offset window states from the same underlying prime corpus; "
            "not an independent-corpus replication."
        ),
    }
    return plan, summary


def plan_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = pd.read_csv(args.state_dataset)
    validate_state_dataset(state)

    plan, summary = build_offset_plan(
        state,
        args.total_segments,
        args.offset_segments,
    )

    summary["state_dataset"] = str(Path(args.state_dataset))
    summary["state_dataset_sha256"] = sha256_file(Path(args.state_dataset))

    atomic_csv(out / "b50_7_offset_window_plan.csv", plan)
    atomic_json(out / "b50_7_offset_window_plan.json", summary)

    print("=== B50.7 OFFSET PLAN ===")
    print(f"original windows : {summary['original_window_count']}")
    print(f"offset windows   : {summary['offset_window_count']}")
    print(f"nominal step     : {summary['original_nominal_step']}")
    print(f"window length    : {summary['original_nominal_length']}")
    print(f"offset           : {summary['applied_offset']}")
    print(f"duplicates       : {summary['exact_duplicate_with_original_count']}")
    print(f"novelty PASS     : {summary['offset_novelty_pass']}")
    print(f"protocol modified: {summary['protocol_modified_from_default']}")
    print(f"plan             : {out / 'b50_7_offset_window_plan.csv'}")

    if not summary["offset_novelty_pass"]:
        raise RuntimeError("INVALID_OFFSET_PLAN")
    return summary


# ---------------------------------------------------------------------------
# Offset operator extraction
# ---------------------------------------------------------------------------


def segment_path(prime_root: Path, segment_index: int) -> Path:
    return prime_root / f"primes_batch_{segment_index:06d}_all_with_1.txt"


def hardlink_window(
    prime_root: Path,
    staging_dir: Path,
    start_segment: int,
    end_segment: int,
) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        for seg in range(start_segment, end_segment + 1):
            src = segment_path(prime_root, seg)
            if not src.exists():
                raise FileNotFoundError(src)
            dst = staging_dir / src.name
            os.link(src, dst)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def compute_operator_for_window(
    b50,
    prime_root: Path,
    staging_root: Path,
    row: pd.Series,
    gap_bins: int,
) -> Dict[str, Any]:
    range_name = str(row["range_name"])
    stage = staging_root / range_name
    start = int(row["start_segment"])
    end = int(row["end_segment"])

    hardlink_window(prime_root, stage, start, end)
    try:
        primes = b50.load_full_range_primes(stage)
        gaps = b50.primes_to_gaps(primes).astype(float)
        if len(gaps) < 100:
            raise RuntimeError(f"Too few gaps in {range_name}: {len(gaps)}")
        P = b50.build_markov_operator(gaps, n_bins=gap_bins)
        vec = np.asarray(P, dtype=np.float64).reshape(-1)
        rec = {
            "range_name": range_name,
            "start_segment": start,
            "end_segment": end,
            "operator_vector": vec,
            "operator_dim": int(len(vec)),
            "n_primes": int(len(primes)),
            "n_gaps": int(len(gaps)),
            "mean_log_gap": float(np.mean(np.log(gaps))),
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
        }
        return rec
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def cache_path(cache_dir: Path, range_name: str) -> Path:
    return cache_dir / f"{range_name}.npz"


def save_cache(path: Path, rec: Dict[str, Any], gap_bins: int) -> None:
    atomic_npz(
        path,
        operator_vector=np.asarray(rec["operator_vector"], dtype=np.float64),
        range_name=np.asarray([rec["range_name"]]),
        start_segment=np.asarray([rec["start_segment"]], dtype=np.int64),
        end_segment=np.asarray([rec["end_segment"]], dtype=np.int64),
        n_primes=np.asarray([rec["n_primes"]], dtype=np.int64),
        n_gaps=np.asarray([rec["n_gaps"]], dtype=np.int64),
        mean_log_gap=np.asarray([rec["mean_log_gap"]], dtype=np.float64),
        gap_mean=np.asarray([rec["gap_mean"]], dtype=np.float64),
        gap_std=np.asarray([rec["gap_std"]], dtype=np.float64),
        gap_bins=np.asarray([gap_bins], dtype=np.int64),
    )


def load_cache(path: Path, expected_name: str, gap_bins: int) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        name = str(z["range_name"][0])
        if name != expected_name:
            raise RuntimeError(f"Cache label mismatch: expected {expected_name}, got {name}")
        if int(z["gap_bins"][0]) != gap_bins:
            raise RuntimeError(f"Cache gap_bins mismatch in {path}")
        return {
            "range_name": name,
            "start_segment": int(z["start_segment"][0]),
            "end_segment": int(z["end_segment"][0]),
            "operator_vector": np.asarray(z["operator_vector"], dtype=np.float64),
            "operator_dim": int(len(z["operator_vector"])),
            "n_primes": int(z["n_primes"][0]),
            "n_gaps": int(z["n_gaps"][0]),
            "mean_log_gap": float(z["mean_log_gap"][0]),
            "gap_mean": float(z["gap_mean"][0]),
            "gap_std": float(z["gap_std"][0]),
        }


def extract_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plan_path = out / "b50_7_offset_window_plan.csv"
    plan_json_path = out / "b50_7_offset_window_plan.json"
    if not plan_path.exists() or not plan_json_path.exists():
        raise FileNotFoundError("Run --mode plan first; B50.7 offset plan is missing")

    plan = pd.read_csv(plan_path)
    plan_summary = json.loads(plan_json_path.read_text(encoding="utf-8"))
    if not bool(plan_summary.get("offset_novelty_pass", False)):
        raise RuntimeError("Frozen offset plan did not pass novelty gate")

    prime_root = Path(args.prime_root)
    if not prime_root.exists():
        raise FileNotFoundError(prime_root)

    b50 = import_b50(Path(args.b50_script))

    with np.load(args.training_operator_stack, allow_pickle=False) as z:
        train_stack = np.asarray(z["P_stack"], dtype=np.float64)
    expected_operator_dim = int(train_stack.shape[1])

    cache_dir = out / "operator_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging_root = Path(args.staging_dir) if args.staging_dir else out / "_staging_hardlinks"
    staging_root.mkdir(parents=True, exist_ok=True)

    vectors = []
    status_rows = []
    meta_rows = []
    t_all = time.time()

    print("=== B50.7 OFFSET OPERATOR EXTRACTION ===")
    print(f"windows       : {len(plan)}")
    print(f"prime root    : {prime_root}")
    print(f"operator dim  : {expected_operator_dim}")
    print(f"reuse cache   : {args.reuse_existing}")
    print(f"staging       : {staging_root}")

    for i, row in plan.iterrows():
        name = str(row["range_name"])
        cp = cache_path(cache_dir, name)
        t0 = time.time()

        if args.reuse_existing and cp.exists():
            rec = load_cache(cp, name, args.gap_bins)
            source = "cache"
        else:
            rec = compute_operator_for_window(
                b50,
                prime_root,
                staging_root,
                row,
                args.gap_bins,
            )
            save_cache(cp, rec, args.gap_bins)
            source = "recomputed"

        if int(rec["operator_dim"]) != expected_operator_dim:
            raise RuntimeError(
                f"Operator dimension mismatch at {name}: {rec['operator_dim']} != {expected_operator_dim}"
            )
        if int(rec["start_segment"]) != int(row["start_segment"]) or int(rec["end_segment"]) != int(row["end_segment"]):
            raise RuntimeError(f"Cached/extracted range bounds mismatch at {name}")

        vectors.append(rec["operator_vector"])
        meta_rows.append({
            "range_index": int(i),
            "range_name": name,
            "start_segment": int(rec["start_segment"]),
            "end_segment": int(rec["end_segment"]),
            "n_primes": int(rec["n_primes"]),
            "n_gaps": int(rec["n_gaps"]),
            "mean_log_gap": float(rec["mean_log_gap"]),
            "gap_mean": float(rec["gap_mean"]),
            "gap_std": float(rec["gap_std"]),
        })
        status_rows.append({
            "range_index": int(i),
            "range_name": name,
            "source": source,
            "operator_dim": int(rec["operator_dim"]),
            "n_primes": int(rec["n_primes"]),
            "n_gaps": int(rec["n_gaps"]),
            "elapsed_seconds": float(time.time() - t0),
            "cache_path": str(cp),
        })

        print(
            f"[{i+1:03d}/{len(plan):03d}] {name} {source} "
            f"time={status_rows[-1]['elapsed_seconds']:.1f}s",
            flush=True,
        )

        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            atomic_csv(out / "b50_7_extraction_status.csv", pd.DataFrame(status_rows))
            atomic_json(out / "b50_7_extraction_checkpoint.json", {
                "version": VERSION,
                "updated": now_s(),
                "completed": int(i + 1),
                "total": int(len(plan)),
                "last_range": name,
                "elapsed_seconds": float(time.time() - t_all),
            })

    shutil.rmtree(staging_root, ignore_errors=True)

    P_ext = np.vstack(vectors)
    labels = np.asarray([r["range_name"] for r in meta_rows])

    if P_ext.shape != (len(plan), expected_operator_dim):
        raise RuntimeError(
            f"Unexpected external stack shape {P_ext.shape}, expected {(len(plan), expected_operator_dim)}"
        )
    if labels.tolist() != plan["range_name"].astype(str).tolist():
        raise RuntimeError("External stack label/order mismatch against frozen plan")

    stack_path = out / "b50_7_offset_operator_stack.npz"
    atomic_npz(
        stack_path,
        P_stack=P_ext,
        labels=labels,
        gap_bins=np.asarray([args.gap_bins], dtype=np.int64),
    )
    atomic_csv(out / "b50_7_operator_metadata.csv", pd.DataFrame(meta_rows))
    atomic_csv(out / "b50_7_extraction_status.csv", pd.DataFrame(status_rows))

    manifest = {
        "version": VERSION,
        "created": now_s(),
        "prime_root": str(prime_root),
        "b50_script": str(Path(args.b50_script)),
        "b50_script_sha256": sha256_file(Path(args.b50_script)),
        "training_operator_stack": str(Path(args.training_operator_stack)),
        "training_operator_stack_sha256": sha256_file(Path(args.training_operator_stack)),
        "plan_csv": str(plan_path),
        "plan_csv_sha256": sha256_file(plan_path),
        "range_count": int(P_ext.shape[0]),
        "operator_dim": int(P_ext.shape[1]),
        "stack_shape": list(P_ext.shape),
        "gap_bins": int(args.gap_bins),
        "labels_exact_against_plan": True,
        "elapsed_seconds": float(time.time() - t_all),
        "hardlink_only_policy": True,
    }
    atomic_json(out / "b50_7_extraction_manifest.json", manifest)

    print(f"[B50.7] stack saved: {stack_path} shape={P_ext.shape}")
    return manifest


# ---------------------------------------------------------------------------
# Frozen training PCA projection
# ---------------------------------------------------------------------------


def frozen_pca_basis(P_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P_train = np.asarray(P_train, dtype=np.float64)
    mean = P_train.mean(axis=0, keepdims=True)
    Xc = P_train - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S[None, :]
    return mean.reshape(-1), Vt, scores


def projection_parity(P_train, mean, Vt, scores, dims) -> Dict[str, Any]:
    max_d = max(dims)
    direct = (P_train - mean[None, :]) @ Vt[:max_d].T
    ref = scores[:, :max_d]
    err = np.abs(direct - ref)
    per_dim_corr = []
    for j in range(max_d):
        per_dim_corr.append(safe_spearman(direct[:, j], ref[:, j]))
    return {
        "max_abs_error": float(np.max(err)),
        "mean_abs_error": float(np.mean(err)),
        "min_component_spearman": float(np.nanmin(per_dim_corr)),
        "pass": bool(
            np.max(err) <= 1e-8
            and np.nanmin(per_dim_corr) >= 0.999999
        ),
    }


# ---------------------------------------------------------------------------
# Frozen front geometry on external windows
# ---------------------------------------------------------------------------


def build_external_configs(
    b504,
    P_ext: np.ndarray,
    train_mean: np.ndarray,
    train_Vt: np.ndarray,
    dims: List[int],
    graph_k_values: List[int],
    diffusion_modes_n: int,
) -> Dict[str, Dict[str, Any]]:
    configs = {}
    Xfull = (P_ext - train_mean[None, :]) @ train_Vt.T

    for d in dims:
        X = np.asarray(Xfull[:, :d], dtype=float)
        D = squareform(pdist(X, metric="euclidean"))
        for gk in graph_k_values:
            cid = f"d{d}_knn{gk}"
            W, radius = unpack_graph_result(b504.build_knn_graph(D, gk))
            eigvals, modes = b504.diffusion_modes(W, diffusion_modes_n)
            eigvals = np.asarray(eigvals, dtype=float)
            modes = np.asarray(modes, dtype=float)
            diff = modes * eigvals[None, :]
            configs[cid] = {
                "dimension": int(d),
                "graph_k": int(gk),
                "X": X,
                "radius": radius,
                "diff": diff,
            }
    return configs


def frozen_channel_scale(training_fronts: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    scale = {}
    for c in CHANNELS:
        if c not in training_fronts.columns:
            raise ValueError(f"Training B50.4 front table missing channel {c}")
        x = training_fronts[c].to_numpy(float)
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))
        s = 1.4826 * mad
        if not np.isfinite(s) or s < 1e-12:
            s = float(np.nanstd(x))
        if not np.isfinite(s) or s < 1e-12:
            s = 0.0
        scale[c] = (med, s)
    return scale


def candidate_raw(X, radius, diff, source_i: int, dest_idx: np.ndarray) -> Dict[str, np.ndarray]:
    if source_i < 1:
        raise ValueError("source_i must be >=1")
    dest_idx = np.asarray(dest_idx, dtype=int)

    prev = X[source_i] - X[source_i - 1]
    v = X[dest_idx] - X[source_i]

    speed = np.linalg.norm(v, axis=1)
    acceleration = np.linalg.norm(v - prev[None, :], axis=1)

    den = np.linalg.norm(prev) * speed
    cos = np.ones(len(dest_idx), dtype=float)
    good = den > 1e-15
    if np.any(good):
        cos[good] = (v[good] @ prev) / den[good]
    turn = 1.0 - np.clip(cos, -1.0, 1.0)

    density_change = np.abs(
        np.log(np.maximum(radius[dest_idx], 1e-15))
        - math.log(max(float(radius[source_i]), 1e-15))
    )

    if diff.shape[1] > 0:
        diffusion_step = np.linalg.norm(
            diff[dest_idx] - diff[source_i][None, :], axis=1
        )
    else:
        diffusion_step = np.zeros(len(dest_idx), dtype=float)

    return {
        "speed": speed,
        "acceleration": acceleration,
        "turn_change": turn,
        "density_scale_change": density_change,
        "diffusion_step": diffusion_step,
    }


def score_raw(raw: Dict[str, np.ndarray], scale: Dict[str, Tuple[float, float]]) -> np.ndarray:
    cols = []
    for c in CHANNELS:
        x = np.asarray(raw[c], dtype=float)
        med, s = scale[c]
        if s <= 1e-15:
            z = np.zeros_like(x)
        else:
            z = (x - med) / s
            z[~np.isfinite(z)] = 0.0
            z = np.maximum(z, 0.0)
        cols.append(z)
    Z = np.column_stack(cols)
    return np.sqrt(np.mean(Z * Z, axis=1))


def physical_raw_sequence(X, radius, diff) -> pd.DataFrame:
    rows = []
    for i in range(1, len(X) - 1):
        raw = candidate_raw(X, radius, diff, i, np.asarray([i + 1], dtype=int))
        rows.append({
            "transition_index": i,
            **{c: float(raw[c][0]) for c in CHANNELS},
        })
    return pd.DataFrame(rows)


def external_raw_parity_audit(front_fn, configs, sample_sources: List[int]) -> Tuple[bool, pd.DataFrame]:
    rows = []
    for cid, c in sorted(configs.items()):
        for i in sample_sources:
            exact = call_front_fn(
                front_fn,
                c["X"][[i - 1, i, i + 1]],
                c["radius"][[i - 1, i, i + 1]],
                c["diff"][[i - 1, i, i + 1]],
            ).sort_values("transition_index").iloc[-1]
            raw = candidate_raw(c["X"], c["radius"], c["diff"], i, np.asarray([i + 1]))
            errs = {ch: abs(float(raw[ch][0]) - float(exact[ch])) for ch in CHANNELS}
            mx = max(errs.values())
            rows.append({
                "config_id": cid,
                "source_i": i,
                "max_abs_error": mx,
                **{f"{k}_abs_error": v for k, v in errs.items()},
                "pass": bool(mx <= 1e-10),
            })
    df = pd.DataFrame(rows)
    return bool(df["pass"].all()), df


# ---------------------------------------------------------------------------
# Matched controls and percentile null
# ---------------------------------------------------------------------------


def build_matched_controls(P_stack, controls_per_transition, min_index_separation):
    P = np.asarray(P_stack, dtype=float)
    n = len(P)
    D = squareform(pdist(P, metric="euclidean"))
    cmap = {}
    rows = []

    for i in range(1, n - 1):
        phys = i + 1
        d0 = float(D[i, phys])
        cand = np.arange(n, dtype=int)
        allowed = (np.abs(cand - i) >= min_index_separation) & (cand != phys)
        cand = cand[allowed]
        if len(cand) < controls_per_transition:
            raise RuntimeError(f"Transition {i}: only {len(cand)} eligible controls")

        dc = D[i, cand]
        err = np.abs(np.log(np.maximum(dc, 1e-15)) - math.log(max(d0, 1e-15)))
        order = np.lexsort((cand, err))[:controls_per_transition]
        chosen = cand[order]
        cmap[i] = chosen.astype(int)

        for rank, p in enumerate(order, start=1):
            rows.append({
                "transition_index": i,
                "source_index": i,
                "physical_dest_index": phys,
                "control_rank": rank,
                "control_dest_index": int(cand[p]),
                "source_control_index_separation": int(abs(int(cand[p]) - i)),
                "physical_distance_full_operator": d0,
                "control_distance_full_operator": float(dc[p]),
                "abs_log_distance_error": float(err[p]),
            })
    return cmap, pd.DataFrame(rows)


def rank_percentiles(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.empty(n, dtype=float)
    for i in range(n):
        less = np.sum(x < x[i])
        equal_other = np.sum(x == x[i]) - 1
        out[i] = (less + 0.5 * equal_other) / max(n - 1, 1)
    return out


def build_replication_cube(
    configs,
    training_fronts,
    cmap,
):
    transition_indices = sorted(cmap)
    config_ids = sorted(configs)
    m = len(next(iter(cmap.values())))

    pct = np.empty((len(transition_indices), len(config_ids), m + 1), dtype=float)
    score = np.empty_like(pct)
    cfg_rows = []

    for ci, cid in enumerate(config_ids):
        c = configs[cid]
        train = training_fronts[training_fronts["config_id"].astype(str) == cid].sort_values("transition_index")
        if train.empty:
            raise RuntimeError(f"Missing frozen training front data for {cid}")
        scale = frozen_channel_scale(train)

        pp = []
        for ti, i in enumerate(transition_indices):
            dest = np.concatenate([np.asarray([i + 1], dtype=int), cmap[i]])
            raw = candidate_raw(c["X"], c["radius"], c["diff"], i, dest)
            sc = score_raw(raw, scale)
            rp = rank_percentiles(sc)
            pct[ti, ci, :] = rp
            score[ti, ci, :] = sc
            pp.append(float(rp[0]))

        cfg_rows.append({
            "config_id": cid,
            "dimension": c["dimension"],
            "graph_k": c["graph_k"],
            "mean_physical_percentile": float(np.mean(pp)),
            "median_physical_percentile": float(np.median(pp)),
            "mean_percentile_effect": float(np.mean(pp) - 0.5),
        })

    return transition_indices, config_ids, pct, score, pd.DataFrame(cfg_rows)


def matched_identity_null_lower(
    consensus_pct: np.ndarray,
    reps: int,
    seed: int,
    mask: Optional[np.ndarray] = None,
):
    X = np.asarray(consensus_pct, dtype=float)
    if mask is not None:
        X = X[np.asarray(mask, dtype=bool)]

    obs = X[:, 0]
    observed_mean = float(np.mean(obs))
    observed_effect = observed_mean - 0.5

    rng = np.random.default_rng(seed)
    rows = np.arange(len(X))
    null = np.empty(reps, dtype=float)
    for r in range(reps):
        choice = rng.integers(0, X.shape[1], size=len(X))
        null[r] = float(np.mean(X[rows, choice]) - 0.5)

    p_lower = float((1 + np.sum(null <= observed_effect)) / (reps + 1))
    p_two = float((1 + np.sum(np.abs(null) >= abs(observed_effect))) / (reps + 1))
    return {
        "n_transitions": int(len(X)),
        "observed_mean_percentile": observed_mean,
        "observed_median_percentile": float(np.median(obs)),
        "observed_mean_effect": observed_effect,
        "null_mean_effect": float(np.mean(null)),
        "null_sd_effect": float(np.std(null)),
        "p_lower": p_lower,
        "p_two_sided_diagnostic": p_two,
    }, null


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    required = {
        "plan_csv": out / "b50_7_offset_window_plan.csv",
        "plan_json": out / "b50_7_offset_window_plan.json",
        "external_stack": out / "b50_7_offset_operator_stack.npz",
        "extraction_manifest": out / "b50_7_extraction_manifest.json",
        "training_stack": Path(args.training_operator_stack),
        "b50_4_protocol": Path(args.b50_4_protocol),
        "b50_4_front_features": Path(args.b50_4_front_features),
        "b50_6_summary": Path(args.b50_6_summary),
        "b50_4_script": Path(args.b50_4_script),
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    plan = pd.read_csv(required["plan_csv"])
    plan_summary = json.loads(required["plan_json"].read_text(encoding="utf-8"))
    extraction_manifest = json.loads(required["extraction_manifest"].read_text(encoding="utf-8"))
    p4 = json.loads(required["b50_4_protocol"].read_text(encoding="utf-8"))
    s6 = json.loads(required["b50_6_summary"].read_text(encoding="utf-8"))
    training_fronts = pd.read_csv(required["b50_4_front_features"])

    if s6.get("final_verdict") != "PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED":
        raise RuntimeError(
            "B50.7 replication target requires B50.6 final verdict "
            "PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED"
        )

    b506_effect = float(s6["primary_specificity"]["observed_mean_effect"])
    if not b506_effect < 0:
        raise RuntimeError("B50.6 replication direction is not negative")

    with np.load(required["training_stack"], allow_pickle=False) as z:
        P_train = np.asarray(z["P_stack"], dtype=np.float64)
        train_labels = [str(x) for x in z["labels"]]

    with np.load(required["external_stack"], allow_pickle=False) as z:
        P_ext = np.asarray(z["P_stack"], dtype=np.float64)
        ext_labels = [str(x) for x in z["labels"]]

    labels_exact = ext_labels == plan["range_name"].astype(str).tolist()
    operator_dim_exact = P_ext.shape[1] == P_train.shape[1]
    plan_novelty = bool(plan_summary.get("offset_novelty_pass", False))
    extraction_exact = bool(extraction_manifest.get("labels_exact_against_plan", False))

    if not (labels_exact and operator_dim_exact and plan_novelty and extraction_exact):
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_EXTRACTION",
            "integrity": {
                "labels_exact": labels_exact,
                "operator_dim_exact": operator_dim_exact,
                "plan_novelty_pass": plan_novelty,
                "extraction_labels_exact": extraction_exact,
            },
        }
        atomic_json(out / "b50_7_summary.json", summary)
        atomic_text(out / "b50_7_verdict.txt", "FINAL VERDICT: INVALID_EXTRACTION\n")
        return summary

    dims = [int(x) for x in p4["dimensions"]]
    graph_k_values = [int(x) for x in p4["graph_k_values"]]
    diffusion_modes_n = int(p4["diffusion_modes"])

    print("=== TDC TOR A / B50.7: EXTERNAL OFFSET-WINDOW REPLICATION ===")
    print(f"training stack      : {P_train.shape}")
    print(f"offset stack        : {P_ext.shape}")
    print(f"offset              : {plan_summary['applied_offset']}")
    print(f"novel windows       : {plan_summary['offset_window_count']}")
    print(f"B50.6 target effect : {b506_effect:.9f}")
    print(f"frozen dimensions   : {dims}")
    print(f"frozen graph k      : {graph_k_values}")
    print()

    protocol = {
        "version": VERSION,
        "created": now_s(),
        "outcome_blind_to_b59": True,
        "replication_target": {
            "source": str(required["b50_6_summary"]),
            "source_sha256": sha256_file(required["b50_6_summary"]),
            "b50_6_effect": b506_effect,
            "frozen_direction": "negative",
        },
        "window_plan": {
            "offset_segments": int(plan_summary["applied_offset"]),
            "window_length": int(plan_summary["original_nominal_length"]),
            "offset_window_count": int(plan_summary["offset_window_count"]),
            "same_underlying_prime_corpus": True,
            "identical_original_windows": int(plan_summary["exact_duplicate_with_original_count"]),
        },
        "representation": {
            "training_pca_basis_frozen_from_original_stack": True,
            "dimensions": dims,
            "graph_k_values": graph_k_values,
            "diffusion_modes": diffusion_modes_n,
            "front_channel_scaling_frozen_from_original_b50_4": True,
        },
        "matching": {
            "controls_per_transition": args.controls_per_transition,
            "min_index_separation": args.min_index_separation,
            "geometry": "external full 4096D operator Euclidean",
        },
        "randomization_replicates": args.randomization_replicates,
        "thresholds": THRESHOLDS,
        "seed": args.seed,
        "scientific_boundary": (
            "Out-of-sample offset-window replication on the same underlying prime corpus; "
            "not independent-corpus replication."
        ),
    }
    atomic_json(out / "b50_7_protocol_manifest.json", protocol)

    # 0. Frozen PCA basis.
    print("[0/6] Frozen training PCA basis", flush=True)
    train_mean, train_Vt, train_scores = frozen_pca_basis(P_train)
    pca_parity = projection_parity(P_train, train_mean, train_Vt, train_scores, dims)
    atomic_json(out / "b50_7_frozen_pca_parity.json", pca_parity)
    print(f"      PCA parity PASS: {pca_parity['pass']}")
    if not pca_parity["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_REPRODUCTION",
            "reason": "Frozen PCA projection parity failed on training stack",
            "pca_parity": pca_parity,
        }
        atomic_json(out / "b50_7_summary.json", summary)
        atomic_text(out / "b50_7_verdict.txt", "FINAL VERDICT: INVALID_REPRODUCTION\n")
        return summary

    # 1. External configs.
    print("[1/6] Projecting offset windows into frozen PCA basis", flush=True)
    b504, front_fn = import_b504(required["b50_4_script"])
    configs = build_external_configs(
        b504,
        P_ext,
        train_mean,
        train_Vt,
        dims,
        graph_k_values,
        diffusion_modes_n,
    )

    sample_sources = np.linspace(1, len(P_ext) - 2, min(5, len(P_ext) - 2), dtype=int).tolist()
    parity_ok, parity_df = external_raw_parity_audit(front_fn, configs, sample_sources)
    atomic_csv(out / "b50_7_external_raw_feature_parity.csv", parity_df)
    print(f"      raw-feature parity PASS: {parity_ok}")
    if not parity_ok:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_REPRODUCTION",
            "reason": "External raw front-feature parity against frozen B50.4 failed",
        }
        atomic_json(out / "b50_7_summary.json", summary)
        atomic_text(out / "b50_7_verdict.txt", "FINAL VERDICT: INVALID_REPRODUCTION\n")
        return summary

    # 2. Matching.
    print("[2/6] External full-operator matched controls", flush=True)
    cmap, matching_df = build_matched_controls(
        P_ext,
        args.controls_per_transition,
        args.min_index_separation,
    )
    atomic_csv(out / "b50_7_matching_audit.csv", matching_df)

    matching_median = float(matching_df["abs_log_distance_error"].median())
    matching_q90 = float(matching_df["abs_log_distance_error"].quantile(0.90))
    matching_pass = bool(
        matching_median <= THRESHOLDS["matching_median_abs_log_distance_error_max"]
        and matching_q90 <= THRESHOLDS["matching_q90_abs_log_distance_error_max"]
    )
    print(
        f"      match median={matching_median:.6f} q90={matching_q90:.6f} PASS={matching_pass}"
    )
    if not matching_pass:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_MATCHING_GEOMETRY",
            "matching": {
                "median_abs_log_distance_error": matching_median,
                "q90_abs_log_distance_error": matching_q90,
                "pass": False,
            },
        }
        atomic_json(out / "b50_7_summary.json", summary)
        atomic_text(out / "b50_7_verdict.txt", "FINAL VERDICT: INVALID_MATCHING_GEOMETRY\n")
        return summary

    # 3. Frozen front score on external matched sets.
    print("[3/6] Frozen front scoring on offset sequence", flush=True)
    tis, cids, pct, score, cfg_df = build_replication_cube(
        configs,
        training_fronts,
        cmap,
    )
    consensus = np.median(pct, axis=1)
    physical_pct = consensus[:, 0]

    transition_rows = []
    for ti, i in enumerate(tis):
        transition_rows.append({
            "transition_index": i,
            "from_range": ext_labels[i],
            "to_range": ext_labels[i + 1],
            "consensus_physical_percentile": float(physical_pct[ti]),
            "consensus_percentile_effect": float(physical_pct[ti] - 0.5),
            "median_physical_front_score": float(np.median(score[ti, :, 0])),
            "median_control_front_score": float(np.median(score[ti, :, 1:])),
            "control_dest_indices": ",".join(str(int(x)) for x in cmap[i]),
        })
    transition_df = pd.DataFrame(transition_rows)
    atomic_csv(out / "b50_7_transition_specificity.csv", transition_df)

    negative_fraction = float(np.mean(cfg_df["mean_percentile_effect"].to_numpy(float) < 0.0))
    config_pass = bool(negative_fraction >= THRESHOLDS["config_negative_fraction_min"])
    cfg_df["negative_effect"] = cfg_df["mean_percentile_effect"] < 0.0
    atomic_csv(out / "b50_7_config_specificity.csv", cfg_df)

    # 4. Primary directional replication null.
    print("[4/6] Directional matched-set identity randomization", flush=True)
    primary, null = matched_identity_null_lower(
        consensus,
        args.randomization_replicates,
        args.seed + 1000,
    )
    primary_pass = bool(
        primary["observed_mean_effect"] <= THRESHOLDS["primary_effect_max"]
        and primary["p_lower"] <= THRESHOLDS["primary_p_lower_max"]
    )
    primary["pass"] = primary_pass
    primary["b50_6_reference_effect"] = b506_effect
    primary["effect_difference_vs_b50_6"] = float(primary["observed_mean_effect"] - b506_effect)
    primary["effect_ratio_vs_b50_6"] = (
        float(primary["observed_mean_effect"] / b506_effect)
        if abs(b506_effect) > 1e-15 else float("nan")
    )
    atomic_csv(out / "b50_7_primary_null.csv", pd.DataFrame({
        "replicate": np.arange(len(null)),
        "mean_percentile_effect": null,
    }))

    # 5. Thirds robustness.
    print("[5/6] Offset-sequence thirds robustness", flush=True)
    n_t = len(tis)
    edges = np.linspace(0, n_t, 4, dtype=int)
    third_rows = []
    negative_count = 0
    nominal_count = 0

    for third in range(3):
        mask = np.zeros(n_t, dtype=bool)
        mask[edges[third]:edges[third + 1]] = True
        res, _ = matched_identity_null_lower(
            consensus,
            args.randomization_replicates,
            args.seed + 2000 + third,
            mask,
        )
        negative = bool(res["observed_mean_effect"] < 0.0)
        nominal = bool(
            res["observed_mean_effect"] <= THRESHOLDS["third_effect_max"]
            and res["p_lower"] <= THRESHOLDS["third_p_lower_max"]
        )
        negative_count += int(negative)
        nominal_count += int(nominal)
        idxs = np.asarray(tis)[mask]
        third_rows.append({
            "third": third + 1,
            "transition_count": int(mask.sum()),
            "first_transition_index": int(idxs[0]),
            "last_transition_index": int(idxs[-1]),
            **res,
            "negative_direction": negative,
            "nominal_replication_pass": nominal,
        })

    thirds_df = pd.DataFrame(third_rows)
    atomic_csv(out / "b50_7_replication_thirds.csv", thirds_df)
    thirds_pass = bool(
        negative_count >= THRESHOLDS["third_negative_required"]
        and nominal_count >= THRESHOLDS["third_nominal_required"]
    )

    # 6. Verdict.
    print("[6/6] Frozen replication verdict", flush=True)
    effect = float(primary["observed_mean_effect"])

    if primary_pass and config_pass and thirds_pass:
        verdict = "EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED"
        reason = (
            "The frozen negative physical-adjacency effect replicates on new offset-window "
            "states, passes matched identity randomization, converges across frozen "
            "representations, and is robust across all three offset-sequence thirds."
        )
    elif primary_pass and config_pass and not thirds_pass:
        verdict = "OFFSET_ADJACENCY_SPECIFICITY_REPLICATED_WITH_WEAK_THIRDS"
        reason = (
            "The full external offset-window sample and representation convergence pass, "
            "but the pre-declared thirds robustness gate does not fully pass."
        )
    elif effect >= 0.05:
        verdict = "OFFSET_REPLICATION_DIRECTION_REVERSED"
        reason = (
            "The offset-window effect is materially positive, opposite to the frozen "
            "negative B50.6 replication direction."
        )
    elif (not primary_pass) and abs(effect) < 0.05:
        verdict = "OFFSET_REPLICATION_NULL_COMPATIBLE"
        reason = (
            "The offset-window physical percentile is not sufficiently displaced from "
            "matched non-adjacent controls under the frozen directional replication test."
        )
    else:
        verdict = "WEAK_OR_INCONSISTENT_OFFSET_REPLICATION"
        reason = (
            "The offset-window replication is mixed under the frozen effect-size, "
            "directional randomization, configuration-convergence, and thirds gates."
        )

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "outcome_blind_to_b59": True,
        "integrity": {
            "offset_novelty_pass": plan_novelty,
            "external_labels_exact": labels_exact,
            "operator_dim_exact": operator_dim_exact,
            "frozen_pca_projection_parity": pca_parity,
            "external_raw_feature_parity_ok": parity_ok,
        },
        "replication_target": {
            "b50_6_final_verdict": s6.get("final_verdict"),
            "b50_6_effect": b506_effect,
            "frozen_direction": "negative",
        },
        "offset_sequence": {
            "window_count": int(len(P_ext)),
            "transition_sources_used": int(len(tis)),
            "offset_segments": int(plan_summary["applied_offset"]),
            "window_length": int(plan_summary["original_nominal_length"]),
            "identical_original_window_count": int(plan_summary["exact_duplicate_with_original_count"]),
            "same_underlying_prime_corpus": True,
        },
        "matching": {
            "controls_per_transition": args.controls_per_transition,
            "min_index_separation": args.min_index_separation,
            "median_abs_log_distance_error": matching_median,
            "q90_abs_log_distance_error": matching_q90,
            "pass": matching_pass,
        },
        "primary_replication": primary,
        "config_convergence": {
            "negative_effect_fraction": negative_fraction,
            "pass": config_pass,
        },
        "thirds_robustness": {
            "negative_thirds": negative_count,
            "nominally_replicated_thirds": nominal_count,
            "pass": thirds_pass,
            "rows": thirds_df.to_dict(orient="records"),
        },
        "thresholds": THRESHOLDS,
        "final_verdict": verdict,
        "final_reason": reason,
        "scientific_boundary": (
            "Out-of-sample offset-window replication on new states from the same underlying "
            "prime corpus. Not independent-corpus replication, not B59 validation, not "
            "individual-prime prediction, and not a theorem."
        ),
    }
    atomic_json(out / "b50_7_summary.json", summary)

    # Plots
    plt.figure(figsize=(8, 5))
    plt.hist(null, bins=45)
    plt.axvline(effect, linewidth=2)
    plt.xlabel("mean percentile effect under matched-set identity null")
    plt.ylabel("count")
    plt.title("B50.7 external offset-window matched identity null")
    plt.tight_layout()
    plt.savefig(out / "b50_7_primary_null.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(
        transition_df["transition_index"],
        transition_df["consensus_physical_percentile"],
        linewidth=1,
    )
    plt.axhline(0.5, linestyle="--", linewidth=0.8)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("offset physical transition index")
    plt.ylabel("physical destination percentile")
    plt.title("B50.7 offset-window adjacency specificity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "b50_7_specificity_by_transition.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    x = np.arange(len(cfg_df))
    plt.plot(x, cfg_df["mean_physical_percentile"], marker="o")
    plt.axhline(0.5, linestyle="--", linewidth=0.8)
    plt.xticks(x, cfg_df["config_id"], rotation=60, ha="right")
    plt.ylabel("mean physical percentile")
    plt.title("B50.7 frozen representation convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "b50_7_config_convergence.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(thirds_df["third"], thirds_df["observed_mean_percentile"], marker="o")
    plt.axhline(0.5, linestyle="--", linewidth=0.8)
    plt.xticks([1, 2, 3])
    plt.xlabel("offset-sequence contiguous third")
    plt.ylabel("mean physical percentile")
    plt.title("B50.7 thirds robustness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "b50_7_thirds_robustness.png", dpi=160)
    plt.close()

    report = f"""B50.7 — External Offset-Window Adjacency Replication
=========================================================

Interpretation scope
--------------------
New offset WINDOW STATES: YES
Identical to original windows: NO
Same underlying prime corpus: YES
Independent-corpus replication: NO

Integrity / novelty
-------------------
training operator stack              = {P_train.shape}
offset operator stack                = {P_ext.shape}
offset segments                      = {plan_summary['applied_offset']}
window length                        = {plan_summary['original_nominal_length']}
identical original windows           = {plan_summary['exact_duplicate_with_original_count']}
offset novelty PASS                  = {plan_novelty}
external labels exact                = {labels_exact}
operator dimension exact             = {operator_dim_exact}
frozen PCA parity PASS               = {pca_parity['pass']}
external raw-feature parity PASS     = {parity_ok}

Frozen replication target
-------------------------
B50.6 verdict                        = {s6.get('final_verdict')}
B50.6 mean percentile effect         = {b506_effect}
frozen replication direction         = negative

Matched-control protocol
------------------------
physical source transitions used     = {len(tis)}
controls per transition              = {args.controls_per_transition}
minimum index separation             = {args.min_index_separation}
matching geometry                    = external full 4096D operator Euclidean
median |log-distance match error|    = {matching_median}
q90 |log-distance match error|       = {matching_q90}
MATCHING PASS                        = {matching_pass}

Primary external replication
----------------------------
mean physical percentile             = {primary['observed_mean_percentile']}
median physical percentile           = {primary['observed_median_percentile']}
mean percentile effect vs 0.5        = {primary['observed_mean_effect']}
B50.6 reference effect               = {b506_effect}
effect difference vs B50.6           = {primary['effect_difference_vs_b50_6']}
effect ratio vs B50.6                = {primary['effect_ratio_vs_b50_6']}
matched identity p_lower             = {primary['p_lower']}
two-sided p diagnostic               = {primary['p_two_sided_diagnostic']}
PRIMARY REPLICATION PASS             = {primary_pass}

Frozen representation convergence
---------------------------------
negative-effect config fraction      = {negative_fraction}
CONFIG CONVERGENCE PASS              = {config_pass}

Offset-sequence thirds robustness
--------------------------------
negative thirds                      = {negative_count}/3
nominally replicated thirds          = {nominal_count}/3
THIRDS ROBUSTNESS PASS               = {thirds_pass}

{thirds_df.to_string(index=False)}

FINAL VERDICT: {verdict}

REASON:
{reason}

Interpretation boundary
-----------------------
This is an out-of-sample OFFSET-WINDOW replication on new window states from
the same underlying prime corpus. It is not an independent-corpus replication.
No B59/B59.1 selector or predictive outcome is used.
No individual-prime prediction, RH, or theorem claim is made.

Key outputs
-----------
b50_7_offset_window_plan.csv
b50_7_offset_window_plan.json
b50_7_offset_operator_stack.npz
b50_7_operator_metadata.csv
b50_7_extraction_status.csv
b50_7_extraction_manifest.json
b50_7_protocol_manifest.json
b50_7_frozen_pca_parity.json
b50_7_external_raw_feature_parity.csv
b50_7_matching_audit.csv
b50_7_transition_specificity.csv
b50_7_config_specificity.csv
b50_7_primary_null.csv
b50_7_replication_thirds.csv
b50_7_summary.json
b50_7_verdict.txt
b50_7_primary_null.png
b50_7_specificity_by_transition.png
b50_7_config_convergence.png
b50_7_thirds_robustness.png
"""
    atomic_text(out / "b50_7_verdict.txt", report)

    print()
    print("=== B50.7 RESULT ===")
    print(f"matching PASS       : {matching_pass}")
    print(f"primary PASS        : {primary_pass}")
    print(f"config PASS         : {config_pass}")
    print(f"thirds PASS         : {thirds_pass}")
    print(f"mean percentile     : {primary['observed_mean_percentile']:.6f}")
    print(f"effect vs 0.5       : {effect:.6f}")
    print(f"B50.6 effect        : {b506_effect:.6f}")
    print(f"p_lower             : {primary['p_lower']:.6g}")
    print(f"FINAL VERDICT       : {verdict}")
    print(f"verdict             : {out / 'b50_7_verdict.txt'}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="B50.7 External Offset-Window Adjacency Replication"
    )
    p.add_argument("--mode", choices=["plan", "extract", "audit", "all"], default="plan")

    p.add_argument("--prime-root", default=r"d:\primes_output")
    p.add_argument("--total-segments", type=int, default=100000)
    p.add_argument(
        "--offset-segments",
        type=int,
        default=None,
        help="Default: half the inferred original nominal start step (expected +75).",
    )

    p.add_argument("--state-dataset", required=True)
    p.add_argument("--training-operator-stack", required=True)
    p.add_argument("--b50-script", required=True)
    p.add_argument("--b50-4-script", required=True)
    p.add_argument("--b50-4-protocol", required=True)
    p.add_argument("--b50-4-front-features", required=True)
    p.add_argument("--b50-6-summary", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--gap-bins", type=int, default=64)
    p.add_argument("--staging-dir", default=None)
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=10)

    p.add_argument("--controls-per-transition", type=int, default=20)
    p.add_argument("--min-index-separation", type=int, default=10)
    p.add_argument("--randomization-replicates", type=int, default=5000)
    p.add_argument("--seed", type=int, default=50707)

    args = p.parse_args()

    if args.total_segments < 100:
        p.error("--total-segments must be >=100")
    if args.gap_bins < 4:
        p.error("--gap-bins must be >=4")
    if args.controls_per_transition < 5:
        p.error("--controls-per-transition must be >=5")
    if args.min_index_separation < 3:
        p.error("--min-index-separation must be >=3")
    if args.randomization_replicates < 999:
        p.error("--randomization-replicates must be >=999")

    return args


def main():
    args = parse_args()
    if args.mode == "plan":
        plan_mode(args)
    elif args.mode == "extract":
        extract_mode(args)
    elif args.mode == "audit":
        audit_mode(args)
    elif args.mode == "all":
        plan_mode(args)
        extract_mode(args)
        audit_mode(args)
    else:
        raise ValueError(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
