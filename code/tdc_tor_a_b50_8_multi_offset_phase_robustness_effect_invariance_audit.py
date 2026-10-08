#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TDC TOR A / B50.8 — Multi-Offset Phase Robustness & Effect Invariance Audit
============================================================================

Purpose
-------
B50.7 replicated the frozen negative physical-adjacency specificity effect on
new +75-segment offset-window states from the same underlying prime corpus.

B50.8 tests whether that result is robust to WINDOW PHASE rather than specific
to the half-step (+75) offset.

The frozen new offsets are:
    +25, +50, +100, +125 segments

The original phase (0) and the already-observed B50.7 +75 offset are NOT used
for selecting the new offsets or thresholds. B50.8 imports the exact B50.7
implementation and reruns the same plan -> extract -> audit contract unchanged
for every frozen new offset.

Primary phase-robustness hypothesis
-----------------------------------
For offset o, define the frozen B50.7 matched-adjacency effect

    Delta_o = E[physical matched-set percentile] - 0.5.

B50.6 and B50.7 established a negative direction. B50.8 therefore freezes:

    1. 4/4 new offsets must have Delta_o < 0.
    2. At least 3/4 new offsets must satisfy BOTH:
           Delta_o <= -0.04
           matched identity p_lower <= 0.01
    3. Median new-offset effect must satisfy:
           median(Delta_o) <= -0.05
    4. At least 3/4 offsets must pass B50.7 representation convergence.
    5. At least 3/4 offsets must pass B50.7 thirds robustness.
    6. Matching and integrity must pass for ALL offsets.

Effect heterogeneity is reported descriptively and is never tuned away.
Because all offsets come from the same underlying prime corpus and overlap in
source segments, the offset effects are statistically dependent. B50.8 does
NOT pretend they are four independent corpora.

Secondary pooled-null diagnostic
--------------------------------
Each B50.7 subrun saves its matched-identity null. B50.8 combines replicate
indices across offsets and reports the mean-offset effect against the mean-null
distribution. This is a diagnostic only; it is not a mandatory verdict gate.

Modes
-----
plan
    Freeze/check all four offset plans. No prime-data read.

extract
    Run the exact B50.7 resumable extraction for all frozen offsets, or one
    execution shard with --only-offset. Each offset has an independent cache.

audit
    Run the exact frozen B50.7 audit on each offset, then aggregate B50.8.

all
    plan -> extract -> audit.

Execution sharding
------------------
--only-offset changes only execution scheduling, not the frozen scientific
protocol. Example: run four extraction jobs sequentially with offsets
25, 50, 100, 125, then run audit without --only-offset.

Final verdicts
--------------
MULTI_OFFSET_PHASE_ROBUST_EFFECT_INVARIANCE
    all mandatory phase-robustness gates pass.

MULTI_OFFSET_DIRECTION_STABLE_WEAK_AMPLITUDE
    all four offsets remain negative but amplitude / nominal replication gate
    is too weak for the full invariance verdict.

MULTI_OFFSET_PARTIAL_REPLICATION
    at least 3/4 offsets are negative, but not all mandatory gates pass.

MULTI_OFFSET_PHASE_SENSITIVE
    at least one materially positive sign reversal occurs (effect >= +0.04).

MULTI_OFFSET_NULL_COMPATIBLE
    median effect is close to null and fewer than 3 offsets meet nominal
    replication criteria.

INVALID_MULTI_OFFSET_PLAN
INVALID_OFFSET_SUBRUN
INVALID_REPRODUCTION
    technical / lineage failures; no scientific conclusion.

Scientific boundary
-------------------
B50.8 is a same-corpus multi-offset window-phase robustness audit. A positive
result is NOT an independent-corpus replication, does NOT validate B59, does
NOT predict individual primes, and does NOT imply RH or any theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "B50.8_multi_offset_phase_robustness_effect_invariance_v1"
FROZEN_OFFSETS = [25, 50, 100, 125]
REFERENCE_OFFSET = 75

THRESHOLDS = {
    "all_negative_required": 4,
    "nominal_offset_required": 3,
    "nominal_effect_max": -0.04,
    "nominal_p_lower_max": 0.01,
    "median_effect_max": -0.05,
    "config_pass_required": 3,
    "thirds_pass_required": 3,
    "material_positive_reversal_min": 0.04,
    "null_compatible_abs_median_effect_max": 0.03,
}


# ---------------------------------------------------------------------------
# Utilities
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


def parse_offsets(s: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
    if not vals:
        raise ValueError("No offsets supplied")
    return vals


def offset_dir(root: Path, offset: int) -> Path:
    sign = "p" if offset >= 0 else "m"
    return root / f"offset_{sign}{abs(offset):03d}"


def import_b507(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("tdc_b507_frozen_b508", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import B50.7 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    required = ["plan_mode", "extract_mode", "audit_mode", "build_offset_plan"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise AttributeError(f"B50.7 missing required API: {missing}")
    return mod


def validate_reference_lineage(args) -> Dict[str, Any]:
    s6 = json.loads(Path(args.b50_6_summary).read_text(encoding="utf-8"))
    s7 = json.loads(Path(args.b50_7_reference_summary).read_text(encoding="utf-8"))

    b506_ok = s6.get("final_verdict") == "PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED"
    b507_ok = s7.get("final_verdict") == "EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED"

    ref_offset = int(s7.get("offset_sequence", {}).get("offset_segments", -999999))
    ref_offset_ok = ref_offset == REFERENCE_OFFSET

    b506_effect = float(
        s6.get("primary_specificity", {}).get("observed_mean_effect", float("nan"))
    )
    b507_effect = float(
        s7.get("primary_replication", {}).get("observed_mean_effect", float("nan"))
    )

    direction_ok = bool(
        np.isfinite(b506_effect)
        and np.isfinite(b507_effect)
        and b506_effect < 0.0
        and b507_effect < 0.0
    )

    return {
        "b50_6_verdict_ok": b506_ok,
        "b50_7_reference_verdict_ok": b507_ok,
        "b50_7_reference_offset": ref_offset,
        "b50_7_reference_offset_ok": ref_offset_ok,
        "b50_6_effect": b506_effect,
        "b50_7_reference_effect": b507_effect,
        "negative_reference_direction_ok": direction_ok,
        "pass": bool(b506_ok and b507_ok and ref_offset_ok and direction_ok),
    }


def make_subargs(args, offset: int) -> SimpleNamespace:
    odir = offset_dir(Path(args.output_dir), offset)
    staging = None
    if args.staging_root:
        staging = str(Path(args.staging_root) / f"offset_{offset:+d}")

    # Deterministic execution seed set before outcomes; different by offset.
    subseed = int(args.seed + 1009 * (FROZEN_OFFSETS.index(offset) + 1))

    return SimpleNamespace(
        mode=None,
        prime_root=args.prime_root,
        total_segments=args.total_segments,
        offset_segments=int(offset),
        state_dataset=args.state_dataset,
        training_operator_stack=args.training_operator_stack,
        b50_script=args.b50_script,
        b50_4_script=args.b50_4_script,
        b50_4_protocol=args.b50_4_protocol,
        b50_4_front_features=args.b50_4_front_features,
        b50_6_summary=args.b50_6_summary,
        output_dir=str(odir),
        gap_bins=args.gap_bins,
        staging_dir=staging,
        reuse_existing=args.reuse_existing,
        checkpoint_every=args.checkpoint_every,
        controls_per_transition=args.controls_per_transition,
        min_index_separation=args.min_index_separation,
        randomization_replicates=args.randomization_replicates,
        seed=subseed,
    )


def selected_offsets(args) -> List[int]:
    if args.only_offset is None:
        return list(args.offsets)
    if int(args.only_offset) not in args.offsets:
        raise ValueError(
            f"--only-offset {args.only_offset} is not in frozen offsets {args.offsets}"
        )
    return [int(args.only_offset)]


def protocol_modified(args) -> bool:
    return list(args.offsets) != FROZEN_OFFSETS


# ---------------------------------------------------------------------------
# PLAN
# ---------------------------------------------------------------------------


def plan_mode(args, b507) -> Dict[str, Any]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    lineage = validate_reference_lineage(args)
    if not lineage["pass"]:
        summary = {
            "version": VERSION,
            "created": now_s(),
            "lineage": lineage,
            "final_verdict": "INVALID_REPRODUCTION",
            "reason": "B50.6/B50.7 reference lineage gate failed before multi-offset planning.",
        }
        atomic_json(root / "b50_8_plan_summary.json", summary)
        raise RuntimeError("INVALID_REPRODUCTION")

    rows = []
    plan_summaries = []
    all_windows: Dict[Tuple[int, int], List[int]] = {}

    print("=== B50.8 MULTI-OFFSET PLAN ===")
    print(f"frozen offsets     : {args.offsets}")
    print(f"reference +75      : PASS, effect={lineage['b50_7_reference_effect']:.6f}")
    print(f"protocol modified  : {protocol_modified(args)}")
    print()

    for offset in args.offsets:
        sub = make_subargs(args, offset)
        print(f"--- offset {offset:+d} ---", flush=True)
        s = b507.plan_mode(sub)
        plan_summaries.append(s)

        p = Path(sub.output_dir) / "b50_7_offset_window_plan.csv"
        pdf = pd.read_csv(p)
        for r in pdf.itertuples():
            key = (int(r.start_segment), int(r.end_segment))
            all_windows.setdefault(key, []).append(offset)

        rows.append({
            "offset": offset,
            "offset_fraction_of_step": float(offset / s["original_nominal_step"]),
            "window_count": int(s["offset_window_count"]),
            "dropped_window_count": int(s["dropped_window_count"]),
            "exact_duplicate_with_original_count": int(s["exact_duplicate_with_original_count"]),
            "novelty_pass": bool(s["offset_novelty_pass"]),
            "subrun_protocol_modified_from_b50_7_default": bool(
                s.get("protocol_modified_from_default", False)
            ),
            "plan_dir": str(Path(sub.output_dir)),
        })

        atomic_json(root / "b50_8_plan_progress.json", {
            "version": VERSION,
            "updated": now_s(),
            "completed_offsets": [int(x["offset"]) for x in rows],
            "requested_execution_offsets": list(args.offsets),
        })

    cross_dupes = []
    for key, offs in all_windows.items():
        if len(set(offs)) > 1:
            cross_dupes.append({
                "start_segment": key[0],
                "end_segment": key[1],
                "offsets": sorted(set(offs)),
            })

    df = pd.DataFrame(rows)
    atomic_csv(root / "b50_8_multi_offset_plan.csv", df)

    all_novel = bool(len(df) > 0 and df["novelty_pass"].all())
    offsets_exact = list(args.offsets) == FROZEN_OFFSETS
    full_plan = True
    cross_duplicate_count = len(cross_dupes)

    plan_pass = bool(
        all_novel
        and cross_duplicate_count == 0
        and offsets_exact
    )

    summary = {
        "version": VERSION,
        "created": now_s(),
        "lineage": lineage,
        "frozen_offsets": FROZEN_OFFSETS,
        "applied_offsets": list(args.offsets),
        "protocol_modified_from_default": protocol_modified(args),
        "execution_only_offset": args.only_offset,
        "full_plan": full_plan,
        "cross_offset_exact_duplicate_count": cross_duplicate_count,
        "cross_offset_duplicates": cross_dupes,
        "all_offset_novelty_pass": all_novel,
        "offsets_exactly_frozen_default": offsets_exact,
        "plan_pass": plan_pass,
        "rows": rows,
        "scientific_boundary": (
            "Four new offset phases from the same underlying prime corpus; "
            "not independent-corpus replication."
        ),
    }
    atomic_json(root / "b50_8_plan_summary.json", summary)

    print()
    print("=== B50.8 PLAN CHECKPOINT ===")
    print(f"planned offsets     : {[int(x) for x in df['offset'].tolist()]}")
    print(f"all novelty PASS    : {all_novel}")
    print(f"cross-offset dupes  : {cross_duplicate_count}")
    print(f"frozen offsets exact: {offsets_exact}")
    print(f"PLAN PASS           : {plan_pass}")

    if not plan_pass:
        raise RuntimeError("INVALID_MULTI_OFFSET_PLAN")
    return summary


# ---------------------------------------------------------------------------
# EXTRACT
# ---------------------------------------------------------------------------


def ensure_full_plan(root: Path) -> Dict[str, Any]:
    p = root / "b50_8_plan_summary.json"
    if not p.exists():
        raise FileNotFoundError("Run B50.8 --mode plan without --only-offset first")
    s = json.loads(p.read_text(encoding="utf-8"))
    if not bool(s.get("full_plan", False)) or not bool(s.get("plan_pass", False)):
        raise RuntimeError("Frozen full B50.8 plan is not valid")
    if list(s.get("applied_offsets", [])) != FROZEN_OFFSETS:
        raise RuntimeError("Frozen B50.8 plan offsets do not match default protocol")
    return s


def extract_mode(args, b507) -> Dict[str, Any]:
    root = Path(args.output_dir)
    ensure_full_plan(root)

    offsets = selected_offsets(args)
    progress_rows = []

    print("=== B50.8 MULTI-OFFSET EXTRACTION ===")
    print(f"execution offsets : {offsets}")
    print("Each offset uses the exact B50.7 resumable extractor.")
    print()

    for offset in offsets:
        sub = make_subargs(args, offset)
        odir = Path(sub.output_dir)
        if not (odir / "b50_7_offset_window_plan.csv").exists():
            raise FileNotFoundError(f"Missing frozen plan for offset {offset:+d}")

        print(f"=== EXTRACT offset {offset:+d} ===", flush=True)
        b507.extract_mode(sub)

        stack_path = odir / "b50_7_offset_operator_stack.npz"
        manifest_path = odir / "b50_7_extraction_manifest.json"
        complete = stack_path.exists() and manifest_path.exists()
        shape = None
        if stack_path.exists():
            with np.load(stack_path, allow_pickle=False) as z:
                P = np.asarray(z["P_stack"])
                shape = list(P.shape)

        progress_rows.append({
            "offset": offset,
            "complete": complete,
            "operator_stack_shape": shape,
            "stack_path": str(stack_path),
            "manifest_path": str(manifest_path),
        })
        atomic_json(root / "b50_8_extraction_progress.json", {
            "version": VERSION,
            "updated": now_s(),
            "rows": progress_rows,
        })

    result = {
        "version": VERSION,
        "updated": now_s(),
        "execution_offsets": offsets,
        "rows": progress_rows,
        "all_execution_offsets_complete": all(x["complete"] for x in progress_rows),
    }
    atomic_json(root / "b50_8_extraction_progress.json", result)
    return result


# ---------------------------------------------------------------------------
# AUDIT / aggregation
# ---------------------------------------------------------------------------


def load_offset_result(root: Path, offset: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    odir = offset_dir(root, offset)
    sp = odir / "b50_7_summary.json"
    npth = odir / "b50_7_primary_null.csv"
    if not sp.exists():
        raise FileNotFoundError(sp)
    if not npth.exists():
        raise FileNotFoundError(npth)
    s = json.loads(sp.read_text(encoding="utf-8"))
    n = pd.read_csv(npth)
    return s, n


def audit_mode(args, b507) -> Dict[str, Any]:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan = ensure_full_plan(root)
    lineage = validate_reference_lineage(args)
    if not lineage["pass"]:
        raise RuntimeError("INVALID_REPRODUCTION")

    if args.only_offset is not None:
        raise ValueError(
            "B50.8 --mode audit is an aggregate frozen test and must run without --only-offset"
        )

    print("=== B50.8 MULTI-OFFSET PHASE ROBUSTNESS AUDIT ===")
    print(f"offsets             : {args.offsets}")
    print(f"B50.6 effect        : {lineage['b50_6_effect']:.6f}")
    print(f"B50.7 +75 effect    : {lineage['b50_7_reference_effect']:.6f}")
    print()

    # Run exact B50.7 audit for every frozen new offset.
    for offset in args.offsets:
        sub = make_subargs(args, offset)
        odir = Path(sub.output_dir)
        if not (odir / "b50_7_offset_operator_stack.npz").exists():
            raise FileNotFoundError(
                f"Missing extraction for offset {offset:+d}; run --mode extract first"
            )
        print(f"=== AUDIT offset {offset:+d} ===", flush=True)
        b507.audit_mode(sub)

    rows = []
    nulls = []
    invalid_subruns = []

    for offset in args.offsets:
        s, ndf = load_offset_result(root, offset)
        verdict = str(s.get("final_verdict", ""))
        integrity = s.get("integrity", {})
        matching = s.get("matching", {})
        primary = s.get("primary_replication", {})
        config = s.get("config_convergence", {})
        thirds = s.get("thirds_robustness", {})

        effect = float(primary.get("observed_mean_effect", float("nan")))
        p_lower = float(primary.get("p_lower", float("nan")))
        config_pass = bool(config.get("pass", False))
        thirds_pass = bool(thirds.get("pass", False))
        match_pass = bool(matching.get("pass", False))

        integrity_pass = bool(
            integrity.get("offset_novelty_pass", False)
            and integrity.get("external_labels_exact", False)
            and integrity.get("operator_dim_exact", False)
            and integrity.get("frozen_pca_projection_parity", False)
            and integrity.get("external_raw_feature_parity_ok", False)
        )

        technical_valid = bool(
            integrity_pass
            and match_pass
            and np.isfinite(effect)
            and np.isfinite(p_lower)
            and not verdict.startswith("INVALID_")
        )

        nominal = bool(
            technical_valid
            and effect <= THRESHOLDS["nominal_effect_max"]
            and p_lower <= THRESHOLDS["nominal_p_lower_max"]
        )

        negative = bool(technical_valid and effect < 0.0)
        material_positive_reversal = bool(
            technical_valid and effect >= THRESHOLDS["material_positive_reversal_min"]
        )

        if not technical_valid:
            invalid_subruns.append(offset)

        rows.append({
            "offset": offset,
            "offset_fraction_of_step": float(offset / 150.0),
            "b50_7_subrun_verdict": verdict,
            "technical_valid": technical_valid,
            "integrity_pass": integrity_pass,
            "matching_pass": match_pass,
            "mean_physical_percentile": float(
                primary.get("observed_mean_percentile", float("nan"))
            ),
            "effect": effect,
            "p_lower": p_lower,
            "negative": negative,
            "material_positive_reversal": material_positive_reversal,
            "nominal_offset_replication": nominal,
            "negative_config_fraction": float(
                config.get("negative_effect_fraction", float("nan"))
            ),
            "config_pass": config_pass,
            "negative_thirds": int(thirds.get("negative_thirds", 0)),
            "nominally_replicated_thirds": int(
                thirds.get("nominally_replicated_thirds", 0)
            ),
            "thirds_pass": thirds_pass,
            "matching_median_abs_log_error": float(
                matching.get("median_abs_log_distance_error", float("nan"))
            ),
            "matching_q90_abs_log_error": float(
                matching.get("q90_abs_log_distance_error", float("nan"))
            ),
        })

        if "mean_percentile_effect" not in ndf.columns:
            raise ValueError(
                f"Offset {offset:+d} null CSV missing mean_percentile_effect"
            )
        nulls.append(ndf["mean_percentile_effect"].to_numpy(float))

    df = pd.DataFrame(rows).sort_values("offset")
    atomic_csv(root / "b50_8_offset_effects.csv", df)

    if invalid_subruns:
        summary = {
            "version": VERSION,
            "finished": now_s(),
            "invalid_offsets": invalid_subruns,
            "final_verdict": "INVALID_OFFSET_SUBRUN",
            "reason": f"Technical/integrity gate failed for offsets {invalid_subruns}.",
        }
        atomic_json(root / "b50_8_summary.json", summary)
        atomic_text(
            root / "b50_8_verdict.txt",
            "FINAL VERDICT: INVALID_OFFSET_SUBRUN\n" + summary["reason"] + "\n",
        )
        return summary

    effects = df["effect"].to_numpy(float)
    negative_count = int(df["negative"].sum())
    nominal_count = int(df["nominal_offset_replication"].sum())
    config_pass_count = int(df["config_pass"].sum())
    thirds_pass_count = int(df["thirds_pass"].sum())
    reversal_count = int(df["material_positive_reversal"].sum())

    mean_effect = float(np.mean(effects))
    median_effect = float(np.median(effects))
    sd_effect = float(np.std(effects, ddof=1)) if len(effects) > 1 else 0.0
    q25 = float(np.quantile(effects, 0.25))
    q75 = float(np.quantile(effects, 0.75))
    iqr = q75 - q25
    effect_range = float(np.max(effects) - np.min(effects))

    # Secondary pooled-null diagnostic. Replicate count may differ in a custom
    # run, so truncate to common length deterministically.
    min_null_n = min(len(x) for x in nulls)
    null_matrix = np.column_stack([x[:min_null_n] for x in nulls])
    pooled_null_mean_effect = np.mean(null_matrix, axis=1)
    pooled_p_lower = float(
        (1 + np.sum(pooled_null_mean_effect <= mean_effect)) / (min_null_n + 1)
    )
    atomic_csv(root / "b50_8_pooled_null.csv", pd.DataFrame({
        "replicate": np.arange(min_null_n),
        "mean_offset_null_effect": pooled_null_mean_effect,
    }))

    phase_gates = {
        "all_four_negative": negative_count >= THRESHOLDS["all_negative_required"],
        "at_least_three_nominal": nominal_count >= THRESHOLDS["nominal_offset_required"],
        "median_effect_at_least_minus_0_05": median_effect <= THRESHOLDS["median_effect_max"],
        "at_least_three_config_pass": config_pass_count >= THRESHOLDS["config_pass_required"],
        "at_least_three_thirds_pass": thirds_pass_count >= THRESHOLDS["thirds_pass_required"],
        "no_material_positive_reversal": reversal_count == 0,
        "all_matching_and_integrity_pass": True,
    }

    full_pass = bool(all(phase_gates.values()))

    if full_pass:
        verdict = "MULTI_OFFSET_PHASE_ROBUST_EFFECT_INVARIANCE"
        reason = (
            "The frozen negative adjacency-specificity effect remains negative at all four "
            "new offset phases, at least three offsets satisfy the pre-declared nominal "
            "replication gate, the median effect remains at or below -0.05, and frozen "
            "representation/thirds robustness gates converge across offsets."
        )
    elif negative_count == 4 and reversal_count == 0:
        verdict = "MULTI_OFFSET_DIRECTION_STABLE_WEAK_AMPLITUDE"
        reason = (
            "All four new phases preserve the negative direction, but one or more frozen "
            "amplitude / nominal replication / convergence gates are too weak for the full "
            "phase-robust invariance verdict."
        )
    elif reversal_count > 0:
        verdict = "MULTI_OFFSET_PHASE_SENSITIVE"
        reason = (
            "At least one new offset shows a materially positive effect, indicating a "
            "phase-dependent sign reversal under the frozen test."
        )
    elif negative_count >= 3:
        verdict = "MULTI_OFFSET_PARTIAL_REPLICATION"
        reason = (
            "Most offsets preserve the negative direction, but the complete frozen "
            "multi-offset phase-robustness contract does not pass."
        )
    elif (
        abs(median_effect) <= THRESHOLDS["null_compatible_abs_median_effect_max"]
        and nominal_count < THRESHOLDS["nominal_offset_required"]
    ):
        verdict = "MULTI_OFFSET_NULL_COMPATIBLE"
        reason = (
            "The median multi-offset effect is close to null and fewer than three offsets "
            "meet the frozen nominal replication gate."
        )
    else:
        verdict = "MULTI_OFFSET_PARTIAL_REPLICATION"
        reason = "The multi-offset result is mixed under the frozen phase-robustness gates."

    reference_effects = {
        "b50_6_original_effect": lineage["b50_6_effect"],
        "b50_7_plus75_effect": lineage["b50_7_reference_effect"],
        "new_offsets_mean_effect": mean_effect,
        "new_offsets_median_effect": median_effect,
        "new_offsets_sd_effect": sd_effect,
        "new_offsets_iqr_effect": iqr,
        "new_offsets_range_effect": effect_range,
        "new_offsets_min_effect": float(np.min(effects)),
        "new_offsets_max_effect": float(np.max(effects)),
        "mean_effect_ratio_vs_b50_7_plus75": (
            float(mean_effect / lineage["b50_7_reference_effect"])
            if abs(lineage["b50_7_reference_effect"]) > 1e-15 else float("nan")
        ),
    }

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "outcome_blind_to_b59": True,
        "same_underlying_prime_corpus": True,
        "independent_corpus_replication": False,
        "lineage": lineage,
        "frozen_offsets": FROZEN_OFFSETS,
        "phase_gates": phase_gates,
        "counts": {
            "negative_offsets": negative_count,
            "nominal_offset_replications": nominal_count,
            "config_pass_offsets": config_pass_count,
            "thirds_pass_offsets": thirds_pass_count,
            "material_positive_reversals": reversal_count,
        },
        "effect_invariance": reference_effects,
        "secondary_pooled_null": {
            "replicate_count": min_null_n,
            "observed_mean_new_offset_effect": mean_effect,
            "null_mean": float(np.mean(pooled_null_mean_effect)),
            "null_sd": float(np.std(pooled_null_mean_effect)),
            "p_lower": pooled_p_lower,
            "mandatory_gate": False,
        },
        "offset_rows": df.to_dict(orient="records"),
        "thresholds": THRESHOLDS,
        "final_verdict": verdict,
        "final_reason": reason,
        "scientific_boundary": (
            "Same-corpus multi-offset window-phase robustness audit. New window states, "
            "but not independent source data; no B59 validation, no individual-prime "
            "prediction, and no theorem claim."
        ),
    }
    atomic_json(root / "b50_8_summary.json", summary)

    # Plots.
    plt.figure(figsize=(9, 5))
    plt.plot(df["offset"], df["effect"], marker="o")
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(THRESHOLDS["nominal_effect_max"], linestyle="--", linewidth=0.8)
    plt.axhline(THRESHOLDS["median_effect_max"], linestyle=":", linewidth=0.8)
    plt.xlabel("offset segments")
    plt.ylabel("mean physical percentile effect vs 0.5")
    plt.title("B50.8 effect across frozen offset phases")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(root / "b50_8_effect_by_offset.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(pooled_null_mean_effect, bins=45)
    plt.axvline(mean_effect, linewidth=2)
    plt.xlabel("mean effect across four offsets under matched-identity null")
    plt.ylabel("count")
    plt.title("B50.8 pooled multi-offset null diagnostic")
    plt.tight_layout()
    plt.savefig(root / "b50_8_pooled_null.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(df["offset"], df["negative_config_fraction"], marker="o", label="negative configs")
    plt.plot(
        df["offset"],
        df["nominally_replicated_thirds"] / 3.0,
        marker="o",
        label="replicated thirds fraction",
    )
    plt.ylim(-0.02, 1.02)
    plt.xlabel("offset segments")
    plt.ylabel("fraction")
    plt.title("B50.8 representation and thirds convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(root / "b50_8_convergence_by_offset.png", dpi=160)
    plt.close()

    report = f"""B50.8 — Multi-Offset Phase Robustness & Effect Invariance Audit
=================================================================

Interpretation scope
--------------------
New offset WINDOW STATES: YES
Same underlying prime corpus: YES
Independent-corpus replication: NO
B59/B59.1 used: NO

Reference checkpoints frozen before B50.8
-----------------------------------------
B50.6 effect                            = {lineage['b50_6_effect']}
B50.7 +75 effect                       = {lineage['b50_7_reference_effect']}
B50.7 reference verdict                = EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED

Frozen new offsets
------------------
offsets                                 = {args.offsets}
protocol modified                       = {protocol_modified(args)}

Per-offset results
------------------
{df.to_string(index=False)}

Multi-offset sign / replication gates
-------------------------------------
negative offsets                        = {negative_count}/4
nominal offset replications             = {nominal_count}/4
config-convergence passes               = {config_pass_count}/4
thirds-robustness passes                = {thirds_pass_count}/4
material positive reversals             = {reversal_count}

Effect invariance
-----------------
mean new-offset effect                  = {mean_effect}
median new-offset effect                = {median_effect}
SD new-offset effect                    = {sd_effect}
IQR new-offset effect                   = {iqr}
range new-offset effect                 = {effect_range}
min / max new-offset effect             = {float(np.min(effects))} / {float(np.max(effects))}
mean effect ratio vs B50.7 +75          = {reference_effects['mean_effect_ratio_vs_b50_7_plus75']}

Frozen mandatory gates
----------------------
4/4 negative                            = {phase_gates['all_four_negative']}
>=3/4 nominal (effect<=-0.04,p<=0.01)   = {phase_gates['at_least_three_nominal']}
median effect <= -0.05                  = {phase_gates['median_effect_at_least_minus_0_05']}
>=3/4 config convergence                = {phase_gates['at_least_three_config_pass']}
>=3/4 thirds robustness                 = {phase_gates['at_least_three_thirds_pass']}
no material positive reversal           = {phase_gates['no_material_positive_reversal']}
all matching/integrity pass             = {phase_gates['all_matching_and_integrity_pass']}

Secondary pooled-null diagnostic
--------------------------------
replicates                               = {min_null_n}
observed mean offset effect              = {mean_effect}
null mean                                = {float(np.mean(pooled_null_mean_effect))}
null SD                                  = {float(np.std(pooled_null_mean_effect))}
p_lower                                  = {pooled_p_lower}
mandatory gate                           = False

FINAL VERDICT: {verdict}

REASON:
{reason}

Interpretation boundary
-----------------------
This is a same-corpus multi-offset window-phase robustness audit. The offset
states are new, but they are derived from the same underlying prime corpus and
are not statistically independent corpora. No B59/B59.1 selector or predictive
outcome is used. No individual-prime prediction, RH, or theorem claim is made.

Key outputs
-----------
b50_8_multi_offset_plan.csv
b50_8_plan_summary.json
b50_8_extraction_progress.json
b50_8_offset_effects.csv
b50_8_pooled_null.csv
b50_8_summary.json
b50_8_verdict.txt
b50_8_effect_by_offset.png
b50_8_pooled_null.png
b50_8_convergence_by_offset.png

Each offset subdirectory also contains the complete frozen B50.7 artifacts.
"""
    atomic_text(root / "b50_8_verdict.txt", report)

    print()
    print("=== B50.8 RESULT ===")
    print(f"negative offsets   : {negative_count}/4")
    print(f"nominal offsets    : {nominal_count}/4")
    print(f"config passes      : {config_pass_count}/4")
    print(f"thirds passes      : {thirds_pass_count}/4")
    print(f"median effect      : {median_effect:.6f}")
    print(f"mean effect        : {mean_effect:.6f}")
    print(f"pooled p_lower     : {pooled_p_lower:.6g}  [secondary]")
    print(f"FINAL VERDICT      : {verdict}")
    print(f"verdict            : {root / 'b50_8_verdict.txt'}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="B50.8 Multi-Offset Phase Robustness & Effect Invariance Audit"
    )
    p.add_argument("--mode", choices=["plan", "extract", "audit", "all"], default="plan")

    p.add_argument("--b50-7-script", required=True)
    p.add_argument("--prime-root", default=r"d:\primes_output")
    p.add_argument("--total-segments", type=int, default=100000)
    p.add_argument("--offsets", default="25,50,100,125")
    p.add_argument(
        "--only-offset",
        type=int,
        default=None,
        help="Execution shard only; does not change frozen scientific protocol.",
    )

    p.add_argument("--state-dataset", required=True)
    p.add_argument("--training-operator-stack", required=True)
    p.add_argument("--b50-script", required=True)
    p.add_argument("--b50-4-script", required=True)
    p.add_argument("--b50-4-protocol", required=True)
    p.add_argument("--b50-4-front-features", required=True)
    p.add_argument("--b50-6-summary", required=True)
    p.add_argument("--b50-7-reference-summary", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--gap-bins", type=int, default=64)
    p.add_argument("--staging-root", default=None)
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=10)

    p.add_argument("--controls-per-transition", type=int, default=20)
    p.add_argument("--min-index-separation", type=int, default=10)
    p.add_argument("--randomization-replicates", type=int, default=5000)
    p.add_argument("--seed", type=int, default=50808)

    args = p.parse_args()
    args.offsets = parse_offsets(args.offsets)

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

    if args.only_offset is not None and args.mode != "extract":
        p.error("--only-offset is allowed only with --mode extract")

    return args


def main():
    args = parse_args()
    b507 = import_b507(Path(args.b50_7_script))

    if args.mode == "plan":
        plan_mode(args, b507)
    elif args.mode == "extract":
        extract_mode(args, b507)
    elif args.mode == "audit":
        audit_mode(args, b507)
    elif args.mode == "all":
        plan_mode(args, b507)
        extract_mode(args, b507)
        audit_mode(args, b507)
    else:
        raise ValueError(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
