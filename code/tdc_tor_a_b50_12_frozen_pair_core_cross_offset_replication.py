#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TDC TOR A / B50.12
Frozen Pair-Core Cross-Offset Replication
=========================================

Purpose
-------
B50.11 exhaustively evaluated all 31 non-empty subsets of the five frozen
B50.4 front channels and found one unique minimal sufficient two-channel core:

    turn_change + density_scale_change

with a same-sequence effect retaining >=90% of the full five-channel B50.6
effect and passing the frozen Holm directional gate.

B50.12 freezes that exact pair and evaluates it, without re-selection, on the
already-existing offset-window state sequences:

    +25, +50, +75, +100, +125 segments

The +75 sequence is the B50.7 external-offset replication reference. The other
four sequences are the B50.8 multi-offset phase-robustness runs.

This script does NOT generate new windows, does NOT rematch controls, and does
NOT search over channels. It reuses:

- the exact B50.7 offset operator stacks;
- the exact B50.7 matched control identities for every offset;
- the frozen original training PCA basis;
- the frozen B50.4 front-channel scaling;
- the exact B50.6 positive robust-z RMS score, restricted only to:
      turn_change
      density_scale_change

Primary question
----------------
Does the pair core selected on the original (offset 0) B50.11 mechanism audit
replicate across the five previously generated offset phases?

Frozen cross-offset gates
-------------------------
Technical:
    all five offset inputs valid;
    full five-channel B50.7 endpoint reproduced exactly at every offset;
    exact B50.7 control identities reused.

Pair-core:
    5/5 offset effects negative;
    >=4/5 offsets satisfy:
        effect <= -0.04
        Holm-adjusted directional p <= 0.05 across the five offsets;
    median offset effect <= -0.05;
    >=4/5 offsets have >=80% frozen configurations with negative pair effect;
    >=4/5 offsets pass thirds robustness:
        3/3 thirds negative
        >=2/3 thirds effect <= -0.04 and raw directional p <= 0.05;
    no material positive reversal (effect >= +0.04).

Scientific boundary
-------------------
This is a frozen representation replication across different window phases of
the SAME underlying prime corpus. It is stronger than a same-sequence
decomposition, but it is NOT independent-corpus replication.

The offset windows and their full five-channel B50.7/B50.8 outcomes existed
before B50.12. Therefore B50.12 is a confirmatory frozen-pair representation
check, not a pristine blinded prospective experiment.

No B59 selector is used. No individual-prime prediction, RH, or theorem claim
is made.

Operational properties
----------------------
- PLAN -> AUDIT split.
- SHA-256 freezes all required B50.10/B50.11/B50.7/B50.8 artifacts.
- Per-offset atomic outputs.
- Per-offset resumability via input/protocol fingerprint.
- No prime-data re-extraction required.
- No new matched-control construction.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "B50.12_frozen_pair_core_cross_offset_replication_v1_1_import_fix"

PAIR_CHANNELS = ["turn_change", "density_scale_change"]
ALL_CHANNELS = [
    "speed",
    "acceleration",
    "turn_change",
    "density_scale_change",
    "diffusion_step",
]
OFFSETS = [25, 50, 75, 100, 125]
REFERENCE_OFFSET = 75

EXPECTED_B5011_VERDICT = "PAIRWISE_CORE_RECAPTURES_FULL_EFFECT"
EXPECTED_B5010_VERDICT = "CHANNEL_ASSOCIATION_WITHOUT_SCORE_LOCALIZATION"
EXPECTED_B507_VERDICT = "EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED"
EXPECTED_B508_VERDICT = "MULTI_OFFSET_PHASE_ROBUST_EFFECT_INVARIANCE"

DEFAULT_REPS = 5000
DEFAULT_SEED = 501012

# Frozen before B50.12 pair-core outcomes.
THRESHOLDS = {
    "full_reproduction_percentile_max_abs_error": 1e-12,
    "full_reproduction_effect_max_abs_error": 1e-12,

    "offset_negative_required": 5,
    "offset_nominal_required": 4,
    "nominal_effect_max": -0.04,
    "holm_directional_p_max": 0.05,
    "median_effect_max": -0.05,

    "config_negative_fraction_min": 0.80,
    "config_pass_required": 4,

    "third_negative_required": 3,
    "third_nominal_required": 2,
    "third_effect_max": -0.04,
    "third_p_lower_max": 0.05,
    "thirds_pass_offsets_required": 4,

    "material_positive_reversal_min": 0.04,

    # Descriptive only.
    "reference_effect_retention_floor": 0.80,
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


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def canonical_hash(obj: Any) -> str:
    b = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def offset_dir_name(offset: int) -> str:
    sign = "p" if offset >= 0 else "m"
    return f"offset_{sign}{abs(offset):03d}"


def import_b507(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "b507_frozen_b5012", str(path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import B50.7 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    required = [
        "frozen_pca_basis",
        "projection_parity",
        "build_external_configs",
        "frozen_channel_scale",
        "candidate_raw",
        "rank_percentiles",
        "build_replication_cube",
        "import_b504",
    ]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise AttributeError(
            f"B50.7 missing required API: {missing}"
        )
    return mod


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, float)
    running = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * p[idx]
        running = max(running, val)
        adjusted_sorted[k] = min(1.0, running)
    out = np.empty(m, float)
    for k, idx in enumerate(order):
        out[idx] = adjusted_sorted[k]
    return out


def matched_identity_null_lower(
    consensus_pct: np.ndarray,
    reps: int,
    seed: int,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    X = np.asarray(consensus_pct, float)
    if mask is not None:
        X = X[np.asarray(mask, bool)]
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("consensus_pct must be transitions x candidates")

    obs = X[:, 0]
    effect = float(np.mean(obs) - 0.5)
    rng = np.random.default_rng(seed)
    rows = np.arange(len(X))
    null = np.empty(reps, float)

    for r in range(reps):
        choice = rng.integers(0, X.shape[1], size=len(X))
        null[r] = float(np.mean(X[rows, choice]) - 0.5)

    p_lower = float(
        (1 + np.sum(null <= effect)) / (reps + 1)
    )
    p_upper = float(
        (1 + np.sum(null >= effect)) / (reps + 1)
    )
    p_two = float(
        (1 + np.sum(np.abs(null) >= abs(effect))) / (reps + 1)
    )

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


def parse_controls(x: Any) -> List[int]:
    vals = [
        int(v.strip())
        for v in str(x).split(",")
        if v.strip()
    ]
    if not vals:
        raise ValueError(f"empty frozen control list: {x!r}")
    return vals


def pair_score(
    raw: Mapping[str, np.ndarray],
    scale: Mapping[str, Tuple[float, float]],
) -> np.ndarray:
    cols = []
    for ch in PAIR_CHANNELS:
        x = np.asarray(raw[ch], float)
        med, s = scale[ch]
        if s <= 1e-15:
            z = np.zeros_like(x)
        else:
            z = (x - med) / s
            z[~np.isfinite(z)] = 0.0
            z = np.maximum(z, 0.0)
        cols.append(z)
    Z = np.column_stack(cols)
    return np.sqrt(np.mean(Z * Z, axis=1))


# ---------------------------------------------------------------------------
# Path / lineage model
# ---------------------------------------------------------------------------

def core_paths(args) -> Dict[str, Path]:
    b10 = Path(args.b50_10_dir)
    b11 = Path(args.b50_11_dir)
    b8 = Path(args.b50_8_dir)
    return {
        "b50_7_script": Path(args.b50_7_script),

        "b50_10_protocol": b10 / "b50_10_protocol_manifest.json",
        "b50_10_summary": b10 / "b50_10_summary.json",

        "b50_11_protocol": b11 / "b50_11_protocol_manifest.json",
        "b50_11_summary": b11 / "b50_11_summary.json",
        "b50_11_minimal": b11 / "b50_11_minimal_sufficient_subsets.csv",
        "b50_11_shapley": b11 / "b50_11_shapley.csv",

        "b50_8_summary": b8 / "b50_8_summary.json",
    }


def offset_input_dir(args, offset: int) -> Path:
    if offset == REFERENCE_OFFSET:
        return Path(args.b50_7_reference_dir)
    return Path(args.b50_8_dir) / offset_dir_name(offset)


def offset_paths(args, offset: int) -> Dict[str, Path]:
    d = offset_input_dir(args, offset)
    return {
        "dir": d,
        "plan_csv": d / "b50_7_offset_window_plan.csv",
        "plan_json": d / "b50_7_offset_window_plan.json",
        "external_stack": d / "b50_7_offset_operator_stack.npz",
        "extraction_manifest": d / "b50_7_extraction_manifest.json",
        "summary": d / "b50_7_summary.json",
        "transition_specificity": d / "b50_7_transition_specificity.csv",
        "matching_audit": d / "b50_7_matching_audit.csv",
    }


def require_files(paths: Mapping[str, Path]) -> None:
    missing = []
    for k, p in paths.items():
        if k == "dir":
            if not p.is_dir():
                missing.append((k, p))
        elif not p.exists():
            missing.append((k, p))
    if missing:
        raise FileNotFoundError(
            "Missing B50.12 input(s):\n"
            + "\n".join(f"{k}: {p}" for k, p in missing)
        )


def load_b5010_frozen_source_paths(
    b5010_protocol_path: Path,
) -> Dict[str, Path]:
    p = json.loads(
        b5010_protocol_path.read_text(encoding="utf-8")
    )
    inputs = p.get("inputs", {})

    required_keys = [
        "operator_stack",
        "b50_4_script",
        "b50_4_protocol",
        "b50_4_front_features",
        "b50_6_summary",
    ]
    out = {}
    for k in required_keys:
        item = inputs.get(k, {})
        path = item.get("path")
        if not path:
            raise KeyError(
                f"B50.10 protocol missing frozen input path {k}"
            )
        out[k] = Path(path)
    return out


def lineage_gate(args) -> Dict[str, Any]:
    c = core_paths(args)
    require_files(c)

    s10 = json.loads(c["b50_10_summary"].read_text(encoding="utf-8"))
    s11 = json.loads(c["b50_11_summary"].read_text(encoding="utf-8"))
    s8 = json.loads(c["b50_8_summary"].read_text(encoding="utf-8"))
    minimal = pd.read_csv(c["b50_11_minimal"])

    v10 = str(s10.get("final_verdict", ""))
    v11 = str(s11.get("final_verdict", ""))
    v8 = str(s8.get("final_verdict", ""))

    min_size = (
        int(s11.get("subset_family", {})
            .get("minimal_subset", {})
            .get("minimal_size"))
        if s11.get("subset_family", {})
        .get("minimal_subset", {})
        .get("minimal_size") is not None
        else None
    )

    min_keys = (
        s11.get("subset_family", {})
        .get("minimal_subset", {})
        .get("minimal_subset_keys", [])
    )

    pair_key = "+".join(PAIR_CHANNELS)
    exact_unique_pair = bool(
        min_size == 2
        and min_keys == [pair_key]
        and len(minimal) == 1
        and str(minimal.iloc[0]["subset_key"]) == pair_key
    )

    pair_effect = (
        float(minimal.iloc[0]["observed_mean_effect"])
        if exact_unique_pair
        else float("nan")
    )
    pair_retention = (
        float(minimal.iloc[0]["abs_effect_retention_fraction"])
        if exact_unique_pair
        else float("nan")
    )

    s10_full = safe_float(
        s10.get("reproduction", {})
        .get("reconstructed_full_effect")
    )

    frozen_offsets_b8 = list(
        s8.get("frozen_offsets", [])
    )
    b8_offsets_ok = frozen_offsets_b8 == [25, 50, 100, 125]

    # Check B50.7 +75 reference.
    p75 = offset_paths(args, 75)
    require_files(p75)
    s75 = json.loads(p75["summary"].read_text(encoding="utf-8"))
    v75 = str(s75.get("final_verdict", ""))
    o75 = int(
        s75.get("offset_sequence", {})
        .get("offset_segments", -999999)
    )

    pass_ = bool(
        v10 == EXPECTED_B5010_VERDICT
        and v11 == EXPECTED_B5011_VERDICT
        and v8 == EXPECTED_B508_VERDICT
        and exact_unique_pair
        and np.isfinite(pair_effect)
        and pair_effect < 0.0
        and np.isfinite(pair_retention)
        and pair_retention >= 0.90
        and np.isfinite(s10_full)
        and s10_full < 0.0
        and b8_offsets_ok
        and v75 == EXPECTED_B507_VERDICT
        and o75 == 75
    )

    return {
        "b50_10_verdict": v10,
        "b50_10_ok": v10 == EXPECTED_B5010_VERDICT,
        "b50_11_verdict": v11,
        "b50_11_ok": v11 == EXPECTED_B5011_VERDICT,
        "b50_8_verdict": v8,
        "b50_8_ok": v8 == EXPECTED_B508_VERDICT,
        "b50_8_offsets": frozen_offsets_b8,
        "b50_8_offsets_ok": b8_offsets_ok,
        "pair_channels": PAIR_CHANNELS,
        "pair_key": pair_key,
        "minimal_size": min_size,
        "minimal_keys": min_keys,
        "exact_unique_pair": exact_unique_pair,
        "b50_11_pair_effect": pair_effect,
        "b50_11_pair_retention": pair_retention,
        "b50_10_full_effect": s10_full,
        "b50_7_plus75_verdict": v75,
        "b50_7_plus75_offset": o75,
        "b50_7_plus75_ok": (
            v75 == EXPECTED_B507_VERDICT and o75 == 75
        ),
        "pass": pass_,
    }


# ---------------------------------------------------------------------------
# PLAN
# ---------------------------------------------------------------------------

def plan_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    c = core_paths(args)
    require_files(c)
    lineage = lineage_gate(args)

    frozen_source_paths = load_b5010_frozen_source_paths(
        c["b50_10_protocol"]
    )
    require_files(frozen_source_paths)

    core_hashes = {
        k: {
            "path": str(v),
            "sha256": sha256_file(v),
        }
        for k, v in {
            **c,
            **{
                f"frozen_{k}": v
                for k, v in frozen_source_paths.items()
            },
        }.items()
    }

    offset_hashes = {}
    offset_rows = []

    for offset in OFFSETS:
        op = offset_paths(args, offset)
        require_files(op)

        ps = json.loads(op["plan_json"].read_text(encoding="utf-8"))
        ss = json.loads(op["summary"].read_text(encoding="utf-8"))

        applied = int(ps.get("applied_offset", -999999))
        summary_offset = int(
            ss.get("offset_sequence", {})
            .get("offset_segments", -999999)
        )

        valid = bool(
            applied == offset
            and summary_offset == offset
            and not str(ss.get("final_verdict", "")).startswith("INVALID_")
        )

        offset_hashes[str(offset)] = {
            k: {
                "path": str(v),
                "sha256": sha256_file(v),
            }
            for k, v in op.items()
            if k != "dir"
        }

        offset_rows.append({
            "offset": offset,
            "input_dir": str(op["dir"]),
            "plan_applied_offset": applied,
            "summary_offset": summary_offset,
            "b50_7_verdict": str(ss.get("final_verdict", "")),
            "technical_lineage_valid": valid,
        })

    offset_df = pd.DataFrame(offset_rows)
    atomic_csv(out / "b50_12_offset_input_plan.csv", offset_df)

    all_offset_lineage = bool(
        offset_df["technical_lineage_valid"].all()
    )

    protocol_modified = bool(
        args.nominal_effect_max
        != THRESHOLDS["nominal_effect_max"]
        or args.median_effect_max
        != THRESHOLDS["median_effect_max"]
    )

    plan_pass = bool(
        lineage["pass"]
        and all_offset_lineage
        and not protocol_modified
    )

    manifest = {
        "version": VERSION,
        "created": now_s(),
        "scientific_role": (
            "frozen two-channel pair-core replication across five "
            "existing offset-window phases"
        ),
        "lineage": lineage,
        "pair_core": {
            "channels": PAIR_CHANNELS,
            "selection_source": str(c["b50_11_summary"]),
            "selection_rule": (
                "unique minimal sufficient subset from exhaustive "
                "31-subset B50.11 audit"
            ),
            "reference_effect_offset0": (
                lineage["b50_11_pair_effect"]
            ),
            "reference_retention_vs_full": (
                lineage["b50_11_pair_retention"]
            ),
            "reselection_allowed": False,
        },
        "offsets": OFFSETS,
        "offset_sources": {
            "75": "B50.7 reference offset directory",
            "25,50,100,125": "B50.8 frozen subrun directories",
        },
        "frozen_source_paths": {
            k: str(v) for k, v in frozen_source_paths.items()
        },
        "core_input_hashes": core_hashes,
        "offset_input_hashes": offset_hashes,
        "technical_reproduction_gate": {
            "full_b50_7_consensus_percentile_max_abs_error": (
                THRESHOLDS[
                    "full_reproduction_percentile_max_abs_error"
                ]
            ),
            "full_b50_7_effect_max_abs_error": (
                THRESHOLDS[
                    "full_reproduction_effect_max_abs_error"
                ]
            ),
            "exact_control_identity_reuse": True,
        },
        "pair_offset_gate": {
            "all_offsets_negative_required": (
                THRESHOLDS["offset_negative_required"]
            ),
            "nominal_offsets_required": (
                THRESHOLDS["offset_nominal_required"]
            ),
            "nominal_effect_max": args.nominal_effect_max,
            "holm_directional_p_max": (
                THRESHOLDS["holm_directional_p_max"]
            ),
            "median_effect_max": args.median_effect_max,
            "config_negative_fraction_min": (
                THRESHOLDS["config_negative_fraction_min"]
            ),
            "config_pass_offsets_required": (
                THRESHOLDS["config_pass_required"]
            ),
            "thirds_pass_offsets_required": (
                THRESHOLDS["thirds_pass_offsets_required"]
            ),
            "material_positive_reversal_min": (
                THRESHOLDS[
                    "material_positive_reversal_min"
                ]
            ),
        },
        "thirds_gate": {
            "negative_required": (
                THRESHOLDS["third_negative_required"]
            ),
            "nominal_required": (
                THRESHOLDS["third_nominal_required"]
            ),
            "effect_max": THRESHOLDS["third_effect_max"],
            "p_lower_max": THRESHOLDS["third_p_lower_max"],
        },
        "randomization_replicates": args.randomization_replicates,
        "seed": args.seed,
        "protocol_modified": protocol_modified,
        "plan_pass": plan_pass,
        "scientific_boundary": (
            "Same underlying prime corpus. Pair frozen from original "
            "offset-0 B50.11 audit. Offset windows/full-score outcomes "
            "pre-existed B50.12, so this is not a pristine blinded "
            "prospective test and not independent-corpus replication."
        ),
    }

    plan_fingerprint_obj = {
        k: v for k, v in manifest.items()
        if k not in ("created",)
    }
    manifest["protocol_fingerprint"] = canonical_hash(
        plan_fingerprint_obj
    )

    atomic_json(out / "b50_12_protocol_manifest.json", manifest)

    report = f"""B50.12 — Frozen Pair-Core Cross-Offset Replication / PLAN
==========================================================

Frozen lineage
--------------
B50.10 verdict                     = {lineage['b50_10_verdict']}
B50.11 verdict                     = {lineage['b50_11_verdict']}
B50.8 verdict                      = {lineage['b50_8_verdict']}
B50.7 +75 verdict                  = {lineage['b50_7_plus75_verdict']}
lineage PASS                       = {lineage['pass']}

Frozen pair core
----------------
channels                           = {PAIR_CHANNELS}
B50.11 offset-0 pair effect        = {lineage['b50_11_pair_effect']}
B50.11 retention vs five-channel   = {lineage['b50_11_pair_retention']}
reselection allowed                = False

Cross-offset sample
-------------------
offsets                            = {OFFSETS}
all offset lineage valid           = {all_offset_lineage}

Frozen mandatory gates
----------------------
5/5 negative                       = required
>=4/5 nominal                      = effect <= {args.nominal_effect_max}
                                     Holm p <= {THRESHOLDS['holm_directional_p_max']}
median effect                      <= {args.median_effect_max}
>=4/5 config convergence           = required
>=4/5 thirds robustness            = required
material positive reversal         = forbidden

Technical gate
--------------
full B50.7 endpoint reproduction   = exact at every offset
matched-control identity           = exact reuse at every offset
new control matching               = forbidden
new channel selection              = forbidden

randomization replicates           = {args.randomization_replicates}
protocol modified                  = {protocol_modified}

PLAN PASS                          = {plan_pass}
protocol fingerprint               = {manifest['protocol_fingerprint']}

Boundary
--------
Same prime corpus; different window phases. Offset full-score outcomes already
existed before this pair-core test. This is a frozen representation replication,
not independent-corpus replication.
"""
    atomic_text(out / "b50_12_plan.txt", report)
    print(report)
    return manifest


def ensure_plan(args) -> Dict[str, Any]:
    p = Path(args.output_dir) / "b50_12_protocol_manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing B50.12 PLAN: {p}. Run --mode plan first."
        )
    plan = json.loads(p.read_text(encoding="utf-8"))
    if not plan.get("plan_pass", False):
        raise RuntimeError("B50.12 PLAN did not pass.")
    if plan.get("protocol_modified", False):
        raise RuntimeError("B50.12 PLAN has protocol_modified=True.")
    return plan


def verify_plan_hashes(
    args,
    plan: Dict[str, Any],
) -> Tuple[bool, pd.DataFrame]:
    rows = []
    ok_all = True

    for group_name in ("core_input_hashes",):
        for key, item in plan[group_name].items():
            p = Path(item["path"])
            exists = p.exists()
            h = sha256_file(p) if exists else None
            exact = exists and h == item["sha256"]
            ok_all &= exact
            rows.append({
                "group": group_name,
                "input": key,
                "path": str(p),
                "exists": exists,
                "frozen_sha256": item["sha256"],
                "current_sha256": h,
                "exact": exact,
            })

    for offset, items in plan["offset_input_hashes"].items():
        for key, item in items.items():
            p = Path(item["path"])
            exists = p.exists()
            h = sha256_file(p) if exists else None
            exact = exists and h == item["sha256"]
            ok_all &= exact
            rows.append({
                "group": f"offset_{offset}",
                "input": key,
                "path": str(p),
                "exists": exists,
                "frozen_sha256": item["sha256"],
                "current_sha256": h,
                "exact": exact,
            })

    df = pd.DataFrame(rows)
    atomic_csv(
        Path(args.output_dir) / "b50_12_frozen_input_audit.csv",
        df,
    )
    return bool(ok_all), df


# ---------------------------------------------------------------------------
# Frozen control-map recovery
# ---------------------------------------------------------------------------

def frozen_control_map(
    transition_df: pd.DataFrame,
    matching_df: pd.DataFrame,
    n_states: int,
) -> Tuple[List[int], Dict[int, np.ndarray], Dict[str, Any]]:
    req_t = {
        "transition_index",
        "consensus_physical_percentile",
        "control_dest_indices",
    }
    req_m = {
        "transition_index",
        "control_rank",
        "control_dest_index",
    }

    if not req_t.issubset(transition_df.columns):
        raise ValueError(
            f"transition table missing {sorted(req_t-set(transition_df.columns))}"
        )
    if not req_m.issubset(matching_df.columns):
        raise ValueError(
            f"matching table missing {sorted(req_m-set(matching_df.columns))}"
        )

    tdf = transition_df.copy()
    tdf["transition_index"] = pd.to_numeric(
        tdf["transition_index"], errors="raise"
    ).astype(int)
    tdf = tdf.sort_values("transition_index", kind="stable")

    tis = tdf["transition_index"].tolist()
    expected = list(range(1, n_states - 1))
    if tis != expected:
        raise RuntimeError(
            f"transition indices not exact 1..{n_states-2}"
        )

    mdf = matching_df.copy()
    for c in (
        "transition_index",
        "control_rank",
        "control_dest_index",
    ):
        mdf[c] = pd.to_numeric(
            mdf[c], errors="raise"
        ).astype(int)

    cmap = {}
    exact_all = True
    counts = set()

    for _, r in tdf.iterrows():
        i = int(r["transition_index"])
        a = parse_controls(r["control_dest_indices"])

        sub = (
            mdf[mdf["transition_index"] == i]
            .sort_values("control_rank", kind="stable")
        )
        b = sub["control_dest_index"].astype(int).tolist()

        exact = a == b
        exact_all &= exact
        if not exact:
            raise RuntimeError(
                f"control identity mismatch at transition {i}"
            )

        if len(set(a)) != len(a):
            raise RuntimeError(
                f"duplicate controls at transition {i}"
            )
        if i + 1 in a:
            raise RuntimeError(
                f"physical destination leaked into controls at transition {i}"
            )
        if min(a) < 0 or max(a) >= n_states:
            raise RuntimeError(
                f"control destination outside state range at transition {i}"
            )

        cmap[i] = np.asarray(a, int)
        counts.add(len(a))

    if len(counts) != 1:
        raise RuntimeError(
            f"non-constant control count: {sorted(counts)}"
        )

    return tis, cmap, {
        "transition_count": len(tis),
        "controls_per_transition": int(next(iter(counts))),
        "exact_identity_between_outputs": bool(exact_all),
    }


# ---------------------------------------------------------------------------
# One-offset reconstruction and pair-core audit
# ---------------------------------------------------------------------------

def reconstruct_offset(
    args,
    plan: Dict[str, Any],
    b507,
    offset: int,
) -> Dict[str, Any]:
    out_root = Path(args.output_dir)
    odir_out = out_root / offset_dir_name(offset)
    odir_out.mkdir(parents=True, exist_ok=True)

    protocol_fp = plan["protocol_fingerprint"]
    summary_path = odir_out / "b50_12_offset_summary.json"
    null_path = odir_out / "b50_12_pair_null.csv"

    # Per-offset input fingerprint.
    offset_inputs = plan["offset_input_hashes"][str(offset)]
    fp_obj = {
        "protocol_fingerprint": protocol_fp,
        "offset": offset,
        "pair_channels": PAIR_CHANNELS,
        "offset_input_hashes": {
            k: v["sha256"] for k, v in offset_inputs.items()
        },
        "core_hashes": {
            k: v["sha256"]
            for k, v in plan["core_input_hashes"].items()
        },
    }
    input_fp = canonical_hash(fp_obj)

    if (
        args.reuse_existing
        and summary_path.exists()
        and null_path.exists()
    ):
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            old.get("input_fingerprint") == input_fp
            and old.get("technical_valid", False)
        ):
            print(
                f"[offset {offset:+d}] reusing frozen B50.12 result",
                flush=True,
            )
            return old

    op = offset_paths(args, offset)
    require_files(op)

    # Resolve B50.10-frozen original representation inputs.
    frozen = {
        k: Path(v)
        for k, v in plan["frozen_source_paths"].items()
    }
    require_files(frozen)

    p4 = json.loads(
        frozen["b50_4_protocol"].read_text(encoding="utf-8")
    )
    training_fronts = pd.read_csv(
        frozen["b50_4_front_features"]
    )

    dims = [int(x) for x in p4["dimensions"]]
    graph_k = [int(x) for x in p4["graph_k_values"]]
    n_modes = int(p4["diffusion_modes"])

    with np.load(
        frozen["operator_stack"], allow_pickle=False
    ) as z:
        P_train = np.asarray(z["P_stack"], float)
        train_labels = [str(x) for x in z["labels"]]

    with np.load(op["external_stack"], allow_pickle=False) as z:
        P_ext = np.asarray(z["P_stack"], float)
        ext_labels = [str(x) for x in z["labels"]]

    plan_df = pd.read_csv(op["plan_csv"])
    plan_json = json.loads(
        op["plan_json"].read_text(encoding="utf-8")
    )
    extraction_manifest = json.loads(
        op["extraction_manifest"].read_text(encoding="utf-8")
    )
    saved_summary = json.loads(
        op["summary"].read_text(encoding="utf-8")
    )
    saved_tdf = pd.read_csv(op["transition_specificity"])
    matching_df = pd.read_csv(op["matching_audit"])

    integrity = {
        "applied_offset_exact": int(
            plan_json.get("applied_offset", -999999)
        ) == offset,
        "external_labels_exact": (
            ext_labels
            == plan_df["range_name"].astype(str).tolist()
        ),
        "operator_dim_exact": (
            P_ext.ndim == 2
            and P_train.ndim == 2
            and P_ext.shape[1] == P_train.shape[1]
        ),
        "extraction_labels_exact": bool(
            extraction_manifest.get(
                "labels_exact_against_plan", False
            )
        ),
        "saved_subrun_not_invalid": not str(
            saved_summary.get("final_verdict", "")
        ).startswith("INVALID_"),
    }
    integrity["pass"] = bool(all(integrity.values()))

    if not integrity["pass"]:
        summary = {
            "version": VERSION,
            "offset": offset,
            "input_fingerprint": input_fp,
            "technical_valid": False,
            "final_status": "INVALID_OFFSET_INTEGRITY",
            "integrity": integrity,
        }
        atomic_json(summary_path, summary)
        return summary

    tis, cmap, control_info = frozen_control_map(
        saved_tdf, matching_df, len(P_ext)
    )

    # Rebuild exact frozen external representation.
    # B50.7 import_b504() returns (module, front_fn).  build_external_configs()
    # expects the module itself, not the tuple.
    b504, _front_fn = b507.import_b504(frozen["b50_4_script"])
    train_mean, train_Vt, train_scores = b507.frozen_pca_basis(
        P_train
    )
    pca_parity = b507.projection_parity(
        P_train,
        train_mean,
        train_Vt,
        train_scores,
        dims,
    )

    configs = b507.build_external_configs(
        b504,
        P_ext,
        train_mean,
        train_Vt,
        dims,
        graph_k,
        n_modes,
    )

    # Full five-channel reproduction against saved B50.7 endpoint.
    (
        tis_full,
        config_ids,
        pct_full,
        score_full,
        cfg_full_df,
    ) = b507.build_replication_cube(
        configs,
        training_fronts,
        cmap,
    )

    if tis_full != tis:
        raise RuntimeError(
            f"offset {offset}: reconstructed transition list differs"
        )

    consensus_full = np.median(pct_full, axis=1)
    saved_phys_pct = (
        saved_tdf.sort_values(
            "transition_index", kind="stable"
        )["consensus_physical_percentile"]
        .to_numpy(float)
    )

    max_pct_err = float(
        np.max(
            np.abs(
                consensus_full[:, 0]
                - saved_phys_pct
            )
        )
    )
    reconstructed_full_effect = float(
        np.mean(consensus_full[:, 0]) - 0.5
    )
    saved_full_effect = float(
        saved_summary.get("primary_replication", {})
        .get("observed_mean_effect", float("nan"))
    )
    full_effect_err = abs(
        reconstructed_full_effect - saved_full_effect
    )

    full_repro_pass = bool(
        pca_parity.get("pass", False)
        and max_pct_err
        <= THRESHOLDS[
            "full_reproduction_percentile_max_abs_error"
        ]
        and full_effect_err
        <= THRESHOLDS[
            "full_reproduction_effect_max_abs_error"
        ]
    )

    atomic_json(
        odir_out / "b50_12_full_reproduction.json",
        {
            "offset": offset,
            "pca_parity": pca_parity,
            "max_consensus_physical_percentile_error": max_pct_err,
            "reconstructed_full_effect": reconstructed_full_effect,
            "saved_b50_7_full_effect": saved_full_effect,
            "full_effect_abs_error": full_effect_err,
            "pass": full_repro_pass,
        },
    )

    if not full_repro_pass:
        summary = {
            "version": VERSION,
            "offset": offset,
            "input_fingerprint": input_fp,
            "technical_valid": False,
            "final_status": "INVALID_FULL_B50_7_REPRODUCTION",
            "full_reproduction": {
                "max_percentile_error": max_pct_err,
                "effect_abs_error": full_effect_err,
                "pca_parity_pass": bool(
                    pca_parity.get("pass", False)
                ),
            },
        }
        atomic_json(summary_path, summary)
        return summary

    # Pair-core score across exact same matched candidate sets.
    n_t = len(tis)
    n_c = len(config_ids)
    n_k = 1 + control_info["controls_per_transition"]

    pair_pct = np.empty((n_t, n_c, n_k), float)
    pair_score_cube = np.empty_like(pair_pct)

    config_rows = []

    for ci, cid in enumerate(config_ids):
        c = configs[cid]
        train = (
            training_fronts[
                training_fronts["config_id"].astype(str)
                == str(cid)
            ]
            .sort_values("transition_index", kind="stable")
        )
        if train.empty:
            raise RuntimeError(
                f"offset {offset}: no training front rows for {cid}"
            )

        scale = b507.frozen_channel_scale(train)
        phys_pp = []

        for ti, i in enumerate(tis):
            dest = np.concatenate(
                [
                    np.asarray([i + 1], int),
                    cmap[i],
                ]
            )
            raw = b507.candidate_raw(
                c["X"],
                c["radius"],
                c["diff"],
                i,
                dest,
            )
            sc = pair_score(raw, scale)
            rp = b507.rank_percentiles(sc)

            pair_score_cube[ti, ci] = sc
            pair_pct[ti, ci] = rp
            phys_pp.append(float(rp[0]))

        effect = float(np.mean(phys_pp) - 0.5)
        config_rows.append({
            "config_id": cid,
            "dimension": int(c["dimension"]),
            "graph_k": int(c["graph_k"]),
            "mean_physical_percentile": float(
                np.mean(phys_pp)
            ),
            "median_physical_percentile": float(
                np.median(phys_pp)
            ),
            "mean_percentile_effect": effect,
            "negative_effect": effect < 0.0,
        })

    cfg_df = pd.DataFrame(config_rows)
    atomic_csv(
        odir_out / "b50_12_pair_config_effects.csv",
        cfg_df,
    )

    consensus_pair = np.median(pair_pct, axis=1)

    primary, null = matched_identity_null_lower(
        consensus_pair,
        args.randomization_replicates,
        args.seed + offset * 1009 + 1000,
    )
    atomic_csv(
        null_path,
        pd.DataFrame({
            "replicate": np.arange(len(null)),
            "mean_percentile_effect": null,
        }),
    )

    # Per-transition artifact.
    rows = []
    for ti, i in enumerate(tis):
        rows.append({
            "transition_index": i,
            "from_range": str(ext_labels[i]),
            "to_range": str(ext_labels[i + 1]),
            "consensus_pair_physical_percentile": float(
                consensus_pair[ti, 0]
            ),
            "pair_percentile_effect": float(
                consensus_pair[ti, 0] - 0.5
            ),
            "median_pair_physical_score": float(
                np.median(pair_score_cube[ti, :, 0])
            ),
            "median_pair_control_score": float(
                np.median(pair_score_cube[ti, :, 1:])
            ),
            "control_dest_indices": ",".join(
                str(int(x)) for x in cmap[i]
            ),
        })
    transition_pair_df = pd.DataFrame(rows)
    atomic_csv(
        odir_out / "b50_12_pair_transition_specificity.csv",
        transition_pair_df,
    )

    # Thirds.
    edges = np.linspace(0, n_t, 4, dtype=int)
    third_rows = []
    negative_thirds = 0
    nominal_thirds = 0

    for third in range(3):
        mask = np.zeros(n_t, bool)
        mask[edges[third]:edges[third + 1]] = True
        res, _ = matched_identity_null_lower(
            consensus_pair,
            args.randomization_replicates,
            args.seed + offset * 1009 + 2000 + third,
            mask=mask,
        )
        neg = res["observed_mean_effect"] < 0.0
        nominal = bool(
            res["observed_mean_effect"]
            <= THRESHOLDS["third_effect_max"]
            and res["p_lower"]
            <= THRESHOLDS["third_p_lower_max"]
        )
        negative_thirds += int(neg)
        nominal_thirds += int(nominal)

        idx = np.asarray(tis)[mask]
        third_rows.append({
            "third": third + 1,
            "transition_count": int(mask.sum()),
            "first_transition_index": int(idx[0]),
            "last_transition_index": int(idx[-1]),
            **res,
            "negative_direction": bool(neg),
            "nominal_replication_pass": nominal,
        })

    thirds_df = pd.DataFrame(third_rows)
    atomic_csv(
        odir_out / "b50_12_pair_thirds.csv",
        thirds_df,
    )

    thirds_pass = bool(
        negative_thirds
        >= THRESHOLDS["third_negative_required"]
        and nominal_thirds
        >= THRESHOLDS["third_nominal_required"]
    )

    negative_config_fraction = float(
        np.mean(
            cfg_df["mean_percentile_effect"].to_numpy(float)
            < 0.0
        )
    )
    config_pass = bool(
        negative_config_fraction
        >= THRESHOLDS["config_negative_fraction_min"]
    )

    ref_effect = float(
        plan["pair_core"]["reference_effect_offset0"]
    )
    effect = float(primary["observed_mean_effect"])
    retention_vs_ref = (
        abs(effect) / abs(ref_effect)
        if abs(ref_effect) > 1e-15
        else float("nan")
    )

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "offset": offset,
        "input_fingerprint": input_fp,
        "technical_valid": True,
        "integrity": integrity,
        "control_identity": control_info,
        "full_reproduction": {
            "pca_parity_pass": bool(
                pca_parity.get("pass", False)
            ),
            "max_consensus_physical_percentile_error": (
                max_pct_err
            ),
            "reconstructed_full_effect": (
                reconstructed_full_effect
            ),
            "saved_b50_7_full_effect": saved_full_effect,
            "effect_abs_error": full_effect_err,
            "pass": full_repro_pass,
        },
        "pair_core": {
            "channels": PAIR_CHANNELS,
            **primary,
            "effect_retention_vs_b50_11_offset0_pair": (
                retention_vs_ref
            ),
            "negative_direction": effect < 0.0,
        },
        "config_convergence": {
            "negative_effect_fraction": (
                negative_config_fraction
            ),
            "pass": config_pass,
        },
        "thirds_robustness": {
            "negative_thirds": negative_thirds,
            "nominally_replicated_thirds": nominal_thirds,
            "pass": thirds_pass,
        },
        "final_status": "VALID_PAIR_CORE_OFFSET_RESULT",
    }
    atomic_json(summary_path, summary)

    return summary


# ---------------------------------------------------------------------------
# AUDIT / aggregate
# ---------------------------------------------------------------------------

def audit_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plan = ensure_plan(args)

    hashes_ok, hash_df = verify_plan_hashes(args, plan)
    print("=== B50.12 FROZEN PAIR-CORE CROSS-OFFSET REPLICATION ===")
    print("frozen input hashes exact :", hashes_ok)
    if not hashes_ok:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_FROZEN_INPUTS",
            "reason": "One or more frozen B50.12 input artifacts changed after PLAN.",
        }
        atomic_json(out / "b50_12_summary.json", summary)
        atomic_text(
            out / "b50_12_verdict.txt",
            "FINAL VERDICT: INVALID_FROZEN_INPUTS\n",
        )
        return summary

    lineage = lineage_gate(args)
    print("lineage PASS              :", lineage["pass"])
    if not lineage["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_LINEAGE",
            "lineage": lineage,
        }
        atomic_json(out / "b50_12_summary.json", summary)
        atomic_text(
            out / "b50_12_verdict.txt",
            "FINAL VERDICT: INVALID_LINEAGE\n",
        )
        return summary

    b507 = import_b507(Path(args.b50_7_script))

    offset_summaries = []
    nulls = []

    for offset in OFFSETS:
        print(
            f"\n--- pair-core offset {offset:+d} ---",
            flush=True,
        )
        s = reconstruct_offset(
            args, plan, b507, offset
        )
        offset_summaries.append(s)

        if not s.get("technical_valid", False):
            continue

        ndf = pd.read_csv(
            out
            / offset_dir_name(offset)
            / "b50_12_pair_null.csv"
        )
        nulls.append(
            ndf["mean_percentile_effect"].to_numpy(float)
        )

    invalid_offsets = [
        int(s.get("offset"))
        for s in offset_summaries
        if not s.get("technical_valid", False)
    ]

    if invalid_offsets:
        summary = {
            "version": VERSION,
            "finished": now_s(),
            "invalid_offsets": invalid_offsets,
            "offset_summaries": offset_summaries,
            "final_verdict": "INVALID_OFFSET_REPRODUCTION",
            "reason": (
                "At least one offset failed frozen technical or "
                "full-score reproduction gates."
            ),
        }
        atomic_json(out / "b50_12_summary.json", summary)
        atomic_text(
            out / "b50_12_verdict.txt",
            "FINAL VERDICT: INVALID_OFFSET_REPRODUCTION\n"
            + f"invalid offsets: {invalid_offsets}\n",
        )
        return summary

    rows = []
    for s in offset_summaries:
        pair = s["pair_core"]
        cfg = s["config_convergence"]
        thirds = s["thirds_robustness"]
        full = s["full_reproduction"]

        rows.append({
            "offset": int(s["offset"]),
            "mean_physical_percentile": float(
                pair["observed_mean_percentile"]
            ),
            "effect": float(
                pair["observed_mean_effect"]
            ),
            "p_lower": float(pair["p_lower"]),
            "p_two_sided": float(pair["p_two_sided"]),
            "negative": bool(
                pair["observed_mean_effect"] < 0.0
            ),
            "retention_vs_b50_11_pair": float(
                pair[
                    "effect_retention_vs_b50_11_offset0_pair"
                ]
            ),
            "negative_config_fraction": float(
                cfg["negative_effect_fraction"]
            ),
            "config_pass": bool(cfg["pass"]),
            "negative_thirds": int(
                thirds["negative_thirds"]
            ),
            "nominal_thirds": int(
                thirds[
                    "nominally_replicated_thirds"
                ]
            ),
            "thirds_pass": bool(thirds["pass"]),
            "full_reproduction_pass": bool(
                full["pass"]
            ),
            "full_reproduction_percentile_error": float(
                full[
                    "max_consensus_physical_percentile_error"
                ]
            ),
            "full_reproduction_effect_error": float(
                full["effect_abs_error"]
            ),
        })

    df = pd.DataFrame(rows).sort_values(
        "offset", kind="stable"
    ).reset_index(drop=True)

    # Holm across the five predeclared offset pair-core hypotheses.
    df["holm_p_lower"] = holm_adjust(
        df["p_lower"].to_numpy(float)
    )
    df["holm_directional_pass"] = (
        df["holm_p_lower"]
        <= THRESHOLDS["holm_directional_p_max"]
    )
    df["nominal_offset_replication"] = (
        (df["effect"] <= args.nominal_effect_max)
        & df["holm_directional_pass"]
    )
    df["material_positive_reversal"] = (
        df["effect"]
        >= THRESHOLDS["material_positive_reversal_min"]
    )

    atomic_csv(out / "b50_12_offset_effects.csv", df)

    effects = df["effect"].to_numpy(float)
    negative_count = int(df["negative"].sum())
    nominal_count = int(
        df["nominal_offset_replication"].sum()
    )
    config_pass_count = int(df["config_pass"].sum())
    thirds_pass_count = int(df["thirds_pass"].sum())
    reversal_count = int(
        df["material_positive_reversal"].sum()
    )

    mean_effect = float(np.mean(effects))
    median_effect = float(np.median(effects))
    sd_effect = float(np.std(effects, ddof=1))
    q25 = float(np.quantile(effects, 0.25))
    q75 = float(np.quantile(effects, 0.75))
    iqr = q75 - q25
    effect_range = float(
        np.max(effects) - np.min(effects)
    )

    ref_effect = float(
        plan["pair_core"]["reference_effect_offset0"]
    )
    median_retention = (
        abs(median_effect) / abs(ref_effect)
        if abs(ref_effect) > 1e-15
        else float("nan")
    )

    gates = {
        "all_five_negative": (
            negative_count
            >= THRESHOLDS["offset_negative_required"]
        ),
        "at_least_four_nominal": (
            nominal_count
            >= THRESHOLDS["offset_nominal_required"]
        ),
        "median_effect_at_least_minus_0_05": (
            median_effect <= args.median_effect_max
        ),
        "at_least_four_config_pass": (
            config_pass_count
            >= THRESHOLDS["config_pass_required"]
        ),
        "at_least_four_thirds_pass": (
            thirds_pass_count
            >= THRESHOLDS[
                "thirds_pass_offsets_required"
            ]
        ),
        "no_material_positive_reversal": (
            reversal_count == 0
        ),
        "all_technical_reproduction_pass": bool(
            df["full_reproduction_pass"].all()
        ),
    }
    full_pass = bool(all(gates.values()))

    # Secondary correlated pooled null across five offsets.
    min_n = min(len(x) for x in nulls)
    null_matrix = np.column_stack(
        [x[:min_n] for x in nulls]
    )
    pooled_null = np.mean(null_matrix, axis=1)
    pooled_p_lower = float(
        (1 + np.sum(pooled_null <= mean_effect))
        / (min_n + 1)
    )
    atomic_csv(
        out / "b50_12_pooled_null.csv",
        pd.DataFrame({
            "replicate": np.arange(min_n),
            "mean_five_offset_null_effect": pooled_null,
        }),
    )

    if full_pass:
        verdict = "PAIR_CORE_CROSS_OFFSET_REPLICATED"
        reason = (
            "The frozen turn_change+density_scale_change pair remains "
            "negative at all five offset phases, at least four offsets "
            "pass the frozen amplitude plus five-test Holm directional "
            "gate, the median effect is <= -0.05, and representation/thirds "
            "robustness converges across at least four offsets."
        )
    elif negative_count == 5 and reversal_count == 0:
        verdict = (
            "PAIR_CORE_CROSS_OFFSET_DIRECTION_STABLE_WEAK_AMPLITUDE"
        )
        reason = (
            "All five offsets preserve the frozen negative pair-core "
            "direction, but one or more amplitude, multiplicity-corrected "
            "significance, config-convergence, thirds, or median-effect "
            "gates are too weak for the full replication verdict."
        )
    elif reversal_count > 0:
        verdict = "PAIR_CORE_CROSS_OFFSET_PHASE_SENSITIVE"
        reason = (
            "At least one offset shows a material positive reversal "
            "under the frozen pair-core score."
        )
    elif negative_count >= 4:
        verdict = "PAIR_CORE_CROSS_OFFSET_PARTIAL_REPLICATION"
        reason = (
            "Most offset phases preserve the negative pair-core "
            "direction, but the complete frozen cross-offset contract "
            "does not pass."
        )
    elif (
        abs(median_effect)
        <= THRESHOLDS[
            "null_compatible_abs_median_effect_max"
        ]
        and nominal_count
        < THRESHOLDS["offset_nominal_required"]
    ):
        verdict = "PAIR_CORE_CROSS_OFFSET_NULL_COMPATIBLE"
        reason = (
            "The cross-offset median pair-core effect is close to null "
            "and fewer than four offsets pass the frozen nominal gate."
        )
    else:
        verdict = "PAIR_CORE_CROSS_OFFSET_PARTIAL_REPLICATION"
        reason = (
            "The frozen pair-core result is mixed across offset phases."
        )

    effect_summary = {
        "b50_11_offset0_pair_effect": ref_effect,
        "five_offset_mean_effect": mean_effect,
        "five_offset_median_effect": median_effect,
        "five_offset_sd_effect": sd_effect,
        "five_offset_iqr_effect": iqr,
        "five_offset_range_effect": effect_range,
        "five_offset_min_effect": float(np.min(effects)),
        "five_offset_max_effect": float(np.max(effects)),
        "median_abs_effect_retention_vs_b50_11_pair": (
            median_retention
        ),
    }

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "pair_core": PAIR_CHANNELS,
        "same_underlying_prime_corpus": True,
        "independent_corpus_replication": False,
        "offset_full_score_outcomes_preexisted_b50_12": True,
        "lineage": lineage,
        "offsets": OFFSETS,
        "offset_rows": df.to_dict(orient="records"),
        "counts": {
            "negative_offsets": negative_count,
            "nominal_offset_replications": nominal_count,
            "config_pass_offsets": config_pass_count,
            "thirds_pass_offsets": thirds_pass_count,
            "material_positive_reversals": reversal_count,
        },
        "phase_gates": gates,
        "effect_invariance": effect_summary,
        "secondary_pooled_null": {
            "replicate_count": min_n,
            "observed_mean_five_offset_effect": mean_effect,
            "null_mean": float(np.mean(pooled_null)),
            "null_sd": float(np.std(pooled_null)),
            "p_lower": pooled_p_lower,
            "mandatory_gate": False,
            "dependence_warning": (
                "offsets overlap in the same deterministic prime corpus; "
                "pooled null is a secondary descriptive diagnostic"
            ),
        },
        "thresholds": THRESHOLDS,
        "final_verdict": verdict,
        "final_reason": reason,
        "scientific_boundary": (
            "Frozen pair-core cross-offset representation replication "
            "within one prime corpus. Not independent-corpus replication, "
            "not a pristine blinded prospective test, not B59 validation, "
            "not individual-prime prediction, and not RH/theorem evidence."
        ),
    }
    atomic_json(out / "b50_12_summary.json", summary)

    # Plots.
    plt.figure(figsize=(9, 5))
    plt.plot(df["offset"], df["effect"], marker="o")
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(
        args.nominal_effect_max,
        linestyle="--",
        linewidth=0.8,
    )
    plt.axhline(
        ref_effect,
        linestyle=":",
        linewidth=1.0,
    )
    plt.xlabel("offset segments")
    plt.ylabel("pair-core mean physical percentile effect vs 0.5")
    plt.title(
        "B50.12 frozen turn+density pair across offset phases"
    )
    plt.tight_layout()
    plt.savefig(
        out / "b50_12_pair_effect_by_offset.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(
        df["offset"],
        df["negative_config_fraction"],
        marker="o",
        label="negative config fraction",
    )
    plt.plot(
        df["offset"],
        df["nominal_thirds"] / 3.0,
        marker="o",
        label="nominal thirds fraction",
    )
    plt.ylim(-0.02, 1.02)
    plt.xlabel("offset segments")
    plt.ylabel("fraction")
    plt.title("B50.12 pair-core convergence diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out / "b50_12_pair_convergence_by_offset.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(pooled_null, bins=45)
    plt.axvline(mean_effect, linewidth=2)
    plt.xlabel("mean five-offset effect under matched-identity null")
    plt.ylabel("count")
    plt.title("B50.12 pooled cross-offset null [secondary]")
    plt.tight_layout()
    plt.savefig(
        out / "b50_12_pooled_null.png",
        dpi=180,
    )
    plt.close()

    report = f"""B50.12 — Frozen Pair-Core Cross-Offset Replication
==================================================

Interpretation scope
--------------------
Frozen pair selected in B50.11: YES
Pair reselection in B50.12: NO
Exact B50.7 matched controls reused: YES
New control matching: NO
New prime-data extraction: NO
Same underlying prime corpus: YES
Independent-corpus replication: NO
Pristine blinded prospective test: NO

Frozen pair core
----------------
channels                              = {PAIR_CHANNELS}
B50.11 offset-0 pair effect           = {ref_effect}
B50.11 pair retention vs full         = {plan['pair_core']['reference_retention_vs_full']}

Technical reproduction
----------------------
all frozen input hashes exact         = {hashes_ok}
all five full B50.7 endpoints exact   = {bool(df['full_reproduction_pass'].all())}
max full-percentile reproduction err  = {float(df['full_reproduction_percentile_error'].max())}
max full-effect reproduction err      = {float(df['full_reproduction_effect_error'].max())}

Per-offset pair-core results
----------------------------
{df.to_string(index=False)}

Cross-offset counts
-------------------
negative offsets                      = {negative_count}/5
nominal offset replications           = {nominal_count}/5
config-convergence passes             = {config_pass_count}/5
thirds-robustness passes              = {thirds_pass_count}/5
material positive reversals           = {reversal_count}

Effect invariance
-----------------
mean five-offset effect               = {mean_effect}
median five-offset effect             = {median_effect}
SD five-offset effect                 = {sd_effect}
IQR five-offset effect                = {iqr}
range five-offset effect              = {effect_range}
min / max five-offset effect          = {float(np.min(effects))} / {float(np.max(effects))}
median |effect| retention vs B50.11   = {median_retention}

Frozen mandatory gates
----------------------
5/5 negative                          = {gates['all_five_negative']}
>=4/5 nominal                         = {gates['at_least_four_nominal']}
  nominal effect <= -0.04
  Holm directional p <= 0.05 over 5 offsets
median effect <= -0.05                = {gates['median_effect_at_least_minus_0_05']}
>=4/5 config convergence              = {gates['at_least_four_config_pass']}
>=4/5 thirds robustness              = {gates['at_least_four_thirds_pass']}
no material positive reversal         = {gates['no_material_positive_reversal']}
all technical reproduction pass       = {gates['all_technical_reproduction_pass']}

Secondary pooled-null diagnostic
--------------------------------
replicates                            = {min_n}
observed mean five-offset effect      = {mean_effect}
null mean                             = {float(np.mean(pooled_null))}
null SD                               = {float(np.std(pooled_null))}
p_lower                               = {pooled_p_lower}
mandatory gate                        = False
dependence                            = SAME CORPUS / OVERLAPPING OFFSET WINDOWS

FINAL VERDICT: {verdict}

REASON:
{reason}

Interpretation boundary
-----------------------
B50.12 tests whether the B50.11-selected two-channel score survives changes of
window phase. The pair is frozen and no controls are changed, but the five
offset sequences are derived from the same deterministic prime corpus and
their full five-channel outcomes existed before B50.12. The result is therefore
a confirmatory cross-phase representation replication, not an independent
source replication or pristine blinded prospective test.

No B59 selector, individual-prime prediction, RH claim, or theorem claim is
made.

Key outputs
-----------
b50_12_protocol_manifest.json
b50_12_plan.txt
b50_12_offset_input_plan.csv
b50_12_frozen_input_audit.csv
b50_12_offset_effects.csv
b50_12_pooled_null.csv
b50_12_summary.json
b50_12_verdict.txt
b50_12_pair_effect_by_offset.png
b50_12_pair_convergence_by_offset.png
b50_12_pooled_null.png

Each offset output directory also contains:
b50_12_full_reproduction.json
b50_12_pair_transition_specificity.csv
b50_12_pair_config_effects.csv
b50_12_pair_null.csv
b50_12_pair_thirds.csv
b50_12_offset_summary.json
"""
    atomic_text(out / "b50_12_verdict.txt", report)

    print()
    print("=== B50.12 RESULT ===")
    print("negative offsets :", f"{negative_count}/5")
    print("nominal offsets  :", f"{nominal_count}/5")
    print("config passes    :", f"{config_pass_count}/5")
    print("thirds passes    :", f"{thirds_pass_count}/5")
    print("median effect    :", f"{median_effect:.8f}")
    print("pooled p_lower   :", pooled_p_lower, "[secondary]")
    print("FINAL VERDICT    :", verdict)
    print("verdict          :", out / "b50_12_verdict.txt")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="B50.12 Frozen Pair-Core Cross-Offset Replication"
    )
    p.add_argument(
        "--mode",
        choices=["plan", "audit", "all"],
        default="plan",
    )

    p.add_argument("--b50-7-script", required=True)
    p.add_argument("--b50-10-dir", required=True)
    p.add_argument("--b50-11-dir", required=True)
    p.add_argument("--b50-7-reference-dir", required=True)
    p.add_argument("--b50-8-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument(
        "--randomization-replicates",
        type=int,
        default=DEFAULT_REPS,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    p.add_argument(
        "--reuse-existing",
        action="store_true",
    )

    # Exposed only for audit transparency. Changing either causes PLAN fail.
    p.add_argument(
        "--nominal-effect-max",
        type=float,
        default=THRESHOLDS["nominal_effect_max"],
    )
    p.add_argument(
        "--median-effect-max",
        type=float,
        default=THRESHOLDS["median_effect_max"],
    )

    a = p.parse_args()

    if a.randomization_replicates < 999:
        p.error("--randomization-replicates must be >=999")

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
            raise RuntimeError(
                "B50.12 PLAN failed; AUDIT not started."
            )
        audit_mode(args)
    else:
        raise ValueError(args.mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
