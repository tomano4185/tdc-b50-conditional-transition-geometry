#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TDC TOR A / B50.11
Channel Redundancy & Interaction Audit
======================================

Purpose
-------
B50.10 established:

    FINAL VERDICT:
    CHANNEL_ASSOCIATION_WITHOUT_SCORE_LOCALIZATION

with exact reproduction of the frozen B50.6 adjacency-specificity endpoint and
three Holm-confirmed raw channels (acceleration, turn_change,
density_scale_change), but no single leave-one-out removal attenuated the full
composite score by the predeclared 25% localization threshold.

B50.11 asks the next mechanistic question without changing controls, geometry,
scaling, or the B50.6 endpoint:

    Which combinations of the five frozen B50.4 channels are sufficient to
    recover the replicated full composite effect, and how redundant /
    interactive are the channels inside the score?

The five frozen channels are:

    speed
    acceleration
    turn_change
    density_scale_change
    diffusion_step

B50.11 evaluates ALL 2^5 - 1 = 31 non-empty channel subsets under the exact
B50.6 positive-robust-z RMS scoring rule, using the exact B50.6 matched control
destinations already frozen inside the B50.10 mechanism cube.

It then computes:

1. all-subset matched-identity randomization tests;
2. Holm correction over the complete 31-subset family;
3. minimal sufficient subsets under a frozen retention criterion;
4. exact 5-player Shapley decomposition of the observed percentile effect;
5. global pairwise Shapley interaction indices;
6. pair-only synergy diagnostics;
7. subset-size envelopes and contiguous-thirds robustness for qualifying
   minimal subsets;
8. exact reconstruction / efficiency gates.

Scientific boundary
-------------------
This is a score-mechanism / interaction audit of the already replicated B50.6
same-corpus adjacency-specificity effect. It does NOT create new controls, does
NOT test a new prime corpus, does NOT validate B59, does NOT predict individual
primes, and is NOT RH/theorem evidence.

Shapley semantics
-----------------
The characteristic function is:

    v(S) = mean physical matched-percentile(S) - 0.5

where S is a channel subset and the score for S is the exact B50.6 composite
restricted to channels in S.

For the empty coalition:

    v(empty) = 0

because an empty score assigns the same constant score to every candidate,
giving tied percentile 0.5 and therefore zero centered effect.

Thus Shapley efficiency is exact by construction:

    sum_i phi_i = v(all five) - v(empty) = v(all five).

Operational properties
----------------------
- PLAN -> AUDIT split.
- SHA-256 frozen B50.10 artifact manifest.
- Exact reuse of B50.10 mechanism cube.
- All 31 subsets declared before observing B50.11 results.
- Atomic writes.
- Resumable subset-score cache.
- No post-hoc subset search outside the declared exhaustive family.

Recommended workflow
--------------------
PLAN:
    python tdc_tor_a_b50_11_channel_redundancy_interaction_audit.py ^
      --mode plan ^
      --b50-10-dir D:\...\b50_10_transition_geometry_mechanism ^
      --output-dir D:\...\b50_11_channel_interactions

AUDIT:
    python tdc_tor_a_b50_11_channel_redundancy_interaction_audit.py ^
      --mode audit ^
      --b50-10-dir D:\...\b50_10_transition_geometry_mechanism ^
      --output-dir D:\...\b50_11_channel_interactions ^
      --reuse-existing
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "B50.11_channel_redundancy_interaction_audit_v1"

CHANNELS = [
    "speed",
    "acceleration",
    "turn_change",
    "density_scale_change",
    "diffusion_step",
]

EXPECTED_B5010_VERDICT = "CHANNEL_ASSOCIATION_WITHOUT_SCORE_LOCALIZATION"
EXPECTED_B506_VERDICT = "PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED"

# Frozen before B50.11 results.
THRESHOLDS = {
    # Exact reconstruction gates.
    "full_effect_abs_error_max": 1e-12,
    "shapley_efficiency_abs_error_max": 1e-12,

    # A subset is "sufficient" only if:
    # - same sign as frozen full effect,
    # - retains at least 90% of |full effect|,
    # - passes Holm correction across all 31 subsets using directional p.
    "subset_retention_min": 0.90,
    "subset_holm_directional_p_max": 0.05,

    # Stronger descriptive "near-full" mark.
    "subset_near_full_retention_min": 0.95,

    # Thirds robustness for minimal sufficient subsets.
    "third_same_direction_required": 3,
    "third_nominal_required": 2,
    "third_retention_min": 0.70,
    "third_directional_p_max": 0.05,

    # Shapley concentration is descriptive only.
    "shapley_top2_abs_share_concentrated_min": 0.75,

    # Pair interaction magnitude label (descriptive only).
    "pair_interaction_material_abs_min": 0.01,
}

DEFAULT_REPS = 5000
DEFAULT_SEED = 501011


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
        return float(x)
    except Exception:
        return default


def rank_percentiles(x: Sequence[float]) -> np.ndarray:
    """
    Exact B50.6 / B50.10 rank semantics:
        (# lower + 0.5 * # tied others) / (n - 1)
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
    Candidate 0 is the physical next state; remaining candidates are the exact
    B50.6 matched controls.
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


def subset_key(indices: Sequence[int]) -> str:
    if not indices:
        return "EMPTY"
    return "+".join(CHANNELS[i] for i in indices)


def subset_mask(indices: Sequence[int]) -> int:
    m = 0
    for i in indices:
        m |= 1 << int(i)
    return m


def indices_from_mask(mask: int) -> Tuple[int, ...]:
    return tuple(i for i in range(len(CHANNELS)) if mask & (1 << i))


def all_nonempty_subsets() -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for r in range(1, len(CHANNELS) + 1):
        out.extend(itertools.combinations(range(len(CHANNELS)), r))
    return out


def score_raw_subset(
    raw_matrix: np.ndarray,
    medians: np.ndarray,
    scales: np.ndarray,
    include_indices: Sequence[int],
) -> np.ndarray:
    """
    Exact B50.6 positive robust-z RMS composite restricted to include_indices.
    """
    raw_matrix = np.asarray(raw_matrix, float)
    idx = list(include_indices)
    if not idx:
        return np.zeros(raw_matrix.shape[0], dtype=float)

    Z = []
    for ch in idx:
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
# B50.10 inputs / lineage
# ---------------------------------------------------------------------------

def b5010_paths(args) -> Dict[str, Path]:
    d = Path(args.b50_10_dir)
    return {
        "b50_10_protocol_manifest": d / "b50_10_protocol_manifest.json",
        "b50_10_summary": d / "b50_10_summary.json",
        "b50_10_cache_manifest": d / "b50_10_cache_manifest.json",
        "b50_10_mechanism_cube": d / "b50_10_mechanism_cube.npz",
        "b50_10_reproduction_audit": d / "b50_10_reproduction_audit.json",
        "b50_10_channel_effects": d / "b50_10_channel_effects.csv",
        "b50_10_leave_one_out": d / "b50_10_leave_one_out.csv",
    }


def require_paths(paths: Mapping[str, Path]) -> None:
    missing = [(k, p) for k, p in paths.items() if not p.exists()]
    if missing:
        msg = "\n".join(f"{k}: {p}" for k, p in missing)
        raise FileNotFoundError("Missing B50.11 input(s):\n" + msg)


def lineage_gate(paths: Mapping[str, Path]) -> Dict[str, Any]:
    summary = json.loads(
        paths["b50_10_summary"].read_text(encoding="utf-8")
    )
    protocol = json.loads(
        paths["b50_10_protocol_manifest"].read_text(encoding="utf-8")
    )
    cache_manifest = json.loads(
        paths["b50_10_cache_manifest"].read_text(encoding="utf-8")
    )
    repro = json.loads(
        paths["b50_10_reproduction_audit"].read_text(encoding="utf-8")
    )

    verdict = str(summary.get("final_verdict", ""))
    b506 = str(summary.get("lineage", {}).get("b50_6_verdict", ""))
    repro_pass = bool(summary.get("reproduction", {}).get("pass", False))
    repro_file_pass = bool(repro.get("pass", False))
    frozen_hashes = bool(summary.get("frozen_input_hashes_exact", False))

    full_effect = safe_float(
        summary.get("reproduction", {}).get("reconstructed_full_effect")
    )
    b506_effect = safe_float(
        summary.get("reproduction", {}).get("frozen_b50_6_full_effect")
    )

    cache_path = paths["b50_10_mechanism_cube"]
    cache_hash_actual = sha256_file(cache_path)
    cache_hash_frozen = str(cache_manifest.get("cache_sha256", ""))
    cache_hash_exact = cache_hash_actual == cache_hash_frozen

    channels_exact = (
        list(cache_manifest.get("channels", [])) == CHANNELS
        and list(protocol.get("frozen_b50_4", {}).get("channels", []))
        == CHANNELS
    )

    pass_ = bool(
        verdict == EXPECTED_B5010_VERDICT
        and b506 == EXPECTED_B506_VERDICT
        and repro_pass
        and repro_file_pass
        and frozen_hashes
        and np.isfinite(full_effect)
        and np.isfinite(b506_effect)
        and abs(full_effect - b506_effect)
        <= THRESHOLDS["full_effect_abs_error_max"]
        and full_effect < 0
        and cache_hash_exact
        and channels_exact
    )

    return {
        "b50_10_verdict": verdict,
        "b50_10_verdict_ok": verdict == EXPECTED_B5010_VERDICT,
        "b50_6_verdict": b506,
        "b50_6_verdict_ok": b506 == EXPECTED_B506_VERDICT,
        "b50_10_reproduction_pass": repro_pass,
        "b50_10_reproduction_file_pass": repro_file_pass,
        "b50_10_frozen_input_hashes_exact": frozen_hashes,
        "full_effect": full_effect,
        "b50_6_effect": b506_effect,
        "full_effect_exact": bool(
            np.isfinite(full_effect)
            and np.isfinite(b506_effect)
            and abs(full_effect - b506_effect)
            <= THRESHOLDS["full_effect_abs_error_max"]
        ),
        "cache_hash_frozen": cache_hash_frozen,
        "cache_hash_actual": cache_hash_actual,
        "cache_hash_exact": cache_hash_exact,
        "channels_exact": channels_exact,
        "pass": pass_,
    }


def load_cube(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        req = {
            "transition_indices",
            "config_ids",
            "dest_indices",
            "raw_cube",
            "cosine_cube",
            "medians",
            "scales",
            "saved_physical_front_scores",
        }
        miss = req - set(z.files)
        if miss:
            raise ValueError(
                f"B50.10 mechanism cube missing arrays: {sorted(miss)}"
            )

        out = {
            "transition_indices": np.asarray(
                z["transition_indices"], int
            ),
            "config_ids": [str(x) for x in z["config_ids"]],
            "dest_indices": np.asarray(z["dest_indices"], int),
            "raw_cube": np.asarray(z["raw_cube"], float),
            "cosine_cube": np.asarray(z["cosine_cube"], float),
            "medians": np.asarray(z["medians"], float),
            "scales": np.asarray(z["scales"], float),
            "saved_physical_front_scores": np.asarray(
                z["saved_physical_front_scores"], float
            ),
        }
    return out


def cube_integrity(cube: Dict[str, Any]) -> Dict[str, Any]:
    raw = cube["raw_cube"]
    T = len(cube["transition_indices"])
    C = len(cube["config_ids"])
    K = cube["dest_indices"].shape[1]

    shape_ok = bool(
        raw.shape == (T, C, K, len(CHANNELS))
        and cube["medians"].shape == (C, len(CHANNELS))
        and cube["scales"].shape == (C, len(CHANNELS))
        and cube["cosine_cube"].shape == (T, C, K)
    )
    transition_unique = (
        len(np.unique(cube["transition_indices"])) == T
    )
    physical_dest_exact = bool(
        np.array_equal(
            cube["dest_indices"][:, 0],
            cube["transition_indices"] + 1,
        )
    )
    finite_raw = bool(np.isfinite(raw).all())

    return {
        "transition_count": T,
        "config_count": C,
        "candidate_count": K,
        "control_count": K - 1,
        "raw_cube_shape": list(raw.shape),
        "shape_ok": shape_ok,
        "transition_unique": transition_unique,
        "physical_dest_exact": physical_dest_exact,
        "finite_raw": finite_raw,
        "pass": bool(
            shape_ok
            and transition_unique
            and physical_dest_exact
            and finite_raw
        ),
    }


# ---------------------------------------------------------------------------
# PLAN
# ---------------------------------------------------------------------------

def protocol_modified(args) -> bool:
    return bool(
        args.subset_retention_min
        != THRESHOLDS["subset_retention_min"]
        or args.subset_holm_p_max
        != THRESHOLDS["subset_holm_directional_p_max"]
    )


def plan_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = b5010_paths(args)
    require_paths(paths)

    lineage = lineage_gate(paths)
    cube = load_cube(paths["b50_10_mechanism_cube"])
    integrity = cube_integrity(cube)

    subsets = all_nonempty_subsets()
    subset_rows = []
    for rank, S in enumerate(subsets, start=1):
        subset_rows.append({
            "subset_order": rank,
            "subset_mask": subset_mask(S),
            "subset_key": subset_key(S),
            "subset_size": len(S),
            "channels": "|".join(CHANNELS[i] for i in S),
            "is_full": len(S) == len(CHANNELS),
        })
    subset_plan_df = pd.DataFrame(subset_rows)
    atomic_csv(out / "b50_11_subset_plan.csv", subset_plan_df)

    hashes = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
    }

    plan_pass = bool(
        lineage["pass"]
        and integrity["pass"]
        and len(subsets) == 31
        and not protocol_modified(args)
    )

    manifest = {
        "version": VERSION,
        "created": now_s(),
        "scientific_role": (
            "exhaustive 31-subset redundancy / interaction audit of the "
            "frozen B50.10 / B50.6 composite score"
        ),
        "b50_10_lineage": lineage,
        "cube_integrity": integrity,
        "inputs": hashes,
        "channels": CHANNELS,
        "subset_family": {
            "nonempty_subset_count": len(subsets),
            "expected_count": 31,
            "all_subsets_predeclared": True,
            "empty_coalition_effect": 0.0,
            "empty_coalition_semantics": (
                "constant tied candidate score -> percentile 0.5 -> "
                "centered effect 0"
            ),
        },
        "subset_sufficiency_gate": {
            "same_sign_as_full": True,
            "absolute_effect_retention_min": args.subset_retention_min,
            "holm_directional_p_max": args.subset_holm_p_max,
            "multiplicity_family_size": 31,
        },
        "shapley_protocol": {
            "players": CHANNELS,
            "characteristic_function": (
                "v(S)=mean physical matched-percentile(S)-0.5"
            ),
            "empty_value": 0.0,
            "efficiency_gate_abs_error_max": (
                THRESHOLDS["shapley_efficiency_abs_error_max"]
            ),
        },
        "pair_interaction_protocol": {
            "global_index": (
                "Shapley interaction index over all contexts "
                "S subset N\\{i,j}"
            ),
            "pair_only_synergy": (
                "v({i,j})-v({i})-v({j})+v(empty)"
            ),
            "role": "descriptive interaction decomposition",
        },
        "thirds_protocol": {
            "applies_to": "minimal sufficient subsets",
            "same_direction_required": (
                THRESHOLDS["third_same_direction_required"]
            ),
            "nominal_required": THRESHOLDS["third_nominal_required"],
            "third_retention_min": THRESHOLDS["third_retention_min"],
            "directional_p_max": (
                THRESHOLDS["third_directional_p_max"]
            ),
        },
        "randomization_replicates": args.randomization_replicates,
        "seed": args.seed,
        "thresholds": THRESHOLDS,
        "protocol_modified": protocol_modified(args),
        "plan_pass": plan_pass,
        "scientific_boundary": (
            "Same-source score decomposition only. No new controls, no "
            "independent corpus, no B59 validation, no individual-prime "
            "prediction, no RH/theorem claim."
        ),
    }
    atomic_json(out / "b50_11_protocol_manifest.json", manifest)

    report = f"""B50.11 — Channel Redundancy & Interaction Audit / PLAN
=====================================================

Frozen lineage
--------------
B50.10 verdict                      = {lineage['b50_10_verdict']}
B50.10 expected verdict             = {EXPECTED_B5010_VERDICT}
B50.6 verdict                       = {lineage['b50_6_verdict']}
full B50 effect                     = {lineage['full_effect']}
B50.10 mechanism-cube hash exact    = {lineage['cache_hash_exact']}
lineage PASS                        = {lineage['pass']}

Mechanism cube
--------------
transitions                         = {integrity['transition_count']}
configurations                      = {integrity['config_count']}
candidates / transition             = {integrity['candidate_count']}
controls / transition               = {integrity['control_count']}
raw cube shape                      = {integrity['raw_cube_shape']}
cube integrity PASS                 = {integrity['pass']}

Frozen exhaustive family
------------------------
channels                            = {CHANNELS}
non-empty subsets                   = {len(subsets)}
expected                            = 31
empty-coalition effect              = 0.0

Subset sufficiency gate
-----------------------
same sign as full                   = required
|effect| retention                  >= {args.subset_retention_min}
Holm directional p                  <= {args.subset_holm_p_max}
Holm family                         = all 31 subsets

Shapley
-------
characteristic function             = centered matched-percentile effect
empty coalition                     = 0.0
efficiency tolerance                = {THRESHOLDS['shapley_efficiency_abs_error_max']}

Randomization
-------------
replicates / subset                 = {args.randomization_replicates}
seed                                = {args.seed}
protocol modified                   = {protocol_modified(args)}

PLAN PASS                           = {plan_pass}

Interpretation
--------------
B50.11 will evaluate all 31 channel subsets. There is no adaptive subset
selection and no new control matching.
"""
    atomic_text(out / "b50_11_plan.txt", report)
    print(report)
    return manifest


def ensure_plan(args) -> Dict[str, Any]:
    p = Path(args.output_dir) / "b50_11_protocol_manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing frozen B50.11 PLAN: {p}. Run --mode plan first."
        )
    plan = json.loads(p.read_text(encoding="utf-8"))
    if not plan.get("plan_pass", False):
        raise RuntimeError("B50.11 PLAN did not pass.")
    if plan.get("protocol_modified", False):
        raise RuntimeError("B50.11 PLAN has protocol_modified=True.")
    return plan


def verify_frozen_hashes(
    args,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    paths = b5010_paths(args)
    rows = []
    all_ok = True
    for key, frozen in plan["inputs"].items():
        p = paths.get(key)
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
    atomic_csv(
        Path(args.output_dir) / "b50_11_frozen_input_audit.csv",
        df,
    )
    return {"pass": bool(all_ok), "rows": rows}


# ---------------------------------------------------------------------------
# Exhaustive subset scoring
# ---------------------------------------------------------------------------

def build_subset_cache(
    args,
    plan: Dict[str, Any],
    cube: Dict[str, Any],
) -> Dict[str, Any]:
    out = Path(args.output_dir)
    cache_path = out / "b50_11_subset_score_cache.npz"
    manifest_path = out / "b50_11_subset_cache_manifest.json"

    input_hashes = {
        k: v["sha256"] for k, v in plan["inputs"].items()
    }
    subsets = all_nonempty_subsets()
    masks = [subset_mask(S) for S in subsets]
    keys = [subset_key(S) for S in subsets]

    if (
        args.reuse_existing
        and cache_path.exists()
        and manifest_path.exists()
    ):
        cm = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            cm.get("version") == VERSION
            and cm.get("input_hashes") == input_hashes
            and cm.get("subset_masks") == masks
            and cm.get("subset_keys") == keys
        ):
            print("[cache] reusing B50.11 all-subset score cache", flush=True)
            with np.load(cache_path, allow_pickle=False) as z:
                return {
                    "subset_masks": z["subset_masks"].astype(int),
                    "subset_keys": [str(x) for x in z["subset_keys"]],
                    "subset_sizes": z["subset_sizes"].astype(int),
                    "consensus_percentiles": np.asarray(
                        z["consensus_percentiles"], float
                    ),
                    "effects": np.asarray(z["effects"], float),
                }

    raw = cube["raw_cube"]
    meds = cube["medians"]
    scales = cube["scales"]
    n_t, n_c, n_k, n_ch = raw.shape

    consensus_all = np.empty(
        (len(subsets), n_t, n_k),
        dtype=np.float64,
    )
    effects = np.empty(len(subsets), dtype=np.float64)

    print(
        f"[2/6] scoring all {len(subsets)} frozen channel subsets",
        flush=True,
    )

    for si, S in enumerate(subsets):
        pct = np.empty((n_t, n_c, n_k), dtype=np.float64)

        for ti in range(n_t):
            for ci in range(n_c):
                score = score_raw_subset(
                    raw[ti, ci],
                    meds[ci],
                    scales[ci],
                    S,
                )
                pct[ti, ci] = rank_percentiles(score)

        consensus = np.median(pct, axis=1)
        consensus_all[si] = consensus
        effects[si] = float(np.mean(consensus[:, 0]) - 0.5)

        print(
            f"  [{si+1:02d}/31] {subset_key(S):<75} "
            f"effect={effects[si]: .8f}",
            flush=True,
        )

    atomic_npz(
        cache_path,
        subset_masks=np.asarray(masks, dtype=int),
        subset_keys=np.asarray(keys, dtype="U256"),
        subset_sizes=np.asarray([len(S) for S in subsets], dtype=int),
        consensus_percentiles=consensus_all,
        effects=effects,
    )

    cm = {
        "version": VERSION,
        "created": now_s(),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "input_hashes": input_hashes,
        "subset_masks": masks,
        "subset_keys": keys,
        "shape_consensus_percentiles": list(consensus_all.shape),
    }
    atomic_json(manifest_path, cm)

    return {
        "subset_masks": np.asarray(masks, int),
        "subset_keys": keys,
        "subset_sizes": np.asarray([len(S) for S in subsets], int),
        "consensus_percentiles": consensus_all,
        "effects": effects,
    }


def subset_tests(
    args,
    cache: Dict[str, Any],
    full_effect: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(args.output_dir)

    rows = []
    null_rows = []
    consensus_all = cache["consensus_percentiles"]

    full_sign = np.sign(full_effect)
    if full_sign == 0:
        raise RuntimeError("Frozen full effect has zero sign.")

    for si, key in enumerate(cache["subset_keys"]):
        mask = int(cache["subset_masks"][si])
        size = int(cache["subset_sizes"][si])
        res, null = matched_identity_null(
            consensus_all[si],
            args.randomization_replicates,
            args.seed + 1000 + si * 211,
        )

        effect = float(res["observed_mean_effect"])
        same_sign = bool(np.sign(effect) == full_sign)
        retention = (
            abs(effect) / abs(full_effect)
            if abs(full_effect) > 1e-15
            else float("nan")
        )
        directional_p = (
            res["p_lower"] if full_sign < 0 else res["p_upper"]
        )

        rows.append({
            "subset_order": si + 1,
            "subset_mask": mask,
            "subset_key": key,
            "subset_size": size,
            **res,
            "same_sign_as_full": same_sign,
            "abs_effect_retention_fraction": float(retention),
            "near_full_retention": bool(
                same_sign
                and retention
                >= THRESHOLDS["subset_near_full_retention_min"]
            ),
            "directional_p": float(directional_p),
        })

        for r, val in enumerate(null):
            null_rows.append({
                "subset_mask": mask,
                "subset_key": key,
                "replicate": r,
                "null_mean_percentile_effect": float(val),
            })

    df = pd.DataFrame(rows)
    df["holm_directional_p"] = holm_adjust(
        df["directional_p"].to_numpy(float)
    )
    df["holm_directional_pass"] = (
        df["holm_directional_p"]
        <= args.subset_holm_p_max
    )
    df["retention_pass"] = (
        df["abs_effect_retention_fraction"]
        >= args.subset_retention_min
    )
    df["sufficient_subset"] = (
        df["same_sign_as_full"]
        & df["retention_pass"]
        & df["holm_directional_pass"]
    )

    null_df = pd.DataFrame(null_rows)
    atomic_csv(out / "b50_11_subset_effects.csv", df)
    atomic_csv(out / "b50_11_subset_null.csv", null_df)
    return df, null_df


# ---------------------------------------------------------------------------
# Full reconstruction gate
# ---------------------------------------------------------------------------

def full_reproduction_gate(
    subset_df: pd.DataFrame,
    frozen_full_effect: float,
) -> Dict[str, Any]:
    full_mask = (1 << len(CHANNELS)) - 1
    row = subset_df[subset_df["subset_mask"] == full_mask]
    if len(row) != 1:
        raise RuntimeError("Cannot identify unique full five-channel subset.")
    effect = float(row.iloc[0]["observed_mean_effect"])
    err = abs(effect - frozen_full_effect)
    return {
        "full_subset_mask": full_mask,
        "full_subset_key": str(row.iloc[0]["subset_key"]),
        "reconstructed_full_effect": effect,
        "frozen_b50_10_full_effect": frozen_full_effect,
        "abs_error": err,
        "pass": bool(
            err <= THRESHOLDS["full_effect_abs_error_max"]
        ),
    }


# ---------------------------------------------------------------------------
# Minimal sufficient subset analysis
# ---------------------------------------------------------------------------

def minimal_subset_analysis(
    subset_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sufficient = subset_df[
        subset_df["sufficient_subset"] == True
    ].copy()

    if sufficient.empty:
        return sufficient, {
            "available": False,
            "minimal_size": None,
            "minimal_count": 0,
            "minimal_subset_keys": [],
        }

    min_size = int(sufficient["subset_size"].min())
    minimal = (
        sufficient[sufficient["subset_size"] == min_size]
        .sort_values(
            [
                "abs_effect_retention_fraction",
                "holm_directional_p",
                "subset_key",
            ],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    info = {
        "available": True,
        "minimal_size": min_size,
        "minimal_count": int(len(minimal)),
        "minimal_subset_keys": minimal["subset_key"].astype(str).tolist(),
        "best_minimal_subset": str(minimal.iloc[0]["subset_key"]),
        "best_minimal_retention": float(
            minimal.iloc[0]["abs_effect_retention_fraction"]
        ),
        "best_minimal_effect": float(
            minimal.iloc[0]["observed_mean_effect"]
        ),
        "best_minimal_holm_directional_p": float(
            minimal.iloc[0]["holm_directional_p"]
        ),
    }
    return minimal, info


def subset_size_envelope(subset_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for size, g in subset_df.groupby("subset_size"):
        # Full effect is negative in this frozen lineage. More negative is
        # stronger in the replicated direction.
        ix_dir = g["observed_mean_effect"].idxmin()
        ix_abs = g["observed_mean_effect"].abs().idxmax()
        best_dir = g.loc[ix_dir]
        best_abs = g.loc[ix_abs]
        rows.append({
            "subset_size": int(size),
            "subset_count": int(len(g)),
            "mean_effect": float(g["observed_mean_effect"].mean()),
            "median_effect": float(g["observed_mean_effect"].median()),
            "min_effect": float(g["observed_mean_effect"].min()),
            "max_effect": float(g["observed_mean_effect"].max()),
            "best_negative_subset": str(best_dir["subset_key"]),
            "best_negative_effect": float(
                best_dir["observed_mean_effect"]
            ),
            "best_absolute_subset": str(best_abs["subset_key"]),
            "best_absolute_effect": float(
                best_abs["observed_mean_effect"]
            ),
            "sufficient_count": int(g["sufficient_subset"].sum()),
        })
    return pd.DataFrame(rows).sort_values("subset_size")


# ---------------------------------------------------------------------------
# Exact Shapley decomposition
# ---------------------------------------------------------------------------

def characteristic_map(
    subset_df: pd.DataFrame,
) -> Dict[int, float]:
    v = {0: 0.0}
    for _, r in subset_df.iterrows():
        v[int(r["subset_mask"])] = float(r["observed_mean_effect"])

    expected = set(range(1 << len(CHANNELS)))
    missing = expected - set(v)
    if missing:
        raise RuntimeError(
            f"Characteristic function missing masks: {sorted(missing)}"
        )
    return v


def exact_shapley(v: Mapping[int, float]) -> pd.DataFrame:
    n = len(CHANNELS)
    fact = math.factorial
    denom = fact(n)
    rows = []

    for i, ch in enumerate(CHANNELS):
        others = [j for j in range(n) if j != i]
        phi = 0.0
        positive_terms = 0
        negative_terms = 0
        terms = []

        for r in range(len(others) + 1):
            for S_tuple in itertools.combinations(others, r):
                S = subset_mask(S_tuple)
                Si = S | (1 << i)
                w = fact(r) * fact(n - r - 1) / denom
                marginal = float(v[Si] - v[S])
                term = w * marginal
                phi += term
                positive_terms += int(marginal > 0)
                negative_terms += int(marginal < 0)
                terms.append(marginal)

        rows.append({
            "channel": ch,
            "shapley_effect_contribution": float(phi),
            "marginal_mean_unweighted": float(np.mean(terms)),
            "marginal_min": float(np.min(terms)),
            "marginal_max": float(np.max(terms)),
            "negative_marginal_contexts": int(negative_terms),
            "positive_marginal_contexts": int(positive_terms),
            "context_count": int(len(terms)),
        })

    df = pd.DataFrame(rows)
    denom_abs = float(
        np.sum(np.abs(df["shapley_effect_contribution"]))
    )
    if denom_abs > 1e-15:
        df["absolute_shapley_share"] = (
            np.abs(df["shapley_effect_contribution"]) / denom_abs
        )
    else:
        df["absolute_shapley_share"] = 0.0

    df["supports_negative_full_effect"] = (
        df["shapley_effect_contribution"] < 0
    )

    return df.sort_values(
        "absolute_shapley_share",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def shapley_efficiency(
    shapley_df: pd.DataFrame,
    full_effect: float,
) -> Dict[str, Any]:
    s = float(shapley_df["shapley_effect_contribution"].sum())
    err = abs(s - full_effect)
    top2 = float(
        shapley_df["absolute_shapley_share"].head(2).sum()
    )
    return {
        "sum_shapley": s,
        "full_effect": full_effect,
        "abs_efficiency_error": err,
        "efficiency_pass": bool(
            err
            <= THRESHOLDS["shapley_efficiency_abs_error_max"]
        ),
        "top2_absolute_share": top2,
        "top2_concentrated": bool(
            top2
            >= THRESHOLDS[
                "shapley_top2_abs_share_concentrated_min"
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Pair interactions
# ---------------------------------------------------------------------------

def pair_only_synergy(
    v: Mapping[int, float],
    i: int,
    j: int,
) -> float:
    mi = 1 << i
    mj = 1 << j
    return float(v[mi | mj] - v[mi] - v[mj] + v[0])


def shapley_pair_interaction(
    v: Mapping[int, float],
    i: int,
    j: int,
) -> float:
    r"""
    Shapley interaction index:
      sum_{S subset N\{i,j}}
      |S|! (n-|S|-2)! / (n-1)!
      * Δ_{ij} v(S)
    """
    n = len(CHANNELS)
    others = [k for k in range(n) if k not in (i, j)]
    denom = math.factorial(n - 1)
    total = 0.0

    for r in range(len(others) + 1):
        for S_tuple in itertools.combinations(others, r):
            S = subset_mask(S_tuple)
            mi = 1 << i
            mj = 1 << j
            delta = (
                v[S | mi | mj]
                - v[S | mi]
                - v[S | mj]
                + v[S]
            )
            w = (
                math.factorial(r)
                * math.factorial(n - r - 2)
                / denom
            )
            total += w * delta

    return float(total)


def pair_interaction_table(
    v: Mapping[int, float],
) -> pd.DataFrame:
    rows = []
    for i, j in itertools.combinations(range(len(CHANNELS)), 2):
        po = pair_only_synergy(v, i, j)
        si = shapley_pair_interaction(v, i, j)
        rows.append({
            "channel_i": CHANNELS[i],
            "channel_j": CHANNELS[j],
            "pair_only_synergy": po,
            "global_shapley_interaction": si,
            "global_interaction_abs": abs(si),
            "interaction_sign": (
                "negative_synergy"
                if si < 0
                else "positive_antagonism"
                if si > 0
                else "zero"
            ),
            "material_interaction": bool(
                abs(si)
                >= THRESHOLDS[
                    "pair_interaction_material_abs_min"
                ]
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "global_interaction_abs",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Thirds robustness for minimal sufficient subsets
# ---------------------------------------------------------------------------

def minimal_subset_thirds(
    args,
    minimal_df: pd.DataFrame,
    cache: Dict[str, Any],
    full_effect: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(args.output_dir)

    if minimal_df.empty:
        empty = pd.DataFrame()
        atomic_csv(out / "b50_11_minimal_subset_thirds.csv", empty)
        atomic_csv(
            out / "b50_11_minimal_subset_thirds_summary.csv", empty
        )
        return empty, empty

    mask_to_index = {
        int(m): i for i, m in enumerate(cache["subset_masks"])
    }
    n = cache["consensus_percentiles"].shape[1]
    edges = np.linspace(0, n, 4, dtype=int)
    full_sign = np.sign(full_effect)

    rows = []
    summaries = []

    for m_i, (_, row) in enumerate(minimal_df.iterrows()):
        mask_val = int(row["subset_mask"])
        key = str(row["subset_key"])
        si = mask_to_index[mask_val]
        X = cache["consensus_percentiles"][si]
        full_subset_effect = float(row["observed_mean_effect"])

        same_count = 0
        nominal_count = 0

        for third in range(3):
            bool_mask = np.zeros(n, dtype=bool)
            bool_mask[edges[third]:edges[third + 1]] = True
            res, _ = matched_identity_null(
                X,
                args.randomization_replicates,
                args.seed + 6000 + m_i * 101 + third,
                mask=bool_mask,
            )
            effect = float(res["observed_mean_effect"])
            same = bool(np.sign(effect) == full_sign)
            same_count += int(same)

            # Retention is relative to the subset's own full-sequence effect,
            # not the five-channel full effect.
            retention = (
                abs(effect) / abs(full_subset_effect)
                if abs(full_subset_effect) > 1e-15
                else float("nan")
            )
            directional_p = (
                res["p_lower"]
                if full_sign < 0
                else res["p_upper"]
            )
            nominal = bool(
                same
                and retention
                >= THRESHOLDS["third_retention_min"]
                and directional_p
                <= THRESHOLDS["third_directional_p_max"]
            )
            nominal_count += int(nominal)

            rows.append({
                "subset_mask": mask_val,
                "subset_key": key,
                "third": third + 1,
                "n": int(bool_mask.sum()),
                "observed_mean_effect": effect,
                "same_sign_as_frozen_full": same,
                "retention_vs_subset_full_effect": float(retention),
                "directional_p": float(directional_p),
                "nominal_third_pass": nominal,
            })

        pass_ = bool(
            same_count
            >= THRESHOLDS["third_same_direction_required"]
            and nominal_count
            >= THRESHOLDS["third_nominal_required"]
        )
        summaries.append({
            "subset_mask": mask_val,
            "subset_key": key,
            "same_direction_thirds": same_count,
            "nominal_thirds": nominal_count,
            "thirds_robustness_pass": pass_,
        })

    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(summaries)
    atomic_csv(out / "b50_11_minimal_subset_thirds.csv", df)
    atomic_csv(
        out / "b50_11_minimal_subset_thirds_summary.csv", sdf
    )
    return df, sdf


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def final_verdict(
    full_repro: Dict[str, Any],
    shapley_eff: Dict[str, Any],
    minimal_info: Dict[str, Any],
    thirds_summary: pd.DataFrame,
    shapley_df: pd.DataFrame,
) -> Tuple[str, str]:
    if not full_repro["pass"]:
        return (
            "INVALID_FULL_SCORE_REPRODUCTION",
            "The all-five subset failed to reproduce the frozen B50.10 / B50.6 full effect exactly.",
        )

    if not shapley_eff["efficiency_pass"]:
        return (
            "INVALID_SHAPLEY_DECOMPOSITION",
            "Exact Shapley contributions do not satisfy the frozen efficiency identity within tolerance.",
        )

    if not minimal_info["available"]:
        return (
            "NO_REDUCED_SUBSET_RECAPTURES_FULL_EFFECT",
            "No non-empty reduced channel subset satisfies the frozen same-sign, >=90% effect-retention, and Holm-directional significance gates.",
        )

    min_size = int(minimal_info["minimal_size"])

    # Full five-channel set is always in the exhaustive family and generally
    # sufficient. We distinguish whether a reduced subset exists.
    if min_size >= len(CHANNELS):
        return (
            "FULL_FIVE_CHANNEL_COMPOSITE_REQUIRED",
            "Only the full five-channel composite satisfies the frozen sufficiency contract; no reduced subset recaptures the replicated effect.",
        )

    robust = False
    if not thirds_summary.empty:
        robust = bool(
            thirds_summary["thirds_robustness_pass"].any()
        )

    top2_conc = bool(shapley_eff["top2_concentrated"])

    if min_size == 1:
        base = (
            "SINGLE_CHANNEL_SCORE_SUFFICIENCY"
            if robust
            else "SINGLE_CHANNEL_SCORE_SUFFICIENCY_WEAK_THIRDS"
        )
        reason = (
            "At least one single-channel composite recaptures >=90% of the frozen full effect "
            "with Holm-corrected directional significance. "
            + (
                "At least one minimal single-channel subset also passes the frozen thirds robustness gate."
                if robust
                else "No minimal single-channel subset passes the frozen thirds robustness gate."
            )
        )
        return base, reason

    if min_size == 2:
        base = (
            "PAIRWISE_CORE_RECAPTURES_FULL_EFFECT"
            if robust
            else "PAIRWISE_CORE_RECAPTURES_FULL_EFFECT_WEAK_THIRDS"
        )
        reason = (
            "A two-channel subset is sufficient under the frozen >=90% retention and Holm-directional gates, "
            "showing that the five-channel score contains substantial redundancy. "
            + (
                "At least one minimal pair also passes the frozen thirds robustness gate."
                if robust
                else "The minimal pairs do not pass the frozen thirds robustness gate."
            )
        )
        return base, reason

    if min_size == 3:
        base = (
            "THREE_CHANNEL_CORE_RECAPTURES_FULL_EFFECT"
            if robust
            else "THREE_CHANNEL_CORE_RECAPTURES_FULL_EFFECT_WEAK_THIRDS"
        )
        reason = (
            "The smallest sufficient composite contains three channels, consistent with a distributed but reducible mechanism. "
            + (
                "At least one minimal triplet passes the frozen thirds robustness gate."
                if robust
                else "Minimal triplets do not pass the frozen thirds robustness gate."
            )
        )
        return base, reason

    if min_size == 4:
        return (
            "FOUR_CHANNEL_NEAR_FULL_COMPOSITE_REQUIRED",
            "The replicated effect can be reduced by one channel but no subset of three or fewer channels satisfies the frozen sufficiency gate.",
        )

    return (
        "INTERACTION_STRUCTURE_RESOLVED_WITHOUT_SIMPLE_CORE",
        "The exhaustive subset and Shapley decomposition is valid, but the result does not reduce to a simpler frozen verdict class.",
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    out: Path,
    subset_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    shapley_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> None:
    # All subset effects sorted.
    d = subset_df.sort_values(
        "observed_mean_effect",
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)
    x = np.arange(len(d))
    plt.figure(figsize=(12, 6))
    plt.bar(x, d["observed_mean_effect"].to_numpy(float))
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(
        float(
            d.loc[
                d["subset_size"] == len(CHANNELS),
                "observed_mean_effect",
            ].iloc[0]
        ),
        linestyle=":",
        linewidth=1.0,
    )
    plt.xticks(x, d["subset_key"].astype(str), rotation=90, fontsize=7)
    plt.ylabel("mean physical percentile effect vs 0.5")
    plt.title("B50.11 all 31 frozen channel subsets")
    plt.tight_layout()
    plt.savefig(out / "b50_11_all_subset_effects.png", dpi=180)
    plt.close()

    # Best effect by subset size.
    plt.figure(figsize=(8, 5))
    plt.plot(
        envelope_df["subset_size"],
        envelope_df["best_negative_effect"],
        marker="o",
    )
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xlabel("subset size")
    plt.ylabel("most negative effect at subset size")
    plt.title("B50.11 subset-size effect envelope")
    plt.tight_layout()
    plt.savefig(out / "b50_11_subset_size_envelope.png", dpi=180)
    plt.close()

    # Shapley.
    s = shapley_df.sort_values(
        "shapley_effect_contribution",
        ascending=True,
    )
    x = np.arange(len(s))
    plt.figure(figsize=(9, 5))
    plt.bar(x, s["shapley_effect_contribution"].to_numpy(float))
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.xticks(x, s["channel"].astype(str), rotation=35, ha="right")
    plt.ylabel("Shapley contribution to centered percentile effect")
    plt.title("B50.11 exact Shapley score decomposition")
    plt.tight_layout()
    plt.savefig(out / "b50_11_shapley_contributions.png", dpi=180)
    plt.close()

    # Pair interaction matrix.
    mat = np.zeros((len(CHANNELS), len(CHANNELS)), dtype=float)
    for _, r in pair_df.iterrows():
        i = CHANNELS.index(str(r["channel_i"]))
        j = CHANNELS.index(str(r["channel_j"]))
        val = float(r["global_shapley_interaction"])
        mat[i, j] = val
        mat[j, i] = val

    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, label="global Shapley pair interaction")
    plt.xticks(
        np.arange(len(CHANNELS)),
        CHANNELS,
        rotation=45,
        ha="right",
    )
    plt.yticks(np.arange(len(CHANNELS)), CHANNELS)
    plt.title("B50.11 pairwise interaction structure")
    plt.tight_layout()
    plt.savefig(out / "b50_11_pair_interaction_matrix.png", dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# AUDIT
# ---------------------------------------------------------------------------

def audit_mode(args) -> Dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plan = ensure_plan(args)
    paths = b5010_paths(args)
    require_paths(paths)

    print("=== B50.11 CHANNEL REDUNDANCY & INTERACTION AUDIT ===")

    hash_audit = verify_frozen_hashes(args, plan)
    print("frozen B50.10 hashes exact :", hash_audit["pass"])
    if not hash_audit["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_FROZEN_INPUTS",
            "reason": "One or more B50.10 frozen input artifacts changed after B50.11 PLAN.",
        }
        atomic_json(out / "b50_11_summary.json", summary)
        atomic_text(
            out / "b50_11_verdict.txt",
            "FINAL VERDICT: INVALID_FROZEN_INPUTS\n",
        )
        return summary

    lineage = lineage_gate(paths)
    if not lineage["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_LINEAGE",
            "reason": "Frozen B50.10/B50.6 lineage gate failed.",
            "lineage": lineage,
        }
        atomic_json(out / "b50_11_summary.json", summary)
        atomic_text(
            out / "b50_11_verdict.txt",
            "FINAL VERDICT: INVALID_LINEAGE\n",
        )
        return summary

    cube = load_cube(paths["b50_10_mechanism_cube"])
    integrity = cube_integrity(cube)
    print("mechanism cube integrity     :", integrity["pass"])
    if not integrity["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_MECHANISM_CUBE",
            "reason": "B50.10 mechanism cube failed B50.11 integrity checks.",
            "cube_integrity": integrity,
        }
        atomic_json(out / "b50_11_summary.json", summary)
        atomic_text(
            out / "b50_11_verdict.txt",
            "FINAL VERDICT: INVALID_MECHANISM_CUBE\n",
        )
        return summary

    print("[1/6] frozen lineage / cube PASS", flush=True)

    cache = build_subset_cache(args, plan, cube)

    print("[3/6] matched-identity null for all 31 subsets", flush=True)
    subset_df, null_df = subset_tests(
        args,
        cache,
        lineage["full_effect"],
    )

    full_repro = full_reproduction_gate(
        subset_df,
        lineage["full_effect"],
    )
    print("full five-channel reproduction:", full_repro["pass"])
    if not full_repro["pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_FULL_SCORE_REPRODUCTION",
            "reason": "The all-five subset does not reproduce B50.10 full effect within frozen tolerance.",
            "full_reproduction": full_repro,
        }
        atomic_json(out / "b50_11_summary.json", summary)
        atomic_text(
            out / "b50_11_verdict.txt",
            "FINAL VERDICT: INVALID_FULL_SCORE_REPRODUCTION\n",
        )
        return summary

    minimal_df, minimal_info = minimal_subset_analysis(subset_df)
    atomic_csv(out / "b50_11_minimal_sufficient_subsets.csv", minimal_df)
    atomic_json(out / "b50_11_minimal_subset_summary.json", minimal_info)

    envelope_df = subset_size_envelope(subset_df)
    atomic_csv(out / "b50_11_subset_size_envelope.csv", envelope_df)

    print("[4/6] exact Shapley decomposition", flush=True)
    v = characteristic_map(subset_df)
    shapley_df = exact_shapley(v)
    shapley_eff = shapley_efficiency(
        shapley_df,
        lineage["full_effect"],
    )
    atomic_csv(out / "b50_11_shapley.csv", shapley_df)
    atomic_json(
        out / "b50_11_shapley_efficiency.json",
        shapley_eff,
    )

    print(
        "Shapley efficiency PASS        :",
        shapley_eff["efficiency_pass"],
    )
    if not shapley_eff["efficiency_pass"]:
        summary = {
            "version": VERSION,
            "final_verdict": "INVALID_SHAPLEY_DECOMPOSITION",
            "reason": "Exact Shapley decomposition failed efficiency identity.",
            "shapley_efficiency": shapley_eff,
        }
        atomic_json(out / "b50_11_summary.json", summary)
        atomic_text(
            out / "b50_11_verdict.txt",
            "FINAL VERDICT: INVALID_SHAPLEY_DECOMPOSITION\n",
        )
        return summary

    print("[5/6] pair interactions / minimal-subset thirds", flush=True)
    pair_df = pair_interaction_table(v)
    atomic_csv(out / "b50_11_pair_interactions.csv", pair_df)

    thirds_df, thirds_summary = minimal_subset_thirds(
        args,
        minimal_df,
        cache,
        lineage["full_effect"],
    )

    print("[6/6] frozen B50.11 verdict", flush=True)
    verdict, reason = final_verdict(
        full_repro,
        shapley_eff,
        minimal_info,
        thirds_summary,
        shapley_df,
    )

    sufficient_count = int(subset_df["sufficient_subset"].sum())
    reduced_sufficient_count = int(
        (
            subset_df["sufficient_subset"]
            & (subset_df["subset_size"] < len(CHANNELS))
        ).sum()
    )

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "lineage": lineage,
        "frozen_input_hashes_exact": hash_audit["pass"],
        "cube_integrity": integrity,
        "full_reproduction": full_repro,
        "subset_family": {
            "tested_nonempty_subsets": int(len(subset_df)),
            "sufficient_subset_count": sufficient_count,
            "reduced_sufficient_subset_count": reduced_sufficient_count,
            "minimal_subset": minimal_info,
            "rows": subset_df.to_dict(orient="records"),
        },
        "shapley": {
            "efficiency": shapley_eff,
            "rows": shapley_df.to_dict(orient="records"),
        },
        "pair_interactions": pair_df.to_dict(orient="records"),
        "minimal_subset_thirds": (
            thirds_summary.to_dict(orient="records")
            if not thirds_summary.empty
            else []
        ),
        "thresholds": THRESHOLDS,
        "final_verdict": verdict,
        "final_reason": reason,
        "scientific_boundary": (
            "Exhaustive score-mechanism decomposition of the frozen B50.10 / "
            "B50.6 same-corpus effect. Subset sufficiency and Shapley "
            "contributions describe score structure, not causal dynamics. "
            "No new corpus, B59 validation, individual-prime prediction, "
            "RH, or theorem claim."
        ),
    }
    atomic_json(out / "b50_11_summary.json", summary)

    make_plots(out, subset_df, envelope_df, shapley_df, pair_df)

    top_pair_rows = pair_df.head(10)
    minimal_text = (
        minimal_df.to_string(index=False)
        if not minimal_df.empty
        else "NONE"
    )
    thirds_text = (
        thirds_summary.to_string(index=False)
        if not thirds_summary.empty
        else "NONE"
    )

    report = f"""B50.11 — Channel Redundancy & Interaction Audit
================================================

Interpretation scope
--------------------
Exact B50.10 mechanism cube reused: YES
Exact B50.6 matched controls retained: YES
All 31 non-empty channel subsets tested: YES
Adaptive subset search: NO
Independent-corpus replication: NO
B59/B59.1 used: NO

Frozen lineage
--------------
B50.10 verdict                       = {lineage['b50_10_verdict']}
B50.6 verdict                        = {lineage['b50_6_verdict']}
frozen B50.10 artifact hashes exact  = {hash_audit['pass']}
mechanism cube integrity             = {integrity['pass']}

Full five-channel reproduction
------------------------------
reconstructed full effect            = {full_repro['reconstructed_full_effect']}
frozen B50.10 full effect            = {full_repro['frozen_b50_10_full_effect']}
absolute error                       = {full_repro['abs_error']}
FULL REPRODUCTION PASS               = {full_repro['pass']}

Exhaustive subset family
------------------------
tested subsets                       = {len(subset_df)}
sufficient subsets                   = {sufficient_count}
reduced sufficient subsets           = {reduced_sufficient_count}
minimal sufficient size              = {minimal_info.get('minimal_size')}
minimal sufficient count             = {minimal_info.get('minimal_count')}
minimal subset keys                  = {minimal_info.get('minimal_subset_keys')}

Minimal sufficient subsets
--------------------------
{minimal_text}

Subset-size envelope
--------------------
{envelope_df.to_string(index=False)}

Exact Shapley decomposition
---------------------------
{shapley_df.to_string(index=False)}

Shapley efficiency
------------------
sum contributions                    = {shapley_eff['sum_shapley']}
full effect                          = {shapley_eff['full_effect']}
absolute efficiency error            = {shapley_eff['abs_efficiency_error']}
efficiency PASS                      = {shapley_eff['efficiency_pass']}
top-2 absolute Shapley share         = {shapley_eff['top2_absolute_share']}
top-2 concentrated diagnostic        = {shapley_eff['top2_concentrated']}

Strongest pair interactions
---------------------------
{top_pair_rows.to_string(index=False)}

Minimal-subset thirds robustness
--------------------------------
{thirds_text}

Frozen sufficiency contract
---------------------------
same sign as full                    = required
|effect| retention                   >= {args.subset_retention_min}
Holm directional p                  <= {args.subset_holm_p_max}
thirds same direction                = {THRESHOLDS['third_same_direction_required']}/3
thirds nominal                       >= {THRESHOLDS['third_nominal_required']}/3

FINAL VERDICT: {verdict}

REASON:
{reason}

Interpretation boundary
-----------------------
B50.11 is an exhaustive decomposition of score sufficiency, redundancy and
interaction inside the already replicated B50.6/B50.10 matched-transition
geometry. A reduced sufficient subset or Shapley contribution is not by itself
a causal law. No independent corpus, B59 validation, individual-prime
prediction, RH, or theorem claim is made.

Key outputs
-----------
b50_11_protocol_manifest.json
b50_11_plan.txt
b50_11_subset_plan.csv
b50_11_frozen_input_audit.csv
b50_11_subset_score_cache.npz
b50_11_subset_cache_manifest.json
b50_11_subset_effects.csv
b50_11_subset_null.csv
b50_11_minimal_sufficient_subsets.csv
b50_11_minimal_subset_summary.json
b50_11_subset_size_envelope.csv
b50_11_shapley.csv
b50_11_shapley_efficiency.json
b50_11_pair_interactions.csv
b50_11_minimal_subset_thirds.csv
b50_11_minimal_subset_thirds_summary.csv
b50_11_summary.json
b50_11_verdict.txt
b50_11_all_subset_effects.png
b50_11_subset_size_envelope.png
b50_11_shapley_contributions.png
b50_11_pair_interaction_matrix.png
"""
    atomic_text(out / "b50_11_verdict.txt", report)

    print()
    print("=== B50.11 RESULT ===")
    print("sufficient subsets  :", sufficient_count)
    print("minimal size        :", minimal_info.get("minimal_size"))
    print("minimal subsets     :", minimal_info.get("minimal_subset_keys"))
    print("Shapley efficiency  :", shapley_eff["efficiency_pass"])
    print("FINAL VERDICT       :", verdict)
    print("verdict             :", out / "b50_11_verdict.txt")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="B50.11 Channel Redundancy & Interaction Audit"
    )
    p.add_argument(
        "--mode",
        choices=["plan", "audit", "all"],
        default="plan",
    )
    p.add_argument("--b50-10-dir", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument(
        "--randomization-replicates",
        type=int,
        default=DEFAULT_REPS,
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--reuse-existing", action="store_true")

    # Exposed only for transparent protocol inspection. Any change causes
    # PLAN PASS=False.
    p.add_argument(
        "--subset-retention-min",
        type=float,
        default=THRESHOLDS["subset_retention_min"],
    )
    p.add_argument(
        "--subset-holm-p-max",
        type=float,
        default=THRESHOLDS["subset_holm_directional_p_max"],
    )

    a = p.parse_args()

    if a.randomization_replicates < 999:
        p.error("--randomization-replicates must be >=999")
    if not (0 < a.subset_retention_min <= 2.0):
        p.error("--subset-retention-min must be in (0,2]")
    if not (0 < a.subset_holm_p_max <= 1):
        p.error("--subset-holm-p-max must be in (0,1]")

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
            raise RuntimeError("B50.11 PLAN failed; AUDIT not started.")
        audit_mode(args)
    else:
        raise ValueError(args.mode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
