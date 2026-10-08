#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TDC TOR A / B50.1 — State Geometry Adequacy & Basin Resolution Audit
====================================================================

Purpose
-------
B50.1 audits whether the *existing B50 state geometry* is capable of producing
a stable, non-degenerate basin partition before any predictive interpretation
is attempted.

This script is deliberately outcome-blind with respect to B59/B59.1:
- it does NOT read B59 scores,
- it does NOT read selected/anti-selected cohorts,
- it does NOT tune parameters to increase B59 lift,
- it does NOT re-read the ~1e12-prime corpus.

Instead it reuses the already-computed B50 state dataset (c1,c2,c3) and imports
the original B50 geometry functions directly from the frozen B50 script.

Why this is the correct next audit
----------------------------------
The B59.1a global physical-order run produced:
    666 states
    665 physical transitions
    0 B50 realized basin crossings

Under original B50 semantics:
    is_jump = bool(basin[i] != basin[i+1])

Therefore the first question is not forecasting performance.  It is whether
the B50 PCA/KDE geometry resolves more than one stable occupied basin.

Primary audit questions
-----------------------
1. BASELINE REPRODUCTION
   Can B50.1 reproduce the original B50 basin assignment from:
       projection      = c1,c2
       grid            = 120
       kde_bandwidth   = 0.65
       min_separation  = 6
       max_minima      = 6
       margin          = 0.5

2. NATIVE c1,c2 RESOLUTION
   Across a frozen, outcome-blind parameter neighborhood:
       - how often are >=2 basins occupied?
       - how often is >=1 physical basin crossing realized?
       - are those assignments stable under parameter perturbation?
       - are extra basins substantive or tiny slivers?

3. PROJECTION SENSITIVITY
   At the baseline KDE parameters, what happens for diagnostic projections:
       c1,c3 and c2,c3?
   These are diagnostics only; the B50 primary geometry remains c1,c2.

4. PCA CAPTURE WARNING
   Record the rank-1 / rank-2 / rank-3 explained-variance capture from the
   original B50 verdict.  B50.1 does not recompute higher PCs because the
   state dataset contains only c1,c2,c3.

Frozen default protocol
-----------------------
Native c1,c2 one-factor sweeps:
    bandwidth      = 0.30,0.40,0.50,0.65,0.80,1.00,1.30
    min_separation = 3,4,6,8,10
    max_minima     = 2,4,6,8,10
    grid           = 80,100,120,160

Frozen local interaction neighborhood:
    bandwidth      = 0.50,0.65,0.80
    min_separation = 4,6,8
    max_minima     = 6
    grid           = 120

Diagnostic projections:
    c1,c3
    c2,c3
at baseline KDE parameters.

The default protocol is intentionally modest so it operates on the 666-row
state dataset rather than repeating the multi-day prime I/O stage.

Audit metrics per configuration
-------------------------------
- number of KDE minima
- number of occupied basins
- realized physical basin crossings
- jump rate
- largest occupied-basin fraction
- smallest occupied-basin fraction
- normalized occupancy entropy
- singleton basin count
- adjusted Rand index (ARI) vs baseline
- jump-location Jaccard vs baseline when defined

Pairwise stability
------------------
For native c1,c2 configurations B50.1 computes a full pairwise ARI matrix.
ARI is label-permutation invariant.

Pre-declared audit classification
---------------------------------
These are engineering adequacy criteria, not mathematical truth criteria.

PERSISTENT_DEGENERACY:
    <= 10% of native c1,c2 configs are non-degenerate
    (non-degenerate := >=2 occupied basins AND >=1 crossing)

STABLE_MULTIBASIN_GEOMETRY:
    >= 60% non-degenerate
    AND median pairwise ARI among non-degenerate configs >= 0.75
    AND >= 60% of non-degenerate configs are "balanced enough"
        largest basin fraction <= 0.98
        smallest occupied basin fraction >= 0.01

PARAMETER_SENSITIVE_MULTIBASIN:
    >= 25% non-degenerate, but stability criteria above fail

SPARSE_MULTIBASIN_RESOLUTION:
    >10% but <25% non-degenerate

INVALID_BASELINE_REPRODUCTION:
    imported original B50 geometry cannot reproduce the saved B50 baseline
    partition semantically (ARI < 0.999999 or physical jump vector differs).

Scientific boundary
-------------------
B50.1 tests adequacy and robustness of the B50 *representation*.  A positive
multibasin verdict does not establish forecasting skill, number-theoretic
structure, or individual-prime prediction.  A negative verdict says that the
current B50 geometry is not a stable realized-crossing endpoint under this
pre-declared audit neighborhood.

Typical use
-----------
python tdc_tor_a_b50_1_state_geometry_adequacy_audit.py ^
  --state-dataset "...\stages\b50\b50_state_dataset.csv" ^
  --verdict-json "...\stages\b50\b50_verdict_scaffold.json" ^
  --b50-script ".\code\tdc_tor_a_b50_predictive_jump_forecasting.py" ^
  --output-dir ".\_replay\b50_1_geometry_adequacy"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "B50.1_state_geometry_adequacy_audit_v1"

BASELINE = {
    "projection": "c1c2",
    "grid": 120,
    "bandwidth": 0.65,
    "min_separation": 6,
    "max_minima": 6,
    "margin": 0.5,
}

DEFAULT_BANDWIDTHS = [0.30, 0.40, 0.50, 0.65, 0.80, 1.00, 1.30]
DEFAULT_MIN_SEPARATIONS = [3, 4, 6, 8, 10]
DEFAULT_MAX_MINIMA = [2, 4, 6, 8, 10]
DEFAULT_GRIDS = [80, 100, 120, 160]
DEFAULT_INTERACTION_BANDWIDTHS = [0.50, 0.65, 0.80]
DEFAULT_INTERACTION_MIN_SEPARATIONS = [4, 6, 8]
DEFAULT_PROJECTIONS = ["c1c2", "c1c3", "c2c3"]


# ---------------------------------------------------------------------------
# IO / utilities
# ---------------------------------------------------------------------------

def now_s() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
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


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def comb2(x: np.ndarray | float | int) -> np.ndarray | float:
    return np.asarray(x, dtype=float) * (np.asarray(x, dtype=float) - 1.0) / 2.0


def adjusted_rand_index(a: Sequence[int], b: Sequence[int]) -> float:
    """
    Label-permutation invariant Adjusted Rand Index without sklearn.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("ARI inputs must have equal length.")
    n = len(a)
    if n < 2:
        return 1.0

    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)

    cont = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(cont, (ia, ib), 1)

    sum_comb = float(np.sum(comb2(cont)))
    row_comb = float(np.sum(comb2(cont.sum(axis=1))))
    col_comb = float(np.sum(comb2(cont.sum(axis=0))))
    total_comb = float(comb2(n))

    if total_comb <= 0:
        return 1.0

    expected = row_comb * col_comb / total_comb
    maximum = 0.5 * (row_comb + col_comb)
    denom = maximum - expected

    if abs(denom) < 1e-15:
        # Both partitions are pairwise-equivalent constants or exact equivalents.
        same_pairs_a = (ia[:, None] == ia[None, :])
        same_pairs_b = (ib[:, None] == ib[None, :])
        return 1.0 if np.array_equal(same_pairs_a, same_pairs_b) else 0.0

    return float((sum_comb - expected) / denom)


def normalized_entropy(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return float("nan")
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    p = counts.astype(float) / counts.sum()
    H = -float(np.sum(p * np.log(p)))
    return float(H / math.log(len(counts)))


def jump_vector(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if len(labels) < 2:
        return np.zeros(0, dtype=np.int8)
    return (labels[:-1] != labels[1:]).astype(np.int8)


def jump_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.sum(a | b)
    if union == 0:
        return float("nan")
    return float(np.sum(a & b) / union)


def import_b50(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("tdc_b50_frozen_engine", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import B50 from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    required = [
        "kde_potential",
        "local_minima",
        "basin_descent_labels",
        "grid_index_for_point",
    ]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise AttributeError(f"B50 script missing required geometry functions: {missing}")
    return mod


# ---------------------------------------------------------------------------
# Audit configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    projection: str
    grid: int
    bandwidth: float
    min_separation: int
    max_minima: int
    margin: float
    families: str

    @property
    def key(self) -> Tuple[Any, ...]:
        return (
            self.projection,
            int(self.grid),
            round(float(self.bandwidth), 12),
            int(self.min_separation),
            int(self.max_minima),
            round(float(self.margin), 12),
        )

    @property
    def config_id(self) -> str:
        return (
            f"{self.projection}"
            f"_g{self.grid}"
            f"_bw{self.bandwidth:.3f}"
            f"_sep{self.min_separation}"
            f"_m{self.max_minima}"
        )


def projection_columns(name: str) -> Tuple[str, str]:
    table = {
        "c1c2": ("c1", "c2"),
        "c1c3": ("c1", "c3"),
        "c2c3": ("c2", "c3"),
    }
    if name not in table:
        raise ValueError(f"Unknown projection {name!r}; allowed={sorted(table)}")
    return table[name]


def add_cfg(
    store: Dict[Tuple[Any, ...], Config],
    cfg: Config,
) -> None:
    old = store.get(cfg.key)
    if old is None:
        store[cfg.key] = cfg
        return

    fam = sorted(set(old.families.split("|")) | set(cfg.families.split("|")))
    store[cfg.key] = Config(
        projection=old.projection,
        grid=old.grid,
        bandwidth=old.bandwidth,
        min_separation=old.min_separation,
        max_minima=old.max_minima,
        margin=old.margin,
        families="|".join(fam),
    )


def build_protocol(args: argparse.Namespace) -> List[Config]:
    store: Dict[Tuple[Any, ...], Config] = {}

    base = Config(
        projection="c1c2",
        grid=args.baseline_grid,
        bandwidth=args.baseline_bandwidth,
        min_separation=args.baseline_min_separation,
        max_minima=args.baseline_max_minima,
        margin=args.margin,
        families="baseline",
    )
    add_cfg(store, base)

    # One-factor bandwidth sweep.
    for bw in args.bandwidths:
        add_cfg(store, Config(
            "c1c2",
            args.baseline_grid,
            float(bw),
            args.baseline_min_separation,
            args.baseline_max_minima,
            args.margin,
            "bandwidth_sweep",
        ))

    # One-factor min-separation sweep.
    for sep in args.min_separations:
        add_cfg(store, Config(
            "c1c2",
            args.baseline_grid,
            args.baseline_bandwidth,
            int(sep),
            args.baseline_max_minima,
            args.margin,
            "min_separation_sweep",
        ))

    # One-factor max-minima sweep.
    for mm in args.max_minima_values:
        add_cfg(store, Config(
            "c1c2",
            args.baseline_grid,
            args.baseline_bandwidth,
            args.baseline_min_separation,
            int(mm),
            args.margin,
            "max_minima_sweep",
        ))

    # One-factor grid sweep.
    for g in args.grids:
        add_cfg(store, Config(
            "c1c2",
            int(g),
            args.baseline_bandwidth,
            args.baseline_min_separation,
            args.baseline_max_minima,
            args.margin,
            "grid_sweep",
        ))

    # Frozen local interaction neighborhood.
    for bw in args.interaction_bandwidths:
        for sep in args.interaction_min_separations:
            add_cfg(store, Config(
                "c1c2",
                args.baseline_grid,
                float(bw),
                int(sep),
                args.baseline_max_minima,
                args.margin,
                "local_interaction",
            ))

    # Diagnostic projection sensitivity only.
    for proj in args.projections:
        add_cfg(store, Config(
            proj,
            args.baseline_grid,
            args.baseline_bandwidth,
            args.baseline_min_separation,
            args.baseline_max_minima,
            args.margin,
            "projection_diagnostic" if proj != "c1c2" else "baseline",
        ))

    configs = list(store.values())
    configs.sort(key=lambda c: (
        0 if "baseline" in c.families and c.projection == "c1c2"
        and c.grid == args.baseline_grid
        and abs(c.bandwidth - args.baseline_bandwidth) < 1e-12
        and c.min_separation == args.baseline_min_separation
        and c.max_minima == args.baseline_max_minima else 1,
        c.projection,
        c.grid,
        c.bandwidth,
        c.min_separation,
        c.max_minima,
    ))
    return configs


# ---------------------------------------------------------------------------
# Geometry evaluation
# ---------------------------------------------------------------------------

def validate_state_dataset(df: pd.DataFrame) -> None:
    required = [
        "range_name", "range_index",
        "c1", "c2", "c3", "basin",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"State dataset missing required columns: {missing}")

    if len(df) < 6:
        raise ValueError("Need at least 6 state rows.")

    idx = pd.to_numeric(df["range_index"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(idx, np.arange(len(df))):
        raise ValueError(
            "State dataset range_index is not exactly 0..n-1 in physical order."
        )

    if df["range_name"].duplicated().any():
        raise ValueError("Duplicate range_name values in state dataset.")

    for c in ("c1", "c2", "c3"):
        vals = pd.to_numeric(df[c], errors="raise").to_numpy(float)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Non-finite values in state coordinate {c}.")


def evaluate_config(
    b50,
    state_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[Dict[str, Any], np.ndarray]:
    xcol, ycol = projection_columns(cfg.projection)
    points = state_df[[xcol, ycol]].to_numpy(dtype=float)

    gx = np.linspace(
        float(points[:, 0].min() - cfg.margin),
        float(points[:, 0].max() + cfg.margin),
        cfg.grid,
    )
    gy = np.linspace(
        float(points[:, 1].min() - cfg.margin),
        float(points[:, 1].max() + cfg.margin),
        cfg.grid,
    )

    t0 = time.time()
    _, _, U, dens = b50.kde_potential(points, gx, gy, cfg.bandwidth)
    minima = b50.local_minima(
        U,
        min_separation=cfg.min_separation,
        max_minima=cfg.max_minima,
    )
    basin_grid = b50.basin_descent_labels(U, minima)

    labels: List[int] = []
    U_occ: List[float] = []
    density_occ: List[float] = []

    for p in points:
        gi, gj = b50.grid_index_for_point(
            float(p[0]), float(p[1]), gx, gy
        )
        lab = int(basin_grid[gi, gj]) if len(minima) else -1
        labels.append(lab)
        U_occ.append(float(U[gi, gj]))
        density_occ.append(float(dens[gi, gj]))

    labels_arr = np.asarray(labels, dtype=int)
    jumps = jump_vector(labels_arr)

    occupied, counts = np.unique(labels_arr, return_counts=True)
    # If no minima, -1 is a sentinel, not a meaningful occupied basin.
    valid_mask = occupied >= 0
    occupied_valid = occupied[valid_mask]
    counts_valid = counts[valid_mask]

    n_occ = int(len(occupied_valid))
    if n_occ > 0:
        fractions = counts_valid.astype(float) / counts_valid.sum()
        largest = float(np.max(fractions))
        smallest = float(np.min(fractions))
        singleton_count = int(np.sum(counts_valid == 1))
    else:
        largest = float("nan")
        smallest = float("nan")
        singleton_count = 0

    rec = {
        "config_id": cfg.config_id,
        "families": cfg.families,
        "projection": cfg.projection,
        "grid": int(cfg.grid),
        "bandwidth": float(cfg.bandwidth),
        "min_separation": int(cfg.min_separation),
        "max_minima": int(cfg.max_minima),
        "margin": float(cfg.margin),
        "n_states": int(len(labels_arr)),
        "n_transitions": int(len(jumps)),
        "n_minima": int(len(minima)),
        "n_occupied_basins": n_occ,
        "n_jumps": int(np.sum(jumps)),
        "jump_rate": float(np.mean(jumps)) if len(jumps) else float("nan"),
        "largest_basin_fraction": largest,
        "smallest_occupied_basin_fraction": smallest,
        "occupancy_entropy_norm": normalized_entropy(labels_arr),
        "singleton_basin_count": singleton_count,
        "mean_U_occ": float(np.mean(U_occ)),
        "std_U_occ": float(np.std(U_occ)),
        "mean_density_occ": float(np.mean(density_occ)),
        "std_density_occ": float(np.std(density_occ)),
        "nondegenerate": bool(n_occ >= 2 and int(np.sum(jumps)) >= 1),
        "balanced_enough": bool(
            n_occ >= 2
            and np.isfinite(largest)
            and np.isfinite(smallest)
            and largest <= 0.98
            and smallest >= 0.01
        ),
        "elapsed_seconds": float(time.time() - t0),
    }
    return rec, labels_arr


# ---------------------------------------------------------------------------
# Stability / interpretation
# ---------------------------------------------------------------------------

def find_baseline_row(results: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    m = (
        (results["projection"] == "c1c2")
        & (results["grid"] == args.baseline_grid)
        & np.isclose(results["bandwidth"], args.baseline_bandwidth)
        & (results["min_separation"] == args.baseline_min_separation)
        & (results["max_minima"] == args.baseline_max_minima)
    )
    rows = results[m]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one baseline row, found {len(rows)}")
    return rows.iloc[0]


def compute_pairwise_stability(
    native_ids: List[str],
    assignments: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for i, a_id in enumerate(native_ids):
        for j, b_id in enumerate(native_ids):
            if j < i:
                continue
            a = assignments[a_id]
            b = assignments[b_id]
            rows.append({
                "config_a": a_id,
                "config_b": b_id,
                "ari": adjusted_rand_index(a, b),
                "jump_jaccard": jump_jaccard(jump_vector(a), jump_vector(b)),
            })
    return pd.DataFrame(rows)


def median_pairwise_ari_for_ids(
    pairwise: pd.DataFrame,
    ids: set[str],
) -> float:
    if len(ids) < 2:
        return float("nan")
    sub = pairwise[
        pairwise["config_a"].isin(ids)
        & pairwise["config_b"].isin(ids)
        & (pairwise["config_a"] != pairwise["config_b"])
    ]
    if sub.empty:
        return float("nan")
    return float(sub["ari"].median())


def classify_native_geometry(
    native: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> Dict[str, Any]:
    n = int(len(native))
    n_nondeg = int(native["nondegenerate"].sum())
    frac_nondeg = n_nondeg / n if n else 0.0

    nondeg = native[native["nondegenerate"]].copy()
    n_bal = int(nondeg["balanced_enough"].sum()) if len(nondeg) else 0
    frac_bal_given_nondeg = n_bal / len(nondeg) if len(nondeg) else 0.0

    nondeg_ids = set(nondeg["config_id"].astype(str))
    median_ari = median_pairwise_ari_for_ids(pairwise, nondeg_ids)

    if frac_nondeg <= 0.10:
        verdict = "PERSISTENT_DEGENERACY"
        reason = (
            "<=10% of native c1,c2 audit configurations resolve both "
            "multiple occupied basins and at least one physical crossing."
        )
    elif (
        frac_nondeg >= 0.60
        and np.isfinite(median_ari)
        and median_ari >= 0.75
        and frac_bal_given_nondeg >= 0.60
    ):
        verdict = "STABLE_MULTIBASIN_GEOMETRY"
        reason = (
            ">=60% of native configurations are non-degenerate, median "
            "pairwise ARI among non-degenerate configurations is >=0.75, "
            "and >=60% of non-degenerate configurations avoid tiny/sliver basins."
        )
    elif frac_nondeg >= 0.25:
        verdict = "PARAMETER_SENSITIVE_MULTIBASIN"
        reason = (
            "Multiple-basin crossings occur in >=25% of native configurations, "
            "but the pre-declared stability/balance criteria are not all met."
        )
    else:
        verdict = "SPARSE_MULTIBASIN_RESOLUTION"
        reason = (
            "Multiple-basin crossings occur in >10% but <25% of native configurations."
        )

    return {
        "native_config_count": n,
        "native_nondegenerate_count": n_nondeg,
        "native_nondegenerate_fraction": frac_nondeg,
        "nondegenerate_balanced_count": n_bal,
        "balanced_fraction_given_nondegenerate": frac_bal_given_nondeg,
        "median_pairwise_ari_nondegenerate": median_ari,
        "verdict": verdict,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    out: Path,
    results: pd.DataFrame,
    pairwise: pd.DataFrame,
    verdict_json: Dict[str, Any],
) -> None:
    # 1) Bandwidth sweep: occupied basins.
    bw = results[
        (results["projection"] == "c1c2")
        & results["families"].str.contains("bandwidth_sweep", regex=False)
    ].sort_values("bandwidth")
    if not bw.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(bw["bandwidth"], bw["n_occupied_basins"], marker="o")
        plt.xlabel("KDE bandwidth")
        plt.ylabel("occupied basins")
        plt.title("B50.1 native c1,c2 basin resolution vs bandwidth")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / "b50_1_bandwidth_occupied_basins.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(bw["bandwidth"], bw["n_jumps"], marker="o")
        plt.xlabel("KDE bandwidth")
        plt.ylabel("physical basin crossings")
        plt.title("B50.1 native c1,c2 realized crossings vs bandwidth")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / "b50_1_bandwidth_crossings.png", dpi=160)
        plt.close()

    # 2) All native configs: basin count vs crossing count.
    native = results[results["projection"] == "c1c2"].copy()
    if not native.empty:
        plt.figure(figsize=(8, 5))
        plt.scatter(native["n_occupied_basins"], native["n_jumps"], s=70)
        plt.xlabel("occupied basins")
        plt.ylabel("physical basin crossings")
        plt.title("B50.1 native geometry adequacy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / "b50_1_native_basins_vs_crossings.png", dpi=160)
        plt.close()

    # 3) Pairwise ARI heatmap.
    ids = list(native["config_id"].astype(str))
    if ids and not pairwise.empty:
        idx = {k: i for i, k in enumerate(ids)}
        M = np.full((len(ids), len(ids)), np.nan, dtype=float)
        for _, r in pairwise.iterrows():
            a, b = str(r["config_a"]), str(r["config_b"])
            if a in idx and b in idx:
                i, j = idx[a], idx[b]
                M[i, j] = float(r["ari"])
                M[j, i] = float(r["ari"])
        plt.figure(figsize=(9, 8))
        im = plt.imshow(M, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, label="Adjusted Rand Index")
        plt.xlabel("native config index")
        plt.ylabel("native config index")
        plt.title("B50.1 pairwise basin-assignment stability")
        plt.tight_layout()
        plt.savefig(out / "b50_1_pairwise_ari.png", dpi=160)
        plt.close()

    # 4) Existing PCA cumulative capture from original verdict.
    vals = [
        verdict_json.get("rank1_explained"),
        verdict_json.get("rank2_cumulative"),
        verdict_json.get("rank3_cumulative"),
    ]
    if all(v is not None and np.isfinite(float(v)) for v in vals):
        plt.figure(figsize=(7, 5))
        plt.bar(["PC1", "PC1+PC2", "PC1+PC2+PC3"], [float(v) for v in vals])
        plt.ylabel("explained variance / cumulative variance")
        plt.ylim(0, 1)
        plt.title("B50.1 original B50 PCA capture")
        plt.tight_layout()
        plt.savefig(out / "b50_1_pca_capture.png", dpi=160)
        plt.close()


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    state_path = Path(args.state_dataset)
    verdict_path = Path(args.verdict_json)
    b50_path = Path(args.b50_script)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not state_path.exists():
        raise FileNotFoundError(state_path)
    if not verdict_path.exists():
        raise FileNotFoundError(verdict_path)
    if not b50_path.exists():
        raise FileNotFoundError(b50_path)

    state_df = pd.read_csv(state_path)
    validate_state_dataset(state_df)

    verdict_json = json.loads(verdict_path.read_text(encoding="utf-8"))
    b50 = import_b50(b50_path)

    configs = build_protocol(args)

    protocol_modified = (
        args.bandwidths != DEFAULT_BANDWIDTHS
        or args.min_separations != DEFAULT_MIN_SEPARATIONS
        or args.max_minima_values != DEFAULT_MAX_MINIMA
        or args.grids != DEFAULT_GRIDS
        or args.interaction_bandwidths != DEFAULT_INTERACTION_BANDWIDTHS
        or args.interaction_min_separations != DEFAULT_INTERACTION_MIN_SEPARATIONS
        or args.projections != DEFAULT_PROJECTIONS
        or args.baseline_grid != BASELINE["grid"]
        or abs(args.baseline_bandwidth - BASELINE["bandwidth"]) > 1e-12
        or args.baseline_min_separation != BASELINE["min_separation"]
        or args.baseline_max_minima != BASELINE["max_minima"]
        or abs(args.margin - BASELINE["margin"]) > 1e-12
    )

    manifest = {
        "version": VERSION,
        "created": now_s(),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "state_dataset": str(state_path),
            "state_dataset_sha256": sha256_file(state_path),
            "verdict_json": str(verdict_path),
            "verdict_json_sha256": sha256_file(verdict_path),
            "b50_script": str(b50_path),
            "b50_script_sha256": sha256_file(b50_path),
        },
        "n_states": int(len(state_df)),
        "n_physical_transitions": int(len(state_df) - 1),
        "baseline": {
            "projection": "c1c2",
            "grid": args.baseline_grid,
            "bandwidth": args.baseline_bandwidth,
            "min_separation": args.baseline_min_separation,
            "max_minima": args.baseline_max_minima,
            "margin": args.margin,
        },
        "protocol_modified_from_default": bool(protocol_modified),
        "b59_or_outcome_labels_used": False,
        "config_count": int(len(configs)),
        "configs": [asdict(c) for c in configs],
        "adequacy_thresholds": {
            "persistent_degeneracy_max_nondegenerate_fraction": 0.10,
            "stable_min_nondegenerate_fraction": 0.60,
            "stable_min_median_pairwise_ari": 0.75,
            "stable_min_balanced_fraction_given_nondegenerate": 0.60,
            "parameter_sensitive_min_nondegenerate_fraction": 0.25,
            "balanced_largest_basin_fraction_max": 0.98,
            "balanced_smallest_occupied_basin_fraction_min": 0.01,
        },
    }
    atomic_json(out / "b50_1_protocol_manifest.json", manifest)

    original_labels = pd.to_numeric(
        state_df["basin"], errors="raise"
    ).astype(int).to_numpy()
    original_jumps = jump_vector(original_labels)

    rows: List[Dict[str, Any]] = []
    assignments: Dict[str, np.ndarray] = {}
    long_rows: List[Dict[str, Any]] = []

    print("=== TDC TOR A / B50.1: STATE GEOMETRY ADEQUACY AUDIT ===")
    print(f"states           : {len(state_df)}")
    print(f"transitions      : {len(state_df)-1}")
    print(f"configs          : {len(configs)}")
    print(f"protocol modified: {protocol_modified}")
    print(f"B50 source       : {b50_path}")
    print()

    for k, cfg in enumerate(configs, start=1):
        print(
            f"[{k:02d}/{len(configs):02d}] {cfg.config_id} "
            f"families={cfg.families}",
            flush=True,
        )
        rec, labels = evaluate_config(b50, state_df, cfg)
        assignments[cfg.config_id] = labels

        rec["ari_vs_original_saved_basin"] = adjusted_rand_index(
            labels, original_labels
        )
        rec["jump_vector_matches_original"] = bool(
            np.array_equal(jump_vector(labels), original_jumps)
        )
        rec["jump_jaccard_vs_original"] = jump_jaccard(
            jump_vector(labels), original_jumps
        )
        rows.append(rec)

        for i, lab in enumerate(labels):
            long_rows.append({
                "config_id": cfg.config_id,
                "projection": cfg.projection,
                "range_index": int(state_df.iloc[i]["range_index"]),
                "range_name": str(state_df.iloc[i]["range_name"]),
                "basin_audit": int(lab),
                "basin_original": int(original_labels[i]),
            })

        print(
            f"    minima={rec['n_minima']} "
            f"occupied={rec['n_occupied_basins']} "
            f"jumps={rec['n_jumps']} "
            f"largest={rec['largest_basin_fraction']:.6g} "
            f"ARI(saved)={rec['ari_vs_original_saved_basin']:.6g} "
            f"time={rec['elapsed_seconds']:.2f}s",
            flush=True,
        )

    results = pd.DataFrame(rows)
    atomic_csv(out / "b50_1_geometry_sweep.csv", results)
    atomic_csv(out / "b50_1_assignments_long.csv", pd.DataFrame(long_rows))

    baseline_row = find_baseline_row(results, args)
    baseline_id = str(baseline_row["config_id"])
    baseline_labels = assignments[baseline_id]

    baseline_ari = adjusted_rand_index(baseline_labels, original_labels)
    baseline_jump_exact = bool(
        np.array_equal(jump_vector(baseline_labels), original_jumps)
    )
    baseline_reproduction_ok = bool(
        baseline_ari >= 0.999999 and baseline_jump_exact
    )

    # ARI vs baseline for every config.
    results["ari_vs_recomputed_baseline"] = [
        adjusted_rand_index(assignments[cid], baseline_labels)
        for cid in results["config_id"].astype(str)
    ]
    atomic_csv(out / "b50_1_geometry_sweep.csv", results)

    native = results[results["projection"] == "c1c2"].copy()
    native_ids = list(native["config_id"].astype(str))
    pairwise = compute_pairwise_stability(native_ids, assignments)
    atomic_csv(out / "b50_1_pairwise_native_stability.csv", pairwise)

    classification = classify_native_geometry(native, pairwise)

    if not baseline_reproduction_ok:
        final_verdict = "INVALID_BASELINE_REPRODUCTION"
        final_reason = (
            "B50.1 could not semantically reproduce the saved original B50 "
            "baseline partition/jump vector using the imported original B50 geometry."
        )
    else:
        final_verdict = classification["verdict"]
        final_reason = classification["reason"]

    projection_rows = results[
        results["families"].str.contains("projection_diagnostic", regex=False)
        | (
            (results["projection"] == "c1c2")
            & results["families"].str.contains("baseline", regex=False)
        )
    ][[
        "config_id",
        "projection",
        "n_minima",
        "n_occupied_basins",
        "n_jumps",
        "largest_basin_fraction",
        "smallest_occupied_basin_fraction",
        "occupancy_entropy_norm",
        "ari_vs_original_saved_basin",
        "ari_vs_recomputed_baseline",
    ]].copy()
    atomic_csv(out / "b50_1_projection_diagnostic.csv", projection_rows)

    pca_capture = {
        "rank1_explained": verdict_json.get("rank1_explained"),
        "rank2_cumulative": verdict_json.get("rank2_cumulative"),
        "rank3_cumulative": verdict_json.get("rank3_cumulative"),
    }

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "protocol_modified_from_default": bool(protocol_modified),
        "outcome_blind_to_b59": True,
        "input_integrity": {
            "n_states": int(len(state_df)),
            "n_transitions": int(len(state_df) - 1),
            "original_saved_occupied_basins": int(
                len(np.unique(original_labels[original_labels >= 0]))
            ),
            "original_saved_jumps": int(np.sum(original_jumps)),
        },
        "baseline_reproduction": {
            "baseline_config_id": baseline_id,
            "ari_vs_saved_basin": baseline_ari,
            "jump_vector_exact": baseline_jump_exact,
            "ok": baseline_reproduction_ok,
            "recomputed_occupied_basins": int(
                baseline_row["n_occupied_basins"]
            ),
            "recomputed_jumps": int(baseline_row["n_jumps"]),
        },
        "pca_capture_from_original_b50": pca_capture,
        "native_c1c2_audit": classification,
        "projection_diagnostic": projection_rows.to_dict(orient="records"),
        "final_verdict": final_verdict,
        "final_reason": final_reason,
        "scientific_boundary": (
            "B50.1 audits the robustness/adequacy of the existing B50 state "
            "representation. It does not validate forecasting skill, B59, "
            "individual-prime prediction, or a number-theoretic theorem."
        ),
        "next_step_rule": {
            "STABLE_MULTIBASIN_GEOMETRY": (
                "Freeze one geometry protocol prospectively, then validate "
                "B59 against realized crossings on held-out physical ranges."
            ),
            "PARAMETER_SENSITIVE_MULTIBASIN": (
                "Do not select a winning parameter from these outcomes. "
                "Design an independent geometry criterion or held-out stability test."
            ),
            "SPARSE_MULTIBASIN_RESOLUTION": (
                "Treat B50 crossing endpoint as weak/fragile; improve representation "
                "before predictive validation."
            ),
            "PERSISTENT_DEGENERACY": (
                "Close current B50 basin endpoint as inadequate for crossing validation "
                "under this audit neighborhood; revisit state representation/dimensionality."
            ),
            "INVALID_BASELINE_REPRODUCTION": (
                "Stop. Resolve implementation/provenance mismatch before interpretation."
            ),
        },
    }
    atomic_json(out / "b50_1_summary.json", summary)

    make_plots(out, results, pairwise, verdict_json)

    report = f"""B50.1 — State Geometry Adequacy & Basin Resolution Audit
==========================================================

Input integrity
---------------
states                         = {len(state_df)}
physical transitions           = {len(state_df)-1}
saved occupied basins          = {summary['input_integrity']['original_saved_occupied_basins']}
saved realized crossings       = {summary['input_integrity']['original_saved_jumps']}
protocol modified from default = {protocol_modified}
B59/B59.1 labels used           = NO

Baseline reproduction
---------------------
config                         = {baseline_id}
ARI vs saved basin             = {baseline_ari}
jump vector exact              = {baseline_jump_exact}
reproduction OK                = {baseline_reproduction_ok}
recomputed occupied basins     = {int(baseline_row['n_occupied_basins'])}
recomputed crossings           = {int(baseline_row['n_jumps'])}

Original B50 PCA capture
------------------------
rank1 explained                = {pca_capture['rank1_explained']}
rank2 cumulative               = {pca_capture['rank2_cumulative']}
rank3 cumulative               = {pca_capture['rank3_cumulative']}

Native c1,c2 frozen audit
-------------------------
configs                        = {classification['native_config_count']}
non-degenerate configs         = {classification['native_nondegenerate_count']}
non-degenerate fraction        = {classification['native_nondegenerate_fraction']}
balanced fraction | nondeg     = {classification['balanced_fraction_given_nondegenerate']}
median ARI | nondeg            = {classification['median_pairwise_ari_nondegenerate']}

FINAL VERDICT: {final_verdict}

REASON:
{final_reason}

Interpretation boundary
-----------------------
This is a representation adequacy / robustness audit.  It is not a B59
validation, not prospective forecasting evidence, not individual-prime
prediction, and not a theorem.

Key outputs
-----------
b50_1_protocol_manifest.json
b50_1_geometry_sweep.csv
b50_1_assignments_long.csv
b50_1_pairwise_native_stability.csv
b50_1_projection_diagnostic.csv
b50_1_summary.json
b50_1_verdict.txt
b50_1_bandwidth_occupied_basins.png
b50_1_bandwidth_crossings.png
b50_1_native_basins_vs_crossings.png
b50_1_pairwise_ari.png
b50_1_pca_capture.png
"""
    atomic_text(out / "b50_1_verdict.txt", report)

    print()
    print("=== B50.1 RESULT ===")
    print(f"baseline reproduction : {baseline_reproduction_ok}")
    print(
        "native nondegenerate  : "
        f"{classification['native_nondegenerate_count']}/"
        f"{classification['native_config_count']} "
        f"({classification['native_nondegenerate_fraction']:.3f})"
    )
    print(
        "median ARI | nondeg   : "
        f"{classification['median_pairwise_ari_nondegenerate']}"
    )
    print(f"FINAL VERDICT         : {final_verdict}")
    print(f"summary               : {out / 'b50_1_summary.json'}")
    print(f"verdict               : {out / 'b50_1_verdict.txt'}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B50.1 State Geometry Adequacy & Basin Resolution Audit"
    )
    p.add_argument("--state-dataset", required=True)
    p.add_argument("--verdict-json", required=True)
    p.add_argument("--b50-script", required=True)
    p.add_argument("--output-dir", required=True)

    # Baseline reproduces original B50 defaults.
    p.add_argument("--baseline-grid", type=int, default=120)
    p.add_argument("--baseline-bandwidth", type=float, default=0.65)
    p.add_argument("--baseline-min-separation", type=int, default=6)
    p.add_argument("--baseline-max-minima", type=int, default=6)
    p.add_argument("--margin", type=float, default=0.5)

    # Frozen outcome-blind audit protocol.
    p.add_argument(
        "--bandwidths",
        default="0.30,0.40,0.50,0.65,0.80,1.00,1.30",
    )
    p.add_argument(
        "--min-separations",
        default="3,4,6,8,10",
    )
    p.add_argument(
        "--max-minima-values",
        default="2,4,6,8,10",
    )
    p.add_argument(
        "--grids",
        default="80,100,120,160",
    )
    p.add_argument(
        "--interaction-bandwidths",
        default="0.50,0.65,0.80",
    )
    p.add_argument(
        "--interaction-min-separations",
        default="4,6,8",
    )
    p.add_argument(
        "--projections",
        default="c1c2,c1c3,c2c3",
    )

    args = p.parse_args()

    args.bandwidths = parse_float_list(args.bandwidths)
    args.min_separations = parse_int_list(args.min_separations)
    args.max_minima_values = parse_int_list(args.max_minima_values)
    args.grids = parse_int_list(args.grids)
    args.interaction_bandwidths = parse_float_list(args.interaction_bandwidths)
    args.interaction_min_separations = parse_int_list(
        args.interaction_min_separations
    )
    args.projections = parse_str_list(args.projections)

    for proj in args.projections:
        projection_columns(proj)

    if args.baseline_grid < 20:
        raise ValueError("--baseline-grid must be >=20")
    if args.baseline_bandwidth <= 0:
        raise ValueError("--baseline-bandwidth must be >0")
    if args.baseline_min_separation < 1:
        raise ValueError("--baseline-min-separation must be >=1")
    if args.baseline_max_minima < 1:
        raise ValueError("--baseline-max-minima must be >=1")
    if args.margin <= 0:
        raise ValueError("--margin must be >0")

    return args


def main() -> int:
    args = parse_args()
    run_audit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
