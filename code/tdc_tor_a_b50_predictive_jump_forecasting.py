#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDC TOR A / B50: predictive jump forecasting

Purpose
-------
B49 showed that observed basin jumps are consistent with barrier-crossing physics.
B50 turns that into a predictive test:

    Can we forecast basin jumps before they happen?

We evaluate pre-jump features only at transition source state s_i:
- U(s_i)
- distance to nearest basin boundary
- barrier_proxy = U_boundary - U_min(basin)
- local noise trace Sigma_k
- barrier/noise
- Kramers-like score exp(-barrier/noise)
- combined hazard score

Core conservative rule
----------------------
With only a few observed transitions, a fitted ML classifier would overfit.
So B50 uses:
1) physics score from B49-style features,
2) leave-one-transition-out thresholding,
3) rank/AUC diagnostics,
4) next-jump alert table,
5) precision/recall sweep.

Run
---
python .\scripts\tdc_torA_b50_predictive_jump_forecasting.py --root d:\1e12_ranges --output-dir torA_b50_output --gap-bins 64

Outputs
-------
CSV/JSON:
- b50_state_dataset.csv
- b50_forecast_events.csv
- b50_leave_one_out_predictions.csv
- b50_threshold_sweep.csv
- b50_model_summary.csv
- b50_verdict_scaffold.json

Figures:
- b50_forecast_score_by_transition.png
- b50_leave_one_out_prediction_overlay.png
- b50_precision_recall_sweep.png
- b50_roc_like_curve.png
- b50_calibration_curve.png
- b50_prejump_feature_panel.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# I/O and operator construction
# -----------------------------

def parse_prime_file(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"\d+", text)
    if not nums:
        return np.array([], dtype=np.int64)
    return np.array([int(x) for x in nums], dtype=np.int64)


def collect_range_prime_files(range_dir: Path) -> list[Path]:
    files = sorted(range_dir.glob("primes_batch_*_all_with_1.txt"))
    if not files:
        files = sorted(range_dir.glob("*.txt"))
    return files


def load_full_range_primes(range_dir: Path) -> np.ndarray:
    files = collect_range_prime_files(range_dir)
    if not files:
        raise FileNotFoundError(f"No TXT prime files found in {range_dir}")
    chunks = []
    for fp in files:
        arr = parse_prime_file(fp)
        if len(arr) > 0:
            chunks.append(arr)
    if not chunks:
        raise ValueError(f"All TXT files in {range_dir} are empty or unreadable.")   
    primes = np.concatenate(chunks)
    primes = primes[primes > 2] 
    primes = np.unique(primes)
    primes.sort()
    return primes


def primes_to_gaps(primes: np.ndarray) -> np.ndarray:
    if len(primes) < 3:
        return np.array([], dtype=np.int64)
    gaps = np.diff(primes)
    return gaps[gaps > 0]


def build_markov_operator(values: np.ndarray, n_bins: int = 64) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        hi = lo + 1e-12
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.digitize(values, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    counts = np.zeros((n_bins, n_bins), dtype=float)
    for a, b in zip(idx[:-1], idx[1:]):
        counts[int(a), int(b)] += 1.0

    P = counts.copy()
    row_sums = P.sum(axis=1, keepdims=True)
    global_hist = counts.sum(axis=0)
    if global_hist.sum() <= 0:
        global_hist = np.ones(n_bins, dtype=float)
    global_hist /= global_hist.sum()

    for i in range(n_bins):
        if row_sums[i, 0] <= 0:
            P[i] = global_hist
        else:
            P[i] /= row_sums[i, 0]
    return P


def pca_decompose(P_stack: np.ndarray):
    mean_P = np.mean(P_stack, axis=0, keepdims=True)
    centered = P_stack - mean_P
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    scores = U * S
    explained = (S ** 2) / np.sum(S ** 2) if np.sum(S ** 2) > 0 else np.zeros_like(S)
    cumulative = np.cumsum(explained)
    return mean_P[0], scores, Vt, explained, cumulative


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# -----------------------------
# KDE potential / basins
# -----------------------------

def kde_potential(points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, bandwidth: float):
    X, Y = np.meshgrid(grid_x, grid_y)
    grid = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    dens = np.zeros(len(grid), dtype=float)
    bw2 = bandwidth * bandwidth
    norm = 1.0 / (2.0 * np.pi * bw2)
    for p in points:
        d2 = np.sum((grid - p[None, :]) ** 2, axis=1)
        dens += norm * np.exp(-0.5 * d2 / bw2)
    dens /= max(len(points), 1)
    U = -np.log(dens + 1e-12)
    return X, Y, U.reshape(X.shape), dens.reshape(X.shape)


def local_minima(U: np.ndarray, min_separation: int = 6, max_minima: int = 6):
    H, W = U.shape
    candidates = []
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            patch = U[i - 1:i + 2, j - 1:j + 2]
            if U[i, j] <= np.min(patch) + 1e-12:
                candidates.append((float(U[i, j]), i, j))
    candidates.sort(key=lambda x: x[0])
    kept = []
    for val, i, j in candidates:
        if all(np.sqrt((i - ii) ** 2 + (j - jj) ** 2) >= min_separation for _, ii, jj in kept):
            kept.append((val, i, j))
        if len(kept) >= max_minima:
            break
    return kept


def basin_descent_labels(U: np.ndarray, minima: list[tuple[float, int, int]]):
    H, W = U.shape
    min_index = {(i, j): k for k, (_, i, j) in enumerate(minima)}
    memo = {}
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def descend(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        if (i, j) in min_index:
            memo[(i, j)] = min_index[(i, j)]
            return memo[(i, j)]
        bi, bj = i, j
        bv = U[i, j]
        for di, dj in neigh:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and U[ni, nj] < bv:
                bv = U[ni, nj]
                bi, bj = ni, nj
        if bi == i and bj == j:
            d = [np.sqrt((i - mi) ** 2 + (j - mj) ** 2) for _, mi, mj in minima]
            lab = int(np.argmin(d)) if d else -1
        else:
            lab = descend(bi, bj)
        memo[(i, j)] = lab
        return lab

    labels = np.full((H, W), -1, dtype=int)
    for i in range(H):
        for j in range(W):
            labels[i, j] = descend(i, j)
    return labels


def grid_index_for_point(x: float, y: float, gx: np.ndarray, gy: np.ndarray):
    j = int(np.argmin(np.abs(gx - x)))
    i = int(np.argmin(np.abs(gy - y)))
    return i, j


def nearest_boundary_info(i: int, j: int, basin_grid: np.ndarray, U: np.ndarray, max_radius: int = 90):
    label = basin_grid[i, j]
    H, W = basin_grid.shape
    for r in range(1, max_radius + 1):
        i0, i1 = max(0, i - r), min(H - 1, i + r)
        j0, j1 = max(0, j - r), min(W - 1, j + r)
        found = []
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                if basin_grid[ii, jj] != label:
                    d = math.sqrt((ii - i) ** 2 + (jj - j) ** 2)
                    found.append((d, ii, jj, U[ii, jj], basin_grid[ii, jj]))
        if found:
            found.sort(key=lambda z: (z[0], z[3]))
            d, ii, jj, u_b, lab_b = found[0]
            return float(d), float(u_b), int(lab_b), int(ii), int(jj)
    return float(max_radius), float("nan"), -1, -1, -1


def basin_minima_values(minima: list[tuple[float, int, int]], gx: np.ndarray, gy: np.ndarray) -> pd.DataFrame:
    rows = []
    for k, (u, i, j) in enumerate(minima):
        rows.append({"basin": k, "min_U": float(u), "min_c1": float(gx[j]), "min_c2": float(gy[i]), "grid_i": i, "grid_j": j})
    return pd.DataFrame(rows)


# -----------------------------
# Noise / hazard features
# -----------------------------

def safe_cov(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if len(X) <= 1:
        return np.eye(3) * eps
    C = np.cov(X.T)
    C = np.asarray(C, dtype=float)
    C += eps * np.eye(3)
    return C


def estimate_basin_noise(state: np.ndarray, basins: list[int], shrinkage: float):
    S_from = state[:-1]
    S_next = state[1:]
    V = S_next - S_from
    bas_from = np.asarray(basins[:-1], dtype=int)
    global_mu = np.mean(V, axis=0)
    global_cov = safe_cov(V - global_mu[None, :])
    global_trace = float(np.trace(global_cov))

    params = {}
    for b in sorted(set(int(x) for x in bas_from if x >= 0)):
        idx = np.where(bas_from == b)[0]
        n = len(idx)
        w = n / (n + shrinkage)
        local_mu = np.mean(V[idx], axis=0) if n > 0 else global_mu
        local_cov = safe_cov(V[idx] - local_mu[None, :]) if n > 1 else global_cov
        cov = (1.0 - w) * global_cov + w * local_cov
        params[b] = {
            "n": n,
            "mu": (1.0 - w) * global_mu + w * local_mu,
            "cov": cov,
            "trace": float(np.trace(cov)),
            "weight": float(w),
        }
    return params, global_trace


def minmax_score(x: np.ndarray, invert: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)):
        return np.zeros_like(x)
    med = np.nanmedian(x)
    x = np.where(np.isfinite(x), x, med)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        y = np.zeros_like(x)
    else:
        y = (x - lo) / (hi - lo)
    return 1.0 - y if invert else y


def build_hazard_score(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()
    boundary_score = minmax_score(df["boundary_distance_grid"].to_numpy(), invert=True)
    U_score = minmax_score(df["U_occ"].to_numpy(), invert=False)
    kramers_score = minmax_score(df["kramers_score_raw"].to_numpy(), invert=False)
    noise_score = minmax_score(df["local_noise_trace"].to_numpy(), invert=False)
    eon_score = minmax_score(df["energy_over_noise"].to_numpy(), invert=True)

    df["boundary_score"] = boundary_score
    df["U_score"] = U_score
    df["kramers_score_norm"] = kramers_score
    df["noise_score"] = noise_score
    df["energy_over_noise_score"] = eon_score
    df["hazard_score"] = (
        0.30 * boundary_score
        + 0.20 * U_score
        + 0.25 * kramers_score
        + 0.15 * noise_score
        + 0.10 * eon_score
    )
    return df


# -----------------------------
# Forecast validation
# -----------------------------

def auc_rank_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins, total = 0.0, 0.0
    for p in pos:
        for n in neg:
            total += 1.0
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return float(wins / total)


def threshold_sweep(y_true: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ths = np.unique(np.quantile(s, np.linspace(0, 1, 101)))
    rows = []
    for th in ths:
        pred = (s >= th).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tpr
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        specificity = tn / max(tn + fp, 1)
        rows.append({
            "threshold": float(th), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tpr": float(tpr), "fpr": float(fpr), "specificity": float(specificity),
        })
    return pd.DataFrame(rows)


def loo_threshold_predictions(y: np.ndarray, h: np.ndarray, target_mode: str = "best_f1") -> pd.DataFrame:
    rows = []
    n = len(y)
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        sweep = threshold_sweep(y[train_idx], h[train_idx])
        if target_mode == "high_precision":
            candidates = sweep[sweep["precision"] >= 0.75]
            if len(candidates) == 0:
                best = sweep.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
            else:
                best = candidates.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
        else:
            best = sweep.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
        th = float(best["threshold"])
        pred = int(h[i] >= th)
        rows.append({
            "transition_index": i,
            "hazard_score": float(h[i]),
            "actual_jump": int(y[i]),
            "loo_threshold": th,
            "predicted_jump": pred,
            "correct": int(pred == y[i]),
            "train_best_f1": float(best["f1"]),
            "train_precision": float(best["precision"]),
            "train_recall": float(best["recall"]),
        })
    return pd.DataFrame(rows)


def reliability_bins(y: np.ndarray, score: np.ndarray, n_bins: int = 4) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi if i == n_bins - 1 else s < hi)
        if mask.sum() == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "mean_score": float("nan"), "empirical_jump_rate": float("nan")})
        else:
            rows.append({
                "bin_lo": float(lo), "bin_hi": float(hi), "n": int(mask.sum()),
                "mean_score": float(np.mean(s[mask])),
                "empirical_jump_rate": float(np.mean(y[mask])),
            })
    return pd.DataFrame(rows)


# -----------------------------
# Plots
# -----------------------------

def plot_forecast_score(df: pd.DataFrame, path: Path):
    plt.figure(figsize=(10, 5))
    colors = ["red" if j else "gray" for j in df["is_jump"]]
    plt.bar(df["transition_index"].astype(str), df["hazard_score"], color=colors)
    plt.xlabel("transition index")
    plt.ylabel("pre-jump forecast hazard score")
    plt.title("B50 predictive jump forecast score; red = observed jump")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_loo_overlay(loo_df: pd.DataFrame, path: Path):
    x = loo_df["transition_index"].to_numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(x, loo_df["hazard_score"], marker="o", label="hazard score")
    plt.plot(x, loo_df["loo_threshold"], marker="x", linestyle="--", label="LOO threshold")
    for _, r in loo_df.iterrows():
        if r["actual_jump"] == 1:
            plt.axvline(r["transition_index"], color="red", alpha=0.18)
    plt.xlabel("transition index")
    plt.ylabel("score / threshold")
    plt.title("B50 leave-one-out jump forecast overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_pr_sweep(sweep_df: pd.DataFrame, path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["threshold"], sweep_df["precision"], marker="o", label="precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], marker="o", label="recall")
    plt.plot(sweep_df["threshold"], sweep_df["f1"], marker="o", label="F1")
    plt.xlabel("threshold")
    plt.ylabel("metric")
    plt.title("B50 precision / recall / F1 threshold sweep")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_roc(sweep_df: pd.DataFrame, auc: float, path: Path):
    plt.figure(figsize=(6, 6))
    plt.plot(sweep_df["fpr"], sweep_df["tpr"], marker="o", label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="random")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("B50 ROC-like predictive curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_calibration(cal_df: pd.DataFrame, path: Path):
    plt.figure(figsize=(6, 6))
    valid = cal_df[cal_df["n"] > 0]
    plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    plt.scatter(valid["mean_score"], valid["empirical_jump_rate"], s=80)
    for _, r in valid.iterrows():
        plt.text(r["mean_score"], r["empirical_jump_rate"], f"n={int(r['n'])}", fontsize=9)
    plt.xlabel("mean forecast score")
    plt.ylabel("empirical jump rate")
    plt.title("B50 calibration curve, coarse bins")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_feature_panel(df: pd.DataFrame, path: Path):
    cols = ["hazard_score", "kramers_score_raw", "boundary_distance_grid", "energy_over_noise"]
    titles = ["hazard", "Kramers", "boundary distance", "barrier/noise"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    for ax, col, title in zip(axes, cols, titles):
        colors = ["red" if j else "gray" for j in df["is_jump"]]
        ax.bar(df["transition_index"].astype(str), df[col], color=colors)
        ax.set_title(title)
        ax.grid(True, axis="y")
    fig.suptitle("B50 pre-jump feature panel; red = observed jump")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="B50: predictive jump forecasting")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="torA_b50_output")
    parser.add_argument("--gap-bins", type=int, default=64)
    parser.add_argument("--grid", type=int, default=120)
    parser.add_argument("--kde-bandwidth", type=float, default=0.65)
    parser.add_argument("--min-separation", type=int, default=6)
    parser.add_argument("--max-minima", type=int, default=6)
    parser.add_argument("--shrinkage", type=float, default=2.0)
    parser.add_argument("--target-mode", choices=["best_f1", "high_precision"], default="best_f1")
    parser.add_argument("--limit-ranges", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    range_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("range_")])
    if args.limit_ranges > 0:
        range_dirs = range_dirs[: args.limit_ranges]
    if len(range_dirs) < 6:
        raise ValueError("Need at least 6 ranges for B50.")

    print("=== TDC TOR A / B50: PREDICTIVE JUMP FORECASTING ===")
    print(f"Root:       {root}")
    print(f"Output dir: {outdir}")
    print(f"Ranges:     {len(range_dirs)}")

    labels, P_list = [], []
    mean_log_gap, gap_mean, gap_std, n_primes, n_gaps = [], [], [], [], []
    for idx, rd in enumerate(range_dirs, start=1):
        print(f"[{idx}/{len(range_dirs)}] {rd.name}")
        primes = load_full_range_primes(rd)
        gaps = primes_to_gaps(primes).astype(float)
        if len(gaps) < 100:
            print("  skip: too few gaps")
            continue
        P = build_markov_operator(gaps, n_bins=args.gap_bins)
        labels.append(rd.name)
        P_list.append(P.reshape(-1))
        mean_log_gap.append(float(np.mean(np.log(gaps))))
        gap_mean.append(float(np.mean(gaps)))
        gap_std.append(float(np.std(gaps)))
        n_primes.append(int(len(primes)))
        n_gaps.append(int(len(gaps)))

    P_stack = np.vstack(P_list)
    _, scores, _, explained, cumulative = pca_decompose(P_stack)
    state = np.column_stack([scores[:, 0], scores[:, 1], scores[:, 2]])
    if safe_corr(state[:, 0], np.array(mean_log_gap)) < 0:
        state[:, 0] *= -1.0
    if safe_corr(state[:, 1], np.array(gap_std)) < 0:
        state[:, 1] *= -1.0

    # KDE basins
    margin = 0.5
    gx = np.linspace(float(state[:, 0].min() - margin), float(state[:, 0].max() + margin), args.grid)
    gy = np.linspace(float(state[:, 1].min() - margin), float(state[:, 1].max() + margin), args.grid)
    _, _, U, dens = kde_potential(state[:, :2], gx, gy, args.kde_bandwidth)
    minima = local_minima(U, min_separation=args.min_separation, max_minima=args.max_minima)
    basin_grid = basin_descent_labels(U, minima)
    minima_df = basin_minima_values(minima, gx, gy)

    basins, U_occ, density, boundary_d, boundary_U, boundary_basin = [], [], [], [], [], []
    for s in state:
        gi, gj = grid_index_for_point(float(s[0]), float(s[1]), gx, gy)
        b = int(basin_grid[gi, gj]) if len(minima) else -1
        d, u_b, lab_b, _, _ = nearest_boundary_info(gi, gj, basin_grid, U)
        basins.append(b)
        U_occ.append(float(U[gi, gj]))
        density.append(float(dens[gi, gj]))
        boundary_d.append(float(d))
        boundary_U.append(float(u_b))
        boundary_basin.append(int(lab_b))

    state_df = pd.DataFrame({
        "range_name": labels,
        "range_index": np.arange(len(labels)),
        "mean_log_gap": mean_log_gap,
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "n_primes": n_primes,
        "n_gaps": n_gaps,
        "c1": state[:, 0], "c2": state[:, 1], "c3": state[:, 2],
        "basin": basins,
        "U_occ": U_occ,
        "density": density,
        "boundary_distance_grid": boundary_d,
        "nearest_boundary_U": boundary_U,
        "nearest_boundary_basin": boundary_basin,
    })
    state_df.to_csv(outdir / "b50_state_dataset.csv", index=False)

    basin_noise, global_noise_trace = estimate_basin_noise(state, basins, args.shrinkage)

    events = []
    min_U_lookup = {int(r["basin"]): float(r["min_U"]) for _, r in minima_df.iterrows()}
    for i in range(len(labels) - 1):
        b = int(basins[i])
        b_next = int(basins[i + 1])
        ds = state[i + 1] - state[i]
        min_U = min_U_lookup.get(b, float("nan"))
        barrier = float(boundary_U[i] - min_U) if np.isfinite(boundary_U[i]) and np.isfinite(min_U) else float("nan")
        local_noise = basin_noise.get(b, {"trace": global_noise_trace})["trace"]
        energy_over_noise = barrier / max(local_noise, 1e-12) if np.isfinite(barrier) else float("nan")
        kramers_score = math.exp(-energy_over_noise) if np.isfinite(energy_over_noise) else float("nan")
        events.append({
            "transition_index": i,
            "transition": f"{labels[i]} -> {labels[i+1]}",
            "from_range": labels[i],
            "to_range": labels[i + 1],
            "from_basin": b,
            "to_basin": b_next,
            "is_jump": bool(b != b_next),
            "from_c1": float(state[i, 0]), "from_c2": float(state[i, 1]), "from_c3": float(state[i, 2]),
            "to_c1": float(state[i + 1, 0]), "to_c2": float(state[i + 1, 1]), "to_c3": float(state[i + 1, 2]),
            "dc1": float(ds[0]), "dc2": float(ds[1]), "dc3": float(ds[2]),
            "step_l2": float(np.linalg.norm(ds)),
            "U_occ": float(U_occ[i]),
            "basin_min_U": float(min_U),
            "nearest_boundary_U": float(boundary_U[i]),
            "boundary_distance_grid": float(boundary_d[i]),
            "nearest_boundary_basin": int(boundary_basin[i]),
            "barrier_proxy": float(barrier),
            "local_noise_trace": float(local_noise),
            "energy_over_noise": float(energy_over_noise),
            "kramers_score_raw": float(kramers_score),
        })

    events_df = build_hazard_score(pd.DataFrame(events))
    events_df.to_csv(outdir / "b50_forecast_events.csv", index=False)

    y = events_df["is_jump"].astype(int).to_numpy()
    h = events_df["hazard_score"].to_numpy()
    auc = auc_rank_score(y, h)
    sweep_df = threshold_sweep(y, h)
    sweep_df.to_csv(outdir / "b50_threshold_sweep.csv", index=False)

    loo_df = loo_threshold_predictions(y, h, target_mode=args.target_mode)
    loo_df.to_csv(outdir / "b50_leave_one_out_predictions.csv", index=False)

    cal_df = reliability_bins(y, h, n_bins=4)
    cal_df.to_csv(outdir / "b50_calibration_bins.csv", index=False)

    # Metrics
    pred = loo_df["predicted_jump"].astype(int).to_numpy()
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(len(y), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    best = sweep_df.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]

    summary_df = pd.DataFrame([
        {"metric": "n_transitions", "value": len(y)},
        {"metric": "n_jumps", "value": int(np.sum(y))},
        {"metric": "jump_rate", "value": float(np.mean(y))},
        {"metric": "auc_rank", "value": auc},
        {"metric": "loo_accuracy", "value": accuracy},
        {"metric": "loo_precision", "value": precision},
        {"metric": "loo_recall", "value": recall},
        {"metric": "loo_specificity", "value": specificity},
        {"metric": "loo_f1", "value": f1},
        {"metric": "best_threshold_in_sample", "value": float(best["threshold"])},
        {"metric": "best_f1_in_sample", "value": float(best["f1"])},
        {"metric": "best_precision_in_sample", "value": float(best["precision"])},
        {"metric": "best_recall_in_sample", "value": float(best["recall"])},
    ])
    summary_df.to_csv(outdir / "b50_model_summary.csv", index=False)

    plot_forecast_score(events_df, outdir / "b50_forecast_score_by_transition.png")
    plot_loo_overlay(loo_df, outdir / "b50_leave_one_out_prediction_overlay.png")
    plot_pr_sweep(sweep_df, outdir / "b50_precision_recall_sweep.png")
    plot_roc(sweep_df, auc, outdir / "b50_roc_like_curve.png")
    plot_calibration(cal_df, outdir / "b50_calibration_curve.png")
    plot_feature_panel(events_df, outdir / "b50_prejump_feature_panel.png")

    verdict_label = "predictive_signal_present" if auc >= 0.75 and recall > 0 and precision > 0 else "predictive_signal_inconclusive"
    if len(y) < 20 or int(np.sum(y)) < 5:
        confidence = "low_sample_warning"
    else:
        confidence = "moderate"

    verdict = {
        "n_ranges": int(len(labels)),
        "n_transitions": int(len(y)),
        "n_jumps": int(np.sum(y)),
        "jump_rate": float(np.mean(y)),
        "rank1_explained": float(explained[0]) if len(explained) > 0 else 0.0,
        "rank2_cumulative": float(cumulative[1]) if len(cumulative) > 1 else 0.0,
        "rank3_cumulative": float(cumulative[2]) if len(cumulative) > 2 else 0.0,
        "auc_rank": float(auc),
        "loo_accuracy": float(accuracy),
        "loo_precision": float(precision),
        "loo_recall": float(recall),
        "loo_specificity": float(specificity),
        "loo_f1": float(f1),
        "best_threshold_in_sample": float(best["threshold"]),
        "best_f1_in_sample": float(best["f1"]),
        "model_equation": "forecast P(jump at i) from pre-transition features: U(s_i), boundary distance, barrier/noise, noise trace, Kramers score",
        "verdict": verdict_label,
        "confidence": confidence,
        "warning": "This is a genuine predictive scaffold, but the current transition count is small. Add more ranges before claiming stable forecasting law."
    }
    with open(outdir / "b50_verdict_scaffold.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    for name in [
        "b50_state_dataset.csv",
        "b50_forecast_events.csv",
        "b50_leave_one_out_predictions.csv",
        "b50_threshold_sweep.csv",
        "b50_calibration_bins.csv",
        "b50_model_summary.csv",
        "b50_verdict_scaffold.json",
        "b50_forecast_score_by_transition.png",
        "b50_leave_one_out_prediction_overlay.png",
        "b50_precision_recall_sweep.png",
        "b50_roc_like_curve.png",
        "b50_calibration_curve.png",
        "b50_prejump_feature_panel.png",
    ]:
        print(f"  - {outdir / name}")

    print("\nDone. B50 built predictive jump forecasting scaffold.")


if __name__ == "__main__":
    main()
