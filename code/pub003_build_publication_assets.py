#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PUB-003 — Machine-Derived Publication Assets & Reproducibility Release Builder

Purpose
-------
Build every numerical publication table/figure directly from the frozen
B50.6–B50.12 CSV/JSON outputs and assemble a public reproducibility release.

This tool:
- does NOT rerun scientific inference;
- does NOT rematch controls;
- does NOT retune thresholds;
- does NOT select channels;
- reads only frozen result artifacts;
- writes publication assets, provenance tables, hashes, and a release bundle.

Required lineage
----------------
B50.6  PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED
B50.7  EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED
B50.8  MULTI_OFFSET_PHASE_ROBUST_EFFECT_INVARIANCE
B50.10 CHANNEL_ASSOCIATION_WITHOUT_SCORE_LOCALIZATION
B50.11 PAIRWISE_CORE_RECAPTURES_FULL_EFFECT
B50.12 PAIR_CORE_CROSS_OFFSET_REPLICATED

Default expected directories under --results-root
-----------------------------------------------
b50_6_adjacency_specificity
b50_7_external_offset
b50_8_multi_offset
b50_10_transition_geometry_mechanism
b50_11_channel_interactions
b50_12_pair_core_cross_offset

Output
------
publication/
  figures/
  tables/
  publication_numbers.json
release/
  README.md
  CITATION.cff
  requirements.txt
  CODE_DATA_AVAILABILITY.md
  REPRODUCIBILITY_LEVELS.md
  data/derived/
  code/
  manifests/
pub003_input_audit.csv
pub003_summary.json
pub003_verdict.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "PUB003_publication_assets_release_v1"
EXPECTED = {
    "b50_6": "PHYSICAL_ADJACENCY_SPECIFICITY_REPLICATED",
    "b50_7": "EXTERNAL_OFFSET_ADJACENCY_SPECIFICITY_REPLICATED",
    "b50_8": "MULTI_OFFSET_PHASE_ROBUST_EFFECT_INVARIANCE",
    "b50_10": "CHANNEL_ASSOCIATION_WITHOUT_SCORE_LOCALIZATION",
    "b50_11": "PAIRWISE_CORE_RECAPTURES_FULL_EFFECT",
    "b50_12": "PAIR_CORE_CROSS_OFFSET_REPLICATED",
}
PAIR_KEY = "turn_change+density_scale_change"
OFFSETS = [25, 50, 75, 100, 125]


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
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True))


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def safe_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def result_dirs(args) -> Dict[str, Path]:
    root = Path(args.results_root)
    return {
        "b50_6": Path(args.b50_6_dir) if args.b50_6_dir else root / "b50_6_adjacency_specificity",
        "b50_7": Path(args.b50_7_dir) if args.b50_7_dir else root / "b50_7_external_offset",
        "b50_8": Path(args.b50_8_dir) if args.b50_8_dir else root / "b50_8_multi_offset",
        "b50_10": Path(args.b50_10_dir) if args.b50_10_dir else root / "b50_10_transition_geometry_mechanism",
        "b50_11": Path(args.b50_11_dir) if args.b50_11_dir else root / "b50_11_channel_interactions",
        "b50_12": Path(args.b50_12_dir) if args.b50_12_dir else root / "b50_12_pair_core_cross_offset",
    }


def paths(args) -> Dict[str, Path]:
    d = result_dirs(args)
    return {
        "b50_6_summary": d["b50_6"] / "b50_6_summary.json",
        "b50_6_transitions": d["b50_6"] / "b50_6_transition_specificity.csv",
        "b50_6_matching": d["b50_6"] / "b50_6_matching_audit.csv",
        "b50_6_config": d["b50_6"] / "b50_6_config_specificity.csv",
        "b50_6_protocol": d["b50_6"] / "b50_6_protocol_manifest.json",

        "b50_7_summary": d["b50_7"] / "b50_7_summary.json",
        "b50_7_transitions": d["b50_7"] / "b50_7_transition_specificity.csv",
        "b50_7_matching": d["b50_7"] / "b50_7_matching_audit.csv",
        "b50_7_protocol": d["b50_7"] / "b50_7_protocol_manifest.json",
        "b50_7_stack": d["b50_7"] / "b50_7_offset_operator_stack.npz",

        "b50_8_summary": d["b50_8"] / "b50_8_summary.json",
        "b50_8_offsets": d["b50_8"] / "b50_8_offset_effects.csv",
        "b50_8_plan": d["b50_8"] / "b50_8_plan_summary.json",

        "b50_10_summary": d["b50_10"] / "b50_10_summary.json",
        "b50_10_channels": d["b50_10"] / "b50_10_channel_effects.csv",
        "b50_10_loo": d["b50_10"] / "b50_10_leave_one_out.csv",
        "b50_10_protocol": d["b50_10"] / "b50_10_protocol_manifest.json",
        "b50_10_cube": d["b50_10"] / "b50_10_mechanism_cube.npz",

        "b50_11_summary": d["b50_11"] / "b50_11_summary.json",
        "b50_11_subset": d["b50_11"] / "b50_11_subset_effects.csv",
        "b50_11_minimal": d["b50_11"] / "b50_11_minimal_sufficient_subsets.csv",
        "b50_11_shapley": d["b50_11"] / "b50_11_shapley.csv",
        "b50_11_pairs": d["b50_11"] / "b50_11_pair_interactions.csv",
        "b50_11_protocol": d["b50_11"] / "b50_11_protocol_manifest.json",

        "b50_12_summary": d["b50_12"] / "b50_12_summary.json",
        "b50_12_offsets": d["b50_12"] / "b50_12_offset_effects.csv",
        "b50_12_protocol": d["b50_12"] / "b50_12_protocol_manifest.json",
    }


def verdict_of(summary: Dict[str, Any]) -> str:
    return str(summary.get("final_verdict", ""))


def load_lineage(args) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    ps = paths(args)
    required_keys = [
        "b50_6_summary", "b50_6_transitions", "b50_6_matching", "b50_6_config",
        "b50_7_summary",
        "b50_8_summary", "b50_8_offsets",
        "b50_10_summary", "b50_10_channels", "b50_10_loo",
        "b50_11_summary", "b50_11_subset", "b50_11_minimal", "b50_11_shapley", "b50_11_pairs",
        "b50_12_summary", "b50_12_offsets",
    ]
    for k in required_keys:
        require(ps[k], k)

    sums = {
        "b50_6": safe_json(ps["b50_6_summary"]),
        "b50_7": safe_json(ps["b50_7_summary"]),
        "b50_8": safe_json(ps["b50_8_summary"]),
        "b50_10": safe_json(ps["b50_10_summary"]),
        "b50_11": safe_json(ps["b50_11_summary"]),
        "b50_12": safe_json(ps["b50_12_summary"]),
    }

    rows = []
    ok = True
    for key in EXPECTED:
        got = verdict_of(sums[key])
        pass_ = got == EXPECTED[key]
        ok &= pass_
        rows.append({
            "stage": key,
            "expected_verdict": EXPECTED[key],
            "actual_verdict": got,
            "exact": pass_,
        })

    return {
        "summaries": sums,
        "lineage_rows": rows,
        "pass": bool(ok),
    }, ps


def write_input_audit(out: Path, ps: Mapping[str, Path], lineage: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, p in ps.items():
        if p.exists():
            rows.append({
                "input": k,
                "path": str(p),
                "bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
                "exists": True,
            })
        else:
            rows.append({
                "input": k,
                "path": str(p),
                "bytes": None,
                "sha256": None,
                "exists": False,
            })
    df = pd.DataFrame(rows)
    atomic_csv(out / "pub003_input_audit.csv", df)
    atomic_csv(out / "pub003_lineage_audit.csv", pd.DataFrame(lineage["lineage_rows"]))
    return df


def extract_primary_numbers(lineage: Dict[str, Any], ps: Mapping[str, Path]) -> Dict[str, Any]:
    s6 = lineage["summaries"]["b50_6"]
    s7 = lineage["summaries"]["b50_7"]
    s11 = lineage["summaries"]["b50_11"]
    s12 = lineage["summaries"]["b50_12"]

    b6_effect = safe_float(s6.get("primary_specificity", {}).get("observed_mean_effect"))
    b6_p = safe_float(s6.get("primary_specificity", {}).get("p_two_sided"))
    if not np.isfinite(b6_p):
        b6_p = safe_float(s6.get("primary_specificity", {}).get("p_lower"))

    b7_effect = safe_float(s7.get("primary_replication", {}).get("observed_mean_effect"))

    minimal = pd.read_csv(ps["b50_11_minimal"])
    if len(minimal) != 1 or str(minimal.iloc[0]["subset_key"]) != PAIR_KEY:
        raise RuntimeError("B50.11 unique minimal pair contract not reproduced.")
    pair0 = float(minimal.iloc[0]["observed_mean_effect"])
    pair0_ret = float(minimal.iloc[0]["abs_effect_retention_fraction"])
    pair0_holm = float(minimal.iloc[0]["holm_directional_p"])

    b12 = pd.read_csv(ps["b50_12_offsets"]).sort_values("offset")
    if b12["offset"].astype(int).tolist() != OFFSETS:
        raise RuntimeError("B50.12 offsets are not exactly [25,50,75,100,125].")

    return {
        "b50_6_full_effect": b6_effect,
        "b50_6_primary_p": b6_p,
        "b50_7_plus75_full_effect": b7_effect,
        "b50_11_pair_effect_offset0": pair0,
        "b50_11_pair_retention": pair0_ret,
        "b50_11_pair_holm_p": pair0_holm,
        "b50_12_pair_mean_effect": safe_float(s12.get("effect_invariance", {}).get("five_offset_mean_effect")),
        "b50_12_pair_median_effect": safe_float(s12.get("effect_invariance", {}).get("five_offset_median_effect")),
        "b50_12_negative_offsets": int(s12.get("counts", {}).get("negative_offsets", -1)),
        "b50_12_nominal_offsets": int(s12.get("counts", {}).get("nominal_offset_replications", -1)),
        "b50_12_thirds_pass_offsets": int(s12.get("counts", {}).get("thirds_pass_offsets", -1)),
    }


def build_tables(pub_tables: Path, lineage: Dict[str, Any], ps: Mapping[str, Path], nums: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    out = {}

    # Primary B50.6 table.
    b6 = lineage["summaries"]["b50_6"]
    primary = b6.get("primary_specificity", {})
    t = pd.DataFrame([{
        "transitions": int(primary.get("n_transitions", len(pd.read_csv(ps["b50_6_transitions"])))),
        "mean_physical_percentile": safe_float(primary.get("observed_mean_percentile")),
        "mean_effect": safe_float(primary.get("observed_mean_effect")),
        "p_lower": safe_float(primary.get("p_lower")),
        "p_two_sided": safe_float(primary.get("p_two_sided")),
        "matching_median_abs_log_error": safe_float(
            b6.get("matching_geometry", {}).get("median_abs_log_distance_error",
                b6.get("matching_geometry", {}).get("median_abs_log_error"))
        ),
        "matching_q90_abs_log_error": safe_float(
            b6.get("matching_geometry", {}).get("q90_abs_log_distance_error",
                b6.get("matching_geometry", {}).get("q90_abs_log_error"))
        ),
        "verdict": verdict_of(b6),
    }])
    atomic_csv(pub_tables / "table_primary_b50_6.csv", t)
    out["primary"] = t

    # Full-score cross-offset table: offset 0 from B50.6, +75 from B50.7,
    # other four from B50.8 frozen offset effects.
    b8 = pd.read_csv(ps["b50_8_offsets"]).copy()
    # Accept either "effect" or "observed_mean_effect".
    eff_col = "effect" if "effect" in b8.columns else "observed_mean_effect"
    rows = [{"offset": 0, "full_score_effect": nums["b50_6_full_effect"], "source_stage": "B50.6"}]
    for off in [25, 50, 100, 125]:
        rr = b8[pd.to_numeric(b8["offset"], errors="coerce").astype("Int64") == off]
        if len(rr) != 1:
            raise RuntimeError(f"Cannot identify unique B50.8 offset {off}.")
        rows.append({
            "offset": off,
            "full_score_effect": float(rr.iloc[0][eff_col]),
            "source_stage": "B50.8",
        })
    rows.append({"offset": 75, "full_score_effect": nums["b50_7_plus75_full_effect"], "source_stage": "B50.7"})
    tfull = pd.DataFrame(rows).sort_values("offset").reset_index(drop=True)
    atomic_csv(pub_tables / "table_full_score_offsets.csv", tfull)
    out["full_offsets"] = tfull

    # B50.10 channel decomposition.
    ch = pd.read_csv(ps["b50_10_channels"]).copy()
    keep = [c for c in [
        "channel", "n_transitions", "observed_mean_percentile", "observed_mean_effect",
        "p_lower", "p_two_sided", "holm_p_two_sided", "effect_size_gate",
        "holm_significance_gate", "channel_confirmed"
    ] if c in ch.columns]
    ch = ch[keep]
    atomic_csv(pub_tables / "table_channel_decomposition.csv", ch)
    out["channels"] = ch

    # B50.11 minimal subset + Shapley + pair interactions.
    min_df = pd.read_csv(ps["b50_11_minimal"])
    atomic_csv(pub_tables / "table_minimal_pair.csv", min_df)
    out["minimal"] = min_df

    shap = pd.read_csv(ps["b50_11_shapley"]).copy()
    atomic_csv(pub_tables / "table_shapley.csv", shap)
    out["shapley"] = shap

    pi = pd.read_csv(ps["b50_11_pairs"]).copy()
    atomic_csv(pub_tables / "table_pair_interactions.csv", pi)
    out["pair_interactions"] = pi

    # B50.12 pair offsets.
    p12 = pd.read_csv(ps["b50_12_offsets"]).sort_values("offset").reset_index(drop=True)
    atomic_csv(pub_tables / "table_pair_core_offsets.csv", p12)
    out["pair_offsets"] = p12

    # Merge full and pair effects for direct comparison.
    pair0 = pd.DataFrame([{"offset": 0, "pair_core_effect": nums["b50_11_pair_effect_offset0"]}])
    pairrest = p12[["offset", "effect"]].rename(columns={"effect": "pair_core_effect"})
    pairall = pd.concat([pair0, pairrest], ignore_index=True).sort_values("offset")
    comp = tfull.merge(pairall, on="offset", how="inner")
    comp["pair_abs_retention_vs_full"] = np.abs(comp["pair_core_effect"]) / np.maximum(np.abs(comp["full_score_effect"]), 1e-15)
    atomic_csv(pub_tables / "table_full_vs_pair_offsets.csv", comp)
    out["comparison"] = comp

    # Headline table.
    headline = pd.DataFrame([
        {"quantity": "B50.6 full-score effect", "value": nums["b50_6_full_effect"]},
        {"quantity": "B50.11 pair effect at offset 0", "value": nums["b50_11_pair_effect_offset0"]},
        {"quantity": "B50.11 pair retention vs full", "value": nums["b50_11_pair_retention"]},
        {"quantity": "B50.11 pair Holm directional p", "value": nums["b50_11_pair_holm_p"]},
        {"quantity": "B50.12 five-offset mean pair effect", "value": nums["b50_12_pair_mean_effect"]},
        {"quantity": "B50.12 five-offset median pair effect", "value": nums["b50_12_pair_median_effect"]},
        {"quantity": "B50.12 negative offsets", "value": nums["b50_12_negative_offsets"]},
        {"quantity": "B50.12 nominal offsets", "value": nums["b50_12_nominal_offsets"]},
        {"quantity": "B50.12 thirds-pass offsets", "value": nums["b50_12_thirds_pass_offsets"]},
    ])
    atomic_csv(pub_tables / "table_headline_results.csv", headline)
    out["headline"] = headline
    return out


def save_figure(fig, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_figures(pub_figs: Path, lineage: Dict[str, Any], ps: Mapping[str, Path], tables: Dict[str, pd.DataFrame]) -> List[str]:
    made = []

    # Figure 1: machine-read lineage/falsification path.
    stages = [
        ("B50.1", "persistent\ndegeneracy"),
        ("B50.2", "severe low-dim\ninformation loss"),
        ("B50.3", "no stable\nregime topology"),
        ("B50.4/5", "continuous/front\nnull limits"),
        ("B50.6-8", "matched adjacency\nreplicated"),
        ("B50.10-12", "pair core\nreplicated"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    x = np.arange(len(stages))
    ax.plot(x, np.zeros_like(x), marker="o")
    for xi, (name, desc) in zip(x, stages):
        ax.text(xi, 0.10 if xi % 2 == 0 else -0.10, f"{name}\n{desc}",
                ha="center", va="bottom" if xi % 2 == 0 else "top", fontsize=9)
    ax.set_xlim(-0.4, len(stages)-0.6)
    ax.set_ylim(-0.33, 0.33)
    ax.axis("off")
    ax.set_title("Falsification and narrowing path to the frozen conditional-transition endpoint")
    save_figure(fig, pub_figs/"figure_1_falsification_path.png", pub_figs/"figure_1_falsification_path.pdf")
    made.append("figure_1_falsification_path")

    # Primary B50.6 physical-percentile distribution.
    trans = pd.read_csv(ps["b50_6_transitions"])
    col = "consensus_physical_percentile"
    if col not in trans.columns:
        raise RuntimeError(f"{ps['b50_6_transitions']} missing {col}")
    vals = pd.to_numeric(trans[col], errors="coerce").dropna().to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(vals, bins=np.linspace(0,1,22))
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.axvline(float(np.mean(vals)), linestyle="-", linewidth=1.5)
    ax.set_xlabel("Consensus physical-destination percentile")
    ax.set_ylabel("Transitions")
    ax.set_title("Matched physical-successor percentile distribution (B50.6)")
    save_figure(fig, pub_figs/"figure_2_primary_successor_percentiles.png", pub_figs/"figure_2_primary_successor_percentiles.pdf")
    made.append("figure_2_primary_successor_percentiles")

    # Channel effects B50.10.
    ch = tables["channels"].copy()
    ch = ch.sort_values("observed_mean_effect")
    fig, ax = plt.subplots(figsize=(7.4,4.6))
    ax.bar(np.arange(len(ch)), ch["observed_mean_effect"].astype(float))
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(ch)))
    ax.set_xticklabels(ch["channel"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Mean physical percentile effect vs 0.5")
    ax.set_title("Raw channel decomposition (B50.10)")
    save_figure(fig, pub_figs/"figure_3_channel_effects.png", pub_figs/"figure_3_channel_effects.pdf")
    made.append("figure_3_channel_effects")

    # Exact Shapley.
    shap = tables["shapley"].copy().sort_values("shapley_effect_contribution")
    fig, ax = plt.subplots(figsize=(7.4,4.6))
    ax.bar(np.arange(len(shap)), shap["shapley_effect_contribution"].astype(float))
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(len(shap)))
    ax.set_xticklabels(shap["channel"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Shapley contribution to centered percentile effect")
    ax.set_title("Exact five-channel Shapley decomposition (B50.11)")
    save_figure(fig, pub_figs/"figure_4_shapley_contributions.png", pub_figs/"figure_4_shapley_contributions.pdf")
    made.append("figure_4_shapley_contributions")

    # Pair-core offset effect.
    pair = tables["pair_offsets"].copy().sort_values("offset")
    fig, ax = plt.subplots(figsize=(7.4,4.6))
    ax.plot(pair["offset"].astype(float), pair["effect"].astype(float), marker="o")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.axhline(-0.04, linestyle=":", linewidth=1)
    ax.set_xlabel("Window offset (segments)")
    ax.set_ylabel("Pair-core mean percentile effect vs 0.5")
    ax.set_title("Frozen turn + density pair across window phases (B50.12)")
    save_figure(fig, pub_figs/"figure_5_pair_core_offsets.png", pub_figs/"figure_5_pair_core_offsets.pdf")
    made.append("figure_5_pair_core_offsets")

    # Full vs pair, including offset 0.
    comp = tables["comparison"].copy().sort_values("offset")
    fig, ax = plt.subplots(figsize=(7.8,4.8))
    ax.plot(comp["offset"].astype(float), comp["full_score_effect"].astype(float), marker="o", label="five-channel full score")
    ax.plot(comp["offset"].astype(float), comp["pair_core_effect"].astype(float), marker="s", label="turn+density pair")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Window offset (segments)")
    ax.set_ylabel("Mean physical percentile effect vs 0.5")
    ax.set_title("Full composite and frozen pair across offsets")
    ax.legend()
    save_figure(fig, pub_figs/"figure_6_full_vs_pair_offsets.png", pub_figs/"figure_6_full_vs_pair_offsets.pdf")
    made.append("figure_6_full_vs_pair_offsets")

    return made


def copy_if_small(src: Path, dst: Path, max_bytes: int) -> Dict[str, Any]:
    rec = {
        "source_path": str(src),
        "exists": src.exists(),
        "sha256": sha256_file(src) if src.exists() else None,
        "bytes": int(src.stat().st_size) if src.exists() else None,
        "copied": False,
        "release_path": None,
    }
    if not src.exists():
        return rec
    if src.stat().st_size <= max_bytes:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rec["copied"] = True
        rec["release_path"] = str(dst)
    return rec


def build_release(args, out: Path, lineage: Dict[str, Any], ps: Mapping[str, Path],
                  pub_tables: Path, pub_figs: Path, nums: Dict[str, Any]) -> Dict[str, Any]:
    release = out / "release"
    data_dir = release / "data" / "derived"
    code_dir = release / "code"
    mani_dir = release / "manifests"
    fig_dir = release / "figures"
    table_dir = release / "tables"
    for d in [data_dir, code_dir, mani_dir, fig_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)

    max_bytes = int(args.max_copy_mib * 1024 * 1024)
    manifest_rows = []

    # Copy all publication tables/figures.
    for p in sorted(pub_tables.glob("*")):
        if p.is_file():
            dst = table_dir / p.name
            shutil.copy2(p, dst)
            manifest_rows.append({
                "category":"publication_table","source_path":str(p),"release_path":str(dst),
                "bytes":p.stat().st_size,"sha256":sha256_file(p),"copied":True
            })
    for p in sorted(pub_figs.glob("*")):
        if p.is_file():
            dst = fig_dir / p.name
            shutil.copy2(p, dst)
            manifest_rows.append({
                "category":"publication_figure","source_path":str(p),"release_path":str(dst),
                "bytes":p.stat().st_size,"sha256":sha256_file(p),"copied":True
            })

    # Small frozen data inputs plus important NPZ if under threshold.
    data_keys = [
        k for k in ps if k not in ("b50_7_stack","b50_10_cube")
    ] + ["b50_7_stack","b50_10_cube"]
    for k in data_keys:
        src = ps[k]
        if not src.exists():
            continue
        dst = data_dir / f"{k}__{src.name}"
        rec = copy_if_small(src, dst, max_bytes)
        manifest_rows.append({"category":"frozen_data","input_key":k, **rec})

    # Include B50.8 four offset subruns and their operator stacks/artifacts.
    b8root = result_dirs(args)["b50_8"]
    for off in [25,50,100,125]:
        od = b8root / f"offset_p{off:03d}"
        if not od.exists():
            continue
        for src in sorted(od.iterdir()):
            if not src.is_file():
                continue
            if src.suffix.lower() not in {".json",".csv",".txt",".npz"}:
                continue
            dst = data_dir / f"b50_8_offset_p{off:03d}" / src.name
            rec = copy_if_small(src, dst, max_bytes)
            manifest_rows.append({"category":"frozen_offset_data","offset":off, **rec})

    # Include scripts: all B50 Python scripts and PUB scripts available locally.
    scripts_root = Path(args.scripts_root)
    if scripts_root.exists():
        for src in sorted(scripts_root.glob("tdc_tor*a_b50*.py")) + sorted(scripts_root.glob("tdc_torA_b50*.py")):
            if src.is_file():
                dst = code_dir / src.name
                rec = copy_if_small(src, dst, max_bytes)
                manifest_rows.append({"category":"code", **rec})
        for src in sorted(scripts_root.glob("pub00*.py")):
            if src.is_file():
                dst = code_dir / src.name
                rec = copy_if_small(src, dst, max_bytes)
                manifest_rows.append({"category":"code", **rec})

    # Also copy this builder itself when it has a real file path.
    try:
        selfp = Path(__file__).resolve()
        if selfp.exists():
            dst = code_dir / selfp.name
            rec = copy_if_small(selfp, dst, max_bytes)
            manifest_rows.append({"category":"code", **rec})
    except Exception:
        pass

    mdf = pd.DataFrame(manifest_rows)
    atomic_csv(mani_dir / "release_manifest.csv", mdf)

    # SHA256SUMS for actually copied files.
    copied_files = []
    for p in release.rglob("*"):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            copied_files.append(p)
    lines = []
    for p in sorted(copied_files):
        rel = p.relative_to(release)
        lines.append(f"{sha256_file(p)}  {rel.as_posix()}")
    atomic_text(mani_dir / "SHA256SUMS.txt", "\n".join(lines) + "\n")

    # Requirements intentionally conservative.
    atomic_text(release / "requirements.txt",
        "numpy\npandas\nscipy\nmatplotlib\nscikit-learn\npython-docx\n"
    )

    citation = """cff-version: 1.2.0
message: "If you use this reproduction package, please cite the associated manuscript."
title: "Conditional Transition Geometry in Operator Representations of Prime-Number Windows — reproduction package"
type: software
authors:
  - family-names: "Tomanek"
    given-names: "Pawel"
version: "PUB-003-v1"
date-released: "2026-09-08"
repository-code: "TO_BE_ASSIGNED"
doi: "TO_BE_MINTED"
"""
    atomic_text(release / "CITATION.cff", citation)

    readme = f"""# Conditional Transition Geometry — public reproduction package

## Scope

This archive reproduces the public B50 experimental endpoint only.

It does **not** include or claim:
- a Hamiltonian of the primes;
- a prime-field or spiral theory;
- a Hilbert–Pólya construction;
- prediction of individual prime numbers;
- evidence for the Riemann hypothesis.

## Frozen public lineage

- B50.6: {EXPECTED['b50_6']}
- B50.7: {EXPECTED['b50_7']}
- B50.8: {EXPECTED['b50_8']}
- B50.10: {EXPECTED['b50_10']}
- B50.11: {EXPECTED['b50_11']}
- B50.12: {EXPECTED['b50_12']}

## Headline endpoint

B50.6 full five-channel matched-successor effect: {nums['b50_6_full_effect']}

B50.11 unique minimal pair:
`turn_change + density_scale_change`

Pair effect at offset 0: {nums['b50_11_pair_effect_offset0']}
Retention versus full score: {nums['b50_11_pair_retention']}

B50.12:
- negative offsets: {nums['b50_12_negative_offsets']}/5
- nominal offset replications: {nums['b50_12_nominal_offsets']}/5
- thirds-robustness passes: {nums['b50_12_thirds_pass_offsets']}/5

## Directory structure

- `data/derived/` — frozen CSV/JSON/NPZ artifacts small enough for the release policy
- `code/` — available frozen B50/PUB source files
- `tables/` — publication tables generated directly from frozen outputs
- `figures/` — PNG (300 dpi) and PDF publication figures generated directly from frozen outputs
- `manifests/` — release manifest and SHA-256 checksums

## Reproducibility levels

See `REPRODUCIBILITY_LEVELS.md`.

## Integrity

Every copied artifact is hashed. Files larger than the configured copy threshold remain
listed in the manifest but are not silently substituted.
"""
    atomic_text(release / "README.md", readme)

    levels = """# Reproducibility levels

## Level A — publication endpoint reproduction

Recompute the manuscript tables and figures from the frozen derived B50.6–B50.12
CSV/JSON artifacts. This is the purpose of PUB-003 and requires no access to the
raw prime corpus.

## Level B — frozen representation reproduction

Recompute the B50.6–B50.12 endpoint from released operator stacks / mechanism cubes,
when those NPZ artifacts are included in the release.

## Level C — raw-corpus rebuild

Rebuild the operator states from the segmented prime corpus. The original segmented
corpus is very large and is not assumed to be deposited in the same archive.
A raw-corpus release should instead provide a source-generation/audit protocol and a
segment hash inventory. Level C is separate from the journal's machine-derived figure
and table contract.

## Independence boundary

Offset replications use overlapping representations of the same deterministic prime
corpus. They are cross-phase representation replications, not independent arithmetic-
interval replications.
"""
    atomic_text(release / "REPRODUCIBILITY_LEVELS.md", levels)

    availability = """# Code and data availability statement

Code, frozen protocol manifests, derived result tables, publication figures, and
reproduction manifests for the reported B50.6–B50.12 endpoint will be deposited in
a versioned public repository and archived with a persistent DOI before or at
publication. The release is designed to reproduce the reported tables and figures
directly from frozen derived artifacts.

The segmented prime corpus is substantially larger than the manuscript-level
release and may be distributed separately or regenerated/audited from the documented
corpus protocol. Cross-offset tests in the article use the same underlying
deterministic prime corpus and are not presented as independent-corpus replications.

The private theoretical research programme is outside the scope of this release.
"""
    atomic_text(release / "CODE_DATA_AVAILABILITY.md", availability)

    return {
        "release_dir": str(release),
        "manifest_rows": int(len(mdf)),
        "copied_files": int(mdf.get("copied", pd.Series(dtype=bool)).fillna(False).sum()) if len(mdf) else 0,
        "copy_threshold_mib": args.max_copy_mib,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="PUB-003 publication asset and release builder")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--scripts-root", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--b50-6-dir")
    ap.add_argument("--b50-7-dir")
    ap.add_argument("--b50-8-dir")
    ap.add_argument("--b50-10-dir")
    ap.add_argument("--b50-11-dir")
    ap.add_argument("--b50-12-dir")

    ap.add_argument("--max-copy-mib", type=float, default=256.0,
                    help="Copy an individual frozen artifact into the release only if it is <= this size.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    pub = out / "publication"
    figs = pub / "figures"
    tables = pub / "tables"
    for d in [out, pub, figs, tables]:
        d.mkdir(parents=True, exist_ok=True)

    print("=== PUB-003 MACHINE-DERIVED PUBLICATION ASSETS ===")
    lineage, ps = load_lineage(args)
    print("frozen lineage exact:", lineage["pass"])
    if not lineage["pass"]:
        atomic_json(out / "pub003_summary.json", {
            "version": VERSION,
            "final_verdict": "INVALID_FROZEN_LINEAGE",
            "lineage": lineage["lineage_rows"],
        })
        atomic_text(out / "pub003_verdict.txt", "FINAL VERDICT: INVALID_FROZEN_LINEAGE\n")
        return 2

    input_df = write_input_audit(out, ps, lineage)
    nums = extract_primary_numbers(lineage, ps)
    atomic_json(pub / "publication_numbers.json", nums)

    tables_map = build_tables(tables, lineage, ps, nums)
    fig_names = build_figures(figs, lineage, ps, tables_map)
    release = build_release(args, out, lineage, ps, tables, figs, nums)

    summary = {
        "version": VERSION,
        "finished": now_s(),
        "frozen_lineage_exact": True,
        "headline_numbers": nums,
        "publication_table_count": len(list(tables.glob("*.csv"))),
        "publication_figure_sets": fig_names,
        "release": release,
        "final_verdict": "PUB003_ASSETS_AND_RELEASE_BUILT",
        "scientific_rerun": False,
        "protocol_modified": False,
        "private_theory_included": False,
    }
    atomic_json(out / "pub003_summary.json", summary)

    verdict = f"""PUB-003 — Machine-Derived Figures, Tables & Reproducibility Release
================================================================

Frozen B50.6–B50.12 lineage exact       = True
Scientific rerun                        = False
Protocol modification                   = False
Channel reselection                     = False
Control rematching                      = False
Unrelated internal material             = False

Publication tables                      = {summary['publication_table_count']}
Publication figure sets                 = {len(fig_names)}

Headline endpoint
-----------------
B50.6 full effect                       = {nums['b50_6_full_effect']}
B50.11 pair effect (offset 0)           = {nums['b50_11_pair_effect_offset0']}
B50.11 pair retention                   = {nums['b50_11_pair_retention']}
B50.12 negative offsets                 = {nums['b50_12_negative_offsets']}/5
B50.12 nominal offsets                  = {nums['b50_12_nominal_offsets']}/5
B50.12 thirds-pass offsets              = {nums['b50_12_thirds_pass_offsets']}/5

Release
-------
directory                               = {release['release_dir']}
manifest rows                           = {release['manifest_rows']}
copied files                            = {release['copied_files']}
per-file copy threshold                 = {release['copy_threshold_mib']} MiB

FINAL VERDICT: PUB003_ASSETS_AND_RELEASE_BUILT

Next action:
Run pub003_patch_manuscript.py against the PUB-002 manuscript and this output directory,
then render/inspect the resulting DOCX before journal-language finalization.
"""
    atomic_text(out / "pub003_verdict.txt", verdict)
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
