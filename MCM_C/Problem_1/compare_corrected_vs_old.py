#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare corrected vs old outputs for MCM 2026 Problem C - Task 1 (Problem_1).

What this script does
- Loads week-level validation metrics (old vs corrected): MCC/ACC + validity flags
- (Optionally) loads certainty summaries (old vs corrected)
- Produces:
  1) A heatmap of ΔMCC_weighted across (season × week)
  2) A bar chart comparing coverage (how many weeks have valid MCC/ACC)
  3) A line chart comparing average certainty (CI width) by week (if certainty inputs provided)
  4) A CSV summary table of key headline numbers

This script is intentionally defensive:
- If a metric column is missing, it will warn and skip the dependent output
- If some seasons/weeks exist in only one run, it will align on the intersection for deltas
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def _parse_week_num(w: str) -> int:
    """
    Accepts 'Week_1', 'week_10', '1', etc. Returns integer week.
    """
    if pd.isna(w):
        return -1
    s = str(w)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def _ensure_bool(series: pd.Series) -> pd.Series:
    """
    Convert common truthy/falsey representations to bool.
    """
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _safe_mean(x: pd.Series) -> float:
    x2 = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x2.values)) if np.isfinite(np.nanmean(x2.values)) else float("nan")


def _save_fig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _align_for_delta(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_cols: List[str],
    value_col: str
) -> pd.DataFrame:
    """
    Inner-join old/new on key_cols and compute delta for value_col: new - old.
    Returns a dataframe with key_cols + [old_value, new_value, delta]
    """
    o = old_df[key_cols + [value_col]].copy()
    n = new_df[key_cols + [value_col]].copy()
    o = o.rename(columns={value_col: f"{value_col}_old"})
    n = n.rename(columns={value_col: f"{value_col}_new"})
    merged = pd.merge(o, n, on=key_cols, how="inner")
    merged["delta"] = pd.to_numeric(merged[f"{value_col}_new"], errors="coerce") - pd.to_numeric(merged[f"{value_col}_old"], errors="coerce")
    return merged


# ----------------------------
# Main plotting routines
# ----------------------------
def plot_delta_mcc_heatmap(old_week: pd.DataFrame, new_week: pd.DataFrame, out_png: str) -> Optional[pd.DataFrame]:
    """
    Heatmap of Δ MCC_weighted across season×week for weeks where both runs have a value.
    """
    required = {"season", "week", "MCC_weighted"}
    if not required.issubset(set(old_week.columns)) or not required.issubset(set(new_week.columns)):
        print("[WARN] Missing required columns for MCC heatmap. Need:", required)
        return None

    key_cols = ["season", "week_num"]
    o = old_week.copy()
    n = new_week.copy()
    o["week_num"] = o["week"].map(_parse_week_num)
    n["week_num"] = n["week"].map(_parse_week_num)

    merged = _align_for_delta(o, n, key_cols=key_cols, value_col="MCC_weighted")
    # keep finite deltas
    merged = merged[np.isfinite(merged["delta"].values)]

    if merged.empty:
        print("[WARN] No overlapping (season,week) with finite MCC_weighted to plot delta heatmap.")
        return None

    # pivot into matrix for imshow
    seasons = sorted(merged["season"].unique(), key=lambda s: int(re.search(r"\d+", s).group(0)) if re.search(r"\d+", s) else 10**9)
    weeks = sorted(merged["week_num"].unique())

    mat = np.full((len(seasons), len(weeks)), np.nan, dtype=float)
    season_to_i = {s: i for i, s in enumerate(seasons)}
    week_to_j = {w: j for j, w in enumerate(weeks)}

    for _, row in merged.iterrows():
        i = season_to_i[row["season"]]
        j = week_to_j[int(row["week_num"])]
        mat[i, j] = float(row["delta"])

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Δ MCC_weighted (corrected - old)")
    plt.yticks(range(len(seasons)), seasons)
    plt.xticks(range(len(weeks)), [str(w) for w in weeks], rotation=90)
    plt.xlabel("Week number")
    plt.ylabel("Season")
    plt.title("Δ MCC_weighted Heatmap (corrected - old)")
    _save_fig(out_png)

    return merged


def plot_coverage_bars(old_week: pd.DataFrame, new_week: pd.DataFrame, out_png: str) -> pd.DataFrame:
    """
    Compare how many weeks are valid for MCC / have ACC_01 available.
    """
    # validity
    def coverage(df: pd.DataFrame) -> Tuple[int, int, int]:
        total = len(df)
        valid_mcc = int(_ensure_bool(df.get("valid_for_mcc", pd.Series([False]*total))).sum()) if "valid_for_mcc" in df.columns else int(df["MCC_weighted"].notna().sum()) if "MCC_weighted" in df.columns else 0
        acc01 = int(pd.to_numeric(df.get("ACC_01", np.nan), errors="coerce").notna().sum()) if "ACC_01" in df.columns else int(pd.to_numeric(df.get("ACC_mean_recomputed", np.nan), errors="coerce").notna().sum()) if "ACC_mean_recomputed" in df.columns else 0
        return total, valid_mcc, acc01

    o_total, o_valid_mcc, o_acc = coverage(old_week)
    n_total, n_valid_mcc, n_acc = coverage(new_week)

    labels = ["total_weeks", "valid_for_mcc", "acc_available"]
    old_vals = [o_total, o_valid_mcc, o_acc]
    new_vals = [n_total, n_valid_mcc, n_acc]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, old_vals, width, label="old")
    plt.bar(x + width/2, new_vals, width, label="corrected")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Count")
    plt.title("Coverage comparison (old vs corrected)")
    plt.legend()
    _save_fig(out_png)

    return pd.DataFrame({"metric": labels, "old": old_vals, "corrected": new_vals})


def plot_certainty_line(old_cert_week: pd.DataFrame, new_cert_week: pd.DataFrame, out_png: str) -> Optional[pd.DataFrame]:
    """
    Compare average CI width by week number (lower = more certain).
    Expects columns: week_num, avg_ci_width (or similar).
    """
    if old_cert_week is None or new_cert_week is None:
        return None

    # Attempt to find the right columns
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # find week column
        if "week_num" not in d.columns:
            if "week" in d.columns:
                d["week_num"] = d["week"].map(_parse_week_num)
            elif "week_number" in d.columns:
                d["week_num"] = pd.to_numeric(d["week_number"], errors="coerce")
        # find avg ci width column
        cand = None
        for c in ["avg_ci_width", "mean_ci_width", "avg_ci", "mean_ci"]:
            if c in d.columns:
                cand = c
                break
        if cand is None and "ci_width" in d.columns:
            # maybe already aggregated differently
            cand = "ci_width"
        if cand is None:
            return pd.DataFrame()
        d = d[["week_num", cand]].copy()
        d[cand] = pd.to_numeric(d[cand], errors="coerce")
        d = d.groupby("week_num", as_index=False)[cand].mean()
        d = d.rename(columns={cand: "avg_ci_width"})
        return d.sort_values("week_num")

    o = normalize(old_cert_week)
    n = normalize(new_cert_week)
    if o.empty or n.empty:
        print("[WARN] Could not identify certainty columns for line plot.")
        return None

    plt.figure()
    plt.plot(o["week_num"], o["avg_ci_width"], label="old")
    plt.plot(n["week_num"], n["avg_ci_width"], label="corrected")
    plt.xlabel("Week number")
    plt.ylabel("Average CI width (lower = more certain)")
    plt.title("Certainty comparison by week (CI width)")
    plt.legend()
    _save_fig(out_png)

    merged = pd.merge(o, n, on="week_num", how="outer", suffixes=("_old", "_corrected"))
    merged["delta"] = merged["avg_ci_width_corrected"] - merged["avg_ci_width_old"]
    return merged


def build_headline_summary(old_week: pd.DataFrame, new_week: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a compact summary of key metrics to paste into the paper.
    """
    def summarize(df: pd.DataFrame, tag: str) -> dict:
        out = {"run": tag, "n_rows": len(df)}
        if "valid_for_mcc" in df.columns:
            out["valid_for_mcc"] = int(_ensure_bool(df["valid_for_mcc"]).sum())
        if "MCC_weighted" in df.columns:
            out["mean_MCC_weighted"] = _safe_mean(df.loc[_ensure_bool(df.get("valid_for_mcc", df["MCC_weighted"].notna())), "MCC_weighted"] if "valid_for_mcc" in df.columns else df["MCC_weighted"])
        if "MCC_unweighted" in df.columns:
            out["mean_MCC_unweighted"] = _safe_mean(df.loc[_ensure_bool(df.get("valid_for_mcc", df["MCC_unweighted"].notna())), "MCC_unweighted"] if "valid_for_mcc" in df.columns else df["MCC_unweighted"])
        # ACC: prefer ACC_01 if available, else ACC_mean_recomputed
        if "ACC_01" in df.columns:
            out["mean_ACC"] = _safe_mean(df["ACC_01"])
            out["ACC_available"] = int(pd.to_numeric(df["ACC_01"], errors="coerce").notna().sum())
        elif "ACC_mean_recomputed" in df.columns:
            out["mean_ACC"] = _safe_mean(df["ACC_mean_recomputed"])
            out["ACC_available"] = int(pd.to_numeric(df["ACC_mean_recomputed"], errors="coerce").notna().sum())
        return out

    rows = [summarize(old_week, "old"), summarize(new_week, "corrected")]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_week_metrics", required=True, help="Path to OLD validation_week_metrics.csv")
    ap.add_argument("--new_week_metrics", required=True, help="Path to CORRECTED validation_week_metrics.csv")
    ap.add_argument("--old_certainty_week", default=None, help="Optional path to OLD validation_certainty_by_week_summary.csv")
    ap.add_argument("--new_certainty_week", default=None, help="Optional path to CORRECTED validation_certainty_by_week_summary.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder for plots/tables")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    old_week = pd.read_csv(args.old_week_metrics)
    new_week = pd.read_csv(args.new_week_metrics)

    # 1) Heatmap Δ MCC_weighted
    heatmap_png = os.path.join(args.out_dir, "delta_mcc_weighted_heatmap.png")
    _ = plot_delta_mcc_heatmap(old_week, new_week, heatmap_png)

    # 2) Coverage bars
    coverage_png = os.path.join(args.out_dir, "coverage_comparison.png")
    coverage_df = plot_coverage_bars(old_week, new_week, coverage_png)

    # 3) Certainty line (optional)
    certainty_df = None
    if args.old_certainty_week and args.new_certainty_week and os.path.exists(args.old_certainty_week) and os.path.exists(args.new_certainty_week):
        old_c = pd.read_csv(args.old_certainty_week)
        new_c = pd.read_csv(args.new_certainty_week)
        certainty_png = os.path.join(args.out_dir, "certainty_ciwidth_by_week.png")
        certainty_df = plot_certainty_line(old_c, new_c, certainty_png)
    else:
        certainty_png = None

    # 4) Headline summary
    headline_df = build_headline_summary(old_week, new_week)

    out_headline = os.path.join(args.out_dir, "headline_summary.csv")
    headline_df.to_csv(out_headline, index=False, encoding="utf-8-sig")

    out_coverage = os.path.join(args.out_dir, "coverage_summary.csv")
    coverage_df.to_csv(out_coverage, index=False, encoding="utf-8-sig")

    if certainty_df is not None:
        out_cert = os.path.join(args.out_dir, "certainty_delta_by_week.csv")
        certainty_df.to_csv(out_cert, index=False, encoding="utf-8-sig")

    print("Saved plots/tables to:", args.out_dir)
    print(" - delta_mcc_weighted_heatmap.png")
    print(" - coverage_comparison.png")
    if certainty_png:
        print(" - certainty_ciwidth_by_week.png")
    print(" - headline_summary.csv")
    print(" - coverage_summary.csv")
    if certainty_df is not None:
        print(" - certainty_delta_by_week.csv")


if __name__ == "__main__":
    main()
