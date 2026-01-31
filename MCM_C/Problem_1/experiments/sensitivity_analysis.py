from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])

def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    # O(n^2) Kendall tau for small n (DWTS weekly roster is small enough)
    n = len(a)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else 0.0


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from estimation.dwts_estimator import load_week_level_json, calibrate_from_week_json, FanVoteParticleFilter, CalibratedParams


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    a_idx = set(np.argsort(-a)[:k].tolist())
    b_idx = set(np.argsort(-b)[:k].tolist())
    inter = len(a_idx & b_idx)
    union = len(a_idx | b_idx)
    return inter / union if union else 1.0


def run_one(season_obj: Dict[str, Any], params: CalibratedParams, n_particles: int, seed: int) -> Dict[str, Any]:
    pf = FanVoteParticleFilter(n_particles=n_particles, params=params, random_seed=seed)
    return pf.estimate_season(season_obj, compute_ci=False)


def compare_to_baseline(baseline: Dict[str, Any], other: Dict[str, Any]) -> pd.DataFrame:
    rows: List[dict] = []
    for wk_key, wk0 in baseline["Weeks"].items():
        if wk_key not in other["Weeks"]:
            continue
        wka = other["Weeks"][wk_key]
        c0 = wk0["Contestants"]
        ca = wka["Contestants"]
        if c0 != ca:
            # only compare identical rosters
            continue
        v0 = np.array(wk0["FanShare_Mean"], dtype=float)
        va = np.array(wka["FanShare_Mean"], dtype=float)
        sp = spearman_corr(v0, va)
        kt = kendall_tau(v0, va)
        jac = jaccard_topk(v0, va, k=3)
        rows.append({
            "week": wk_key,
            "spearman": float(sp) if sp is not None else np.nan,
            "kendall": float(kt) if kt is not None else np.nan,
            "top3_jaccard": float(jac),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_json", required=True)
    ap.add_argument("--season", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_particles", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    week_data = load_week_level_json(args.week_json)
    base_params = calibrate_from_week_json(week_data)
    season_obj = week_data[args.season]

    baseline = run_one(season_obj, base_params, args.n_particles, args.seed)

    # parameter grid (multipliers)
    sigma_mults = [0.5, 0.8, 1.0, 1.25, 1.6]
    mem_shifts = [-0.10, -0.05, 0.0, 0.05, 0.10]
    results = []

    for sm in sigma_mults:
        for ds in mem_shifts:
            params = CalibratedParams(**base_params.__dict__)
            params.sigma_proc = float(np.clip(base_params.sigma_proc * sm, 0.01, 0.6))
            params.memory_coeff = float(np.clip(base_params.memory_coeff + ds, 0.1, 0.99))

            out = run_one(season_obj, params, args.n_particles, args.seed)
            cmp_df = compare_to_baseline(baseline, out)
            results.append({
                "sigma_mult": sm,
                "memory_shift": ds,
                "avg_spearman": float(cmp_df["spearman"].mean()),
                "avg_kendall": float(cmp_df["kendall"].mean()),
                "avg_top3_jaccard": float(cmp_df["top3_jaccard"].mean()),
            })

    res_df = pd.DataFrame(results).sort_values(["sigma_mult", "memory_shift"])
    out_csv = os.path.join(args.out_dir, f"sensitivity_{args.season}.csv")
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "baseline_params": base_params.__dict__,
        "grid": {"sigma_mults": sigma_mults, "memory_shifts": mem_shifts},
        "notes": "Stability measured relative to baseline run using rank-based correlations and Top-3 overlap."
    }
    out_json = os.path.join(args.out_dir, f"sensitivity_{args.season}_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:", out_csv)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
