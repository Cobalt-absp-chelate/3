from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from estimation.dwts_estimator import (
    load_week_level_json,
    calibrate_from_week_json,
    FanVoteParticleFilter,
    softmax_stable,
    judge_percent,
    ranks_desc,
    zscore,
)


def simulate_elimination_percent(judge_scores: np.ndarray, fan_share: np.ndarray) -> Tuple[List[int], List[int]]:
    """Return (eliminated_idx, survivor_idx) for a single-elimination week under Percent rule."""
    combined = judge_percent(judge_scores) + fan_share
    elim = int(np.argmin(combined))
    surv = [i for i in range(len(combined)) if i != elim]
    return [elim], surv


def simulate_elimination_rank(judge_scores: np.ndarray, fan_share: np.ndarray) -> Tuple[List[int], List[int]]:
    """Return (eliminated_idx, survivor_idx) under Rank-sum rule."""
    jr = ranks_desc(judge_scores)
    fr = ranks_desc(fan_share)
    comb = jr + fr
    elim = int(np.argmax(comb))  # worst
    surv = [i for i in range(len(comb)) if i != elim]
    return [elim], surv


def generate_synthetic_theta(
    weeks: Dict[str, Any],
    params,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic latent theta trajectories aligned to each week roster.
    Uses the same dynamic structure as the estimator:
      theta_t = memory*theta_{t-1} + (1-memory)*zscore(J_t) + eps
    """
    # sort weeks
    week_nums = sorted(int(k.split("_")[1]) for k in weeks.keys())
    theta_by_week: Dict[str, np.ndarray] = {}
    prev_theta_mean = None
    prev_cont = None

    for wn in week_nums:
        wk_key = f"Week_{wn}"
        wk = weeks[wk_key]
        cont = wk["Contestants"]
        J = np.asarray(wk["Judge_Scores"], dtype=float)
        signal = zscore(J)

        if prev_theta_mean is None:
            theta = rng.normal(0.0, params.init_sigma, size=len(cont))
        else:
            prev_map = {c: i for i, c in enumerate(prev_cont)}
            aligned = np.zeros(len(cont))
            for i, c in enumerate(cont):
                if c in prev_map:
                    aligned[i] = prev_theta_mean[prev_map[c]]
            theta = aligned

        eps = rng.normal(0.0, params.sigma_proc, size=len(cont))
        theta = params.memory_coeff * theta + (1.0 - params.memory_coeff) * signal + eps

        theta_by_week[wk_key] = theta
        prev_theta_mean = theta
        prev_cont = cont

    return theta_by_week


def run_synthetic_test(
    week_data: Dict[str, Any],
    season_key: str,
    n_particles: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    params = calibrate_from_week_json(week_data)
    rng = np.random.default_rng(seed)

    season = copy.deepcopy(week_data[season_key])
    season_type = season.get("Type", "Percent")
    weeks = season["Weeks"]

    # 1) generate synthetic truth
    theta_true = generate_synthetic_theta(weeks, params, rng)
    f_true: Dict[str, np.ndarray] = {wk: softmax_stable(th) for wk, th in theta_true.items()}

    # 2) simulate eliminations
    for wk_key, wk in weeks.items():
        cont = wk["Contestants"]
        J = np.asarray(wk["Judge_Scores"], dtype=float)
        f = f_true[wk_key]
        if season_type.lower().startswith("percent"):
            elim_idx, surv_idx = simulate_elimination_percent(J, f)
        else:
            elim_idx, surv_idx = simulate_elimination_rank(J, f)
        wk["Eliminated"] = [cont[i] for i in elim_idx]
        wk["Survivors"] = [cont[i] for i in surv_idx]

    # 3) recover using PF
    pf = FanVoteParticleFilter(n_particles=n_particles, params=params, random_seed=seed)
    recovered = pf.estimate_season(season, compute_ci=True)

    # 4) score RMSE + coverage
    rows = []
    for wk_key, wk_out in recovered["Weeks"].items():
        cont = wk_out["Contestants"]
        mean = np.array(wk_out["FanShare_Mean"], dtype=float)
        lo = np.array(wk_out["FanShare_CI_Lower"], dtype=float)
        hi = np.array(wk_out["FanShare_CI_Upper"], dtype=float)
        truth = f_true[wk_key]
        rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
        coverage = float(np.mean((truth >= lo) & (truth <= hi)))
        rows.append({
            "season": season_key,
            "week": wk_key,
            "rmse": rmse,
            "coverage_95": coverage,
            "MCC_unweighted": wk_out.get("MCC_unweighted"),
            "MCC_weighted": wk_out.get("MCC_weighted"),
            "ACC_mean": wk_out.get("ACC_mean"),
            "n_contestants": len(cont),
        })

    df = pd.DataFrame(rows).sort_values(["week"])
    meta = {
        "season_type": season_type,
        "calibrated_params": params.__dict__,
        "notes": "RMSE/coverage computed against synthetic ground truth fan shares.",
    }
    return df, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_json", required=True)
    ap.add_argument("--season", required=True, help='Season key like "S27"')
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_particles", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    week_data = load_week_level_json(args.week_json)
    df, meta = run_synthetic_test(week_data, args.season, args.n_particles, args.seed)

    out_csv = os.path.join(args.out_dir, f"synthetic_validation_{args.season}.csv")
    out_json = os.path.join(args.out_dir, f"synthetic_validation_{args.season}_meta.json")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("RMSE mean:", df["rmse"].mean(), "Coverage mean:", df["coverage_95"].mean())


if __name__ == "__main__":
    main()
