from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
# allow importing sibling packages when run from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, Any, List

import pandas as pd

from dwts_estimator import (
    load_week_level_json,
    calibrate_from_week_json,
    FanVoteParticleFilter,
)


def flatten_to_rows(estimates: Dict[str, Any]) -> List[dict]:
    rows: List[dict] = []
    for season_key, season_out in estimates.items():
        season_type = season_out.get("Type", "")
        weeks = season_out.get("Weeks", {})
        for wk_key, wk_out in weeks.items():
            contestants = wk_out["Contestants"]
            fan_mean = wk_out["FanShare_Mean"]
            fan_std = wk_out["FanShare_Std"]
            lo = wk_out["FanShare_CI_Lower"]
            hi = wk_out["FanShare_CI_Upper"]
            for i, name in enumerate(contestants):
                rows.append({
                    "season": season_key,
                    "season_type": season_type,
                    "week": wk_key,
                    "contestant": name,
                    "fan_mean": fan_mean[i],
                    "fan_std": fan_std[i],
                    "ci_lower": lo[i],
                    "ci_upper": hi[i],
                    "ci_width": hi[i] - lo[i],
                    "MCC_unweighted_week": wk_out.get("MCC_unweighted"),
                    "MCC_weighted_week": wk_out.get("MCC_weighted"),
                    "ACC_mean_week": wk_out.get("ACC_mean"),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_json", required=True, help="Path to week_level_data_corrected.json")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--n_particles", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    week_data = load_week_level_json(args.week_json)
    params = calibrate_from_week_json(week_data)

    pf = FanVoteParticleFilter(n_particles=args.n_particles, params=params, random_seed=args.seed)

    all_estimates: Dict[str, Any] = {"_calibrated_params": params.__dict__}
    for season_key, season_obj in week_data.items():
        all_estimates[season_key] = pf.estimate_season(season_obj, compute_ci=True)

    out_json = os.path.join(args.out_dir, "fan_voting_estimates.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_estimates, f, ensure_ascii=False, indent=2)

    # summary CSV
    rows = flatten_to_rows({k: v for k, v in all_estimates.items() if not k.startswith("_")})
    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "fan_voting_summary.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Saved:", out_json)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
