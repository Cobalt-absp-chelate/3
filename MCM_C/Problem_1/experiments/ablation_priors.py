from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any, List

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from estimation.dwts_estimator import load_week_level_json, calibrate_from_week_json, FanVoteParticleFilter, CalibratedParams


def summarize_season(out: Dict[str, Any]) -> Dict[str, float]:
    mcc = []
    acc = []
    ciw = []
    for wk_key, wk in out["Weeks"].items():
        if wk.get("ACC_mean", -1) != -1:
            acc.append(wk["ACC_mean"])
        if wk.get("MCC_weighted") is not None and not (wk["MCC_weighted"] != wk["MCC_weighted"]):  # not nan
            mcc.append(wk["MCC_weighted"])
        lo = np.array(wk["FanShare_CI_Lower"], dtype=float)
        hi = np.array(wk["FanShare_CI_Upper"], dtype=float)
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ciw.append(float(np.mean(hi - lo)))
    return {
        "avg_ACC_mean": float(np.mean(acc)) if acc else float("nan"),
        "avg_MCC_weighted": float(np.mean(mcc)) if mcc else float("nan"),
        "avg_CI_width": float(np.mean(ciw)) if ciw else float("nan"),
        "n_weeks_scored": int(len(acc)),
    }


def run_variant(season_obj: Dict[str, Any], params: CalibratedParams, n_particles: int, seed: int) -> Dict[str, Any]:
    pf = FanVoteParticleFilter(n_particles=n_particles, params=params, random_seed=seed)
    return pf.estimate_season(season_obj, compute_ci=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week_json", required=True)
    ap.add_argument("--season", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_particles", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    week_data = load_week_level_json(args.week_json)
    base_params = calibrate_from_week_json(week_data)
    season_obj = week_data[args.season]

    variants = [
        ("A_no_inertia", {"memory_coeff": 0.0}),
        ("B_baseline", {"memory_coeff": base_params.memory_coeff}),
        ("C_high_inertia", {"memory_coeff": float(np.clip(base_params.memory_coeff + 0.15, 0.0, 0.95))}),
    ]

    rows: List[dict] = []
    for name, overrides in variants:
        params = CalibratedParams(**base_params.__dict__)
        for k, v in overrides.items():
            setattr(params, k, v)
        out = run_variant(season_obj, params, args.n_particles, args.seed)
        summ = summarize_season(out)
        rows.append({"variant": name, **overrides, **summ})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, f"ablation_priors_{args.season}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {"baseline_params": base_params.__dict__, "variants": variants}
    out_json = os.path.join(args.out_dir, f"ablation_priors_{args.season}_meta.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:", out_csv)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
