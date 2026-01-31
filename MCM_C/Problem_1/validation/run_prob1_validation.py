from __future__ import annotations

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from validation_metrics import (
    load_week_level_json,
    estimates_to_week_table,
    estimates_to_certainty_table,
    summarize_certainty_by_week,
    summarize_certainty_by_contestant,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimates_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--week_json",
        required=False,
        default=None,
        help="(Recommended) Path to week_level_data_corrected.json to enrich validation with reasons/counts and recompute ACC as 0/1/NaN.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.estimates_json, "r", encoding="utf-8") as f:
        estimates = json.load(f)

    week_data = load_week_level_json(args.week_json) if args.week_json else None

    week_df = estimates_to_week_table(estimates, week_data=week_data)
    cert_df = estimates_to_certainty_table(estimates)
    by_week = summarize_certainty_by_week(cert_df)
    by_cont = summarize_certainty_by_contestant(cert_df)

    week_path = os.path.join(args.out_dir, "validation_week_metrics.csv")
    cert_path = os.path.join(args.out_dir, "validation_certainty_by_contestant_week.csv")
    week_sum_path = os.path.join(args.out_dir, "validation_certainty_by_week_summary.csv")
    cont_sum_path = os.path.join(args.out_dir, "validation_certainty_by_contestant_summary.csv")

    week_df.to_csv(week_path, index=False, encoding="utf-8-sig")
    cert_df.to_csv(cert_path, index=False, encoding="utf-8-sig")
    by_week.to_csv(week_sum_path, index=False, encoding="utf-8-sig")
    by_cont.to_csv(cont_sum_path, index=False, encoding="utf-8-sig")

    print("Saved:", week_path)
    print("Saved:", cert_path)
    print("Saved:", week_sum_path)
    print("Saved:", cont_sum_path)


if __name__ == "__main__":
    main()
