from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Set
import json
import numpy as np
import pandas as pd


def load_week_level_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_percent(judge_scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    js = np.asarray(judge_scores, dtype=float)
    s = float(np.sum(js))
    if s <= eps:
        return np.ones_like(js) / len(js)
    return js / s


def ranks_desc(values: np.ndarray) -> np.ndarray:
    """Rank 1 = best (largest value). Deterministic tie-break by index."""
    v = np.asarray(values, dtype=float)
    order = np.lexsort((np.arange(len(v)), -v))
    ranks = np.empty(len(v), dtype=int)
    ranks[order] = np.arange(1, len(v) + 1)
    return ranks


def normalize_elim_survivors(contestants: List[str], eliminated: Any, survivors: Any) -> Tuple[List[str], List[str], str, bool]:
    """
    Normalize eliminated/survivors lists and infer missing side when logically possible.

    Returns (elim_list, surv_list, reason, valid_for_metrics).

    valid_for_metrics means we can define a margin-based ordering check between eliminated and survivors.
    """
    # to list
    if eliminated is None:
        elim = []
    elif isinstance(eliminated, str):
        elim = [eliminated]
    else:
        elim = list(eliminated)

    if survivors is None:
        surv = []
    elif isinstance(survivors, str):
        surv = [survivors]
    else:
        surv = list(survivors)

    roster: Set[str] = set(contestants)
    elim = [x for x in elim if x in roster]
    surv = [x for x in surv if x in roster]

    if not elim and not surv:
        return [], [], "no_elimination_info", False

    # infer complement if one side missing
    if elim and not surv:
        surv = [c for c in contestants if c not in set(elim)]
        surv = [x for x in surv if x in roster]
        if surv:
            return elim, surv, "survivors_inferred", True
        return elim, [], "survivors_missing", False

    if surv and not elim:
        elim = [c for c in contestants if c not in set(surv)]
        elim = [x for x in elim if x in roster]
        if elim:
            return elim, surv, "eliminated_inferred", True
        return [], surv, "eliminated_missing", False

    return elim, surv, "ok", True


def compute_acc_from_posterior_mean(
    season_type: str,
    judge_scores: List[float],
    fan_mean: List[float],
    eliminated_names: List[str],
) -> Optional[int]:
    """
    Recompute ACC from posterior-mean fan share and observed eliminated names.

    Returns 0/1, or None if cannot be evaluated (e.g., no eliminated info or roster mismatch).
    """
    contestants_n = len(fan_mean)
    if contestants_n == 0 or len(judge_scores) != contestants_n:
        return None
    if not eliminated_names:
        return None

    js = np.asarray(judge_scores, dtype=float)
    fm = np.asarray(fan_mean, dtype=float)

    # choose predicted eliminated under the SAME rule used in estimator
    if season_type.lower().startswith("percent"):
        comb = judge_percent(js) + fm
        pred = int(np.argmin(comb))
    else:
        jr = ranks_desc(js)
        fr = ranks_desc(fm)
        comb_rank = jr + fr
        pred = int(np.argmax(comb_rank))  # worst = largest sum rank

    # eliminated_names are aligned to contestants order outside this function
    return 1 if pred in eliminated_names else 0


def estimates_to_week_table(
    estimates: Dict[str, Any],
    week_data: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Produce per-week table of MCC/ACC and (optionally) diagnostic columns from week_level_data_corrected.json.
    """
    rows: List[dict] = []
    for season_key, season_out in estimates.items():
        if season_key.startswith("_"):
            continue
        season_type = season_out.get("Type", "")
        weeks_est = season_out.get("Weeks", {})

        for wk_key, wk_out in weeks_est.items():
            row = {
                "season": season_key,
                "season_type": season_type,
                "week": wk_key,
                "MCC_unweighted": wk_out.get("MCC_unweighted"),
                "MCC_weighted": wk_out.get("MCC_weighted"),
                "ACC_mean": wk_out.get("ACC_mean"),
            }

            # If week_data provided, enrich with reason / counts and recompute ACC from posterior mean
            if week_data is not None and season_key in week_data:
                wk_src = week_data[season_key].get("Weeks", {}).get(wk_key, {})
                contestants = wk_src.get("Contestants", [])
                eliminated = wk_src.get("Eliminated", [])
                survivors = wk_src.get("Survivors", [])

                elim, surv, reason, valid = normalize_elim_survivors(contestants, eliminated, survivors)

                row.update({
                    "n_contestants": len(contestants),
                    "n_eliminated": len(elim),
                    "n_survivors": len(surv),
                    "valid_for_mcc": bool(valid),
                    "reason": reason,
                })

                # recompute ACC using posterior mean (wk_out has FanShare_Mean/Judge_Scores)
                fan_mean = wk_out.get("FanShare_Mean", None)
                judge_scores = wk_out.get("Judge_Scores", None)

                if isinstance(fan_mean, list) and isinstance(judge_scores, list) and contestants:
                    # map eliminated names -> indices in roster so we can check predicted index membership
                    name_to_idx = {c: i for i, c in enumerate(contestants)}
                    elim_idx = [name_to_idx[n] for n in elim if n in name_to_idx]
                    # we store eliminated_idx list for membership test
                    acc = None
                    if elim_idx:
                        # compute pred index then check in elim_idx
                        js = np.asarray(judge_scores, dtype=float)
                        fm = np.asarray(fan_mean, dtype=float)
                        if len(js) == len(fm) and len(fm) == len(contestants):
                            if season_type.lower().startswith("percent"):
                                comb = judge_percent(js) + fm
                                pred = int(np.argmin(comb))
                            else:
                                jr = ranks_desc(js)
                                fr = ranks_desc(fm)
                                pred = int(np.argmax(jr + fr))
                            acc = 1 if pred in elim_idx else 0
                    row["ACC_mean_recomputed"] = acc

                # If estimator used -1 as sentinel, convert to NaN-like None in the table
                if row["ACC_mean"] == -1:
                    row["ACC_mean"] = None

                # Flag cases where MCC is missing but the week looks evaluable.
                if valid and (row["MCC_unweighted"] is None or (isinstance(row["MCC_unweighted"], float) and np.isnan(row["MCC_unweighted"]))):
                    row["reason"] = (row["reason"] + "|mcc_missing_in_estimates").strip("|")

            rows.append(row)

    df = pd.DataFrame(rows)

    # If recomputed ACC exists, prefer it in a clean final column ACC_01 (0/1/NaN)
    if "ACC_mean_recomputed" in df.columns:
        df["ACC_01"] = df["ACC_mean_recomputed"]
    else:
        # fallback: sanitize ACC_mean to {0,1,NaN}
        df["ACC_01"] = df["ACC_mean"].where(df["ACC_mean"].isin([0, 1]))

    return df


def estimates_to_certainty_table(estimates: Dict[str, Any]) -> pd.DataFrame:
    rows: List[dict] = []
    for season_key, season_out in estimates.items():
        if season_key.startswith("_"):
            continue
        season_type = season_out.get("Type", "")
        for wk_key, wk_out in season_out.get("Weeks", {}).items():
            contestants = wk_out["Contestants"]
            mean = wk_out["FanShare_Mean"]
            std = wk_out["FanShare_Std"]
            lo = wk_out["FanShare_CI_Lower"]
            hi = wk_out["FanShare_CI_Upper"]
            for i, name in enumerate(contestants):
                width = (hi[i] - lo[i])
                rows.append({
                    "season": season_key,
                    "season_type": season_type,
                    "week": wk_key,
                    "contestant": name,
                    "fan_mean": mean[i],
                    "fan_std": std[i],
                    "ci_lower": lo[i],
                    "ci_upper": hi[i],
                    "ci_width": width,
                    "certainty_inv_std": 1.0 / (std[i] + 1e-9),
                    "certainty_1_minus_ci_width": 1.0 - width,
                })
    return pd.DataFrame(rows)


def summarize_certainty_by_week(certainty_df: pd.DataFrame) -> pd.DataFrame:
    g = certainty_df.groupby(["season", "season_type", "week"], as_index=False).agg(
        avg_ci_width=("ci_width", "mean"),
        median_ci_width=("ci_width", "median"),
        avg_fan_std=("fan_std", "mean"),
        n_contestants=("contestant", "count"),
    )
    return g


def summarize_certainty_by_contestant(certainty_df: pd.DataFrame) -> pd.DataFrame:
    g = certainty_df.groupby(["contestant"], as_index=False).agg(
        avg_ci_width=("ci_width", "mean"),
        avg_fan_std=("fan_std", "mean"),
        n_points=("ci_width", "count"),
    ).sort_values("avg_ci_width", ascending=False)
    return g
