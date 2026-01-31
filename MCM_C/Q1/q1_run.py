import numpy as np
import pandas as pd
import os
from scipy.optimize import linprog
from q1_rules import build_pairs
from q1_clean import clean_dwts
from q1_config import season_method
from q1_utils import safe_normalize
from q1_week_solver import solve_one_week
from q1_intervals import uncertainty_intervals

def _min_feasible_delta(A_ids, E_ids, J_vec, method, xi_star, v_star, Rj_vec=None):
    N = len(A_ids)
    pairs = build_pairs(A_ids, E_ids, J_vec, method, Rj_vec=Rj_vec)
    M = len(pairs)
    if M == 0:
        return 0.0

    xi_star = np.asarray(xi_star, dtype=float)
    if xi_star.ndim == 0 or xi_star.shape[0] != M:
        return 0.0

    v_star = np.asarray(v_star, dtype=float)
    if v_star.shape[0] != N:
        return 0.0

    nvar = N + 1
    A_eq = np.zeros((1, nvar))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    A_ub = []
    b_ub = []

    for r, (k_idx, s_idx, rhs) in enumerate(pairs):
        row = np.zeros(nvar)
        row[s_idx] = -1.0
        row[k_idx] = 1.0
        A_ub.append(row)
        b_ub.append(-(float(rhs) - float(xi_star[r])))

    for i in range(N):
        row1 = np.zeros(nvar)
        row1[i] = 1.0
        row1[-1] = -1.0
        A_ub.append(row1)
        b_ub.append(float(v_star[i]))

        row2 = np.zeros(nvar)
        row2[i] = -1.0
        row2[-1] = -1.0
        A_ub.append(row2)
        b_ub.append(-float(v_star[i]))

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)

    c = np.zeros(nvar)
    c[-1] = 1.0

    bounds = [(0.0, 1.0)] * N + [(0.0, None)]

    sol = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not sol.success:
        return 0.0
    return float(sol.x[-1])

def run_q1(csv_path: str, out_csv: str = "q1_fan_vote_estimates.csv"):
    pack = clean_dwts(csv_path)
    long_df = pack["long_df"].copy()
    out_rows = []

    for season, sdf in long_df.groupby("season"):
        season = int(season)
        method = season_method(season)
        weeks = sorted(sdf["week"].unique())
        v_prev_map = None
        J_prev_map = None
        diffs_hist = []

        for w in weeks:
            w = int(w)
            wdf = sdf[sdf["week"] == w].copy()
            A_ids = wdf["contestant_id"].tolist()
            N = len(A_ids)
            J_vec = wdf["judge_percent"].to_numpy(float)
            Rj_vec = wdf["judge_rank"].to_numpy(float) if method == "rank" else None
            elim_week_int = wdf["elim_week"].apply(lambda x: int(x) if np.isfinite(x) else -999).to_numpy()
            E_ids = wdf.loc[elim_week_int == w, "contestant_id"].tolist()

            v_prev = None
            J_prev = None
            if v_prev_map is not None:
                arr = np.array([v_prev_map.get(cid, 0.0) for cid in A_ids], dtype=float)
                v_prev = safe_normalize(arr)
            if J_prev_map is not None:
                arrJ = np.array([J_prev_map.get(cid, 0.0) for cid in A_ids], dtype=float)
                J_prev = safe_normalize(arrJ)

            sol = solve_one_week(
                A_ids=A_ids, E_ids=E_ids, J_vec=J_vec, method=method,
                v_prev=v_prev, J_prev=J_prev, diffs_hist=diffs_hist, Rj_vec=Rj_vec
            )

            v_star = sol["v"]
            slack_sum = sol["slack_sum"]
            xi_star = sol["xi"]
            alpha = sol["alpha"]
            diffs_hist = sol["diffs_hist"]
            pairs_used = sol.get("pairs", None)

            if v_prev is None:
                m_vec = J_vec.copy()
            else:
                m_vec = (1.0 - float(alpha)) * v_prev + float(alpha) * J_vec

            tau = float(np.sum((v_star - m_vec) ** 2))
            delta_bound = float(np.sqrt(tau / max(N, 1)))

            eps_floor = float(np.sqrt(np.finfo(float).eps))
            delta_bound = max(delta_bound, eps_floor)

            pairs_tmp = pairs_used if pairs_used is not None else build_pairs(A_ids, E_ids, J_vec, method, Rj_vec=Rj_vec)
            if len(pairs_tmp) > 0:
                xi_star_arr = np.asarray(xi_star, dtype=float)
                if xi_star_arr.ndim == 1 and xi_star_arr.shape[0] == len(pairs_tmp):
                    viol_max = 0.0
                    for r, (k_idx, s_idx, rhs) in enumerate(pairs_tmp):
                        viol = (float(rhs) - float(xi_star_arr[r])) - (float(v_star[s_idx]) - float(v_star[k_idx]))
                        if viol > viol_max:
                            viol_max = viol
                    viol_max = max(0.0, viol_max)
                    delta_bound = max(delta_bound, viol_max + eps_floor)

            delta_feas = _min_feasible_delta(A_ids, E_ids, J_vec, method, xi_star, v_star, Rj_vec=Rj_vec)
            delta_bound = max(delta_bound, delta_feas + eps_floor)

            vmin, vmax = uncertainty_intervals(
                A_ids, E_ids, J_vec, method,
                xi_star=xi_star, v_star=v_star, delta_bound=delta_bound, Rj_vec=Rj_vec,
                pairs=pairs_used, eps_floor=eps_floor
            )

            U = vmax - vmin
            U = np.maximum(U, 0.0)

            Umax = np.nanmax(U) if np.any(np.isfinite(U)) else np.nan
            certainty = 1.0 - (U / Umax) if np.isfinite(Umax) and Umax > 0 else np.ones_like(U)

            CI = max(0.0, 1.0 - slack_sum / max(N, 1))

            for i, cid in enumerate(A_ids):
                out_rows.append({
                    "season": season,
                    "week": w,
                    "method": method,
                    "contestant_id": cid,
                    "judge_percent": float(J_vec[i]),
                    "fan_vote_est": float(v_star[i]),
                    "fan_vote_min": float(vmin[i]) if np.isfinite(vmin[i]) else np.nan,
                    "fan_vote_max": float(vmax[i]) if np.isfinite(vmax[i]) else np.nan,
                    "uncertainty_width": float(U[i]) if np.isfinite(U[i]) else np.nan,
                    "certainty_score_0_1": float(certainty[i]) if np.isfinite(certainty[i]) else np.nan,
                    "week_slack_sum": float(slack_sum),
                    "week_consistency_CI": float(CI),
                    "alpha_data_driven": float(alpha),
                    "tau_local": float(tau),
                    "delta_local": float(delta_bound),
                    "is_eliminated_this_week": bool(cid in set(E_ids)),
                })

            v_prev_map = {cid: float(v_star[i]) for i, cid in enumerate(A_ids)}
            J_prev_map = {cid: float(J_vec[i]) for i, cid in enumerate(A_ids)}

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] saved: {out_csv}, rows={len(out_df)}")
    return out_df

if __name__ == "__main__":
    out_dir = r"C:\Users\11411\Desktop\Python\MCM_C\Q1\Q1_output"
    os.makedirs(out_dir, exist_ok=True)
    run_q1(
        r"C:\Users\11411\Desktop\Python\MCM_C\data\2026_MCM_Problem_C_Data.csv",
        r"C:\Users\11411\Desktop\Python\MCM_C\Q1\Q1_output\q1_fan_vote_estimates.csv"
    )
