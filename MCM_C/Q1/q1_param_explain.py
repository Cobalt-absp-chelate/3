# q1_param_explain.py
from q1_config import OMEGA_PERCENT, RANK_SEASONS

def explain():
    print("=== Q1 Parameters / Symbols Explanation ===\n")

    print("1) Fixed by problem statement / data definition")
    print(f"  - omega (w) = {OMEGA_PERCENT}")
    print("    Meaning: in percent seasons, total score S = w*V + (1-w)*J, and w=0.5 (judge and fan are equally weighted).")
    print(f"  - Season method mapping: rank seasons = {sorted(list(RANK_SEASONS))}, percent seasons = others (3-27).")
    print("")

    print("2) Computed from data (NOT hand-tuned)")
    print("  - A_t: active contestants in week t, defined by total_judge>0 in that week.")
    print("  - N_t = |A_t|: number of active contestants in week t.")
    print("  - J_t: judge_percent vector, computed by total_judge / sum(total_judge) within (season,week).")
    print("  - R^J_t: judge_rank derived from total_judge descending (1 is best), with average tie handling.")
    print("  - E_t: eliminated set in week t, parsed from results or fallback to last positive-score week.")
    print("  - delta (rank relaxation scale) = 1/N_t (computed each week).")
    print("")

    print("3) Variables in optimization (inverse optimization / reconstruction)")
    print("  - V_t: unknown fan vote share vector in week t (decision variables v_i,t).")
    print("  - xi_{k,s,t}: nonnegative slack variables for elimination constraints (k eliminated, s safe).")
    print("")

    print("4) Constraints (hard logic from elimination fact)")
    print("  - Simplex: sum_i v_i,t = 1 and v_i,t >= 0.")
    print("  - Percent season (w=0.5): for k in E_t, s in S_t:  v_s - v_k >= j_k - j_s  (with slack).")
    print("  - Rank season (relaxed): for k in E_t, s in S_t:    v_s - v_k >= delta*(R^J_s - R^J_k) (with slack).")
    print("")

    print("5) Objective / 'choose a point' rule without hand-tuned C")
    print("  - Stage 1 (LP): minimize sum xi, obtaining slack_opt (minimum total relaxation needed).")
    print("    Interpretation: slack_opt quantifies abnormality/controversy of that week.")
    print("  - Stage 2 (QP): fix sum xi = slack_opt, then minimize ||V_t - m_t||^2 where")
    print("        m_t = (1-alpha_t)*V_{t-1} + alpha_t*J_t")
    print("    This enforces temporal smoothness (memory) + performance resonance (stimulus).")
    print("")

    print("6) alpha_t (data-driven, no guessing)")
    print("  - d_t = ||J_t - J_{t-1}||^2 (computed after aligning sets).")
    print("  - scale = median of historical positive d values (within that season run).")
    print("  - alpha_t = d_t / (d_t + scale), with boundary handling for d_t=0 or scale=0.")
    print("")

    print("7) Uncertainty intervals (LP boundary probing)")
    print("  - For each i, compute v_i^min and v_i^max over the feasible set with sum xi fixed to slack_opt.")
    print("  - Width U_i,t = v_i^max - v_i^min; narrower => more certain estimate.")
    print("")

if __name__ == "__main__":
    explain()
