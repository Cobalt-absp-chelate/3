import numpy as np
from scipy.optimize import linprog
from q1_rules import build_pairs

def uncertainty_intervals(A_ids, E_ids, J_vec, method: str,
                          xi_star, v_star, delta_bound: float,
                          Rj_vec=None, pairs=None, eps_floor=None):
    N = len(A_ids)

    v_star = np.asarray(v_star, dtype=float)
    if v_star.shape[0] != N:
        raise ValueError(f"v_star length mismatch: got {v_star.shape[0]}, expected {N}")

    delta = float(max(delta_bound, 0.0))
    lb = np.maximum(0.0, v_star - delta)
    ub = np.minimum(1.0, v_star + delta)
    bounds = list(zip(lb.tolist(), ub.tolist()))

    if eps_floor is None:
        eps_floor = float(np.sqrt(np.finfo(float).eps))
    else:
        eps_floor = float(eps_floor)

    if pairs is None:
        pairs = build_pairs(A_ids, E_ids, J_vec, method, Rj_vec=Rj_vec)

    M = len(pairs)
    if M == 0:
        return lb, ub

    xi_star = np.asarray(xi_star, dtype=float)
    if xi_star.ndim == 0:
        raise ValueError("xi_star must be a vector (per-pair slack), got scalar.")
    if xi_star.shape[0] != M:
        raise ValueError(f"xi_star length mismatch: got {xi_star.shape[0]}, expected {M}")

    A_eq = np.zeros((1, N))
    A_eq[0, :] = 1.0
    b_eq = np.array([1.0])

    A_ub = np.zeros((M, N))
    b_ub = np.zeros(M)
    for r, (k_idx, s_idx, rhs) in enumerate(pairs):
        A_ub[r, s_idx] = -1.0
        A_ub[r, k_idx] = +1.0
        b_ub[r] = -(float(rhs) - float(xi_star[r]))

    def solve_lp(c, b_ub_local):
        return linprog(c, A_ub=A_ub, b_ub=b_ub_local,
                       A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method="highs")

    vmin = np.full(N, np.nan)
    vmax = np.full(N, np.nan)

    b_ub_relaxed = b_ub + eps_floor

    for i in range(N):
        c = np.zeros(N)
        c[i] = 1.0
        sol = solve_lp(c, b_ub)
        if not sol.success:
            sol = solve_lp(c, b_ub_relaxed)
        if sol.success:
            vmin[i] = sol.x[i]
        else:
            vmin[i] = lb[i]

        c2 = np.zeros(N)
        c2[i] = -1.0
        sol2 = solve_lp(c2, b_ub)
        if not sol2.success:
            sol2 = solve_lp(c2, b_ub_relaxed)
        if sol2.success:
            vmax[i] = sol2.x[i]
        else:
            vmax[i] = ub[i]

    return vmin, vmax
