# q1_week_solver.py
import numpy as np
from scipy.optimize import linprog, minimize, LinearConstraint, Bounds

from q1_rules import build_pairs
from q1_utils import project_to_simplex, safe_normalize

def compute_alpha_data_driven(J_t: np.ndarray, J_prev: np.ndarray, diffs_hist: list) -> float:
    """
    alpha 完全由数据计算，不拍：
      d = ||J_t - J_prev||^2
      scale = median(历史 d>0)
      alpha = d / (d + scale)
    """
    d = float(np.sum((J_t - J_prev) ** 2))
    diffs_hist.append(d)

    pos = [x for x in diffs_hist if x > 0]
    scale = float(np.median(pos)) if len(pos) > 0 else d

    if d == 0.0:
        return 0.0
    if scale == 0.0:
        return 1.0
    return d / (d + scale)

def solve_one_week(A_ids, E_ids, J_vec, method: str,
                   v_prev=None, J_prev=None, diffs_hist=None,
                   Rj_vec=None):
    """
    两阶段：
      Stage1 (LP): min sum xi  s.t. 单纯形 + 淘汰约束(带xi)
      Stage2 (QP): 在 sum xi = slack_opt 下，min ||v - m||^2
                   m = (1-alpha)v_prev + alpha*J
    """
    N = len(A_ids)
    diffs_hist = diffs_hist if diffs_hist is not None else []

    pairs = build_pairs(A_ids, E_ids, J_vec, method, Rj_vec=Rj_vec)
    M = len(pairs)

    # ---------- 计算 alpha 和 m ----------
    if v_prev is None or J_prev is None:
        alpha = 1.0
        m = J_vec.copy()
    else:
        alpha = compute_alpha_data_driven(J_vec, J_prev, diffs_hist)
        m = (1.0 - alpha) * v_prev + alpha * J_vec

    # 没有淘汰约束：直接投影到单纯形
    if M == 0:
        v_star = project_to_simplex(m)
        return {"v": v_star, "xi": np.zeros(0), "slack_sum": 0.0,
                "alpha": alpha, "pairs": pairs, "diffs_hist": diffs_hist}

    # 变量 x = [v(0..N-1), xi(0..M-1)]
    # ---------- Stage 1: LP 最小总松弛 ----------
    c = np.concatenate([np.zeros(N), np.ones(M)])

    # sum v = 1
    A_eq = np.zeros((1, N + M))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # v_s - v_k + xi >= rhs  <=>  -v_s + v_k - xi <= -rhs
    A_ub = np.zeros((M, N + M))
    b_ub = np.zeros(M)
    for r, (k_idx, s_idx, rhs) in enumerate(pairs):
        A_ub[r, s_idx] = -1.0
        A_ub[r, k_idx] = +1.0
        A_ub[r, N + r] = -1.0
        b_ub[r] = -rhs

    bounds = [(0.0, 1.0)] * N + [(0.0, None)] * M

    lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bounds, method="highs")
    if not lp.success:
        raise RuntimeError(f"Stage1 LP failed: {lp.message}")

    x_lp = lp.x
    slack_opt = float(np.sum(x_lp[N:]))

    # ---------- Stage 2: QP（固定最小松弛） ----------
    # 目标：||v - m||^2
    def fun(x):
        v = x[:N]
        return float(np.sum((v - m) ** 2))

    def jac(x):
        v = x[:N]
        g = np.zeros_like(x)
        g[:N] = 2.0 * (v - m)
        return g

    # 约束1：sum v = 1
    lc1 = LinearConstraint(A_eq, lb=b_eq, ub=b_eq)

    # 约束2：v_s - v_k + xi >= rhs
    A_con = np.zeros((M, N + M))
    lb = np.zeros(M)
    ub = np.full(M, np.inf)
    for r, (k_idx, s_idx, rhs) in enumerate(pairs):
        A_con[r, s_idx] = +1.0
        A_con[r, k_idx] = -1.0
        A_con[r, N + r] = +1.0
        lb[r] = rhs
    lc2 = LinearConstraint(A_con, lb=lb, ub=ub)

    # 约束3：sum xi = slack_opt
    A_xi = np.zeros((1, N + M))
    A_xi[0, N:] = 1.0
    lc3 = LinearConstraint(A_xi, lb=[slack_opt], ub=[slack_opt])

    bnds = Bounds(
        lb=np.concatenate([np.zeros(N), np.zeros(M)]),
        ub=np.concatenate([np.ones(N), np.full(M, np.inf)])
    )

    # 用 LP 解做初值，保证可行
    x0 = x_lp.copy()

    res = minimize(fun, x0, method="trust-constr", jac=jac,
                   constraints=[lc1, lc2, lc3], bounds=bnds,
                   options={"verbose": 0, "maxiter": 2000})
    if not res.success:
        # 兜底：SLSQP
        res = minimize(
            fun, x0, method="SLSQP", jac=jac,
            constraints=[
                {"type": "eq", "fun": lambda x: np.sum(x[:N]) - 1.0},
                {"type": "eq", "fun": lambda x: np.sum(x[N:]) - slack_opt},
                {"type": "ineq", "fun": lambda x: (A_con @ x) - lb},
            ],
            bounds=bounds, options={"maxiter": 2000}
        )
        if not res.success:
            raise RuntimeError(f"Stage2 QP failed: {res.message}")

    x_star = res.x
    v_star = x_star[:N]
    xi_star = x_star[N:]

    return {"v": v_star, "xi": xi_star, "slack_sum": slack_opt,
            "alpha": alpha, "pairs": pairs, "diffs_hist": diffs_hist}
