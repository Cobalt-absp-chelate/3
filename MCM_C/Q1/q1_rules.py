# q1_rules.py
import numpy as np
from q1_config import OMEGA_PERCENT

def build_pairs(A_ids, E_ids, J_vec, method: str, Rj_vec=None):
    """
    返回 pairs: [(k_idx, s_idx, rhs), ...]
    约束形式统一为：v_s - v_k + xi_{k,s} >= rhs
    """
    N = len(A_ids)
    id2idx = {cid: i for i, cid in enumerate(A_ids)}
    S_ids = [cid for cid in A_ids if cid not in set(E_ids)]

    pairs = []
    if len(E_ids) == 0 or len(S_ids) == 0:
        return pairs

    if method == "percent":
        # ω=0.5 -> v_s - v_k >= j_k - j_s
        # 注意：这里 rhs 直接用 (j_k - j_s)，完全对应你的推导
        for k in E_ids:
            for s in S_ids:
                k_idx = id2idx[k]
                s_idx = id2idx[s]
                rhs = float(J_vec[k_idx] - J_vec[s_idx])
                pairs.append((k_idx, s_idx, rhs))
    elif method == "rank":
        # δ = 1/Nt（由当周人数计算）
        if Rj_vec is None:
            raise ValueError("rank赛制必须提供 Rj_vec")
        delta = 1.0 / float(N)
        for k in E_ids:
            for s in S_ids:
                k_idx = id2idx[k]
                s_idx = id2idx[s]
                rhs = float(delta * (Rj_vec[s_idx] - Rj_vec[k_idx]))
                pairs.append((k_idx, s_idx, rhs))
    else:
        raise ValueError("method must be 'percent' or 'rank'")

    return pairs
