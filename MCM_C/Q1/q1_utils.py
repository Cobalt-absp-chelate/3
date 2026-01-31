# q1_utils.py
import numpy as np

def project_to_simplex(y: np.ndarray) -> np.ndarray:
    """
    欧氏投影到概率单纯形 {v>=0, sum v=1}
    用于：无淘汰约束时的闭式投影、以及数值稳健的兜底。
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    v = np.maximum(y - theta, 0.0)
    return v

def safe_normalize(x: np.ndarray) -> np.ndarray:
    s = float(np.sum(x))
    if s <= 0:
        return None
    return x / s
