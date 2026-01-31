# q1_clean.py
import re
import numpy as np
import pandas as pd

WEEK_JUDGE_PATTERN = re.compile(r"^week(\d+)_judge(\d+)_score$", re.IGNORECASE)

def _infer_week_judge_cols(df: pd.DataFrame):
    """返回 {week: [col1,col2,...]}，并给出最大周数"""
    week_to_cols = {}
    max_week = 0
    for c in df.columns:
        m = WEEK_JUDGE_PATTERN.match(c)
        if m:
            w = int(m.group(1))
            max_week = max(max_week, w)
            week_to_cols.setdefault(w, []).append(c)
    # 保持judge列顺序
    for w in week_to_cols:
        week_to_cols[w] = sorted(week_to_cols[w], key=lambda x: int(WEEK_JUDGE_PATTERN.match(x).group(2)))
    return week_to_cols, max_week

def _parse_elim_week_from_results(s: str):
    """从 results 字段解析淘汰周。解析不到返回 None。"""
    if not isinstance(s, str):
        return None
    # 常见："Eliminated Week 2"
    m = re.search(r"Eliminated\s+Week\s+(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 有些可能写 Withdrew Week X / Disqualified Week X
    m2 = re.search(r"(Withdrew|Disqualified)\s+Week\s+(\d+)", s, flags=re.IGNORECASE)
    if m2:
        return int(m2.group(2))
    # 决赛/名次：1st Place, Runner Up, 3rd Place...
    if re.search(r"(Place|Runner\s*Up|Finalist|Winner|Champion)", s, flags=re.IGNORECASE):
        return None
    return None

def clean_dwts_data(csv_path: str) -> dict:
    """
    输出一个字典，包含：
      - raw_df: 原始清洗后的宽表
      - long_week_df: 长表 (season, week, contestant_id, total_judge, active, elim_week)
      - meta: (week_to_cols, max_week)
    """
    df = pd.read_csv(csv_path)

    # 统一缺失值
    df = df.replace(["N/A", "NA", "n/a", ""], np.nan)

    # 建 contestant_id（避免重名；你也可以改成更严格的键）
    for col in ["celebrity_name", "ballroom_partner"]:
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")
    df["contestant_id"] = df["celebrity_name"].astype(str).str.strip() + " | " + df["ballroom_partner"].astype(str).str.strip()

    # week/judge 列
    week_to_cols, max_week = _infer_week_judge_cols(df)

    # scores 转数值
    for w, cols in week_to_cols.items():
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 淘汰周：先用 results 解析
    if "results" not in df.columns:
        raise ValueError("missing required column: results")
    df["elim_week"] = df["results"].apply(_parse_elim_week_from_results)

    # 若仍为 None，则用“最后一个>0周”兜底（符合题目说明：淘汰后记录0分）:contentReference[oaicite:1]{index=1}
    # 注意：有些赛季可能中途退赛/特殊周，这里作为保守兜底。
    def fallback_elim_week(row):
        if pd.notna(row["elim_week"]):
            return int(row["elim_week"])
        last_pos = None
        for w in range(1, max_week + 1):
            cols = week_to_cols.get(w, [])
            if not cols:
                continue
            total = np.nansum([row[c] for c in cols])
            if np.isfinite(total) and total > 0:
                last_pos = w
        # 如果拿不到正分，说明这条可能是异常/全缺失
        if last_pos is None:
            return np.nan
        # last_pos 是“最后一次登场并计分的周”，等价于“淘汰发生在 last_pos 周末”或“决赛周”
        # 无法区分决赛/淘汰：若 results 没写名次，默认当作淘汰周
        return float(last_pos)

    df["elim_week"] = df.apply(fallback_elim_week, axis=1)

    # 生成长表：每周总评委分
    long_rows = []
    for _, row in df.iterrows():
        season = int(row["season"])
        cid = row["contestant_id"]
        elim_week = row["elim_week"]
        for w in range(1, max_week + 1):
            cols = week_to_cols.get(w, [])
            if not cols:
                continue
            vals = [row[c] for c in cols]
            total = np.nansum(vals) if np.any(pd.notna(vals)) else np.nan

            # active：该周有计分且>0（题目说明淘汰后为0）:contentReference[oaicite:2]{index=2}
            active = (np.isfinite(total) and total > 0)

            long_rows.append({
                "season": season,
                "week": w,
                "contestant_id": cid,
                "total_judge": float(total) if np.isfinite(total) else np.nan,
                "active": bool(active),
                "elim_week": float(elim_week) if pd.notna(elim_week) else np.nan,
            })

    long_df = pd.DataFrame(long_rows)

    # 只保留 active 选手（A_t）
    long_active = long_df[long_df["active"]].copy()

    # 计算每个(season,week)的评委占比 J_t
    long_active["judge_percent"] = long_active.groupby(["season", "week"])["total_judge"].transform(
        lambda x: x / np.nansum(x.values)
    )

    # 计算评委名次 R^J（rank赛季用）：分数高名次小（1最好）
    def rank_with_ties_desc(x: pd.Series):
        # method='average'：并列给平均名次，避免不稳定
        return x.rank(ascending=False, method="average")

    long_active["judge_rank"] = long_active.groupby(["season", "week"])["total_judge"].transform(rank_with_ties_desc)

    return {
        "raw_df": df,
        "long_week_df": long_active,
        "meta": {"week_to_cols": week_to_cols, "max_week": max_week},
    }
