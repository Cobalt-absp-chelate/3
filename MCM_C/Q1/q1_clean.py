# q1_clean.py
import re
import numpy as np
import pandas as pd

WEEK_JUDGE_PATTERN = re.compile(r"^week(\d+)_judge(\d+)_score$", re.IGNORECASE)

def infer_week_judge_cols(df: pd.DataFrame):
    """返回 week_to_cols: {week:[col,...]} 与 max_week"""
    week_to_cols = {}
    max_week = 0
    for c in df.columns:
        m = WEEK_JUDGE_PATTERN.match(c)
        if m:
            w = int(m.group(1))
            max_week = max(max_week, w)
            week_to_cols.setdefault(w, []).append(c)
    for w in week_to_cols:
        week_to_cols[w] = sorted(week_to_cols[w], key=lambda x: int(WEEK_JUDGE_PATTERN.match(x).group(2)))
    return week_to_cols, max_week

def parse_elim_week_from_results(s: str):
    """解析 results 字段中的淘汰周；解析不到返回 None。"""
    if not isinstance(s, str):
        return None
    m = re.search(r"Eliminated\s+Week\s+(\d+)", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(Withdrew|Disqualified)\s+Week\s+(\d+)", s, flags=re.IGNORECASE)
    if m2:
        return int(m2.group(2))
    # 决赛名次信息：不强行给淘汰周
    if re.search(r"(Place|Runner\s*Up|Finalist|Winner|Champion)", s, flags=re.IGNORECASE):
        return None
    return None

def clean_dwts(csv_path: str) -> dict:
    """
    输出：
      raw_df: 原始宽表（加 contestant_id、elim_week）
      long_df: 长表，只保留 active，含 judge_percent、judge_rank
      meta: week_to_cols, max_week
    """
    df = pd.read_csv(csv_path)
    df = df.replace(["N/A", "NA", "n/a", ""], np.nan)

    # contestant_id：用“明星|舞伴”拼接（避免重名）
    if "celebrity_name" not in df.columns or "ballroom_partner" not in df.columns:
        raise ValueError("CSV缺少 celebrity_name 或 ballroom_partner")
    df["contestant_id"] = df["celebrity_name"].astype(str).str.strip() + " | " + df["ballroom_partner"].astype(str).str.strip()

    week_to_cols, max_week = infer_week_judge_cols(df)
    if not week_to_cols:
        raise ValueError("没有找到 weekX_judgeY_score 形式的评分列")

    # 分数转数值
    for w, cols in week_to_cols.items():
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 淘汰周：先从 results 解析
    if "results" not in df.columns:
        raise ValueError("CSV缺少 results 字段")
    df["elim_week"] = df["results"].apply(parse_elim_week_from_results)

    # 兜底：如果解析不到，用“最后一个正分周”作为淘汰/最后出现周
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
        return int(last_pos) if last_pos is not None else np.nan

    df["elim_week"] = df.apply(fallback_elim_week, axis=1)

    # 宽表 → 长表（每周总评委分）
    rows = []
    for _, r in df.iterrows():
        season = int(r["season"])
        cid = r["contestant_id"]
        elim_week = r["elim_week"]
        for w in range(1, max_week + 1):
            cols = week_to_cols.get(w, [])
            if not cols:
                continue
            vals = [r[c] for c in cols]
            total = np.nansum(vals) if np.any(pd.notna(vals)) else np.nan
            active = (np.isfinite(total) and total > 0)  # 题面：淘汰后为0
            if active:
                rows.append({
                    "season": season,
                    "week": w,
                    "contestant_id": cid,
                    "total_judge": float(total),
                    "elim_week": int(elim_week) if np.isfinite(elim_week) else np.nan
                })

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError("长表为空：请检查数据是否读取正确")

    # J_t：评委占比（归一化）
    long_df["judge_percent"] = long_df.groupby(["season", "week"])["total_judge"].transform(
        lambda x: x / float(np.sum(x.values))
    )

    # R^J：评委名次（分数高 → 名次小）
    long_df["judge_rank"] = long_df.groupby(["season", "week"])["total_judge"].transform(
        lambda x: x.rank(ascending=False, method="average")
    )

    return {"raw_df": df, "long_df": long_df, "meta": {"week_to_cols": week_to_cols, "max_week": max_week}}
