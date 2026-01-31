# q1_config.py

# 题面：percent 赛制中总分 = 0.5*粉丝百分比 + 0.5*评委百分比
OMEGA_PERCENT = 0.5

# 题面 Appendix：seasons 1,2 和 28-34 用 rank；3-27 用 percent
RANK_SEASONS = set([1, 2] + list(range(28, 35)))

def season_method(season: int) -> str:
    """返回 'rank' 或 'percent'。"""
    return "rank" if season in RANK_SEASONS else "percent"
