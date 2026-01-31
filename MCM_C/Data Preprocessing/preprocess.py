import pandas as pd
import numpy as np
import warnings
import re
import os
from pathlib import Path

warnings.filterwarnings('ignore')


class DWTSDataPreprocessor:
    """
    Dancing with the Stars Data Preprocessing System
    Enhanced with model-friendly features per data processing suggestions
    """

    PROFESSION_SUPER_CATEGORIES = {
        # 多关键词对应一个大类，覆盖常见职业类别
        'music': ['singer', 'rapper', 'musician', 'band', 'dj', 'music'],
        'screen': ['actor', 'actress', 'tv personality', 'host', 'producer', 'presenter'],
        'athlete': ['athlete', 'racing', 'olympian', 'astronaut', 'football', 'basketball', 'baseball',
                    'soccer', 'hockey', 'skater', 'runner', 'boxer', 'swimmer'],
        'reality': ['reality', 'influencer', 'social media', 'social media personality', 'reality star'],
        'model': ['model', 'beauty pageant'],
        'comedian': ['comedian'],
        'politician': ['politician'],
        'other': []  # 不符合以上分类，默认归入'Other'
    }

    def __init__(self, data_path: str):
        """Initialize and load data"""
        print("=" * 60)
        print("Dancing with the Stars - Data Preprocessing")
        print("=" * 60)

        # Load data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.df = pd.read_csv(data_path)
        self.original_shape = self.df.shape

        print(f"✓ Data loaded successfully: {self.original_shape[0]} rows × {self.original_shape[1]} columns")
        print(f"✓ Season range: S{self.df['season'].min()} - S{self.df['season'].max()}")

    def identify_score_structure(self):
        """Step 1: Identify scoring structure"""
        print("\n" + "=" * 60)
        print("Step 1: Identify Scoring Structure")
        print("=" * 60)

        # Identify all score columns
        self.score_cols = [col for col in self.df.columns
                           if 'week' in col.lower() and 'judge' in col.lower() and 'score' in col.lower()]

        # Extract week numbers
        week_pattern = re.compile(r'week(\d+)_')
        self.weeks = sorted(set([
            int(week_pattern.search(col).group(1))
            for col in self.score_cols
            if week_pattern.search(col)
        ]))

        print(f"✓ Found {len(self.score_cols)} score columns")
        print(f"✓ Week range: Week 1 - Week {max(self.weeks)}")

        return self

    def clean_basic_data(self):
        """Step 2: Basic data cleaning"""
        print("\n" + "=" * 60)
        print("Step 2: Basic Data Cleaning")
        print("=" * 60)

        # Remove duplicates
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"✓ Removed duplicate rows: {before - len(self.df)}")

        # Process age
        self.df['celebrity_age_during_season'] = pd.to_numeric(
            self.df['celebrity_age_during_season'], errors='coerce'
        )

        # Standardize placement
        self.df['final_placement'] = pd.to_numeric(self.df['placement'], errors='coerce')

        # Clean names
        self.df['celebrity_name'] = self.df['celebrity_name'].str.strip()
        self.df['ballroom_partner'] = self.df['ballroom_partner'].str.strip()

        print(f"✓ Basic cleaning completed, {len(self.df)} records retained")

        return self

    def process_judge_scores(self):
        """Step 3: Process judge scores (core functionality)"""
        print("\n" + "=" * 60)
        print("Step 3: Process Judge Scores")
        print("=" * 60)

        # Convert scores to numeric
        for col in self.score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Handle 0 scores after elimination (per data note 7)
        zero_count = 0
        for col in self.score_cols:
            zero_mask = (self.df[col] == 0)
            zero_count += zero_mask.sum()
            self.df.loc[zero_mask, col] = np.nan

        print(f"✓ Converted {zero_count} post-elimination 0 scores to NaN")

        # Calculate weekly statistics including judge disagreement and judge count
        for week in self.weeks:
            week_cols = [col for col in self.score_cols if col.startswith(f'week{week}_')]

            if len(week_cols) > 0:
                # Weekly total score
                self.df[f'week{week}_total'] = self.df[week_cols].sum(axis=1, skipna=True)
                # Weekly average score
                self.df[f'week{week}_avg'] = self.df[week_cols].mean(axis=1, skipna=True)
                # Number of judges per week
                self.df[f'week{week}_judges'] = self.df[week_cols].notna().sum(axis=1)
                # Judge disagreement (standard deviation of scores)
                self.df[f'week{week}_judge_std'] = self.df[week_cols].std(axis=1, skipna=True)
                # Judge score range (max - min)
                self.df[f'week{week}_judge_range'] = (
                    self.df[week_cols].max(axis=1, skipna=True) -
                    self.df[week_cols].min(axis=1, skipna=True)
                )

                # ——— 新增 ———
                # 满分依据：3个评委则满分30，4个则40，动态计算
                # 只考虑judge数量>0的行以免除NaN误算
                max_score_per_judge_num = {3: 30, 4: 40}
                judges = self.df[f'week{week}_judges']
                total_scores = self.df[f'week{week}_total']

                normalized_ratio = []

                for idx, (score, num_judges) in enumerate(zip(total_scores, judges)):
                    if pd.isna(score) or num_judges == 0 or pd.isna(num_judges):
                        normalized_ratio.append(np.nan)
                    else:
                        max_score = max_score_per_judge_num.get(int(num_judges), 10 * int(num_judges))
                        # 避免除0错误
                        ratio = score / max_score if max_score > 0 else np.nan
                        normalized_ratio.append(ratio)

                self.df[f'week{week}_judge_score_ratio'] = normalized_ratio

        # Statistics on judge count variation
        judge_counts = {}
        for week in self.weeks[:10]:  # Check first 10 weeks
            week_cols = [col for col in self.score_cols if col.startswith(f'week{week}_')]
            judge_counts[week] = len(week_cols)

        print(f"✓ Judge count statistics (first 10 weeks): {judge_counts}")

        # Calculate overall judge disagreement metrics
        std_cols = [col for col in self.df.columns if '_judge_std' in col]
        range_cols = [col for col in self.df.columns if '_judge_range' in col]

        self.df['avg_judge_disagreement'] = self.df[std_cols].mean(axis=1, skipna=True)
        self.df['max_judge_disagreement'] = self.df[std_cols].max(axis=1, skipna=True)
        self.df['avg_judge_range'] = self.df[range_cols].mean(axis=1, skipna=True)

        print(f"✓ Average judge disagreement (std): {self.df['avg_judge_disagreement'].mean():.3f}")
        print(f"✓ Average judge score range: {self.df['avg_judge_range'].mean():.3f}")

        return self

    def analyze_judge_trends(self):
        """Step 3.5: Analyze judge scoring trends over weeks (NEW)"""
        print("\n" + "=" * 60)
        print("Step 3.5: Analyze Judge Scoring Trends")
        print("=" * 60)

        # Get weekly average score columns (judge_score_ratio优先)
        ratio_cols = [col for col in self.df.columns if col.endswith('_judge_score_ratio') and 'week' in col]
        ratio_cols = sorted(ratio_cols, key=lambda x: int(re.search(r'week(\d+)', x).group(1)))

        # 备用：仍保留avg列趋势计算
        avg_cols = [col for col in self.df.columns if col.endswith('_avg') and 'week' in col]
        avg_cols = sorted(avg_cols, key=lambda x: int(re.search(r'week(\d+)', x).group(1)))

        def calculate_trend(cols):
            # 返回线性趋势斜率、R2，覆盖率(当前已有数据周数)
            def trend_per_row(row):
                scores = []
                weeks = []

                for i, col in enumerate(cols):
                    val = row[col]
                    if pd.notna(val) and val > 0:
                        scores.append(val)
                        weeks.append(i+1)
                if len(scores) < 2:
                    return 0, 0, 0
                try:
                    coeffs = np.polyfit(weeks, scores, 1)
                    slope = coeffs[0]
                    y_pred = np.polyval(coeffs, weeks)
                    ss_res = np.sum((np.array(scores)-y_pred)**2)
                    ss_tot = np.sum((np.array(scores)-np.mean(scores))**2)
                    r2 = 1 - (ss_res/ss_tot) if ss_tot != 0 else 0
                    return slope, r2, len(scores)
                except:
                    return 0, 0, len(scores)
            return trend_per_row

        # 使用归一化后的评分比例计算趋势更合理
        trend_results = self.df.apply(calculate_trend(ratio_cols), axis=1)
        self.df['score_trend_slope'] = [x[0] for x in trend_results]
        self.df['score_trend_r2'] = [x[1] for x in trend_results]
        self.df['weeks_with_scores'] = [x[2] for x in trend_results]

        # 新增 最近两周动量特征  last_2_weeks_momentum
        def last_2_weeks_momentum(row):
            vals = []
            for col in ratio_cols[-2:]:
                val = row[col]
                if pd.isna(val):
                    return np.nan
                vals.append(val)
            if len(vals) < 2:
                return np.nan
            return vals[-1] - vals[-2]

        self.df['last_2_weeks_momentum'] = self.df.apply(last_2_weeks_momentum, axis=1)

        # Categorize trends
        def categorize_trend(slope, r2, weeks):
            if weeks < 3:
                return 'Insufficient_Data'
            elif r2 < 0.3:  # Low correlation
                return 'No_Clear_Trend'
            elif slope > 0.05:  # 斜率阈值调小到0.05更敏感
                return 'Improving'
            elif slope < -0.05:
                return 'Declining'
            else:
                return 'Stable'

        self.df['trend_category'] = self.df.apply(
            lambda row: categorize_trend(
                row['score_trend_slope'],
                row['score_trend_r2'],
                row['weeks_with_scores']
            ), axis=1
        )

        # Analyze overall judge scoring patterns by week (ratio均值)
        weekly_avg_scores = {}
        weekly_std_scores = {}

        for week in self.weeks[:10]:
            col = f'week{week}_judge_score_ratio'
            if col in self.df.columns:
                valid_scores = self.df[col].dropna()
                if len(valid_scores) > 0:
                    weekly_avg_scores[week] = valid_scores.mean()
                    weekly_std_scores[week] = valid_scores.std()

        if len(weekly_avg_scores) >= 3:
            try:
                weeks_list = list(weekly_avg_scores.keys())
                scores_list = list(weekly_avg_scores.values())
                overall_trend = np.polyfit(weeks_list, scores_list, 1)[0]
                print(f"✓ Overall judge scoring trend (ratio): {overall_trend:+.4f} per week")
            except:
                overall_trend = 0
                print("✓ Overall judge scoring trend: Calculation failed")

        trend_dist = self.df['trend_category'].value_counts()
        print(f"✓ Individual contestant trends:")
        for trend, count in trend_dist.items():
            print(f"  • {trend}: {count} contestants")

        print(f"✓ Average trend slope: {self.df['score_trend_slope'].mean():.4f}")
        print(f"✓ Average trend R²: {self.df['score_trend_r2'].mean():.3f}")

        return self

    def analyze_scoring_systems(self):
        """Step 4: Analyze scoring systems (per problem requirements)"""
        print("\n" + "=" * 60)
        print("Step 4: Analyze Scoring Systems")
        print("=" * 60)

        # Define scoring system based on problem statement
        def get_scoring_system(season):
            if season <= 2 or season >= 28:
                return 'Rank'  # Seasons 1-2 and 28+: Rank-based
            elif 3 <= season <= 27:
                return 'Percent'  # Seasons 3-27: Percentage-based
            else:
                return 'Unknown'

        self.df['scoring_system'] = self.df['season'].apply(get_scoring_system)

        # Mark special features
        self.df['has_judge_elimination'] = (self.df['season'] >= 28).astype(int)
        self.df['is_allstar_season'] = (self.df['season'] == 15).astype(int)

        system_counts = self.df['scoring_system'].value_counts()
        print(f"✓ Scoring system distribution:")
        for system, count in system_counts.items():
            print(f"  • {system}: {count} contestants")

        return self

    def extract_contestant_features(self):
        """Step 5: Extract contestant features (based on actual data)"""
        print("\n" + "=" * 60)
        print("Step 5: Extract Contestant Features")
        print("=" * 60)

        # Age groups
        self.df['age_group'] = pd.cut(
            self.df['celebrity_age_during_season'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['Under 25', '25-34', '35-44', '45-54', '55+']
        )

        # Profession classification映射到超级大类
        def classify_profession_super(industry):
            if pd.isna(industry):
                return 'Unknown'
            industry_lower = str(industry).lower()
            for category, keywords in self.PROFESSION_SUPER_CATEGORIES.items():
                if any(word in industry_lower for word in keywords):
                    return category.capitalize()
            return 'Other'

        self.df['profession_category'] = self.df['celebrity_industry'].apply(classify_profession_super)

        # Geographic features
        self.df['is_us_contestant'] = (
            self.df['celebrity_homecountry/region'] == 'United States'
        ).astype(int)

        print(f"✓ Profession distribution:")
        for prof, count in self.df['profession_category'].value_counts().items():
            print(f"  • {prof}: {count} contestants")

        # 采样简单性别推断（示例，基于名字首字母或自定义字典，后期优化可增强）
        # 这里只做示例，实际应替换或用nlp包
        def infer_gender(name):
            if pd.isna(name):
                return 'Unknown'
            name = name.lower()
            # 绝对简单示例用字典
            male_names = {'jerry', 'bobby', 'apolo', 'warren', 'chris', 'adam', 'john', 'michael', 'drew'}
            female_names = {'kelly', 'britt', 'kristi', 'pamela', 'sharna', 'cheryl', 'erin', 'pamela', 'toni'}
            first_name = name.split()[0]
            if first_name in male_names:
                return 'Male'
            elif first_name in female_names:
                return 'Female'
            else:
                return 'Unknown'

        self.df['gender'] = self.df['celebrity_name'].apply(infer_gender)

        return self

    def calculate_performance_metrics(self):
        """Step 6: Calculate performance metrics"""
        print("\n" + "=" * 60)
        print("Step 6: Calculate Performance Metrics")
        print("=" * 60)

        # Use judge_score_ratio cols 优先
        ratio_cols = [col for col in self.df.columns if col.endswith('_judge_score_ratio') and 'week' in col]

        self.df['overall_avg_score'] = self.df[ratio_cols].mean(axis=1, skipna=True)
        self.df['max_score'] = self.df[ratio_cols].max(axis=1, skipna=True)
        self.df['min_score'] = self.df[ratio_cols].min(axis=1, skipna=True)
        self.df['score_range'] = self.df['max_score'] - self.df['min_score']

        self.df['weeks_competed'] = self.df[ratio_cols].notna().sum(axis=1)

        self.df['reached_finale'] = (self.df['final_placement'] <= 3).astype(int)

        print(f"✓ Average normalized score: {self.df['overall_avg_score'].mean():.3f}")
        print(f"✓ Average weeks competed: {self.df['weeks_competed'].mean():.2f}")
        print(f"✓ Finalists: {self.df['reached_finale'].sum()} contestants")

        return self

    def analyze_partner_effects(self):
        """Step 7: Analyze partner effects"""
        print("\n" + "=" * 60)
        print("Step 7: Analyze Partner Effects")
        print("=" * 60)

        # 统计全部赛季内合作次数，用于partner_experience
        partner_counts = self.df['ballroom_partner'].value_counts()
        self.df['partner_experience'] = self.df['ballroom_partner'].map(partner_counts)

        # 计算舞伴历史平均排名和胜数，支持模型量化舞伴实力（这里统计历史所有赛季数据）
        partner_stats = self.df.groupby('ballroom_partner').agg(
            historical_avg_rank=pd.NamedAgg(column='final_placement', aggfunc='mean'),
            historical_wins=pd.NamedAgg(column='final_placement', aggfunc=lambda x: (x == 1).sum())
        )

        self.df = self.df.merge(partner_stats, on='ballroom_partner', how='left')

        # 舞伴当前赛季局部统计，可选，由于样本量可能不足
        # 多年前数据有用，但这里先保留历史整体

        # Partner win rate (from this and past data)
        partner_wins = self.df[self.df['final_placement'] == 1].groupby('ballroom_partner').size()
        self.df['partner_wins_current'] = self.df['ballroom_partner'].map(partner_wins).fillna(0)

        print(f"✓ Partner experience and historical performance extracted")

        # Top partners by wins (当前数据)
        top_partners = partner_wins.sort_values(ascending=False).head(5)
        print(f"✓ Top partners by current wins:")
        for partner, wins in top_partners.items():
            print(f"  • {partner}: {int(wins)} wins")

        return self

    def identify_controversial_contestants(self):
        """Step 8: Identify controversial contestants (per problem examples)"""
        print("\n" + "=" * 60)
        print("Step 8: Identify Controversial Contestants")
        print("=" * 60)

        controversial_cases = [
            ('Jerry Rice', 2),
            ('Billy Ray Cyrus', 4),
            ('Bristol Palin', 11),
            ('Bobby Bones', 27)
        ]

        self.df['is_controversial_case'] = 0
        found_cases = 0

        for name, season in controversial_cases:
            last_name = name.split()[-1]
            mask = (
                self.df['celebrity_name'].str.contains(last_name, case=False, na=False) &
                (self.df['season'] == season)
            )
            if mask.any():
                self.df.loc[mask, 'is_controversial_case'] = 1
                found_cases += 1
                matched_name = self.df.loc[mask, 'celebrity_name'].iloc[0]
                print(f"  ✓ Found controversial case: {matched_name} (S{season})")

        # Discrepancy between judge avg rank and final placement
        self.df['judge_rank_in_season'] = self.df.groupby('season')['overall_avg_score'].rank(ascending=False, method='min')
        self.df['rank_discrepancy'] = self.df['judge_rank_in_season'] - self.df['final_placement']
        self.df['is_controversial'] = (self.df['rank_discrepancy'] > 3).astype(int)

        print(f"✓ Problem cases found: {found_cases}/4")
        print(f"✓ System-identified controversial: {self.df['is_controversial'].sum()} contestants")

        return self

    def analyze_profession_success(self):
        """Step 9: Analyze profession vs success relationship"""
        print("\n" + "=" * 60)
        print("Step 9: Analyze Profession vs Success")
        print("=" * 60)

        profession_wins = self.df[self.df['final_placement'] == 1].groupby('profession_category').size()
        profession_avg_rank = self.df.groupby('profession_category')['final_placement'].mean()
        profession_avg_score = self.df.groupby('profession_category')['overall_avg_score'].mean()
        profession_finale_rate = self.df.groupby('profession_category')['reached_finale'].mean()
        profession_avg_judge_disagreement = self.df.groupby('profession_category')['avg_judge_disagreement'].mean()
        profession_avg_score_trend = self.df.groupby('profession_category')['score_trend_slope'].mean()

        print(f"✓ Success analysis by profession:")
        for profession in self.df['profession_category'].unique():
            if pd.notna(profession):
                wins = profession_wins.get(profession, 0)
                avg_rank = profession_avg_rank.get(profession, 0)
                avg_score = profession_avg_score.get(profession, 0)
                finale_rate = profession_finale_rate.get(profession, 0)
                avg_disagreement = profession_avg_judge_disagreement.get(profession, 0)
                avg_trend = profession_avg_score_trend.get(profession, 0)

                print(f"  • {profession}:")
                print(f"    - Wins: {int(wins)}")
                print(f"    - Avg rank: {avg_rank:.1f}")
                print(f"    - Avg score: {avg_score:.3f}")
                print(f"    - Finale rate: {finale_rate:.1%}")
                print(f"    - Avg judge disagreement: {avg_disagreement:.3f}")
                print(f"    - Avg score trend: {avg_trend:.3f}")

        return self

    def build_status_column(self):
        """
        新增：status列区分Active/Eliminated/Skip
        依据：if all judge scores NaN and not eliminated -> Skip
             if eliminated (final_placement > current week) -> Eliminated
             else Active
        这里假设淘汰的周后分数为NaN，否则状态为Skip或Active区分较复杂需结合业务逻辑优化
        """
        print("\n" + "=" * 60)
        print("Step X: Build Status Column (Active/Eliminated/Skip)")
        print("=" * 60)

        def determine_status(row, week):
            # 本周judge分数列
            score_cols = [col for col in self.score_cols if col.startswith(f'week{week}_')]
            scores = row[score_cols]

            # 如果本周所有judge分数均为NaN，且final_placement>对应周（可调整逻辑）
            if scores.isna().all():
                if row['final_placement'] is not None and row['final_placement'] <= week:
                    return 'Eliminated'
                else:
                    return 'Skip'
            else:
                return 'Active'

        # 对每周生成一列状态，后期建模可用
        for week in self.weeks:
            col_name = f'week{week}_status'
            self.df[col_name] = self.df.apply(lambda r: determine_status(r, week), axis=1)

        print(f"✓ Generated status columns for weeks: week1_status ... week{max(self.weeks)}_status")

        return self

    def create_week_level_data(self):
        """
        重点新增：创建按赛季-周层级数据字典，方便粒子滤波模型时间循环读取
        格式：
        {
            "S1": {
                "Type": "Rank" / "Percent",
                "Week_1": {
                    "Contestants": [...],
                    "Judge_Scores": [...],
                    "Features_Matrix": [...],
                    "Eliminated": [...],
                    "Survivors": [...]
                },
                "Week_2": {...}
            },
            ...
        }
        """
        print("\n" + "=" * 60)
        print("Step X: Create Week-Level Data for Modeling")
        print("=" * 60)

        week_level_data = {}
        # 使用judge_score_ratio作为核心分数
        ratio_cols_all = [col for col in self.df.columns if col.endswith('_judge_score_ratio') and 'week' in col]

        for season in sorted(self.df['season'].unique()):
            season_str = f"S{season}"
            season_df = self.df[self.df['season'] == season]
            scoring_type = season_df['scoring_system'].iloc[0] if len(season_df)>0 else 'Unknown'

            week_level_data[season_str] = {
                "Type": scoring_type,
                "Weeks": {}
            }

            # Identify weeks that actually have score ratios for this season.
            # NOTE: avoid backslashes inside f-string expressions for compatibility.
            weeks_in_season_list = []
            for c in ratio_cols_all:
                m_w = re.search(r'week(\d+)', c)
                if not m_w:
                    continue
                w = int(m_w.group(1))
                colname = f'week{w}_judge_score_ratio'
                if colname in season_df.columns and season_df[colname].notna().any():
                    weeks_in_season_list.append(w)
            weeks_in_season = sorted(set(weeks_in_season_list))

            last_survivors = set(season_df['celebrity_name'].unique())

            # ---- FIX: parse elimination week from 'results' (NOT from placement) ----
            # results examples: "Eliminated Week 2", "1st Place", "3rd Place", etc.
            elim_week_map = {}
            tmp = season_df[['celebrity_name', 'results']].drop_duplicates()
            for _, rr in tmp.iterrows():
                name = rr['celebrity_name']
                res = str(rr.get('results', ''))
                m_elim = re.search(r'Eliminated\s*Week\s*(\d+)', res, flags=re.IGNORECASE)
                if m_elim:
                    elim_week_map[name] = int(m_elim.group(1))
                else:
                    elim_week_map[name] = None

            for week in weeks_in_season:
                week_col = f'week{week}_judge_score_ratio'

                # 这周参赛（有非空分数）选手
                this_week_df = season_df[season_df[week_col].notna()]

                contestants = this_week_df['celebrity_name'].tolist()
                scores = this_week_df[week_col].tolist()

                # 参赛选手特征矩阵 示例特征: age, gender编码, profession编码, partner_experience, partner_historical_avg_rank
                # 性别映射
                gender_map = {'Male': 0, 'Female': 1, 'Unknown': -1}
                prof_map = {k: i for i, k in enumerate(sorted(self.df['profession_category'].dropna().unique()))}

                def get_features(row):
                    gender_num = gender_map.get(row['gender'], -1)
                    prof_num = prof_map.get(row['profession_category'], -1)
                    return [
                        row['celebrity_age_during_season'] if pd.notna(row['celebrity_age_during_season']) else -1,
                        gender_num,
                        prof_num,
                        row.get('partner_experience', 0),
                        row.get('historical_avg_rank', 1000)  # 1000作为默认大排名表示未知
                    ]

                features_matrix = np.array([get_features(r) for _, r in this_week_df.iterrows()])

                # ---- FIX: elimination is defined by 'results' text (Eliminated Week k) ----
                # Only mark elimination among current-week contestants to keep alignment.
                eliminated = [n for n in contestants if elim_week_map.get(n, None) == week]

                # Survivors = current contestants not eliminated (also intersect with last_survivors for robustness)
                survivors = [n for n in contestants if (n in last_survivors) and (n not in set(eliminated))]


                week_level_data[season_str]["Weeks"][f"Week_{week}"] = {
                    "Contestants": contestants,
                    "Judge_Scores": scores,
                    "Features_Matrix": features_matrix.tolist(),
                    "Eliminated": eliminated,
                    "Survivors": survivors
                }

                # 更新幸存者集合
                last_survivors = set(survivors)

        self.week_level_data = week_level_data
        print(f"✓ Week-level hierarchical data structure created for modeling")
        return self

    def calculate_performance_metrics(self):
        # 重载，防止覆盖新增的last_2_weeks_momentum
        print("\n" + "=" * 60)
        print("Step 6: Calculate Performance Metrics")
        print("=" * 60)

        ratio_cols = [col for col in self.df.columns if col.endswith('_judge_score_ratio') and 'week' in col]

        self.df['overall_avg_score'] = self.df[ratio_cols].mean(axis=1, skipna=True)
        self.df['max_score'] = self.df[ratio_cols].max(axis=1, skipna=True)
        self.df['min_score'] = self.df[ratio_cols].min(axis=1, skipna=True)
        self.df['score_range'] = self.df['max_score'] - self.df['min_score']

        self.df['weeks_competed'] = self.df[ratio_cols].notna().sum(axis=1)

        self.df['reached_finale'] = (self.df['final_placement'] <= 3).astype(int)

        print(f"✓ Average normalized score: {self.df['overall_avg_score'].mean():.3f}")
        print(f"✓ Average weeks competed: {self.df['weeks_competed'].mean():.2f}")
        print(f"✓ Finalists: {self.df['reached_finale'].sum()} contestants")

        return self

    def create_final_dataset(self):
        """Step 10: Create final dataset"""
        print("\n" + "=" * 60)
        print("Step 10: Create Final Dataset")
        print("=" * 60)

        # 基础核心列+新增列
        core_features = [
            'celebrity_name', 'ballroom_partner', 'season', 'celebrity_age_during_season',
            'celebrity_industry', 'celebrity_homecountry/region',
            'results', 'final_placement', 'reached_finale',
            'age_group', 'profession_category', 'is_us_contestant', 'gender',
            'overall_avg_score', 'max_score', 'min_score', 'score_range',
            'weeks_competed',
            'score_trend_slope', 'score_trend_r2', 'weeks_with_scores', 'trend_category',
            'avg_judge_disagreement', 'max_judge_disagreement', 'avg_judge_range',
            'partner_experience', 'partner_wins_current', 'historical_avg_rank', 'historical_wins',
            'is_controversial_case', 'is_controversial', 'rank_discrepancy',
            'scoring_system', 'has_judge_elimination', 'is_allstar_season',
            'last_2_weeks_momentum'
        ]

        weekly_features = []
        for week in range(1, min(11, max(self.weeks) + 1)):
            week_features = [
                f'week{week}_total', f'week{week}_avg', f'week{week}_judges',
                f'week{week}_judge_std', f'week{week}_judge_range',
                f'week{week}_judge_score_ratio',
                f'week{week}_status',  # weeks状态列，辅助分析
            ]
            weekly_features.extend([f for f in week_features if f in self.df.columns])

        all_features = core_features + weekly_features
        available_features = [f for f in all_features if f in self.df.columns]

        self.df_processed = self.df[available_features].copy()

        # 移除没有得分的选手行
        score_cols_final = [col for col in self.df_processed.columns if '_judge_score_ratio' in col]
        has_any_score = self.df_processed[score_cols_final].notna().any(axis=1)
        self.df_processed = self.df_processed[has_any_score]

        print(f"✓ Final dataset: {self.df_processed.shape[0]} rows × {self.df_processed.shape[1]} columns")
        print(f"✓ Core features: {len(core_features)}")
        print(f"✓ Weekly features: {len(weekly_features)}")

        return self

    def generate_summary_report(self):
        """Step 11: Generate summary report"""
        print("\n" + "=" * 60)
        print("Data Preprocessing Summary Report")
        print("=" * 60)

        print(f"\n📊 Data Overview:")
        print(f"  • Original data: {self.original_shape[0]} rows")
        print(f"  • Processed data: {self.df_processed.shape[0]} rows")
        print(f"  • Retention rate: {self.df_processed.shape[0] / self.original_shape[0] * 100:.1f}%")
        print(f"  • Feature count: {self.df_processed.shape[1]}")

        print(f"\n🏆 Season Distribution:")
        season_dist = self.df_processed['scoring_system'].value_counts()
        for system, count in season_dist.items():
            print(f"  • {system}: {count} contestants")

        print(f"\n👥 Contestant Composition:")
        prof_dist = self.df_processed['profession_category'].value_counts()
        for prof, count in prof_dist.items():
            print(f"  • {prof}: {count} contestants")

        print(f"\n⚡ Controversial Contestants:")
        print(f"  • Problem-mentioned cases: {self.df_processed['is_controversial_case'].sum()}")
        print(f"  • System-identified: {self.df_processed['is_controversial'].sum()}")

        print(f"\n📈 Key Statistics:")
        print(f"  • Average age: {self.df_processed['celebrity_age_during_season'].mean():.1f} years")
        print(f"  • Average score: {self.df_processed['overall_avg_score'].mean():.3f}")
        print(f"  • Average weeks competed: {self.df_processed['weeks_competed'].mean():.1f} weeks")

        print(f"\n📊 Judge Scoring Patterns:")
        print(f"  • Average judge disagreement: {self.df_processed['avg_judge_disagreement'].mean():.3f}")
        print(f"  • Average judge score range: {self.df_processed['avg_judge_range'].mean():.3f}")

        print(f"\n📈 Score Trends:")
        trend_dist = self.df_processed['trend_category'].value_counts()
        for trend, count in trend_dist.items():
            print(f"  • {trend}: {count} contestants")

        print(f"\n🎯 New Features Added:")
        print(f"  • Judge score normalization (ratio): ✓")
        print(f"  • Last 2 weeks momentum: ✓")
        print(f"  • Partner historical performance: ✓")
        print(f"  • Status tracking (Active/Eliminated/Skip): ✓")
        print(f"  • Week-level hierarchical data structure: ✓")

        return self

    def save_data(self, output_dir: str):
        """Step 12: Save processed data"""
        print(f"\n💾 Saving data to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Save main processed data
        main_file = os.path.join(output_dir, 'processed_dwts_data.csv')
        self.df_processed.to_csv(main_file, index=False)

        # Save controversial cases separately
        controversial = self.df_processed[
            (self.df_processed['is_controversial_case'] == 1) |
            (self.df_processed['is_controversial'] == 1)
            ][['celebrity_name', 'season', 'final_placement', 'overall_avg_score', 'rank_discrepancy',
               'score_trend_slope', 'avg_judge_disagreement', 'last_2_weeks_momentum']]

        controversial_file = os.path.join(output_dir, 'controversial_cases.csv')
        controversial.to_csv(controversial_file, index=False)

        # Save profession analysis (enhanced)
        profession_stats = self.df_processed.groupby('profession_category').agg({
            'celebrity_name': 'count',
            'final_placement': 'mean',
            'overall_avg_score': 'mean',
            'reached_finale': 'mean',
            'avg_judge_disagreement': 'mean',
            'score_trend_slope': 'mean',
            'last_2_weeks_momentum': 'mean'
        }).rename(columns={
            'celebrity_name': 'count',
            'final_placement': 'avg_placement',
            'overall_avg_score': 'avg_score',
            'reached_finale': 'finale_rate',
            'avg_judge_disagreement': 'avg_judge_disagreement',
            'score_trend_slope': 'avg_score_trend',
            'last_2_weeks_momentum': 'avg_momentum'
        })

        profession_file = os.path.join(output_dir, 'profession_analysis.csv')
        profession_stats.to_csv(profession_file)

        # Save judge scoring analysis (enhanced)
        judge_analysis = self.df_processed[[
            'celebrity_name', 'season', 'overall_avg_score', 'final_placement',
            'avg_judge_disagreement', 'max_judge_disagreement', 'avg_judge_range',
            'score_trend_slope', 'score_trend_r2', 'trend_category',
            'last_2_weeks_momentum'
        ]].copy()

        judge_file = os.path.join(output_dir, 'judge_scoring_analysis.csv')
        judge_analysis.to_csv(judge_file, index=False)

        # Save partner analysis (NEW)
        partner_stats = self.df_processed.groupby('ballroom_partner').agg({
            'celebrity_name': 'count',
            'final_placement': 'mean',
            'overall_avg_score': 'mean',
            'reached_finale': 'mean',
            'historical_avg_rank': 'first',
            'historical_wins': 'first',
            'partner_experience': 'first'
        }).rename(columns={
            'celebrity_name': 'partnerships_count',
            'final_placement': 'avg_placement',
            'overall_avg_score': 'avg_score',
            'reached_finale': 'finale_rate'
        })

        partner_file = os.path.join(output_dir, 'partner_analysis.csv')
        partner_stats.to_csv(partner_file)

        # Save week-level hierarchical data for modeling (NEW)
        import json
        week_level_file = os.path.join(output_dir, 'week_level_data_corrected.json')
        with open(week_level_file, 'w') as f:
            json.dump(self.week_level_data, f, indent=2)

        # Save scoring system comparison data (NEW)
        scoring_comparison = self.df_processed.groupby(['scoring_system', 'profession_category']).agg({
            'celebrity_name': 'count',
            'final_placement': 'mean',
            'overall_avg_score': 'mean',
            'avg_judge_disagreement': 'mean',
            'rank_discrepancy': 'mean'
        }).rename(columns={
            'celebrity_name': 'count',
            'final_placement': 'avg_placement',
            'overall_avg_score': 'avg_score',
            'avg_judge_disagreement': 'avg_disagreement',
            'rank_discrepancy': 'avg_discrepancy'
        })

        scoring_file = os.path.join(output_dir, 'scoring_system_comparison.csv')
        scoring_comparison.to_csv(scoring_file)

        print(f"✓ Main data saved: {main_file}")
        print(f"✓ Controversial cases saved: {controversial_file}")
        print(f"✓ Profession analysis saved: {profession_file}")
        print(f"✓ Judge scoring analysis saved: {judge_file}")
        print(f"✓ Partner analysis saved: {partner_file}")
        print(f"✓ Week-level data saved: {week_level_file}")
        print(f"✓ Scoring system comparison saved: {scoring_file}")

        return self

    def run_complete_pipeline(self, output_dir: str):
        """Run complete preprocessing pipeline"""
        print("🚀 Starting enhanced data preprocessing...")

        try:
            (self.identify_score_structure()
             .clean_basic_data()
             .process_judge_scores()
             .analyze_judge_trends()
             .analyze_scoring_systems()
             .extract_contestant_features()
             .calculate_performance_metrics()
             .analyze_partner_effects()
             .identify_controversial_contestants()
             .analyze_profession_success()
             .build_status_column()  # NEW
             .create_week_level_data()  # NEW
             .create_final_dataset()
             .generate_summary_report()
             .save_data(output_dir))

            print(f"\n🎉 Enhanced preprocessing completed! Data saved to: {output_dir}")
            print(f"\n🔧 New Features Summary:")
            print(f"  • Judge score normalization (handles 3/4 judge variations)")
            print(f"  • Scoring system clearly marked as 'Rank' or 'Percent'")
            print(f"  • Profession categories simplified to 5 super-categories")
            print(f"  • Partner historical performance metrics")
            print(f"  • Last 2 weeks momentum for trend analysis")
            print(f"  • Week-level hierarchical data structure for time-series modeling")
            print(f"  • Status tracking for each week (Active/Eliminated/Skip)")
            print(f"  • Enhanced analysis files for different modeling approaches")

            return self.df_processed

        except Exception as e:
            print(f"\n❌ Error during preprocessing: {str(e)}")
            raise

    def get_modeling_ready_data(self):
        """
        返回建模友好的数据结构
        """
        if not hasattr(self, 'df_processed'):
            raise ValueError("Please run preprocessing pipeline first")

        if not hasattr(self, 'week_level_data'):
            raise ValueError("Week-level data not created. Please run complete pipeline")

        return {
            'contestant_level_data': self.df_processed,
            'week_level_data': self.week_level_data,
            'feature_info': {
                'numerical_features': [
                    'celebrity_age_during_season', 'overall_avg_score', 'weeks_competed',
                    'score_trend_slope', 'score_trend_r2', 'avg_judge_disagreement',
                    'partner_experience', 'historical_avg_rank', 'last_2_weeks_momentum'
                ],
                'categorical_features': [
                    'profession_category', 'gender', 'scoring_system', 'trend_category'
                ],
                'target_variables': [
                    'final_placement', 'reached_finale'
                ],
                'weekly_score_features': [
                    col for col in self.df_processed.columns
                    if '_judge_score_ratio' in col
                ]
            }
        }

    # Main execution
if __name__ == "__main__":
    # Define paths
    DATA_PATH = r"C:\Users\11411\Desktop\Python\MCM_C\data\2026_MCM_Problem_C_Data.csv"
    OUTPUT_DIR = r"C:\Users\11411\Desktop\Python\MCM_C\data\processed data"

    # Run enhanced preprocessing
    preprocessor = DWTSDataPreprocessor(DATA_PATH)
    processed_data = preprocessor.run_complete_pipeline(OUTPUT_DIR)

    # Get modeling-ready data structure
    modeling_data = preprocessor.get_modeling_ready_data()

    # Display results
    print(f"\n📋 Enhanced Processing Complete!")
    print(f"📋 Data shape: {processed_data.shape}")
    print(f"📋 Sample new columns: {[col for col in processed_data.columns if any(x in col for x in ['ratio', 'momentum', 'historical', 'status'])][:10]}")
    print(f"📋 Week-level data seasons: {list(modeling_data['week_level_data'].keys())[:5]}...")
    print(f"📋 Ready for fan vote estimation modeling! 🎯")