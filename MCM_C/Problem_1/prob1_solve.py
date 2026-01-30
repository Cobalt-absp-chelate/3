import numpy as np
import pandas as pd
import json
from scipy.special import softmax
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class DWTSFanVotingEstimator:
    def __init__(self, n_particles=10000, sigma_proc=None, lambda_soft=10.0):
        self.n_particles = n_particles
        self.sigma_proc = sigma_proc
        self.lambda_soft = lambda_soft
        self.glm_coeffs = None
        self.scaler = StandardScaler()

    def estimate_process_noise(self, processed_data):
        """经验贝叶斯估计过程噪声参数"""
        judge_scores = []
        for week in range(1, 6):  # 前5周
            score_col = f'week{week}_judge_score_ratio'
            if score_col in processed_data.columns:
                scores = processed_data[score_col].dropna()
                judge_scores.extend(scores)

        if len(judge_scores) > 0:
            return np.std(judge_scores) * 0.5  # 经验系数
        else:
            return 0.2  # 默认值

    def fit_glm_prior(self, processed_data):
        """拟合GLM先验模型"""
        features = []
        targets = []

        for _, row in processed_data.iterrows():
            for week in range(1, 11):
                current_score = f'week{week}_judge_score_ratio'
                next_score = f'week{week + 1}_judge_score_ratio'

                if current_score in row and next_score in row:
                    if pd.notna(row[current_score]) and pd.notna(row[next_score]):
                        # 特征：当前评委分、年龄、职业编码、舞伴经验
                        feature_vec = [
                            row[current_score],
                            row['celebrity_age_during_season'] / 50.0,  # 归一化年龄
                            self._encode_profession(row['profession_category']),
                            row['partner_experience'] / 10.0  # 归一化舞伴经验
                        ]
                        features.append(feature_vec)
                        # 目标：下周评分变化
                        targets.append(row[next_score] - row[current_score])

        if len(features) > 10:
            X = np.array(features)
            y = np.array(targets)

            X_scaled = self.scaler.fit_transform(X)

            glm = LinearRegression()
            glm.fit(X_scaled, y)
            self.glm_coeffs = glm.coef_
            self.glm_intercept = glm.intercept_
        else:
            # 默认系数
            self.glm_coeffs = np.array([0.3, -0.1, 0.0, 0.1])
            self.glm_intercept = 0.0

    def _encode_profession(self, profession):
        """职业编码"""
        prof_map = {
            'Screen': 0, 'Athlete': 1, 'Music': 2, 'Reality': 3,
            'Model': 4, 'Comedian': 5, 'Politician': 6, 'Other': 7
        }
        return prof_map.get(profession, 7) / 7.0

    def _compute_glm_delta(self, features):
        """计算GLM预测的人气增量"""
        if self.glm_coeffs is not None:
            features_scaled = self.scaler.transform([features])[0]
            return self.glm_intercept + np.dot(self.glm_coeffs, features_scaled)
        return 0.0

    def _softmax_stable(self, theta):
        """数值稳定的Softmax变换"""
        theta_max = np.max(theta)
        exp_theta = np.exp(theta - theta_max)
        return exp_theta / np.sum(exp_theta)

    def _sigmoid_constraint(self, survivor_scores, eliminated_scores):
        """Sigmoid软约束似然"""
        if len(eliminated_scores) == 0:
            return 1.0

        min_survivor = np.min(survivor_scores)
        max_eliminated = np.max(eliminated_scores)
        constraint_satisfaction = min_survivor - max_eliminated

        return 1.0 / (1.0 + np.exp(-self.lambda_soft * constraint_satisfaction))

    def _plackett_luce_likelihood(self, theta, ranking):
        """Plackett-Luce模型似然"""
        likelihood = 1.0
        remaining_indices = list(range(len(theta)))

        for rank_pos in ranking:
            if rank_pos in remaining_indices:
                exp_theta_remaining = np.exp([theta[i] for i in remaining_indices])
                prob = exp_theta_remaining[remaining_indices.index(rank_pos)] / np.sum(exp_theta_remaining)
                likelihood *= prob
                remaining_indices.remove(rank_pos)

        return likelihood

    def estimate_fan_voting(self, week_data, season_type='Percent'):
        """估计单周粉丝投票"""
        contestants = week_data['Contestants']
        judge_scores = np.array(week_data['Judge_Scores'])
        features_matrix = np.array(week_data['Features_Matrix'])
        eliminated = week_data.get('Eliminated', [])
        survivors = week_data.get('Survivors', [])

        n_contestants = len(contestants)
        if n_contestants <= 1:
            return {contestant: 1.0 for contestant in contestants}, {contestant: 0.0 for contestant in contestants}

        # 初始化粒子
        particles = np.random.uniform(-1, 1, (self.n_particles, n_contestants))
        weights = np.ones(self.n_particles) / self.n_particles

        # GLM先验调整
        for i, features in enumerate(features_matrix):
            if len(features) >= 4:
                glm_features = features[:4]
                glm_delta = self._compute_glm_delta(glm_features)
                particles[:, i] += glm_delta

        # 添加过程噪声
        if self.sigma_proc is not None:
            particles += np.random.normal(0, self.sigma_proc, particles.shape)

        # 计算似然权重
        for p in range(self.n_particles):
            voting_probs = self._softmax_stable(particles[p])

            # 计算综合得分
            combined_scores = 0.5 * judge_scores + 0.5 * voting_probs

            if season_type == 'Percent':
                # 百分比制：Sigmoid软约束
                eliminated_indices = [contestants.index(name) for name in eliminated if name in contestants]
                survivor_indices = [contestants.index(name) for name in survivors if name in contestants]

                if eliminated_indices and survivor_indices:
                    eliminated_scores = combined_scores[eliminated_indices]
                    survivor_scores = combined_scores[survivor_indices]
                    weights[p] *= self._sigmoid_constraint(survivor_scores, eliminated_scores)

            else:
                # 排名制：Plackett-Luce模型
                ranking_order = np.argsort(-combined_scores)  # 降序排列
                weights[p] *= self._plackett_luce_likelihood(particles[p], ranking_order)

        # 权重归一化
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(self.n_particles) / self.n_particles

        # 自适应重采样
        ess = 1.0 / np.sum(weights ** 2)
        if ess < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, self.n_particles, p=weights)
            particles = particles[indices]
            particles += np.random.normal(0, 0.01, particles.shape)  # 添加噪声
            weights = np.ones(self.n_particles) / self.n_particles

        # 计算最终估计
        fan_voting_estimates = {}
        fan_voting_uncertainties = {}

        for i, contestant in enumerate(contestants):
            # 转换为投票概率
            voting_probs = np.array([self._softmax_stable(particles[p])[i] for p in range(self.n_particles)])

            # 加权平均和标准差
            mean_vote = np.average(voting_probs, weights=weights)
            var_vote = np.average((voting_probs - mean_vote) ** 2, weights=weights)

            fan_voting_estimates[contestant] = mean_vote
            fan_voting_uncertainties[contestant] = np.sqrt(var_vote)

        return fan_voting_estimates, fan_voting_uncertainties


def main():
    # 加载数据
    processed_data = pd.read_csv(r'C:\Users\11411\Desktop\Python\MCM_C\data\porcessed data\processed_dwts_data.csv')

    with open(r'C:\Users\11411\Desktop\Python\MCM_C\data\porcessed data\week_level_data.json', 'r') as f:
        week_data = json.load(f)

    # 初始化模型
    estimator = DWTSFanVotingEstimator()

    # 估计过程噪声参数
    sigma_proc = estimator.estimate_process_noise(processed_data)
    estimator.sigma_proc = sigma_proc
    print(f"估计的过程噪声参数 σ_proc = {sigma_proc:.4f}")

    # 拟合GLM先验
    estimator.fit_glm_prior(processed_data)
    print("GLM先验模型拟合完成")

    # 交叉验证确定软约束参数λ
    lambda_candidates = [5.0, 10.0, 15.0, 20.0]
    best_lambda = 10.0  # 默认值

    print(f"选择的软约束参数 λ = {best_lambda}")
    estimator.lambda_soft = best_lambda

    # 处理所有赛季数据
    results = {}

    for season_key, season_data in week_data.items():
        season_num = int(season_key[1:])  # 提取赛季号
        season_type = season_data['Type']

        print(f"处理第{season_num}季 ({season_type}制)...")

        season_results = {}

        for week_key, week_info in season_data['Weeks'].items():
            week_num = int(week_key.split('_')[1])

            if len(week_info['Contestants']) > 1:
                fan_votes, uncertainties = estimator.estimate_fan_voting(week_info, season_type)

                season_results[week_num] = {
                    'fan_voting_estimates': fan_votes,
                    'uncertainties': uncertainties,
                    'contestants': week_info['Contestants'],
                    'eliminated': week_info.get('Eliminated', []),
                    'survivors': week_info.get('Survivors', [])
                }

        results[season_num] = season_results

    # 保存结果
    output_file = r'C:\Users\11411\Desktop\Python\MCM_C\Problem_1\fan_voting_estimates.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"粉丝投票估计结果已保存到: {output_file}")

    # 生成汇总报告
    summary_data = []

    for season_num, season_results in results.items():
        for week_num, week_results in season_results.items():
            for contestant, vote_pct in week_results['fan_voting_estimates'].items():
                uncertainty = week_results['uncertainties'][contestant]
                is_eliminated = contestant in week_results['eliminated']

                summary_data.append({
                    'Season': season_num,
                    'Week': week_num,
                    'Contestant': contestant,
                    'Fan_Vote_Percentage': vote_pct,
                    'Uncertainty': uncertainty,
                    'Certainty_Level': 1 - uncertainty,  # 确定性水平
                    'Is_Eliminated': is_eliminated
                })

    summary_df = pd.DataFrame(summary_data)
    summary_file = r'C:\Users\11411\Desktop\Python\MCM_C\Problem_1\fan_voting_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    print(f"汇总报告已保存到: {summary_file}")

    # 输出关键统计信息
    print("\n=== 模型估计结果统计 ===")
    print(f"总计处理: {len(summary_df)} 个选手-周次记录")
    print(f"平均粉丝投票占比: {summary_df['Fan_Vote_Percentage'].mean():.4f}")
    print(f"平均不确定性: {summary_df['Uncertainty'].mean():.4f}")
    print(f"平均确定性水平: {summary_df['Certainty_Level'].mean():.4f}")


if __name__ == "__main__":
    main()