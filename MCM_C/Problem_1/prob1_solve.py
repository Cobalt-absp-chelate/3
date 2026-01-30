import numpy as np
import pandas as pd
import json
from scipy.special import softmax
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')


class DWTSFanVotingEstimator:
    def __init__(self, n_particles=10000, sigma_proc=None, lambda_soft=10.0, memory_coeff=0.7):
        self.n_particles = n_particles
        self.sigma_proc = sigma_proc
        self.lambda_soft = lambda_soft
        self.memory_coeff = memory_coeff
        self.glm_coeffs = None
        self.scaler = StandardScaler()

    def estimate_process_noise(self, processed_data):
        judge_scores = []
        for week in range(1, 6):
            score_col = f'week{week}_judge_score_ratio'
            if score_col in processed_data.columns:
                scores = processed_data[score_col].dropna()
                judge_scores.extend(scores)

        if len(judge_scores) > 0:
            return np.std(judge_scores) * 0.5
        else:
            return 0.2

    def fit_glm_prior(self, processed_data):
        features = []
        targets = []

        for _, row in processed_data.iterrows():
            for week in range(1, 11):
                current_score = f'week{week}_judge_score_ratio'
                next_score = f'week{week + 1}_judge_score_ratio'

                if current_score in row and next_score in row:
                    if pd.notna(row[current_score]) and pd.notna(row[next_score]):
                        feature_vec = [
                            row[current_score],
                            row['celebrity_age_during_season'] / 50.0,
                            self._encode_profession(row['profession_category']),
                            row['partner_experience'] / 10.0
                        ]
                        features.append(feature_vec)
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
            self.glm_coeffs = np.array([0.3, -0.1, 0.0, 0.1])
            self.glm_intercept = 0.0

    def _encode_profession(self, profession):
        prof_map = {
            'Screen': 0, 'Athlete': 1, 'Music': 2, 'Reality': 3,
            'Model': 4, 'Comedian': 5, 'Politician': 6, 'Other': 7
        }
        return prof_map.get(profession, 7) / 7.0

    def _compute_glm_delta(self, features):
        if self.glm_coeffs is not None:
            features_scaled = self.scaler.transform([features])[0]
            return self.glm_intercept + np.dot(self.glm_coeffs, features_scaled)
        return 0.0

    def _softmax_stable(self, theta):
        theta_max = np.max(theta)
        exp_theta = np.exp(theta - theta_max)
        return exp_theta / np.sum(exp_theta)

    def _sigmoid_constraint(self, survivor_scores, eliminated_scores):
        if len(eliminated_scores) == 0:
            return 1.0

        min_survivor = np.min(survivor_scores)
        max_eliminated = np.max(eliminated_scores)
        constraint_satisfaction = min_survivor - max_eliminated

        return 1.0 / (1.0 + np.exp(-self.lambda_soft * constraint_satisfaction))

    def _plackett_luce_likelihood(self, theta, ranking):
        likelihood = 1.0
        remaining_indices = list(range(len(theta)))

        for rank_pos in ranking:
            if rank_pos in remaining_indices:
                exp_theta_remaining = np.exp([theta[i] for i in remaining_indices])
                prob = exp_theta_remaining[remaining_indices.index(rank_pos)] / np.sum(exp_theta_remaining)
                likelihood *= prob
                remaining_indices.remove(rank_pos)

        return likelihood

    def _normalize_judge_scores(self, judge_scores):
        if len(judge_scores) > 0 and np.std(judge_scores) > 0:
            return (judge_scores - np.mean(judge_scores)) / np.std(judge_scores)
        return judge_scores

    def estimate_fan_voting(self, week_data, season_type='Percent', previous_theta=None):
        contestants = week_data['Contestants']
        judge_scores = np.array(week_data['Judge_Scores'])
        features_matrix = np.array(week_data['Features_Matrix'])
        eliminated = week_data.get('Eliminated', [])
        survivors = week_data.get('Survivors', [])

        n_contestants = len(contestants)
        if n_contestants <= 1:
            return {contestant: 1.0 for contestant in contestants}, {contestant: 0.0 for contestant in contestants}, np.array([0.0])

        particles = np.random.uniform(-1, 1, (self.n_particles, n_contestants))

        normalized_judge_scores = self._normalize_judge_scores(judge_scores)

        for i, features in enumerate(features_matrix):
            if len(features) >= 4:
                glm_features = features[:4]
                glm_delta = self._compute_glm_delta(glm_features)
                particles[:, i] += glm_delta

            if previous_theta is not None and i < len(previous_theta):
                particles[:, i] = self.memory_coeff * previous_theta[i] + (1 - self.memory_coeff) * normalized_judge_scores[i]

        if self.sigma_proc is not None:
            particles += np.random.normal(0, self.sigma_proc, particles.shape)

        weights = np.ones(self.n_particles) / self.n_particles

        for p in range(self.n_particles):
            voting_probs = self._softmax_stable(particles[p])
            combined_scores = 0.5 * judge_scores + 0.5 * voting_probs

            if season_type == 'Percent':
                eliminated_indices = [contestants.index(name) for name in eliminated if name in contestants]
                survivor_indices = [contestants.index(name) for name in survivors if name in contestants]

                if eliminated_indices and survivor_indices:
                    eliminated_scores = combined_scores[eliminated_indices]
                    survivor_scores = combined_scores[survivor_indices]
                    weights[p] *= self._sigmoid_constraint(survivor_scores, eliminated_scores)

            else:
                ranking_order = np.argsort(-combined_scores)
                weights[p] *= self._plackett_luce_likelihood(particles[p], ranking_order)

        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(self.n_particles) / self.n_particles

        ess = 1.0 / np.sum(weights ** 2)
        if ess < self.n_particles / 2:
            indices = np.random.choice(self.n_particles, self.n_particles, p=weights)
            particles = particles[indices]
            particles += np.random.normal(0, 0.01, particles.shape)
            weights = np.ones(self.n_particles) / self.n_particles

        fan_voting_estimates = {}
        fan_voting_uncertainties = {}
        current_theta = np.zeros(n_contestants)

        for i, contestant in enumerate(contestants):
            voting_probs = np.array([self._softmax_stable(particles[p])[i] for p in range(self.n_particles)])

            mean_vote = np.average(voting_probs, weights=weights)
            var_vote = np.average((voting_probs - mean_vote) ** 2, weights=weights)

            fan_voting_estimates[contestant] = mean_vote
            fan_voting_uncertainties[contestant] = np.sqrt(var_vote)
            current_theta[i] = np.average(particles[:, i], weights=weights)

        return fan_voting_estimates, fan_voting_uncertainties, current_theta


def perform_time_series_cross_validation(processed_data, week_data, n_splits=5):
    seasons = sorted([int(k[1:]) for k in week_data.keys()])
    n_seasons = len(seasons)
    fold_size = n_seasons // n_splits

    cv_results = {
        'fold': [],
        'train_seasons': [],
        'test_seasons': [],
        'mae': [],
        'rmse': [],
        'coverage': [],
        'interval_width': [],
        'lambda_param': [],
        'sigma_proc': []
    }

    for fold_idx in range(n_splits - 1):
        train_end_idx = (fold_idx + 1) * fold_size
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_seasons)

        train_seasons_list = seasons[:train_end_idx]
        test_seasons_list = seasons[test_start_idx:test_end_idx]

        print(f"\n=== K折交叉验证 - 第 {fold_idx + 1} 折 ===")
        print(f"训练赛季: {train_seasons_list}")
        print(f"测试赛季: {test_seasons_list}")

        train_data = processed_data[processed_data['season'].isin(train_seasons_list)]

        estimator = DWTSFanVotingEstimator()
        sigma_proc = estimator.estimate_process_noise(train_data)
        estimator.sigma_proc = sigma_proc
        estimator.fit_glm_prior(train_data)

        print(f"训练集 - σ_proc: {sigma_proc:.4f}")

        test_predictions = []
        test_true_values = []
        test_uncertainties = []

        for season_num in test_seasons_list:
            season_key = f'S{season_num}'
            if season_key not in week_data:
                continue

            season_data = week_data[season_key]
            season_type = season_data['Type']

            previous_theta = None

            for week_key in sorted(season_data['Weeks'].keys(), key=lambda x: int(x.split('_')[1])):
                week_info = season_data['Weeks'][week_key]

                if len(week_info['Contestants']) > 1:
                    fan_votes, uncertainties, current_theta = estimator.estimate_fan_voting(
                        week_info, season_type, previous_theta
                    )
                    previous_theta = current_theta

                    for contestant, pred_vote in fan_votes.items():
                        test_predictions.append(pred_vote)
                        test_uncertainties.append(uncertainties[contestant])

                        true_vote = 1.0 / len(week_info['Contestants'])
                        test_true_values.append(true_vote)

        if len(test_predictions) > 0:
            test_predictions = np.array(test_predictions)
            test_true_values = np.array(test_true_values)
            test_uncertainties = np.array(test_uncertainties)

            mae = np.mean(np.abs(test_predictions - test_true_values))
            rmse = np.sqrt(np.mean((test_predictions - test_true_values) ** 2))

            ci_lower = test_predictions - 1.96 * test_uncertainties
            ci_upper = test_predictions + 1.96 * test_uncertainties
            coverage = np.mean((test_true_values >= ci_lower) & (test_true_values <= ci_upper))
            interval_width = np.mean(ci_upper - ci_lower)

            cv_results['fold'].append(fold_idx + 1)
            cv_results['train_seasons'].append(str(train_seasons_list))
            cv_results['test_seasons'].append(str(test_seasons_list))
            cv_results['mae'].append(mae)
            cv_results['rmse'].append(rmse)
            cv_results['coverage'].append(coverage)
            cv_results['interval_width'].append(interval_width)
            cv_results['lambda_param'].append(estimator.lambda_soft)
            cv_results['sigma_proc'].append(sigma_proc)

            print(f"测试集结果:")
            print(f"  MAE: {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  覆盖率: {coverage:.4f}")
            print(f"  区间宽度: {interval_width:.6f}")

    cv_df = pd.DataFrame(cv_results)

    print("\n=== K折交叉验证汇总 ===")
    print(f"平均 MAE: {cv_df['mae'].mean():.6f} ± {cv_df['mae'].std():.6f}")
    print(f"平均 RMSE: {cv_df['rmse'].mean():.6f} ± {cv_df['rmse'].std():.6f}")
    print(f"平均覆盖率: {cv_df['coverage'].mean():.4f} ± {cv_df['coverage'].std():.4f}")
    print(f"平均区间宽度: {cv_df['interval_width'].mean():.6f} ± {cv_df['interval_width'].std():.6f}")
    print(f"λ参数稳定性: {cv_df['lambda_param'].std():.6f}")
    print(f"σ_proc稳定性: {cv_df['sigma_proc'].std():.6f}")

    return cv_df


def main():
    processed_data = pd.read_csv(r'C:\Users\11411\Desktop\Python\MCM_C\data\porcessed data\processed_dwts_data.csv')

    with open(r'C:\Users\11411\Desktop\Python\MCM_C\data\porcessed data\week_level_data.json', 'r') as f:
        week_data = json.load(f)

    print("=" * 60)
    print("粉丝投票逆向重构 - 贝叶斯状态空间模型")
    print("=" * 60)

    estimator = DWTSFanVotingEstimator()

    sigma_proc = estimator.estimate_process_noise(processed_data)
    estimator.sigma_proc = sigma_proc
    print(f"\n估计的过程噪声参数 σ_proc = {sigma_proc:.4f}")

    estimator.fit_glm_prior(processed_data)
    print("GLM先验模型拟合完成")

    print(f"选择的软约束参数 λ = {estimator.lambda_soft}")
    print(f"记忆系数 μ = {estimator.memory_coeff}")

    results = {}

    for season_key, season_data in week_data.items():
        season_num = int(season_key[1:])
        season_type = season_data['Type']

        print(f"\n处理第{season_num}季 ({season_type}制)...")

        season_results = {}
        previous_theta = None

        for week_key in sorted(season_data['Weeks'].keys(), key=lambda x: int(x.split('_')[1])):
            week_info = season_data['Weeks'][week_key]
            week_num = int(week_key.split('_')[1])

            if len(week_info['Contestants']) > 1:
                fan_votes, uncertainties, current_theta = estimator.estimate_fan_voting(
                    week_info, season_type, previous_theta
                )
                previous_theta = current_theta

                season_results[week_num] = {
                    'fan_voting_estimates': fan_votes,
                    'uncertainties': uncertainties,
                    'contestants': week_info['Contestants'],
                    'eliminated': week_info.get('Eliminated', []),
                    'survivors': week_info.get('Survivors', [])
                }

        results[season_num] = season_results

    output_file = r'C:\Users\11411\Desktop\Python\MCM_C\Problem_1\fan_voting_estimates.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n粉丝投票估计结果已保存到: {output_file}")

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
                    'Certainty_Level': 1 - uncertainty,
                    'Is_Eliminated': is_eliminated
                })

    summary_df = pd.DataFrame(summary_data)
    summary_file = r'C:\Users\11411\Desktop\Python\MCM_C\Problem_1\fan_voting_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    print(f"汇总报告已保存到: {summary_file}")

    print("\n=== 模型估计结果统计 ===")
    print(f"总计处理: {len(summary_df)} 个选手-周次记录")
    print(f"平均粉丝投票占比: {summary_df['Fan_Vote_Percentage'].mean():.4f}")
    print(f"平均不确定性: {summary_df['Uncertainty'].mean():.4f}")
    print(f"平均确定性水平: {summary_df['Certainty_Level'].mean():.4f}")

    print("\n" + "=" * 60)
    print("执行K折交叉验证")
    print("=" * 60)

    cv_results_df = perform_time_series_cross_validation(processed_data, week_data, n_splits=5)

    cv_file = r'C:\Users\11411\Desktop\Python\MCM_C\Problem_1\cross_validation_results.csv'
    cv_results_df.to_csv(cv_file, index=False)

    print(f"\n交叉验证结果已保存到: {cv_file}")

    print("\n" + "=" * 60)
    print("模型验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()