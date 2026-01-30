import numpy as np
import pandas as pd
import json
from scipy.optimize import minimize, LinearConstraint, Bounds
import warnings

warnings.filterwarnings('ignore')


class StaticOptimizationModel:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.results = {}

    def normalize_judge_scores(self, judge_scores):
        total = np.sum(judge_scores)
        if total > 0:
            return judge_scores / total
        return judge_scores

    def build_constraints(self, n_contestants, eliminated_indices, survivor_indices, judge_scores_normalized):
        constraints = []

        for e_idx in eliminated_indices:
            for s_idx in survivor_indices:
                def constraint_func(v, e=e_idx, s=s_idx, j=judge_scores_normalized):
                    return (j[s] + v[s]) - (j[e] + v[e]) - self.epsilon

                constraints.append({
                    'type': 'ineq',
                    'fun': constraint_func
                })

        def sum_constraint(v):
            return np.sum(v) - 1.0

        constraints.append({
            'type': 'eq',
            'fun': sum_constraint
        })

        return constraints

    def objective_function(self, v, judge_scores_normalized):
        return np.sum((v - judge_scores_normalized) ** 2)

    def solve_week(self, week_data, season_type='Percent'):
        contestants = week_data['Contestants']
        judge_scores = np.array(week_data['Judge_Scores'])
        eliminated = week_data.get('Eliminated', [])
        survivors = week_data.get('Survivors', [])

        n_contestants = len(contestants)

        if n_contestants <= 1:
            return {contestant: 1.0 for contestant in contestants}, True

        judge_scores_normalized = self.normalize_judge_scores(judge_scores)

        eliminated_indices = [contestants.index(name) for name in eliminated if name in contestants]
        survivor_indices = [contestants.index(name) for name in survivors if name in contestants]

        if len(eliminated_indices) == 0 or len(survivor_indices) == 0:
            return {contestant: judge_scores_normalized[i] for i, contestant in enumerate(contestants)}, True

        constraints = self.build_constraints(n_contestants, eliminated_indices, survivor_indices,
                                             judge_scores_normalized)

        bounds = Bounds(lb=np.zeros(n_contestants), ub=np.ones(n_contestants))

        initial_guess = judge_scores_normalized.copy()

        try:
            result = minimize(
                self.objective_function,
                initial_guess,
                args=(judge_scores_normalized,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 1000}
            )

            if result.success:
                solution = result.x
                solution = np.maximum(solution, 0)
                solution = solution / np.sum(solution)

                fan_votes = {contestant: solution[i] for i, contestant in enumerate(contestants)}
                return fan_votes, True
            else:
                return {contestant: judge_scores_normalized[i] for i, contestant in enumerate(contestants)}, False

        except Exception as e:
            return {contestant: judge_scores_normalized[i] for i, contestant in enumerate(contestants)}, False

    def solve_season(self, season_data):
        season_results = {}

        for week_key in sorted(season_data['Weeks'].keys(), key=lambda x: int(x.split('_')[1])):
            week_info = season_data['Weeks'][week_key]
            week_num = int(week_key.split('_')[1])
            season_type = season_data['Type']

            if len(week_info['Contestants']) > 1:
                fan_votes, is_feasible = self.solve_week(week_info, season_type)
                season_results[week_num] = {
                    'fan_voting_estimates': fan_votes,
                    'contestants': week_info['Contestants'],
                    'eliminated': week_info.get('Eliminated', []),
                    'survivors': week_info.get('Survivors', []),
                    'is_feasible': is_feasible
                }

        return season_results

    def compute_trajectory_smoothness(self, season_results):
        smoothness_metrics = {}

        for contestant in set([c for week in season_results.values() for c in week['contestants']]):
            votes = []
            weeks = []

            for week_num in sorted(season_results.keys()):
                if contestant in season_results[week_num]['fan_voting_estimates']:
                    votes.append(season_results[week_num]['fan_voting_estimates'][contestant])
                    weeks.append(week_num)

            if len(votes) > 1:
                votes = np.array(votes)
                diffs = np.abs(np.diff(votes))
                total_variation = np.sum(diffs)
                smoothness_metrics[contestant] = total_variation

        return smoothness_metrics


def main():
    processed_data = pd.read_csv(r'C:\Users\11411\Desktop\Python\MCM_C\data\porcessed data\processed_dwts_data.csv')

    with open(r'C:\Users\11411\Desktop\Python\MCM_C\data\porcessed data\week_level_data.json', 'r') as f:
        week_data = json.load(f)

    print("=" * 70)
    print("静态凸优化模型 - 粉丝投票逆向重构")
    print("=" * 70)

    model = StaticOptimizationModel(epsilon=1e-6)

    all_results = {}
    feasibility_report = {}
    smoothness_report = {}

    for season_key, season_data in week_data.items():
        season_num = int(season_key[1:])
        season_type = season_data['Type']

        print(f"\n处理第{season_num}季 ({season_type}制)...")

        season_results = model.solve_season(season_data)
        all_results[season_num] = season_results

        feasible_count = sum(1 for week in season_results.values() if week['is_feasible'])
        total_weeks = len(season_results)
        feasibility_report[season_num] = {
            'total_weeks': total_weeks,
            'feasible_weeks': feasible_count,
            'infeasible_weeks': total_weeks - feasible_count,
            'feasibility_rate': feasible_count / total_weeks if total_weeks > 0 else 0
        }

        smoothness_metrics = model.compute_trajectory_smoothness(season_results)
        smoothness_report[season_num] = smoothness_metrics

        print(f"  可行性: {feasible_count}/{total_weeks} 周次可行")
        if smoothness_metrics:
            avg_smoothness = np.mean(list(smoothness_metrics.values()))
            print(f"  平均轨迹波动率: {avg_smoothness:.4f}")

    output_file = r'C:\Users\11411\Desktop\Python\MCM_C\Problem_1\OUTPUT_EZ_Problem_1\fan_voting_estimates.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n粉丝投票估计结果已保存到: {output_file}")

    summary_data = []
    for season_num, season_results in all_results.items():
        for week_num, week_results in season_results.items():
            for contestant, vote_pct in week_results['fan_voting_estimates'].items():
                is_eliminated = contestant in week_results['eliminated']
                summary_data.append({
                    'Season': season_num,
                    'Week': week_num,
                    'Contestant': contestant,
                    'Fan_Vote_Percentage': vote_pct,
                    'Is_Eliminated': is_eliminated,
                    'Is_Feasible': week_results['is_feasible']
                })

    summary_df = pd.DataFrame(summary_data)
    summary_file = r'C:\Users\11411\Desktop\Python\MCM_C\Problem_1\OUTPUT_EZ_Problem_1\fan_voting_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    print(f"汇总报告已保存到: {summary_file}")

    feasibility_df = pd.DataFrame([
        {
            'Season': season_num,
            'Total_Weeks': info['total_weeks'],
            'Feasible_Weeks': info['feasible_weeks'],
            'Infeasible_Weeks': info['infeasible_weeks'],
            'Feasibility_Rate': info['feasibility_rate']
        }
        for season_num, info in feasibility_report.items()
    ])

    feasibility_file = r'C:\Users\11411\Desktop\Python\MCM_C\Problem_1\OUTPUT_EZ_Problem_1\feasibility_report.csv'
    feasibility_df.to_csv(feasibility_file, index=False)

    print(f"可行性报告已保存到: {feasibility_file}")

    print("\n" + "=" * 70)
    print("模型统计信息")
    print("=" * 70)
    print(f"总计处理: {len(summary_df)} 个选手-周次记录")
    print(f"平均粉丝投票占比: {summary_df['Fan_Vote_Percentage'].mean():.4f}")
    print(f"投票占比标准差: {summary_df['Fan_Vote_Percentage'].std():.4f}")
    print(f"投票占比最小值: {summary_df['Fan_Vote_Percentage'].min():.6f}")
    print(f"投票占比最大值: {summary_df['Fan_Vote_Percentage'].max():.6f}")

    feasible_rate = feasibility_df['Feasibility_Rate'].mean()
    print(f"\n整体可行性率: {feasible_rate:.2%}")
    print(f"完全可行赛季数: {(feasibility_df['Feasibility_Rate'] == 1.0).sum()}")
    print(f"存在不可行周次的赛季数: {(feasibility_df['Feasibility_Rate'] < 1.0).sum()}")

    infeasible_seasons = feasibility_df[feasibility_df['Feasibility_Rate'] < 1.0]
    if len(infeasible_seasons) > 0:
        print("\n存在不可行周次的赛季:")
        for _, row in infeasible_seasons.iterrows():
            print(f"  第{int(row['Season'])}季: {int(row['Infeasible_Weeks'])}个不可行周次")

    print("\n" + "=" * 70)
    print("轨迹平滑度分析")
    print("=" * 70)

    all_smoothness_values = []
    for season_num, smoothness_dict in smoothness_report.items():
        if smoothness_dict:
            all_smoothness_values.extend(smoothness_dict.values())

    if all_smoothness_values:
        print(f"平均轨迹波动率: {np.mean(all_smoothness_values):.4f}")
        print(f"轨迹波动率标准差: {np.std(all_smoothness_values):.4f}")
        print(f"轨迹波动率最大值: {np.max(all_smoothness_values):.4f}")
        print(f"轨迹波动率最小值: {np.min(all_smoothness_values):.4f}")

    print("\n" + "=" * 70)
    print("模型求解完成")
    print("=" * 70)


if __name__ == "__main__":
    main()