from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


def softmax_stable(x: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D arrays."""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    e = np.exp(x - m)
    s = np.sum(e)
    if s <= 0:
        # fallback to uniform
        return np.ones_like(x) / len(x)
    return e / s


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # clip to avoid overflow
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sd = np.std(x)
    if sd < eps:
        return np.zeros_like(x)
    return (x - mu) / (sd + eps)


def judge_percent(judge_scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert judge raw/normalized scores to per-week percent that sums to 1."""
    js = np.asarray(judge_scores, dtype=float)
    s = np.sum(js)
    if s <= eps:
        return np.ones_like(js) / len(js)
    return js / s


def ranks_desc(values: np.ndarray) -> np.ndarray:
    """
    Rank 1 = best (largest value). Ties are broken deterministically by stable sort index.
    Returns integer ranks in [1..n].
    """
    v = np.asarray(values, dtype=float)
    # stable sort by (-v, index)
    order = np.lexsort((np.arange(len(v)), -v))
    ranks = np.empty(len(v), dtype=int)
    ranks[order] = np.arange(1, len(v) + 1)
    return ranks


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: List[float]) -> np.ndarray:
    """
    Compute weighted quantiles for 1D samples.
    qs in [0,1].
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0)
    s = np.sum(w)
    if s <= 0:
        # fall back to unweighted
        return np.quantile(v, qs)
    w = w / s
    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]
    cdf = np.cumsum(w_sorted)
    return np.interp(qs, cdf, v_sorted)


@dataclass
class CalibratedParams:
    """
    All parameters calibrated from observed judges' score dynamics ONLY.
    """
    memory_coeff: float          # lambda in theta_t = lambda*theta_{t-1} + (1-lambda)*signal + noise
    sigma_proc: float            # process noise std on theta
    lambda_soft_percent: float   # sigmoid strength for Percent seasons
    lambda_soft_rank: float      # sigmoid strength for Rank seasons
    init_sigma: float            # std for initial theta particles
    ess_threshold: float         # resampling trigger as fraction of N


def calibrate_from_week_json(week_data: Dict[str, Any]) -> CalibratedParams:
    """
    Calibrate parameters based on judges' score dynamics and elimination margins.
    No fan vote ground truth is used.

    Approach:
    - memory_coeff: average lag-1 correlation of judges' z-scored vectors between consecutive weeks
      over overlapping contestants, mapped to [0.40, 0.95].
    - sigma_proc: std of week-to-week differences in z-scored judge scores for overlapping contestants,
      scaled to a reasonable range.
    - lambda_soft_percent: choose gamma so that a 'typical positive margin' in judge-percent
      produces sigmoid(gamma*margin) ≈ 0.9.
    - lambda_soft_rank: choose gamma so that a margin of 1 in rank-score space produces sigmoid ≈ 0.9.
    """
    corr_list: List[float] = []
    diff_list: List[float] = []
    margin_percent_list: List[float] = []

    for season_key, season_obj in week_data.items():
        weeks = season_obj.get("Weeks", {})
        # sort by numeric week
        week_nums = []
        for wk in weeks.keys():
            try:
                week_nums.append(int(wk.split("_")[1]))
            except Exception:
                continue
        week_nums.sort()

        prev = None
        for wn in week_nums:
            wk = weeks.get(f"Week_{wn}", {})
            contestants = wk.get("Contestants", [])
            js = np.asarray(wk.get("Judge_Scores", []), dtype=float)
            if len(contestants) != len(js) or len(js) == 0:
                prev = None
                continue

            # Percent-margin (judge-only) for calibration
            eliminated = wk.get("Eliminated", [])
            survivors = wk.get("Survivors", [])
            if isinstance(eliminated, str):
                eliminated = [eliminated]
            if isinstance(survivors, str):
                survivors = [survivors]
            if eliminated and survivors:
                jp = judge_percent(js)
                name_to_idx = {c: i for i, c in enumerate(contestants)}
                elim_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
                surv_idx = [name_to_idx[s] for s in survivors if s in name_to_idx]
                if elim_idx and surv_idx:
                    # margin: min survivor score - max eliminated score
                    m = np.min(jp[surv_idx]) - np.max(jp[elim_idx])
                    margin_percent_list.append(float(m))

            if prev is not None:
                prev_cont, prev_js = prev
                # overlap contestants
                prev_map = {c: i for i, c in enumerate(prev_cont)}
                overlap = [c for c in contestants if c in prev_map]
                if len(overlap) >= 3:
                    cur_vals = np.array([js[contestants.index(c)] for c in overlap], dtype=float)
                    prev_vals = np.array([prev_js[prev_map[c]] for c in overlap], dtype=float)
                    cur_z = zscore(cur_vals)
                    prev_z = zscore(prev_vals)
                    # correlation
                    if np.std(cur_z) > 1e-12 and np.std(prev_z) > 1e-12:
                        corr = float(np.corrcoef(cur_z, prev_z)[0, 1])
                        if not np.isnan(corr):
                            corr_list.append(corr)
                    # differences
                    diff = cur_z - prev_z
                    diff_list.extend(diff.tolist())

            prev = (contestants, js)

    # memory_coeff
    if corr_list:
        corr_mean = float(np.clip(np.mean(corr_list), -1.0, 1.0))
    else:
        corr_mean = 0.3  # weak dependence when nothing available
    # map corr to [0.40, 0.95]
    memory_coeff = float(np.clip(0.40 + 0.55 * max(0.0, corr_mean), 0.40, 0.95))

    # sigma_proc from diff std
    if diff_list:
        dstd = float(np.std(diff_list))
    else:
        dstd = 0.15
    # keep within a safe numeric range
    sigma_proc = float(np.clip(dstd, 0.03, 0.30))

    # lambda_soft_percent from typical positive judge-percent margin
    pos_margins = [m for m in margin_percent_list if m > 1e-6]
    if pos_margins:
        typical_margin = float(np.median(pos_margins))
        lambda_soft_percent = float(np.clip(math.log(0.9 / 0.1) / typical_margin, 1.0, 80.0))
    else:
        lambda_soft_percent = 15.0

    # lambda_soft_rank so that margin 1 => sigmoid≈0.9
    lambda_soft_rank = float(math.log(0.9 / 0.1))  # ≈ 2.197

    # initial theta dispersion: tie to sigma_proc
    init_sigma = float(np.clip(2.0 * sigma_proc, 0.08, 0.60))
    ess_threshold = 0.50

    return CalibratedParams(
        memory_coeff=memory_coeff,
        sigma_proc=sigma_proc,
        lambda_soft_percent=lambda_soft_percent,
        lambda_soft_rank=lambda_soft_rank,
        init_sigma=init_sigma,
        ess_threshold=ess_threshold,
    )


@dataclass
class WeekEstimate:
    contestants: List[str]
    fan_mean: np.ndarray
    fan_std: np.ndarray
    fan_ci_lower: np.ndarray
    fan_ci_upper: np.ndarray
    mcc_unweighted: float
    mcc_weighted: float
    acc_mean: int  # whether posterior mean reproduces elimination


class FanVoteParticleFilter:
    """
    Particle filter over latent popularity theta_{i,t}.
    fan_share_{i,t} = softmax(theta_t)[i]
    """

    def __init__(
        self,
        n_particles: int,
        params: CalibratedParams,
        random_seed: int = 42,
    ) -> None:
        if n_particles < 500:
            raise ValueError("n_particles should be at least 500 for stability.")
        self.n = int(n_particles)
        self.params = params
        self.rng = np.random.default_rng(random_seed)

    def _init_particles(self, n_contestants: int) -> np.ndarray:
        return self.rng.normal(loc=0.0, scale=self.params.init_sigma, size=(self.n, n_contestants))

    def _propagate(self, particles: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        State transition:
        theta_t = memory*theta_{t-1} + (1-memory)*signal + eps
        with eps ~ N(0, sigma_proc^2)
        """
        memory = self.params.memory_coeff
        noise = self.rng.normal(0.0, self.params.sigma_proc, size=particles.shape)
        return memory * particles + (1.0 - memory) * signal[None, :] + noise

    def _likelihood_percent(
        self,
        fan_share: np.ndarray,
        judge_scores: np.ndarray,
        eliminated_idx: List[int],
        survivor_idx: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft likelihood for Percent seasons:
        Combined = judge_percent + fan_share.
        Condition: eliminated have the lowest combined scores.
        Use margin = min_survivor - max_eliminated and likelihood = sigmoid(gamma*margin).

        Returns (likelihood, margin).
        """
        jp = judge_percent(judge_scores)
        combined = jp[None, :] + fan_share  # (n_particles, n_contestants)
        elim_score = np.max(combined[:, eliminated_idx], axis=1)
        surv_score = np.min(combined[:, survivor_idx], axis=1)
        margin = surv_score - elim_score
        like = sigmoid(self.params.lambda_soft_percent * margin)
        return like, margin

    def _likelihood_rank(
        self,
        fan_share: np.ndarray,
        judge_scores: np.ndarray,
        eliminated_idx: List[int],
        survivor_idx: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Soft likelihood for Rank seasons:
        fan_rank + judge_rank, eliminated should have worst (max) combined rank.
        Work in rank-score space where higher is better: rank_score = -(fan_rank + judge_rank)
        Condition: eliminated have the minimum rank_score.
        margin = min_survivor_rank_score - max_eliminated_rank_score.

        Returns (likelihood, margin).
        """
        n_particles, _ = fan_share.shape
        judge_r = ranks_desc(judge_scores)  # 1..n
        fan_ranks = np.empty_like(fan_share, dtype=int)
        for p in range(n_particles):
            fan_ranks[p, :] = ranks_desc(fan_share[p, :])
        combined_rank = fan_ranks + judge_r[None, :]
        rank_score = -combined_rank.astype(float)  # higher better
        elim_score = np.max(rank_score[:, eliminated_idx], axis=1)
        surv_score = np.min(rank_score[:, survivor_idx], axis=1)
        margin = surv_score - elim_score
        like = sigmoid(self.params.lambda_soft_rank * margin)
        return like, margin

    def _effective_sample_size(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        return 1.0 / (np.sum(w ** 2) + 1e-12)

    def _resample(self, particles: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Systematic resampling.
        """
        n = self.n
        w = weights / (np.sum(weights) + 1e-12)
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative_sum = np.cumsum(w)
        indices = np.searchsorted(cumulative_sum, positions, side="left")
        new_particles = particles[indices, :].copy()
        new_weights = np.ones(n, dtype=float) / n
        return new_particles, new_weights

    def estimate_season(
        self,
        season_obj: Dict[str, Any],
        compute_ci: bool = True,
        ci_alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run PF for one season; returns per-week estimates and metrics.

        season_obj structure (from week_level_data_corrected.json):
        {
          "Type": "Rank"|"Percent",
          "Weeks": {
            "Week_1": {"Contestants": [...], "Judge_Scores":[...], "Eliminated":[...], "Survivors":[...]},
            ...
          }
        }
        """
        season_type = season_obj.get("Type", "Percent")
        weeks = season_obj.get("Weeks", {})
        # order weeks
        week_nums = []
        for wk in weeks.keys():
            try:
                week_nums.append(int(wk.split("_")[1]))
            except Exception:
                continue
        week_nums.sort()

        outputs: Dict[str, Any] = {"Type": season_type, "Weeks": {}}

        particles: Optional[np.ndarray] = None
        weights: Optional[np.ndarray] = None
        prev_theta_mean: Optional[np.ndarray] = None
        prev_contestants: Optional[List[str]] = None

        for wn in week_nums:
            wk_key = f"Week_{wn}"
            wk = weeks.get(wk_key, {})
            contestants = wk.get("Contestants", [])
            judge_scores = np.asarray(wk.get("Judge_Scores", []), dtype=float)

            if len(contestants) == 0 or len(judge_scores) != len(contestants):
                continue

            # Build signal from current judge scores only (behavioral signal), standardized within week
            signal = zscore(judge_scores)

            # Initialize / align particles if roster changes
            if particles is None:
                particles = self._init_particles(len(contestants))
                weights = np.ones(self.n, dtype=float) / self.n
                prev_theta_mean = np.mean(particles, axis=0)
                prev_contestants = list(contestants)
            else:
                # align previous theta mean into current roster; new contestants start at 0
                prev_map = {c: i for i, c in enumerate(prev_contestants or [])}
                aligned_mean = np.zeros(len(contestants), dtype=float)
                for i, c in enumerate(contestants):
                    if c in prev_map and prev_theta_mean is not None:
                        aligned_mean[i] = float(prev_theta_mean[prev_map[c]])
                # reinitialize particles around aligned_mean (preserve uncertainty)
                particles = aligned_mean[None, :] + self.rng.normal(0.0, self.params.init_sigma, size=(self.n, len(contestants)))
                weights = np.ones(self.n, dtype=float) / self.n
                prev_contestants = list(contestants)

            # propagate
            particles = self._propagate(particles, signal)

            # convert to fan shares per particle
            fan_share = np.zeros_like(particles)
            for p in range(self.n):
                fan_share[p, :] = softmax_stable(particles[p, :])

            # observation: eliminated/survivors
            eliminated = wk.get("Eliminated", [])
            survivors = wk.get("Survivors", [])
            if isinstance(eliminated, str):
                eliminated = [eliminated]
            if isinstance(survivors, str):
                survivors = [survivors]
            name_to_idx = {c: i for i, c in enumerate(contestants)}
            eliminated_idx = [name_to_idx[e] for e in eliminated if e in name_to_idx]
            survivor_idx = [name_to_idx[s] for s in survivors if s in name_to_idx]

            # If no elimination info (e.g., no one eliminated), skip update and just report prior
            if not eliminated_idx or not survivor_idx:
                like = np.ones(self.n, dtype=float)
                mcc_unw = float("nan")
                mcc_w = float("nan")
                acc_mean = -1
            else:
                if season_type.lower().startswith("percent"):
                    like, margin = self._likelihood_percent(fan_share, judge_scores, eliminated_idx, survivor_idx)
                else:
                    like, margin = self._likelihood_rank(fan_share, judge_scores, eliminated_idx, survivor_idx)

                # MCC computed BEFORE normalization of weights (works either way)
                indicator = (margin > 0.0).astype(float)  # particle reproduces elimination ordering
                # Unweighted MCC: fraction of particles consistent with elimination ordering
                mcc_unw = float(np.mean(indicator))
                # Weighted MCC: posterior probability mass on consistent particles
                w_raw = (weights if weights is not None else np.ones(self.n) / self.n) * like
                w_sum = float(np.sum(w_raw) + 1e-12)
                mcc_w = float(np.sum((w_raw / w_sum) * indicator))

                # Update weights
                weights = w_raw / w_sum

                # ACC for posterior mean fan share
                fan_mean_tmp = np.sum(fan_share * weights[:, None], axis=0)
                if season_type.lower().startswith("percent"):
                    comb = judge_percent(judge_scores) + fan_mean_tmp
                    pred_elim = int(np.argmin(comb))
                else:
                    jr = ranks_desc(judge_scores)
                    fr = ranks_desc(fan_mean_tmp)
                    comb_rank = jr + fr
                    pred_elim = int(np.argmax(comb_rank))
                acc_mean = 1 if pred_elim in eliminated_idx else 0

                # resample if needed
                ess = self._effective_sample_size(weights)
                if ess < self.params.ess_threshold * self.n:
                    particles, weights = self._resample(particles, weights)

            # summarize posterior over fan_share
            # NOTE: fan_share derived from particles BEFORE resample; that's fine for reporting.
            # If you prefer, recompute fan_share after resample; impact is usually minor.
            if weights is None:
                weights = np.ones(self.n, dtype=float) / self.n
            fan_mean = np.sum(fan_share * weights[:, None], axis=0)
            fan_var = np.sum(((fan_share - fan_mean[None, :]) ** 2) * weights[:, None], axis=0)
            fan_std = np.sqrt(np.maximum(fan_var, 0.0))

            if compute_ci:
                lo = np.zeros(len(contestants), dtype=float)
                hi = np.zeros(len(contestants), dtype=float)
                for i in range(len(contestants)):
                    qs = weighted_quantile(fan_share[:, i], weights, [ci_alpha / 2, 1 - ci_alpha / 2])
                    lo[i], hi[i] = float(qs[0]), float(qs[1])
            else:
                lo, hi = np.full(len(contestants), np.nan), np.full(len(contestants), np.nan)

            outputs["Weeks"][wk_key] = {
                "Contestants": contestants,
                "Judge_Scores": judge_scores.tolist(),
                "FanShare_Mean": fan_mean.tolist(),
                "FanShare_Std": fan_std.tolist(),
                "FanShare_CI_Lower": lo.tolist(),
                "FanShare_CI_Upper": hi.tolist(),
                "MCC_unweighted": mcc_unw,
                "MCC_weighted": mcc_w,
                "ACC_mean": int(acc_mean),
            }

            prev_theta_mean = np.mean(particles, axis=0)

        return outputs


def load_week_level_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
