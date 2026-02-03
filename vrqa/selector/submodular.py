# vrqa/selector/submodular.py
from __future__ import annotations

from collections import defaultdict
from math import inf
from typing import Dict, List, Optional

from vrqa.utils.similarity_utils import compute_similarity


# ====== default objective weights (match your paper narrative) ======
DEFAULT_LAMBDA_COV = 0.25
DEFAULT_LAMBDA_ANS = 0.45
DEFAULT_LAMBDA_ANCH = 0.10
DEFAULT_LAMBDA_DIV = 0.20

MMR_LAMBDA_DEFAULT = 0.20

# ====== per-view thresholds / targets (you asked to keep these tables) ======
_tau_by_view = {
    "definition": 0.76, "mechanism": 0.76, "application": 0.76, "comparison": 0.76,
    "cause": 0.76, "effect": 0.76, "condition": 0.76, "example": 0.74,
    "quantitative": 0.80, "temporal": 0.78,
}
_target_pass_rate = {
    "definition": 0.30, "mechanism": 0.30, "application": 0.30, "comparison": 0.28,
    "cause": 0.28, "effect": 0.28, "condition": 0.25, "example": 0.35,
    "quantitative": 0.22, "temporal": 0.24,
}
_view_stats = {v: {"ok": 0, "tot": 0} for v in _tau_by_view.keys()}


class AdaptiveSelectorController:
    """
    Controller for adapting:
      - per-view pick quota (pick_per_view)
      - selector size M (submodular M)
    """

    def __init__(
        self,
        *,
        view_alpha: float = 0.25,
        selector_alpha: float = 0.20,
        view_high_thr: float = 0.60,
        view_low_thr: float = 0.25,
        selector_high_thr: float = 0.65,
        selector_low_thr: float = 0.35,
        min_pick: int = 1,
        max_pick: int = 4,
        min_selector: int = 1,
        max_selector: int = 5,
        warmup_rounds: int = 3,
    ):
        self.view_alpha = float(view_alpha)
        self.selector_alpha = float(selector_alpha)
        self.view_high_thr = float(view_high_thr)
        self.view_low_thr = float(view_low_thr)
        self.selector_high_thr = float(selector_high_thr)
        self.selector_low_thr = float(selector_low_thr)
        self.min_pick = int(min_pick)
        self.max_pick = int(max_pick)
        self.min_selector = int(min_selector)
        self.max_selector = int(max_selector)
        self.warmup_rounds = int(max(1, warmup_rounds))

        self._view_stats = defaultdict(lambda: {"ema": 0.0, "cnt": 0.0})
        self._selector_success_ema = 0.0
        self._selector_cnt = 0

    def suggest_view_quota(self, views: List[str], base_pick: int) -> Dict[str, int]:
        quotas: Dict[str, int] = {}
        for v in views:
            stat = self._view_stats[v]
            q = int(base_pick)
            if stat["cnt"] >= self.warmup_rounds:
                ema = stat["ema"]
                if ema >= self.view_high_thr:
                    q = min(self.max_pick, q + 1)
                elif ema <= self.view_low_thr:
                    q = max(self.min_pick, q - 1)
            quotas[v] = max(self.min_pick, q)
        return quotas

    def update_view_outcome(self, chosen_views: List[str], accepted_views: List[str]) -> None:
        accepted = set(accepted_views)
        for v in chosen_views:
            val = 1.0 if v in accepted else 0.0
            stat = self._view_stats[v]
            stat["ema"] = val if stat["cnt"] <= 0 else (1 - self.view_alpha) * stat["ema"] + self.view_alpha * val
            stat["cnt"] += 1.0

    def suggest_selector_m(self, score_norms: List[float], base_m: int, pool_size: int) -> int:
        if pool_size <= 0:
            return max(self.min_selector, min(base_m, self.max_selector))

        m = int(base_m)
        if score_norms:
            high_ratio = sum(1 for s in score_norms if s >= self.selector_high_thr) / len(score_norms)
            mid_ratio = sum(1 for s in score_norms if s >= self.selector_low_thr) / len(score_norms)
        else:
            high_ratio, mid_ratio = 0.0, 0.0

        success_ema = self._selector_success_ema if self._selector_cnt else 0.0

        if high_ratio > 0.6 or success_ema > 0.55:
            m = min(self.max_selector, m + 1)
        elif mid_ratio < 0.3 and success_ema < 0.35:
            m = max(self.min_selector, m - 1)

        return max(self.min_selector, min(m, pool_size))

    def update_selector_success(self, success_rate: float) -> None:
        success_rate = max(0.0, min(1.0, float(success_rate)))
        self._selector_success_ema = (
            success_rate if self._selector_cnt <= 0
            else (1 - self.selector_alpha) * self._selector_success_ema + self.selector_alpha * success_rate
        )
        self._selector_cnt += 1


class SubmodularSelector:
    """
    Greedy selection on:
      J = λ_anch*Anchor + λ_ans*Answerability + λ_div*Diversity + λ_cov*Coverage
    """

    def __init__(
        self,
        *,
        mmr_mode: bool = True,
        mmr_lambda: float = MMR_LAMBDA_DEFAULT,
        lambda_cov: float = DEFAULT_LAMBDA_COV,
        lambda_ans: float = DEFAULT_LAMBDA_ANS,
        lambda_anch: float = DEFAULT_LAMBDA_ANCH,
        lambda_div: float = DEFAULT_LAMBDA_DIV,
    ):
        self.mmr_mode = bool(mmr_mode)
        self.mmr_lambda = float(mmr_lambda)

        self.l_cov = float(lambda_cov)
        self.l_ans = float(lambda_ans)
        self.l_anch = float(lambda_anch)
        self.l_div = float(lambda_div)

    @staticmethod
    def _clip01(x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.0
        return 0.0 if x < 0 else 1.0 if x > 1 else x

    def select(self, items: List[dict], M: int = 3) -> List[dict]:
        if not items:
            return []
        M = max(1, min(int(M), len(items)))

        Q = [str(it.get("question", "")).strip() for it in items]
        V = [(it.get("meta") or {}).get("view", "unknown") for it in items]

        # ✅ no pool min-max, no 10-A: use ans_score as passed in (建议你统一传 total_score/10 到 ans_score)
        A = [float(it.get("ans_score", 0.0)) for it in items]

        raw_align = [float(((it.get("anchor") or {}).get("align")) or 0.0) for it in items]
        raw_bridge = [float(((it.get("anchor") or {}).get("bridge")) or 0.0) for it in items]
        raw_conf = [float(((it.get("anchor") or {}).get("conflict")) or 0.0) for it in items]

        AnchorScore = []
        for a, b, c in zip(raw_align, raw_bridge, raw_conf):
            base = (max(0.0, a) * max(0.0, b)) ** 0.5 - 0.3 * max(0.0, c)
            AnchorScore.append(self._clip01(base))

        selected_idx: List[int] = []
        covered_views = set()
        sims_cache: Dict[tuple, float] = {}

        def sim(i: int, j: int) -> float:
            if i == j:
                return 1.0
            if i > j:
                i, j = j, i
            key = (i, j)
            if key not in sims_cache:
                try:
                    sims_cache[key] = float(compute_similarity(Q[i], Q[j]))
                except Exception:
                    sims_cache[key] = 0.0
            return sims_cache[key]

        def score_of_set(idxs: List[int]) -> float:
            if not idxs:
                return 0.0
            cov_raw = float(len(set(V[i] for i in idxs)))
            pool_views = len(set(V))
            cov_norm = cov_raw / max(1.0, float(min(M, pool_views)))

            ans_mean = sum(A[i] for i in idxs) / len(idxs)
            anch_mean = sum(AnchorScore[i] for i in idxs) / len(idxs)

            pair_sims = []
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pair_sims.append(sim(idxs[a], idxs[b]))
            diversity = (sum((1.0 - s) for s in pair_sims) / len(pair_sims)) if pair_sims else 0.0

            return (self.l_anch * anch_mean + self.l_ans * ans_mean + self.l_div * diversity + self.l_cov * cov_norm)

        def marginal_gain(i: int, current: List[int]) -> float:
            if i in current:
                return -inf
            base = score_of_set(current)
            new = score_of_set(current + [i])
            gain = new - base
            if self.mmr_mode and current:
                max_sim_to_S = max(sim(i, j) for j in current)
                gain -= self.mmr_lambda * max_sim_to_S
            return gain

        for _ in range(M):
            best_i, best_g = None, -inf
            for i in range(len(items)):
                g = marginal_gain(i, selected_idx)
                # encourage coverage early
                if not selected_idx:
                    g += (0.15 if V[i] not in covered_views else 0.0)
                if g > best_g:
                    best_g, best_i = g, i
            if best_i is None or best_g <= -1e6:
                break
            selected_idx.append(best_i)
            covered_views.add(V[best_i])

        return [items[i] for i in selected_idx]


def get_tau_by_view() -> Dict[str, float]:
    return dict(_tau_by_view)


def get_target_pass_rate() -> Dict[str, float]:
    return dict(_target_pass_rate)


def get_view_stats() -> Dict[str, dict]:
    return {k: dict(v) for k, v in _view_stats.items()}
