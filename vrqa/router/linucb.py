# vrqa/router/linucb.py
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

from .film import FiLMArmEncoder
from .features import build_view_semantic_embeddings
from vrqa.prompts.qa_prompts import get_view_descriptions


def _ensure_numpy() -> bool:
    global np
    if np is not None:
        return True
    try:
        import numpy as _np  # type: ignore
        np = _np  # type: ignore
        return True
    except Exception:
        return False


class LinUCBRouter:
    """
    LinUCB router:
      ucb(v) = mu(v) + alpha * sigma(v)

    ✅ 删除：任何基于 N_v 的 bonus（你原来 usage_bonus*sqrt(log t / N_v) 那套）。
    """

    def __init__(
        self,
        arms: List[str],
        d: int,
        *,
        alpha: float = 1.4,
        discount: float = 0.90,
        enable_film: bool = True,
        film_lr: float = 1e-3,
        train_linucb: bool = True,
        train_film: bool = True,
        film_state_init: Optional[Dict[str, Any]] = None,
    ):
        self.arms = list(arms)
        self.K = len(self.arms)

        self.base_d = int(d)
        self.base_stats_dim = 0
        self.h_dim = self.base_d
        self.d = self.h_dim  # x already equals h by default

        self.alpha = float(alpha)
        self.discount = float(discount)

        self.enable_film = bool(enable_film)
        self.train_linucb = bool(train_linucb)
        self.train_film = bool(train_film)

        self._use_np = _ensure_numpy()
        self.step_t = 0
        self.arm_counts = defaultdict(int)  # only for logging/analysis, NOT in UCB

        self.A: List[Any] = []
        self.b: List[Any] = []
        for _ in range(self.K):
            if self._use_np and np is not None:
                self.A.append(np.eye(self.d))
                self.b.append(np.zeros((self.d,)))
            else:
                self.A.append([[1.0 if i == j else 0.0 for j in range(self.d)] for i in range(self.d)])
                self.b.append([0.0 for _ in range(self.d)])

        self.film: Optional[FiLMArmEncoder]
        if self._use_np and np is not None and self.enable_film and self.h_dim > 0:
            e_init = None
            try:
                view_desc = get_view_descriptions() or {}
                if view_desc:
                    ev_map = build_view_semantic_embeddings(view_desc, out_dim=16)
                    if ev_map:
                        e_init = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in ev_map.items()}
            except Exception:
                e_init = None

            self.film = FiLMArmEncoder(
                arms=self.arms,
                h_dim=self.h_dim,
                e_dim=16,
                hidden_dim=64,
                lr=film_lr,
                gamma_scale=0.3,
                beta_scale=0.3,
                e_init=e_init,
                freeze_e=False,
            )
            if film_state_init:
                self.film.load_state(film_state_init)
        else:
            self.film = None

    def _split_base_and_h(self, x: List[float]):
        if self._use_np and np is not None:
            arr = np.asarray(x, dtype=float).reshape(-1)
            if arr.size < self.base_d:
                arr = np.pad(arr, (0, self.base_d - arr.size))
            else:
                arr = arr[: self.base_d]
            base_stats = arr[: self.base_stats_dim]
            h = arr[self.base_stats_dim : self.base_stats_dim + self.h_dim]
            return base_stats, h

        vec = list(x[: self.base_d])
        if len(vec) < self.base_d:
            vec += [0.0] * (self.base_d - len(vec))
        base_stats = vec[: self.base_stats_dim]
        h = vec[self.base_stats_dim : self.base_stats_dim + self.h_dim]
        return base_stats, h

    def _compose_xvec_for_arm(self, x: List[float], arm: str):
        base_stats, h = self._split_base_and_h(x)

        if self._use_np and np is not None:
            if self.film is not None and self.enable_film:
                h_v = self.film.forward(h, arm)
            else:
                h_v = h
            xvec = np.concatenate([base_stats, h_v])
            xvec = xvec[: self.d] if xvec.size >= self.d else np.pad(xvec, (0, self.d - xvec.size))
            return xvec, base_stats, h

        # non-numpy fallback
        h_v = h
        full = (list(base_stats) + list(h_v))[: self.d]
        if len(full) < self.d:
            full += [0.0] * (self.d - len(full))
        return full, base_stats, h

    def _solve_theta(self, A, b):
        if self._use_np and np is not None:
            try:
                return np.linalg.solve(A, b)
            except Exception:
                return np.linalg.pinv(A) @ b
        return b

    def select(self, x: List[float], *, k: int = 3, return_debug: bool = False):
        """
        Return top-k views by UCB.

        debug fields:
          mu_list, sigma_list, ucb_list, alpha_sigma_list, count_list
        """
        self.step_t += 1

        mu_list: List[float] = []
        sigma_list: List[float] = []
        ucb_list: List[float] = []
        alpha_sigma_list: List[float] = []
        count_list: List[int] = []

        scores: List[Tuple[float, str]] = []
        for i, arm in enumerate(self.arms):
            theta = self._solve_theta(self.A[i], self.b[i])

            if self._use_np and np is not None:
                xvec, _, _ = self._compose_xvec_for_arm(x, arm)
                mu = float(xvec @ theta)
                invA = np.linalg.pinv(self.A[i])
                var = float(xvec @ invA @ xvec)
            else:
                xvec, _, _ = self._compose_xvec_for_arm(x, arm)
                mu = sum(float(xvec[j]) * float(theta[j]) for j in range(self.d))
                var = sum(float(xx) * float(xx) for xx in xvec)

            sigma = math.sqrt(max(var, 1e-9))

            ucb = mu + self.alpha * sigma  # ✅ ONLY alpha exploration
            scores.append((ucb, arm))

            mu_list.append(mu)
            sigma_list.append(sigma)
            ucb_list.append(ucb)
            alpha_sigma_list.append(self.alpha * sigma)
            count_list.append(int(self.arm_counts.get(arm, 0)))

        scores.sort(key=lambda t: t[0], reverse=True)
        kk = max(1, min(int(k), len(scores)))
        chosen = [arm for _, arm in scores[:kk]]

        if return_debug:
            return chosen, mu_list, sigma_list, ucb_list, alpha_sigma_list, count_list
        return chosen

    def update(self, arm: str, x: List[float], reward: float) -> None:
        if arm not in self.arms:
            return
        idx = self.arms.index(arm)
        r = float(reward)

        xvec_full, base_stats, h = self._compose_xvec_for_arm(x, arm)

        if self.train_linucb:
            if self._use_np and np is not None:
                xcol = np.asarray(xvec_full, dtype=float).reshape(self.d, 1)
                self.A[idx] = self.discount * self.A[idx] + (xcol @ xcol.T)
                self.b[idx] = self.discount * self.b[idx] + (r * xcol.reshape(self.d))
            else:
                # very slow fallback
                A_i = self.A[idx]
                b_i = self.b[idx]
                for p in range(self.d):
                    for q in range(self.d):
                        A_i[p][q] = self.discount * A_i[p][q] + float(xvec_full[p]) * float(xvec_full[q])
                for p in range(self.d):
                    b_i[p] = self.discount * b_i[p] + r * float(xvec_full[p])
                self.A[idx], self.b[idx] = A_i, b_i

        self.arm_counts[arm] += 1

        if self._use_np and np is not None and self.film is not None and self.enable_film and self.train_film:
            theta = self._solve_theta(self.A[idx], self.b[idx])
            self.film.sgd_step(base_stats, h, arm, theta, r)

    def get_state(self) -> Dict[str, Any]:
        if self._use_np and np is not None:
            A_state = [Ai.tolist() for Ai in self.A]
            b_state = [bi.tolist() for bi in self.b]
        else:
            A_state = self.A
            b_state = self.b

        state: Dict[str, Any] = {
            "arms": list(self.arms),
            "d": int(self.d),
            "base_d": int(self.base_d),
            "alpha": float(self.alpha),
            "discount": float(self.discount),
            "enable_film": bool(self.enable_film),
            "train_linucb": bool(self.train_linucb),
            "train_film": bool(self.train_film),
            "step_t": int(self.step_t),
            "arm_counts": dict(self.arm_counts),
            "A": A_state,
            "b": b_state,
        }
        if self.film is not None:
            state["film"] = self.film.get_state()
        return state

    def load_state(self, state: Dict[str, Any]) -> bool:
        if not isinstance(state, dict):
            return False
        if "arms" in state and list(state["arms"]) != self.arms:
            return False
        if "d" in state and int(state["d"]) != self.d:
            return False

        try:
            A_state = state["A"]
            b_state = state["b"]
            if not (isinstance(A_state, list) and isinstance(b_state, list)):
                return False

            if self._use_np and np is not None:
                self.A = [np.asarray(mat, dtype=float) for mat in A_state]
                self.b = [np.asarray(vec, dtype=float) for vec in b_state]
            else:
                self.A = A_state
                self.b = b_state

            if "alpha" in state:
                self.alpha = float(state["alpha"])
            if "discount" in state:
                self.discount = float(state["discount"])
            if "enable_film" in state:
                self.enable_film = bool(state["enable_film"])
            if "train_linucb" in state:
                self.train_linucb = bool(state["train_linucb"])
            if "train_film" in state:
                self.train_film = bool(state["train_film"])
            if "step_t" in state:
                self.step_t = int(state["step_t"])

            if "arm_counts" in state and isinstance(state["arm_counts"], dict):
                self.arm_counts = defaultdict(int, {k: int(v) for k, v in state["arm_counts"].items()})

            film_state = state.get("film")
            if self.film is not None and isinstance(film_state, dict):
                self.film.load_state(film_state)
        except Exception:
            return False
        return True


class CombinatorialUCBRouter:
    """
    A light-weight router used when you want "value-only" routing
    (no context features). This is *not* using N_v exploration either.
    """

    def __init__(self, views: List[str], *, K: int = 3, eps: float = 0.05, T: float = -1.0):
        self.views = list(views)
        self.K = int(K)
        self.eps = float(eps)
        self.T = float(T)  # if >0 use softmax sampling
        self.t = 0
        self.avgR = defaultdict(float)

    def select(self, banned_views: Optional[List[str]] = None) -> List[str]:
        banned = set(banned_views or [])
        cand = [v for v in self.views if v not in banned]
        if not cand:
            return []

        self.t += 1
        picks: List[str] = []

        # epsilon random picks
        while len(picks) < self.K and random.random() < self.eps:
            remaining = [x for x in cand if x not in picks]
            if not remaining:
                break
            picks.append(random.choice(remaining))

        if len(picks) >= self.K:
            return picks[: self.K]

        remaining = [v for v in cand if v not in picks]
        scored = [(v, float(self.avgR[v])) for v in remaining]

        if self.T is not None and self.T > 0:
            # softmax sampling
            vs = [s for _, s in scored]
            m = max(vs) if vs else 0.0
            exps = [math.exp((s - m) / self.T) for _, s in scored]
            Z = sum(exps) or 1.0
            probs = [e / Z for e in exps]
            while len(picks) < self.K and scored:
                idx = random.choices(range(len(scored)), weights=probs, k=1)[0]
                picks.append(scored[idx][0])
                scored.pop(idx)
                probs.pop(idx)
                Z = sum(probs) or 1.0
        else:
            scored.sort(key=lambda x: x[1], reverse=True)
            picks.extend([v for v, _ in scored[: (self.K - len(picks))]])

        return picks[: self.K]

    def update(self, per_view_gain: Dict[str, float], *, decay: float = 0.0) -> None:
        for v, g in (per_view_gain or {}).items():
            g = float(g)
            if decay <= 0:
                # running mean (simple)
                self.avgR[v] = 0.9 * self.avgR[v] + 0.1 * g
            else:
                self.avgR[v] = (1 - decay) * self.avgR[v] + decay * g

    def get_state(self) -> Dict[str, Any]:
        return {"views": list(self.views), "K": self.K, "eps": self.eps, "T": self.T, "t": self.t, "avgR": dict(self.avgR)}

    def load_state(self, state: Dict[str, Any]) -> bool:
        if not isinstance(state, dict):
            return False
        if "views" in state and list(state["views"]) != self.views:
            return False
        try:
            if "K" in state:
                self.K = int(state["K"])
            if "eps" in state:
                self.eps = float(state["eps"])
            if "T" in state:
                self.T = float(state["T"])
            if "t" in state:
                self.t = int(state["t"])
            avg = state.get("avgR") or {}
            self.avgR = defaultdict(float, {str(k): float(v) for k, v in avg.items()})
        except Exception:
            return False
        return True


def ucb_gains_from_pairs(history_selected_pairs: List[Dict[str, Any]], *, gamma: float = 0.6) -> Dict[str, float]:
    """
    Per-view gains for router_ucb.update(...) or router_linucb.update(...).

    ✅ A only comes from scorer outputs:
      - total_score in [0,10]
      - we use A_norm = clip(total_score/10, 0, 1) for stability (NO 10-A, NO min-max).

    H comes from gate_G if present; otherwise uses a soft heuristic:
      H_raw = 0.6 * score_norm + 0.4 * max(sim_to_candidate, sim_to_sentence)
    """
    def clip01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    gains: Dict[str, float] = {}
    for p in history_selected_pairs or []:
        meta = p.get("meta") or {}
        v = meta.get("view", "unknown")

        total_score = float(p.get("total_score", 0.0))
        A_raw = total_score
        A_norm = clip01(A_raw / 10.0)

        if p.get("gate_G") is not None:
            H_raw = float(p.get("gate_G", 0.0))
        else:
            score_norm = clip01(total_score / 10.0)
            sim_c = clip01(float(p.get("similarity_with_candidate", 0.0)))
            sd = p.get("similarity_detail") or {}
            sim_s = clip01(float(sd.get("sim_to_sentence", 0.0))) if isinstance(sd, dict) else clip01(float(p.get("similarity_to_sentence", 0.0)))
            H_raw = 0.6 * score_norm + 0.4 * max(sim_c, sim_s)

        H_norm = clip01(H_raw)
        r_q = (H_norm ** gamma) * (A_norm ** (1.0 - gamma))

        # optional diagnostics
        p["A_raw"] = A_raw
        p["A_norm"] = A_norm
        p["H_raw"] = H_raw
        p["r_q"] = r_q

        gains[v] = gains.get(v, 0.0) + float(r_q)

    return gains
