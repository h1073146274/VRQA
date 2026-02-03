from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore


class FiLMArmEncoder:
    """
    FiLM-based arm-aware encoder: h_v = gamma_v * h + beta_v
    gamma_v = 1 + gamma_scale * tanh(...)
    beta_v  = beta_scale * (...)
    """

    def __init__(
        self,
        arms: List[str],
        h_dim: int,
        *,
        e_dim: int = 16,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma_scale: float = 0.3,
        beta_scale: float = 0.3,
        e_init: Optional[Dict[str, Any]] = None,
        freeze_e: bool = False,
        seed: int = 42,
    ):
        if np is None:
            raise RuntimeError("FiLMArmEncoder requires numpy")

        self.arms = list(arms)
        self.h_dim = int(h_dim)
        self.e_dim = int(e_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.freeze_e = bool(freeze_e)

        rng = np.random.RandomState(seed)

        self.e: Dict[str, "np.ndarray"] = {}
        for v in self.arms:
            if e_init is not None and v in e_init:
                try:
                    ev = np.asarray(e_init[v], dtype=float).reshape(-1)
                    if ev.size != self.e_dim:
                        ev = rng.normal(0.0, 0.02, size=(self.e_dim,))
                    self.e[v] = ev
                except Exception:
                    self.e[v] = rng.normal(0.0, 0.02, size=(self.e_dim,))
            else:
                self.e[v] = rng.normal(0.0, 0.02, size=(self.e_dim,))

        self.W1 = rng.normal(0.0, 0.02, size=(self.e_dim, self.hidden_dim))
        self.b1 = np.zeros((self.hidden_dim,))
        self.W2 = rng.normal(0.0, 0.02, size=(self.hidden_dim, 2 * self.h_dim))
        self.b2 = np.zeros((2 * self.h_dim,))

    def forward(self, h: "np.ndarray", arm: str) -> "np.ndarray":
        e_v = self.e.get(arm)
        if e_v is None:
            return h

        z1 = e_v @ self.W1 + self.b1
        z1 = np.maximum(z1, 0.0)
        z2 = z1 @ self.W2 + self.b2

        gamma_raw = z2[: self.h_dim]
        beta_raw = z2[self.h_dim :]

        gamma = 1.0 + self.gamma_scale * np.tanh(gamma_raw)
        beta = self.beta_scale * beta_raw
        return gamma * h + beta

    def get_params(self, h: "np.ndarray", arm: str) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        e_v = self.e.get(arm)
        h_arr = np.asarray(h, dtype=float).reshape(-1)
        if e_v is None:
            gamma = np.ones_like(h_arr)
            beta = np.zeros_like(h_arr)
            return h_arr, gamma, beta

        z1 = e_v @ self.W1 + self.b1
        z1 = np.maximum(z1, 0.0)
        z2 = z1 @ self.W2 + self.b2

        gamma_raw = z2[: self.h_dim]
        beta_raw = z2[self.h_dim :]

        gamma = 1.0 + self.gamma_scale * np.tanh(gamma_raw)
        beta = self.beta_scale * beta_raw
        h_v = gamma * h_arr + beta
        return h_v, gamma, beta

    def sgd_step(
        self,
        base_stats: "np.ndarray",
        h: "np.ndarray",
        arm: str,
        theta_v: "np.ndarray",
        reward: float,
    ) -> None:
        """
        One-step SGD on FiLM params, using bandit regression error.
        """
        e_v = self.e.get(arm)
        if e_v is None:
            return

        z1 = e_v @ self.W1 + self.b1
        z1_act = np.maximum(z1, 0.0)
        z2 = z1_act @ self.W2 + self.b2

        gamma_raw = z2[: self.h_dim]
        beta_raw = z2[self.h_dim :]

        gamma = 1.0 + self.gamma_scale * np.tanh(gamma_raw)
        beta = self.beta_scale * beta_raw
        h_v = gamma * h + beta

        x_full = np.concatenate([base_stats, h_v])
        r_hat = float(theta_v @ x_full)
        diff = (r_hat - float(reward))

        dL_dx_full = 2.0 * diff * theta_v
        tail = dL_dx_full[len(base_stats) :]

        dL_dgamma = tail * h
        dL_dbeta = tail

        # backprop through tanh / linear
        dgamma_raw = dL_dgamma * self.gamma_scale * (1.0 - np.tanh(gamma_raw) ** 2)
        dbeta_raw = dL_dbeta * self.beta_scale

        dz2 = np.concatenate([dgamma_raw, dbeta_raw], axis=0)

        dW2 = np.outer(z1_act, dz2)
        db2 = dz2

        dz1_act = dz2 @ self.W2.T
        dz1 = dz1_act.copy()
        dz1[z1 <= 0.0] = 0.0

        dW1 = np.outer(e_v, dz1)
        db1 = dz1
        de_v = dz1 @ self.W1.T

        lr = self.lr
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        if not self.freeze_e:
            self.e[arm] = e_v - lr * de_v

    def get_state(self) -> Dict[str, Any]:
        if np is None:
            return {}
        return {
            "arms": list(self.arms),
            "h_dim": int(self.h_dim),
            "e_dim": int(self.e_dim),
            "hidden_dim": int(self.hidden_dim),
            "lr": float(self.lr),
            "gamma_scale": float(self.gamma_scale),
            "beta_scale": float(self.beta_scale),
            "freeze_e": bool(self.freeze_e),
            "e": {k: v.tolist() for k, v in self.e.items()},
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }

    def load_state(self, state: Dict[str, Any]) -> bool:
        if np is None or not isinstance(state, dict):
            return False
        try:
            if "lr" in state:
                self.lr = float(state["lr"])
            if "gamma_scale" in state:
                self.gamma_scale = float(state["gamma_scale"])
            if "beta_scale" in state:
                self.beta_scale = float(state["beta_scale"])
            if "freeze_e" in state:
                self.freeze_e = bool(state["freeze_e"])

            e_state = state.get("e")
            if isinstance(e_state, dict):
                for k in self.arms:
                    if k in e_state:
                        self.e[k] = np.asarray(e_state[k], dtype=float).reshape(-1)

            for name in ["W1", "b1", "W2", "b2"]:
                if name in state:
                    setattr(self, name, np.asarray(state[name], dtype=float))
        except Exception:
            return False
        return True
