"""VS-ePL-KRLS: Variable Step-Size evolving Participatory Learning with KRLS.

Implementation follows Queiroz et al. (SBA / Evolving Systems 2021), with
ambiguities of the PDF recorded as explicit config flags.

Default threshold convention is ``tabela``:
    tau^k = 1 - beta^{k-1}   (tau0 = 0.82 with beta0 = 0.18)
    gamma^k = beta^{k-1}     (gamma0 = 0.18)

The paper body writes the opposite (tau = beta, gamma = 1 - beta). Use
``threshold_convention="texto"`` to test that variant.

Center update uses the standard ePL form
    v <- v + alpha * rho^{1-a} * (x - v)
because the PDF/OCR of Eq. 5 is unusable (element-wise power of the center).
This matches Lima et al. (2010) and the co-author reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


@dataclass
class VSePLKRLSConfig:
    alpha: float = 0.01
    beta0: float = 0.18
    gamma0: float = 0.18
    tau0: float = 0.82
    sigma: float = 0.05
    lam: float = 1e-4
    nu0: float = 0.50
    gamma_bar: float = 0.006
    alpha_vs1: float = 0.88
    alpha_vs2: float = 0.74
    threshold_convention: str = "tabela"
    use_variable_step: bool = True
    novelty_coef: float = 0.1
    d_max: int = 40
    max_rules: int = 12
    beta_min: float = 1e-4
    beta_max: float = 0.999
    nu_min: float = 1e-3
    nu_max: float = 10.0
    rho_floor: float = 1e-12


@dataclass
class Rule:
    center: np.ndarray
    arousal: float = 0.0
    dictionary: np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    nu: np.ndarray = field(default_factory=lambda: np.ones(1) * 0.5)
    theta: np.ndarray = field(default_factory=lambda: np.zeros(1))
    Q: np.ndarray = field(default_factory=lambda: np.ones((1, 1)))
    P_krls: np.ndarray = field(default_factory=lambda: np.ones((1, 1)))
    P_lm: np.ndarray = field(default_factory=lambda: np.ones((1, 1)))

    @property
    def n_dict(self) -> int:
        return int(self.dictionary.shape[0])


class VSePLKRLS:
    def __init__(self, config: Optional[VSePLKRLSConfig] = None):
        self.cfg = config or VSePLKRLSConfig()
        self.rules: List[Rule] = []
        self.beta = float(self.cfg.beta0)
        self.n_seen = 0
        self.last_error = 0.0
        self.history_n_rules: List[int] = []
        self.history_beta: List[float] = []
        self.history_error: List[float] = []

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    def _tau(self) -> float:
        if self.cfg.threshold_convention == "texto":
            return float(np.clip(self.beta, 0.0, 1.0))
        return float(np.clip(1.0 - self.beta, 0.0, 1.0))

    def _gamma(self) -> float:
        if self.cfg.threshold_convention == "texto":
            return float(np.clip(1.0 - self.beta, 0.0, 1.0))
        return float(np.clip(self.beta, 0.0, 1.0))

    def _kernel_vec(self, rule: Rule, x: np.ndarray) -> np.ndarray:
        diff = rule.dictionary - x[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        nu = np.maximum(rule.nu, self.cfg.nu_min)
        return np.exp(-dist2 / (2.0 * nu))

    def _local_output(self, rule: Rule, x: np.ndarray) -> float:
        if rule.n_dict == 0:
            return 0.0
        kvec = self._kernel_vec(rule, x)
        return float(np.dot(rule.theta, kvec))

    def _compatibilities(self, x: np.ndarray) -> np.ndarray:
        m = max(float(x.shape[0]), 1.0)
        rhos = []
        for rule in self.rules:
            dist = float(np.linalg.norm(x - rule.center))
            rhos.append(1.0 - dist / m)
        return np.clip(np.asarray(rhos, dtype=float), 0.0, 1.0)

    def _normalized_activation(self, rhos: np.ndarray) -> np.ndarray:
        mu = np.maximum(rhos, self.cfg.rho_floor)
        s = float(mu.sum())
        if s <= 0:
            return np.ones_like(mu) / max(len(mu), 1)
        return mu / s

    def predict_one(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if not self.rules:
            return 0.0
        rhos = self._compatibilities(x)
        lam = self._normalized_activation(rhos)
        yhat = 0.0
        for i, rule in enumerate(self.rules):
            yhat += lam[i] * self._local_output(rule, x)
        return float(yhat)

    def _init_rule(self, x: np.ndarray, y: float) -> Rule:
        kappa = 1.0
        q = 1.0 / (self.cfg.lam + kappa)
        p0 = 1.0 / max(self.cfg.sigma, 1e-8)
        return Rule(
            center=x.copy(),
            arousal=0.0,
            dictionary=x.copy().reshape(1, -1),
            nu=np.array([self.cfg.nu0], dtype=float),
            theta=np.array([q * y], dtype=float),
            Q=np.array([[q]], dtype=float),
            P_krls=np.array([[1.0]], dtype=float),
            P_lm=np.array([[p0]], dtype=float),
        )

    def _update_center(self, rule: Rule, x: np.ndarray, rho: float) -> None:
        gain = self.cfg.alpha * (max(rho, self.cfg.rho_floor) ** (1.0 - rule.arousal))
        rule.center = rule.center + gain * (x - rule.center)

    def _pairwise_compat(self, i: int, j: int) -> float:
        return float(1.0 - np.mean(np.abs(self.rules[i].center - self.rules[j].center)))

    def _merge_redundant(self) -> None:
        if len(self.rules) < 2:
            return
        gamma = self._gamma()
        merged = True
        steps = 0
        while merged and len(self.rules) > 1 and steps < 3:
            steps += 1
            merged = False
            n = len(self.rules)
            best = None
            best_rho = -1.0
            for i in range(n):
                for j in range(i + 1, n):
                    rho_ij = self._pairwise_compat(i, j)
                    if rho_ij > gamma and rho_ij > best_rho:
                        best_rho = rho_ij
                        best = (i, j)
            if best is None:
                break
            i, j = best
            keep, drop = (i, j) if i < j else (j, i)
            self.rules[keep].center = 0.5 * (self.rules[keep].center + self.rules[drop].center)
            del self.rules[drop]
            merged = True

    def _novelty_add(self, rule: Rule, x: np.ndarray, y: float, local_yhat: float) -> bool:
        if rule.n_dict >= self.cfg.d_max:
            return False
        diffs = rule.dictionary - x[None, :]
        dist = np.sqrt(np.sum(diffs * diffs, axis=1))
        psi = float(dist.min())
        nu_ref = float(np.mean(rule.nu))
        delta = self.cfg.novelty_coef * max(nu_ref, self.cfg.nu_min)
        if psi < delta:
            return False
        g = self._kernel_vec(rule, x)
        z = rule.Q @ g
        r = float(self.cfg.lam + 1.0 - z @ g)
        if (not np.isfinite(r)) or r < 1e-3:
            return False
        err_without = abs(y - local_yhat)
        return True if err_without > 1e-6 else psi >= 2.0 * delta

    def _krls_update_existing(self, rule: Rule, x: np.ndarray, y: float, local_yhat: float) -> None:
        g = self._kernel_vec(rule, x)
        z = rule.Q @ g
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(rule.P_krls)):
            rule.P_krls = np.eye(rule.n_dict)
            return
        pmax = float(np.max(np.abs(rule.P_krls)))
        if pmax > 1e6:
            rule.P_krls = np.eye(rule.n_dict)
        denom = 1.0 + float(z @ (rule.P_krls @ z))
        if (not np.isfinite(denom)) or denom < 1e-8:
            return
        q = (rule.P_krls @ z) / denom
        Pz = rule.P_krls @ z
        P_new = rule.P_krls - np.outer(Pz, Pz) / denom
        e = y - local_yhat
        if abs(e) > 50:
            e = np.sign(e) * 50.0
        delta = rule.Q @ q * e
        if (not np.all(np.isfinite(P_new))) or (not np.all(np.isfinite(delta))):
            return
        rule.P_krls = P_new
        rule.theta = np.clip(rule.theta + delta, -1e3, 1e3)

    def _krls_grow(self, rule: Rule, x: np.ndarray, y: float, local_yhat: float) -> None:
        g = self._kernel_vec(rule, x)
        z = rule.Q @ g
        r = float(self.cfg.lam + 1.0 - z @ g)
        if (not np.isfinite(r)) or r < 1e-3:
            return
        e = y - local_yhat
        n = rule.n_dict
        Q_new = np.zeros((n + 1, n + 1), dtype=float)
        Q_new[:n, :n] = rule.Q + np.outer(z, z) / r
        Q_new[:n, n] = -z / r
        Q_new[n, :n] = -z / r
        Q_new[n, n] = 1.0 / r
        theta_new = np.zeros(n + 1, dtype=float)
        theta_new[:n] = rule.theta - z * (e / r)
        theta_new[n] = e / r
        if not np.all(np.isfinite(Q_new)) or not np.all(np.isfinite(theta_new)):
            return
        P_new = np.zeros((n + 1, n + 1), dtype=float)
        P_new[:n, :n] = rule.P_krls
        P_new[n, n] = 1.0
        P_lm_new = np.zeros((n + 1, n + 1), dtype=float)
        P_lm_new[:n, :n] = rule.P_lm
        P_lm_new[n, n] = float(self.cfg.sigma)
        nu_new = np.append(rule.nu, self.cfg.nu0)
        rule.dictionary = np.vstack([rule.dictionary, x[None, :]])
        rule.Q = Q_new
        rule.theta = np.clip(theta_new, -1e3, 1e3)
        rule.P_krls = P_new
        rule.P_lm = P_lm_new
        rule.nu = nu_new

    def _update_kernel_size(self, rule: Rule, x: np.ndarray, lam_i: float, e_local: float) -> None:
        if rule.n_dict == 0 or not np.isfinite(e_local):
            return
        kvec = self._kernel_vec(rule, x)
        diff = rule.dictionary - x[None, :]
        dist2 = np.sum(diff * diff, axis=1)
        nu = np.maximum(rule.nu, self.cfg.nu_min)
        grad = rule.theta * kvec * dist2 / (2.0 * nu * nu)
        grad = lam_i * np.clip(grad, -50.0, 50.0)
        if not np.all(np.isfinite(grad)):
            return
        if not np.all(np.isfinite(rule.P_lm)):
            rule.P_lm = np.eye(rule.n_dict) * float(self.cfg.sigma)
        Pg = rule.P_lm @ grad
        denom = 1.0 + float(grad @ Pg)
        if not np.isfinite(denom) or denom <= 1e-12:
            return
        P_new = rule.P_lm - np.outer(Pg, Pg) / denom
        step = P_new @ grad
        if not np.all(np.isfinite(P_new)) or not np.all(np.isfinite(step)):
            return
        rule.P_lm = P_new
        rule.nu = np.clip(rule.nu + 0.1 * step * e_local, self.cfg.nu_min, self.cfg.nu_max)

    def _update_beta(self, error: float) -> None:
        if not self.cfg.use_variable_step:
            return
        if abs(error) > self.cfg.gamma_bar:
            self.beta = self.beta / max(self.cfg.alpha_vs1, 1e-8)
        else:
            self.beta = self.beta * self.cfg.alpha_vs2
        self.beta = float(np.clip(self.beta, self.cfg.beta_min, self.cfg.beta_max))

    def update(self, x: np.ndarray, y: float) -> float:
        x = np.asarray(x, dtype=float).ravel()
        y = float(y)
        if not self.rules:
            self.rules.append(self._init_rule(x, y))
            self.n_seen += 1
            self.history_n_rules.append(1)
            self.history_beta.append(self.beta)
            self.history_error.append(0.0)
            return y

        yhat = self.predict_one(x)
        error = y - yhat
        rhos = self._compatibilities(x)
        for i, rule in enumerate(self.rules):
            rule.arousal = float(
                np.clip(rule.arousal + self.beta * (1.0 - rhos[i] - rule.arousal), 0.0, 1.0)
            )

        i_star = int(np.argmax(rhos))
        tau = self._tau()
        if (
            float(np.max([r.arousal for r in self.rules])) > tau
            and self.n_rules < self.cfg.max_rules
        ):
            self.rules.append(self._init_rule(x, y))
            created = True
        else:
            created = False
            self._update_center(self.rules[i_star], x, float(rhos[i_star]))

        self._merge_redundant()
        rhos = self._compatibilities(x)
        lam = self._normalized_activation(rhos)
        i_star = int(np.argmax(rhos))
        winner = self.rules[i_star]
        local_yhat = self._local_output(winner, x)
        if not created or winner is not self.rules[-1]:
            if self._novelty_add(winner, x, y, local_yhat):
                self._krls_grow(winner, x, y, local_yhat)
            else:
                self._krls_update_existing(winner, x, y, local_yhat)
            e_local = y - self._local_output(winner, x)
            self._update_kernel_size(winner, x, float(lam[i_star]), e_local)

        for rule in self.rules:
            if not np.all(np.isfinite(rule.theta)):
                rule.theta = np.nan_to_num(rule.theta, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.all(np.isfinite(rule.Q)):
                rule.Q = np.eye(rule.n_dict) / max(self.cfg.lam, 1e-8)
        self._update_beta(error)
        self.last_error = float(error)
        self.n_seen += 1
        self.history_n_rules.append(self.n_rules)
        self.history_beta.append(self.beta)
        self.history_error.append(float(error))
        return yhat

    def fit_online(self, X: np.ndarray, y: np.ndarray, predict_before_update: bool = True) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        preds = np.zeros(len(y), dtype=float)
        for k in range(len(y)):
            if predict_before_update and self.rules:
                preds[k] = self.predict_one(X[k])
                self.update(X[k], y[k])
            else:
                preds[k] = self.update(X[k], y[k])
        return preds

    def reset(self) -> None:
        self.rules = []
        self.beta = float(self.cfg.beta0)
        self.n_seen = 0
        self.last_error = 0.0
        self.history_n_rules = []
        self.history_beta = []
        self.history_error = []
