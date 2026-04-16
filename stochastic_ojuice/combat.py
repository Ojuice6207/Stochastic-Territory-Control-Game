"""
Combat Model: Logistic Probability with Battlefield Noise
==========================================================

Implements the parameterised logistic combat function:

    η        = k * (S_att - S_def - δ) + ε,    ε ~ N(0, σ_ε²)
    P(win)   = sigmoid(η) = 1 / (1 + exp(-η))

Noise ε is injected into the *linear predictor* (not clamped to [0,1]),
so the outcome distribution is the logit-normal family:
    P(win) has mean ≈ sigmoid(k*(ΔS - δ)) and variance driven by σ_ε.

This is more principled than P + clip(ε, …) because:
1. Bounds [0,1] are guaranteed without clipping.
2. The Fisher information for Bayesian updating is well-defined.
3. Gradient ∂P/∂S_att = k·P(1-P) > 0 everywhere — monotonicity is exact.

Vectorised API
--------------
All public methods accept scalar *or* ndarray inputs and return the
same shape, enabling batch EV evaluation across all candidate attacks
in O(1) NumPy calls.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from .types import CombatParams


# ---------------------------------------------------------------------------
# Core sigmoid helper (numerically stable)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid: avoids exp overflow for large |x|.
    Uses the identity: sigmoid(-x) = 1 - sigmoid(x).
    """
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ---------------------------------------------------------------------------
# Combat model
# ---------------------------------------------------------------------------

@dataclass
class CombatModel:
    """
    Logistic-noise combat probability and resolution engine.

    Parameters
    ----------
    params : CombatParams
        Hyper-parameters (k, delta, noise_sigma, cost fractions).

    Usage
    -----
    >>> model = CombatModel(CombatParams(k=1.5, delta=0.3))
    >>> p = model.success_probability(S_att=5.0, S_def=4.0)
    >>> outcome, new_att, new_def = model.resolve(S_att, S_def, rng=rng)
    """

    params: CombatParams

    def __post_init__(self) -> None:
        self._k     = float(self.params.k)
        self._delta = float(self.params.delta)
        self._ns    = float(self.params.noise_sigma)

    # ------------------------------------------------------------------
    # Probability queries (no sampling)
    # ------------------------------------------------------------------

    def linear_predictor(
        self,
        S_att: np.ndarray | float,
        S_def: np.ndarray | float,
    ) -> np.ndarray:
        """
        η_mean = k * (S_att - S_def - delta)

        This is the *noiseless* linear predictor — the mean of the
        stochastic η before battlefield noise is added.
        """
        delta_s = np.asarray(S_att, dtype=np.float64) - np.asarray(S_def, dtype=np.float64)
        return self._k * (delta_s - self._delta)

    def success_probability(
        self,
        S_att: np.ndarray | float,
        S_def: np.ndarray | float,
        *,
        include_noise_variance: bool = False,
    ) -> np.ndarray:
        """
        Expected P(success) marginalised over battlefield noise ε.

        If noise_sigma == 0:
            E[P] = sigmoid(η_mean)

        If noise_sigma > 0 and include_noise_variance=True:
            Uses the Gaussian-logistic approximation:
            E[sigmoid(η + ε)] ≈ sigmoid(η / sqrt(1 + c²·σ_ε²))
            where c = π/√8 ≈ 1.1284 (Gaussian CDF approximation).

            This avoids Monte Carlo integration for the EV function while
            remaining within ~1% of the exact value.

        Parameters
        ----------
        S_att, S_def : scalar or ndarray
            Attacker and defender strengths (true or estimated).
        include_noise_variance : bool
            If True, marginalises over ε analytically (recommended for
            the EV function).  If False, evaluates at ε=0 (faster but
            ignores uncertainty).

        Returns
        -------
        p : ndarray, same shape as broadcast(S_att, S_def)
        """
        eta = self.linear_predictor(S_att, S_def)

        if include_noise_variance and self._ns > 0:
            # Gaussian-logistic approximation (Mackay 1992 / Bishop 2006)
            c = np.pi / np.sqrt(8.0)
            eta = eta / np.sqrt(1.0 + (c * self._ns) ** 2)

        return _sigmoid(eta)

    def probability_grid(
        self,
        s_range: Tuple[float, float],
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate P(success) on a (S_att, S_def) grid for visualisation.

        Returns
        -------
        S_att_grid, S_def_grid, P_grid : each shape (n_points, n_points)
        """
        s = np.linspace(*s_range, n_points)
        Sa, Sd = np.meshgrid(s, s, indexing="ij")
        P = self.success_probability(Sa, Sd, include_noise_variance=True)
        return Sa, Sd, P

    # ------------------------------------------------------------------
    # Stochastic resolution (single engagement)
    # ------------------------------------------------------------------

    def resolve(
        self,
        S_att: float,
        S_def: float,
        rng: np.random.Generator,
    ) -> Tuple[bool, float, float]:
        """
        Resolve a single combat engagement stochastically.

        Draws ε ~ N(0, σ_ε²), computes η = k*(ΔS - δ) + ε,
        then draws outcome ~ Bernoulli(sigmoid(η)).

        Regardless of outcome, the attacker pays `attack_cost_frac`
        of their troops.  On failure, an additional `failure_drain_frac`
        is also lost.

        Parameters
        ----------
        S_att, S_def : float
            True (authoritative) strengths from GameState.
        rng : np.random.Generator

        Returns
        -------
        success : bool
        S_att_after : float
            Attacker's strength after the engagement.
        S_def_after : float
            Defender's remaining strength (0 if captured).
        """
        # Draw battlefield noise
        eps = rng.normal(0.0, self._ns) if self._ns > 0 else 0.0
        eta = self._k * (S_att - S_def - self._delta) + eps
        p   = float(_sigmoid(eta))

        success = bool(rng.random() < p)

        # Troop losses
        fixed_loss   = self.params.attack_cost_frac * S_att
        failure_loss = self.params.failure_drain_frac * S_att if not success else 0.0

        S_att_after = max(0.0, S_att - fixed_loss - failure_loss)
        S_def_after = 0.0 if success else S_def  # defender wiped if captured

        return success, S_att_after, S_def_after

    # ------------------------------------------------------------------
    # Vectorised batch resolution (for Monte Carlo / RL training)
    # ------------------------------------------------------------------

    def resolve_batch(
        self,
        S_att: np.ndarray,
        S_def: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resolve B engagements simultaneously.

        Parameters
        ----------
        S_att, S_def : ndarray, shape (B,)

        Returns
        -------
        successes    : bool ndarray, shape (B,)
        S_att_after  : float ndarray, shape (B,)
        S_def_after  : float ndarray, shape (B,)
        """
        S_att = np.asarray(S_att, dtype=np.float64)
        S_def = np.asarray(S_def, dtype=np.float64)
        B = S_att.shape[0]

        eps  = rng.normal(0.0, self._ns, size=B) if self._ns > 0 else np.zeros(B)
        eta  = self._k * (S_att - S_def - self._delta) + eps
        p    = _sigmoid(eta)

        successes = rng.random(B) < p

        fixed_loss   = self.params.attack_cost_frac * S_att
        failure_loss = np.where(~successes, self.params.failure_drain_frac * S_att, 0.0)

        S_att_after = np.maximum(0.0, S_att - fixed_loss - failure_loss)
        S_def_after = np.where(successes, 0.0, S_def)

        return successes, S_att_after, S_def_after

    # ------------------------------------------------------------------
    # Bayesian likelihood (for particle filter weight update)
    # ------------------------------------------------------------------

    def log_likelihood(
        self,
        outcome: bool,
        S_att: float,
        S_def_particles: np.ndarray,
    ) -> np.ndarray:
        """
        Log P(outcome | S_att, S_def_particles) for each particle.

        Used by the particle filter in PlayerAgent to update beliefs
        about enemy strength after observing a combat result.

        Because we marginalise over ε analytically using the
        Gaussian-logistic approximation, this is a smooth function of
        S_def — gradients are well-behaved for importance sampling.

        Parameters
        ----------
        outcome : bool
            True = attacker won, False = attacker lost.
        S_att : float
            Attacker's *known* strength.
        S_def_particles : ndarray, shape (N_particles,)
            Candidate defender strengths from particle distribution.

        Returns
        -------
        log_w : ndarray, shape (N_particles,)
            Unnormalised log-weights (log-likelihoods).
        """
        p = self.success_probability(
            S_att,
            S_def_particles,
            include_noise_variance=True,
        )  # shape (N_particles,)

        # Clip for numerical stability before log
        p = np.clip(p, 1e-10, 1.0 - 1e-10)

        if outcome:
            return np.log(p)
        else:
            return np.log(1.0 - p)
