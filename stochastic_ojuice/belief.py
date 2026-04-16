"""
POMDP Belief State Tracking
============================

Two complementary Bayesian estimators, one per hidden variable type:

  1. KalmanBeliefNode  — tracks the true (hidden) OU value V_t of a
     territory using a scalar Kalman filter.  Closed-form updates, O(1)
     per step.  The OU dynamics play the role of the linear state-space
     model's transition, so the Kalman filter is *exact* (no linearisation
     required).

  2. ParticleStrengthEstimator — tracks the hidden troop strength S_j of
     a single enemy node using a particle filter with N weighted samples.
     The non-Gaussian logistic likelihood means conjugate Kalman updates
     are unavailable; particles are the correct tool.

Together these implement the factorised belief update:

    b_{t+1}(V, S) ≈ ∏_i KF(V_i) · ∏_j PF(S_j)

which is tractable because V and S are conditionally independent given
observations (they enter via different sensors with separate noise).

Mathematical details
---------------------

Kalman filter for OU value (per node):
    State:   x_t = V_t   (scalar)
    Dynamics: x_{t+1} = mu + phi*(x_t - mu) + w_t,  w_t ~ N(0, Q)
              where phi = exp(-theta*dt), Q = sigma^2*(1-phi^2)/(2*theta)
    Obs:     z_t = x_t + v_t,  v_t ~ N(0, R)  (noisy territory value)

    Predict:  m_pred = mu + phi*(m - mu)
              P_pred = phi^2 * P + Q
    Update:   K = P_pred / (P_pred + R)
              m_new = m_pred + K*(z - m_pred)
              P_new = (1 - K) * P_pred

Particle filter for enemy strength (per enemy node):
    Prior particles {S^(k), w^(k)}, k=1..N_p
    Predict: S^(k) += drift noise (strength changes slowly between turns)
    Update:  w^(k) *= P(combat_outcome | S_att, S^(k))
             w^(k) /= sum(w)
    Resample: systematic resampling when N_eff = 1/sum(w^2) < N_p/2
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Scalar Kalman Filter for OU-driven territory values
# ---------------------------------------------------------------------------

@dataclass
class KalmanBeliefNode:
    """
    Scalar Kalman filter tracking V_t for one territory node.

    Maintains the Gaussian belief N(m, P) where m = E[V_t | obs]
    and P = Var[V_t | obs].

    Parameters
    ----------
    theta, mu, sigma : float
        OU parameters (must match the true dynamics — or be estimated).
    obs_noise_var : float
        Observation noise variance R = σ_obs^2.
    init_mean : float
        Prior mean m_0 (typically the OU equilibrium mu).
    init_var : float
        Prior variance P_0 (typically the OU stationary variance σ²/2θ).
    """
    theta: float
    mu: float
    sigma: float
    obs_noise_var: float
    m: float = 0.0   # posterior mean
    P: float = 1.0   # posterior variance

    def __post_init__(self) -> None:
        if self.P <= 0:
            raise ValueError("Initial variance P must be > 0.")

    def _precompute(self, dt: float) -> Tuple[float, float]:
        """phi = exp(-theta*dt),  Q = sigma^2*(1-phi^2)/(2*theta)."""
        phi = np.exp(-self.theta * dt)
        two_theta = 2.0 * self.theta
        with np.errstate(invalid="ignore"):
            Q = (self.sigma ** 2 / two_theta) * (1.0 - phi ** 2) if two_theta > 1e-10 else self.sigma ** 2 * dt
        return float(phi), float(Q)

    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """
        Time-update: propagate belief through OU dynamics without an observation.
        Call this once per game step *before* update().
        """
        phi, Q = self._precompute(dt)
        self.m = self.mu + phi * (self.m - self.mu)
        self.P = phi ** 2 * self.P + Q

    def update(self, z: float) -> float:
        """
        Measurement-update: incorporate a noisy observation z ~ V_t + N(0, R).

        Returns the Kalman gain K (useful for diagnostics).
        """
        K = self.P / (self.P + self.obs_noise_var)
        self.m = self.m + K * (z - self.m)
        self.P = (1.0 - K) * self.P
        return K

    def step(self, dt: float, z: Optional[float] = None) -> None:
        """
        Combined predict + optional update.

        Parameters
        ----------
        dt : float
        z  : float or None
            If None (territory not observed this turn), only predict.
        """
        self.predict(dt)
        if z is not None:
            self.update(z)

    @property
    def posterior_std(self) -> float:
        """Standard deviation of the current belief."""
        return float(np.sqrt(max(self.P, 0.0)))

    def credible_interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Return the (alpha)% equal-tails credible interval.

        Example: alpha=0.95 → (m - 1.96*std, m + 1.96*std).
        """
        from scipy.stats import norm
        z = norm.ppf(0.5 + alpha / 2.0)
        half_width = z * self.posterior_std
        return (self.m - half_width, self.m + half_width)


# ---------------------------------------------------------------------------
# Particle Filter for enemy troop strength
# ---------------------------------------------------------------------------

@dataclass
class ParticleStrengthEstimator:
    """
    Sequential Monte Carlo (particle filter) for hidden enemy strength.

    Maintains N_p weighted samples {S^(k), w^(k)}.  The non-conjugate
    logistic likelihood from CombatModel requires particle methods —
    no closed-form Gaussian update exists.

    Parameters
    ----------
    n_particles : int
        Number of particles.  Typical: 200–2000.
    init_mean, init_std : float
        Parameters for the initial Gaussian particle cloud.
    strength_drift_std : float
        Per-step random-walk std on each particle (models the fact that
        enemy troops reinforce/deploy between turns).
    min_strength : float
        Hard lower bound for troop strength (clamped after drift).
    """

    n_particles: int = 500
    init_mean: float = 5.0
    init_std: float = 2.0
    strength_drift_std: float = 0.3
    min_strength: float = 0.1

    # Internal state — not set by user
    particles: np.ndarray = field(init=False)
    weights: np.ndarray   = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()  # internal RNG; override via reset()
        self.reset(mean=self.init_mean, std=self.init_std)

    def reset(
        self,
        mean: float,
        std: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Re-initialise the particle cloud (call when node becomes visible)."""
        if rng is not None:
            self._rng = rng
        self.particles = np.maximum(
            self.min_strength,
            self._rng.normal(mean, std, self.n_particles),
        )
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    # ------------------------------------------------------------------
    # Filter steps
    # ------------------------------------------------------------------

    def predict(self, rng: np.random.Generator) -> None:
        """
        Propagate particles with random-walk process noise.

        Models slow changes in enemy troop count between turns
        (reinforcements, attrition from other fronts, etc.).
        """
        self.particles += rng.normal(0.0, self.strength_drift_std, self.n_particles)
        self.particles = np.maximum(self.min_strength, self.particles)

    def update(
        self,
        log_likelihoods: np.ndarray,
    ) -> None:
        """
        Weight-update using pre-computed log-likelihoods.

        Parameters
        ----------
        log_likelihoods : ndarray, shape (n_particles,)
            log P(observation | particle_k) from CombatModel.log_likelihood().

        Uses the log-sum-exp trick for numerical stability.
        """
        log_w = np.log(self.weights + 1e-300) + log_likelihoods
        # Stable normalisation
        log_w -= log_w.max()
        w = np.exp(log_w)
        total = w.sum()
        if total < 1e-300:
            # All weights collapsed — reinitialise with prior
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        else:
            self.weights = w / total

    def resample_if_needed(
        self,
        rng: np.random.Generator,
        threshold_frac: float = 0.5,
    ) -> bool:
        """
        Systematic resampling when effective sample size drops below threshold.

        N_eff = 1 / Σ w_k²  < threshold_frac * N_p → resample.

        Returns True if resampling occurred.
        """
        n_eff = 1.0 / (np.sum(self.weights ** 2) + 1e-300)
        if n_eff < threshold_frac * self.n_particles:
            self._systematic_resample(rng)
            return True
        return False

    def _systematic_resample(self, rng: np.random.Generator) -> None:
        """
        O(N) systematic resampling (lower variance than multinomial).

        Reference: Kitagawa (1996) — positions the resampling grid with
        a single uniform draw then advances through the CDF.
        """
        N = self.n_particles
        cumsum = np.cumsum(self.weights)
        positions = (rng.random() + np.arange(N)) / N  # single U[0,1] draw

        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights   = np.full(N, 1.0 / N)

    # ------------------------------------------------------------------
    # Posterior summary statistics
    # ------------------------------------------------------------------

    @property
    def mean(self) -> float:
        """Posterior mean E[S | obs]."""
        return float(np.dot(self.weights, self.particles))

    @property
    def variance(self) -> float:
        """Posterior variance Var[S | obs]."""
        mu = self.mean
        return float(np.dot(self.weights, (self.particles - mu) ** 2))

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))

    def quantile(self, q: float) -> float:
        """Weighted quantile of the particle distribution."""
        sort_idx = np.argsort(self.particles)
        sorted_p = self.particles[sort_idx]
        sorted_w = self.weights[sort_idx]
        cumw = np.cumsum(sorted_w)
        return float(sorted_p[np.searchsorted(cumw, q)])

    def credible_interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        lo = (1.0 - alpha) / 2.0
        hi = 1.0 - lo
        return (self.quantile(lo), self.quantile(hi))


# ---------------------------------------------------------------------------
# Aggregated per-node belief state for a single agent
# ---------------------------------------------------------------------------

@dataclass
class NodeBelief:
    """
    Complete belief about one territory node from one agent's perspective.

    Combines a Kalman filter (value) and particle filter (enemy strength)
    into a single object that the agent stores per node.

    Attributes
    ----------
    node_id : int
    value_kf : KalmanBeliefNode
        Kalman filter for territory value.
    strength_pf : ParticleStrengthEstimator
        Particle filter for enemy troop strength (only meaningful if the
        node is not owned by this agent).
    last_seen_step : int
        Step counter when this node was last observed.  Used to decide
        how stale the belief is.
    """

    node_id: int
    value_kf: KalmanBeliefNode
    strength_pf: ParticleStrengthEstimator
    last_seen_step: int = 0

    # ------------------------------------------------------------------
    # Convenience summary for the EV function
    # ------------------------------------------------------------------

    def value_mean(self) -> float:
        return self.value_kf.m

    def value_var(self) -> float:
        return self.value_kf.P

    def strength_mean(self) -> float:
        return self.strength_pf.mean

    def strength_std(self) -> float:
        return self.strength_pf.std
