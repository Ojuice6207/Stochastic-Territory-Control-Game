"""
Ornstein-Uhlenbeck (OU) Process Engine
=======================================

Provides exact (not Euler-Maruyama) vectorised simulation of independent
OU processes, one per territory node.  Can simulate a single step or an
entire batch of Monte Carlo paths simultaneously.

Mathematical background
-----------------------
The OU SDE:

    dV_t = θ(μ - V_t) dt + σ dW_t

admits a closed-form transition distribution (the SDE is linear/affine):

    V_{t+Δt} | V_t  ~  N( m(V_t), v )

where:
    m(V_t) = μ + (V_t - μ) · exp(-θ Δt)          (conditional mean)
    v       = (σ²/2θ) · (1 - exp(-2θ Δt))          (conditional variance)

Because this is *exact*, there is no discretisation error regardless of
how large Δt is — critical when running coarse-grained gameplay steps.

For a batch of B Monte Carlo scenarios across N nodes we generate
Z ~ N(0,1) of shape (B, N) and compute the update in a single broadcast:

    V_{next}[b, i] = mu[i] + phi[i] * (V[b, i] - mu[i]) + sigma_dt[i] * Z[b, i]

where phi[i] = exp(-theta[i] * dt) and sigma_dt[i] = sqrt(v[i]).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OUProcess:
    """
    Vectorised exact simulator for N independent OU processes.

    Parameters
    ----------
    theta : np.ndarray, shape (N,)
        Per-process mean-reversion speeds.
    mu : np.ndarray, shape (N,)
        Per-process long-run means.
    sigma : np.ndarray, shape (N,)
        Per-process volatilities.

    Usage
    -----
    >>> proc = OUProcess(theta, mu, sigma)
    >>> V_next = proc.step(V_current, dt=0.1, rng=rng)           # (N,)
    >>> paths  = proc.simulate_batch(V0, dt=0.1, n_steps=50,
    ...                              n_scenarios=1000, rng=rng)   # (1000, 50, N)
    """

    theta: np.ndarray  # (N,)
    mu: np.ndarray     # (N,)
    sigma: np.ndarray  # (N,)

    def __post_init__(self) -> None:
        self.theta = np.asarray(self.theta, dtype=np.float64)
        self.mu    = np.asarray(self.mu,    dtype=np.float64)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
        assert self.theta.shape == self.mu.shape == self.sigma.shape, \
            "theta, mu, sigma must have the same shape"
        self._n = self.theta.size

    # ------------------------------------------------------------------
    # Pre-compute dt-dependent constants (cached for repeated same-dt use)
    # ------------------------------------------------------------------

    def _precompute(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (phi, sigma_dt) for the given time-step.

        phi       = exp(-theta * dt)                  shape (N,)
        sigma_dt  = sqrt( sigma^2 / (2*theta) * (1 - phi^2) )  shape (N,)

        For theta ≈ 0 (very slow reversion) we use a first-order Taylor
        expansion to avoid 0/0: sigma_dt ≈ sigma * sqrt(dt).
        """
        phi = np.exp(-self.theta * dt)  # (N,)

        # Numerically stable variance: handle theta → 0 via Taylor series
        two_theta_dt = 2.0 * self.theta * dt
        # For small x: (1 - e^{-x})/x → 1, so var → sigma^2 * dt
        with np.errstate(invalid="ignore", divide="ignore"):
            var_factor = np.where(
                two_theta_dt > 1e-8,
                (1.0 - phi**2) / (2.0 * self.theta),
                dt,  # Taylor limit
            )

        sigma_dt = self.sigma * np.sqrt(var_factor)  # (N,)
        return phi, sigma_dt

    # ------------------------------------------------------------------
    # Single-step update (in-place capable)
    # ------------------------------------------------------------------

    def step(
        self,
        V: np.ndarray,
        dt: float,
        rng: np.random.Generator,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Advance all N processes by one time-step dt.

        Parameters
        ----------
        V : np.ndarray, shape (N,) or (B, N)
            Current values.  B is an optional batch dimension.
        dt : float
            Time increment (same units as theta^{-1}).
        rng : np.random.Generator
        out : np.ndarray, optional
            Pre-allocated output array (for zero-copy game loops).

        Returns
        -------
        V_next : np.ndarray, same shape as V
            Next values drawn from the exact transition distribution.
        """
        V = np.asarray(V, dtype=np.float64)
        phi, sigma_dt = self._precompute(dt)

        # Broadcast phi/sigma_dt over optional batch dimension
        if V.ndim == 2:
            phi      = phi[np.newaxis, :]      # (1, N)
            sigma_dt = sigma_dt[np.newaxis, :] # (1, N)
            mu       = self.mu[np.newaxis, :]  # (1, N)
        else:
            mu = self.mu

        noise = rng.standard_normal(V.shape)
        result = mu + phi * (V - mu) + sigma_dt * noise

        if out is not None:
            out[...] = result
            return out
        return result

    # ------------------------------------------------------------------
    # Full Monte Carlo batch  (B scenarios × T steps × N nodes)
    # ------------------------------------------------------------------

    def simulate_batch(
        self,
        V0: np.ndarray,
        dt: float,
        n_steps: int,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate `n_scenarios` independent paths of `n_steps` each.

        Parameters
        ----------
        V0 : np.ndarray, shape (N,)
            Initial values for all nodes (same across scenarios).
        dt, n_steps, n_scenarios : scalars

        Returns
        -------
        paths : np.ndarray, shape (n_scenarios, n_steps+1, N)
            paths[:, 0, :] == V0 (broadcast); paths[:, t, :] is the
            value at time t*dt.
        """
        phi, sigma_dt = self._precompute(dt)

        N = self._n
        paths = np.empty((n_scenarios, n_steps + 1, N), dtype=np.float64)
        paths[:, 0, :] = V0[np.newaxis, :]  # broadcast V0 over scenarios

        # Pre-draw all noise at once (single large allocation — faster
        # than n_steps small calls for typical n_steps up to ~1000)
        noise = rng.standard_normal((n_scenarios, n_steps, N))

        mu_b  = self.mu[np.newaxis, :]        # (1, N)
        phi_b = phi[np.newaxis, :]             # (1, N)
        sd_b  = sigma_dt[np.newaxis, :]        # (1, N)

        for t in range(n_steps):
            V_prev = paths[:, t, :]            # (B, N)
            paths[:, t + 1, :] = (
                mu_b + phi_b * (V_prev - mu_b) + sd_b * noise[:, t, :]
            )

        return paths

    # ------------------------------------------------------------------
    # Analytical statistics (no simulation needed)
    # ------------------------------------------------------------------

    def stationary_mean(self) -> np.ndarray:
        """Long-run mean: E[V_∞] = mu.  Shape (N,)."""
        return self.mu.copy()

    def stationary_variance(self) -> np.ndarray:
        """Long-run variance: Var[V_∞] = sigma^2 / (2*theta).  Shape (N,)."""
        return (self.sigma ** 2) / (2.0 * self.theta)

    def conditional_mean(self, V: np.ndarray, dt: float) -> np.ndarray:
        """E[V_{t+dt} | V_t] = mu + phi*(V_t - mu).  Shape same as V."""
        phi, _ = self._precompute(dt)
        return self.mu + phi * (np.asarray(V) - self.mu)

    def conditional_variance(self, dt: float) -> np.ndarray:
        """Var[V_{t+dt} | V_t] = sigma_dt^2.  Shape (N,) — independent of V_t."""
        _, sigma_dt = self._precompute(dt)
        return sigma_dt ** 2

    def half_life(self) -> np.ndarray:
        """
        Time for deviation from mu to decay by half.
        half_life = ln(2) / theta.  Useful for tuning gameplay pacing.
        """
        return np.log(2.0) / self.theta
