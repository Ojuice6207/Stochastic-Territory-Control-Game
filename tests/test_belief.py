"""
Unit tests for the POMDP belief state components.

Validates:
  - KalmanBeliefNode converges toward truth over repeated observations
  - ParticleStrengthEstimator shifts posterior toward true strength
    after several combat outcomes
  - Systematic resampling preserves particle mean
"""
import numpy as np
import pytest
from stochastic_ojuice.belief import KalmanBeliefNode, ParticleStrengthEstimator
from stochastic_ojuice.combat import CombatModel
from stochastic_ojuice.types import CombatParams


# ---------------------------------------------------------------------------
# Kalman filter tests
# ---------------------------------------------------------------------------

class TestKalmanBeliefNode:

    def test_posterior_narrows_with_observations(self):
        """Variance should decrease as we observe the OU process."""
        kf = KalmanBeliefNode(
            theta=0.5, mu=10.0, sigma=1.0,
            obs_noise_var=0.25,
            m=10.0, P=4.0,
        )
        P_initial = kf.P
        rng = np.random.default_rng(0)
        for _ in range(50):
            z = rng.normal(10.0, 0.5)
            kf.step(dt=0.1, z=z)
        assert kf.P < P_initial / 2.0, "Variance should shrink substantially"

    def test_mean_tracks_truth(self):
        """
        Over 200 steps with true V=10, posterior mean should settle near 10.
        """
        kf = KalmanBeliefNode(
            theta=0.5, mu=10.0, sigma=1.0,
            obs_noise_var=0.25,
            m=5.0, P=10.0,  # start far from truth
        )
        rng = np.random.default_rng(42)
        true_V = 10.0
        for _ in range(200):
            z = true_V + rng.normal(0, 0.5)
            kf.step(dt=0.1, z=z)
        assert abs(kf.m - true_V) < 1.0, f"KF mean {kf.m:.2f} too far from truth {true_V}"

    def test_predict_only_increases_variance(self):
        """Without observations, uncertainty should grow."""
        kf = KalmanBeliefNode(
            theta=0.5, mu=10.0, sigma=2.0,
            obs_noise_var=0.25,
            m=10.0, P=0.01,  # start very certain
        )
        P_before = kf.P
        for _ in range(20):
            kf.predict(dt=0.1)
        assert kf.P > P_before, "Uncertainty must grow without observations"

    def test_credible_interval_coverage(self):
        """95% CI should cover the true value ≥90% of the time."""
        rng  = np.random.default_rng(7)
        hits = 0
        N    = 500
        for _ in range(N):
            kf = KalmanBeliefNode(
                theta=1.0, mu=5.0, sigma=1.0,
                obs_noise_var=1.0, m=5.0, P=2.0,
            )
            true_v = rng.normal(5.0, 1.0)
            for _ in range(30):
                kf.step(dt=0.1, z=true_v + rng.normal(0, 1.0))
            lo, hi = kf.credible_interval(0.95)
            if lo <= true_v <= hi:
                hits += 1
        coverage = hits / N
        assert coverage >= 0.85, f"CI coverage {coverage:.2f} too low (expected ≥0.85)"


# ---------------------------------------------------------------------------
# Particle filter tests
# ---------------------------------------------------------------------------

class TestParticleStrengthEstimator:

    def _build_pf(self, n=500, seed=0):
        pf = ParticleStrengthEstimator(
            n_particles=n,
            init_mean=5.0,
            init_std=3.0,
        )
        pf.reset(mean=5.0, std=3.0, rng=np.random.default_rng(seed))
        return pf

    def test_initial_mean(self):
        pf = self._build_pf()
        assert abs(pf.mean - 5.0) < 0.5

    def test_update_shifts_mean_toward_truth(self):
        """
        After 30 successful attacks by a strong attacker against a weak
        defender (S_true=3), the posterior mean should move below the prior
        mean (originally 5.0) because failures/successes reveal low S_def.
        """
        rng   = np.random.default_rng(42)
        pf    = self._build_pf(n=1000, seed=42)
        model = CombatModel(CombatParams(k=2.0, delta=0.0, noise_sigma=0.0))
        S_att = 8.0
        S_true = 3.0  # true defender strength (agent doesn't know this)

        prior_mean = pf.mean

        for _ in range(30):
            # Simulate outcome from ground truth
            success, _, _ = model.resolve(S_att, S_true, rng)
            log_liks = model.log_likelihood(success, S_att, pf.particles)
            pf.update(log_liks)
            pf.resample_if_needed(rng)
            pf.predict(rng)

        posterior_mean = pf.mean
        # The posterior should have shifted toward S_true=3 from prior 5
        assert posterior_mean < prior_mean, (
            f"Posterior mean {posterior_mean:.2f} should be below prior {prior_mean:.2f}"
        )

    def test_resample_preserves_mean(self):
        """Systematic resampling should preserve the weighted mean."""
        pf  = self._build_pf(n=500, seed=0)
        rng = np.random.default_rng(1)
        # Force non-uniform weights
        log_w = rng.normal(0, 1, 500)
        log_w -= log_w.max()
        pf.weights = np.exp(log_w)
        pf.weights /= pf.weights.sum()

        mean_before = pf.mean
        pf._systematic_resample(rng)
        mean_after = pf.mean
        assert abs(mean_after - mean_before) < 0.5, "Resampling must preserve approximate mean"

    def test_weights_normalised_after_update(self):
        pf  = self._build_pf()
        rng = np.random.default_rng(5)
        model = CombatModel(CombatParams())
        log_liks = model.log_likelihood(True, 8.0, pf.particles)
        pf.update(log_liks)
        assert abs(pf.weights.sum() - 1.0) < 1e-9
