"""
Unit tests for the OU stochastic engine.

Validates:
  - Exact discretisation moments (conditional mean and variance)
  - Vectorised batch simulation produces correct shapes
  - Stationary distribution convergence
"""
import numpy as np
import pytest
from stochastic_ojuice.stochastic import OUProcess


@pytest.fixture
def ou():
    theta = np.array([0.5, 1.0, 2.0])
    mu    = np.array([5.0, 10.0, 2.0])
    sigma = np.array([1.0, 2.0, 0.5])
    return OUProcess(theta, mu, sigma)


def test_conditional_mean(ou):
    """E[V_{t+dt}|V_t] = mu + phi*(V_t - mu)"""
    V0 = np.array([6.0, 8.0, 3.0])
    dt = 0.2
    phi = np.exp(-ou.theta * dt)
    expected_mean = ou.mu + phi * (V0 - ou.mu)
    assert np.allclose(ou.conditional_mean(V0, dt), expected_mean)


def test_conditional_variance_exact(ou):
    """
    Var[V_{t+dt}|V_t] = sigma^2/(2*theta) * (1 - exp(-2*theta*dt))
    Should be independent of V_t.
    """
    dt = 0.1
    phi = np.exp(-ou.theta * dt)
    expected_var = (ou.sigma**2 / (2*ou.theta)) * (1 - phi**2)
    assert np.allclose(ou.conditional_variance(dt), expected_var, rtol=1e-6)


def test_step_shape(ou):
    """Single-step returns same shape as input."""
    rng = np.random.default_rng(0)
    V = np.array([5.0, 10.0, 2.0])
    V_next = ou.step(V, dt=0.1, rng=rng)
    assert V_next.shape == V.shape


def test_batch_step_shape(ou):
    """Batch step (B, N) maintains (B, N) shape."""
    rng = np.random.default_rng(0)
    B, N = 100, 3
    V_batch = np.ones((B, N)) * 5.0
    V_next = ou.step(V_batch, dt=0.1, rng=rng)
    assert V_next.shape == (B, N)


def test_simulate_batch_shape(ou):
    rng = np.random.default_rng(0)
    V0 = np.array([5.0, 10.0, 2.0])
    paths = ou.simulate_batch(V0, dt=0.1, n_steps=50, n_scenarios=1000, rng=rng)
    assert paths.shape == (1000, 51, 3)


def test_stationary_moments(ou):
    """
    After many steps from an arbitrary start, empirical moments should
    converge to stationary mean and variance within 5%.
    """
    rng = np.random.default_rng(42)
    V0  = np.zeros(3)
    # Long run: T = 50 half-lives of slowest process (theta=0.5, t1/2=ln2/0.5≈1.4)
    paths = ou.simulate_batch(V0, dt=0.05, n_steps=2000, n_scenarios=5000, rng=rng)

    empirical_mean = paths[:, -1, :].mean(axis=0)
    empirical_var  = paths[:, -1, :].var(axis=0)

    np.testing.assert_allclose(empirical_mean, ou.stationary_mean(), rtol=0.05)
    np.testing.assert_allclose(empirical_var,  ou.stationary_variance(), rtol=0.10)


def test_half_life(ou):
    """half_life = ln(2) / theta."""
    expected = np.log(2) / ou.theta
    np.testing.assert_allclose(ou.half_life(), expected)
