"""
Unit tests for the logistic combat model.

Validates:
  - Monotonicity: P(success) strictly increases with ΔS
  - Bounds: P(success) ∈ (0, 1)
  - Symmetry at δ: P(ΔS = δ) = 0.5
  - Log-likelihood sign consistency
  - Batch resolution shape
"""
import numpy as np
import pytest
from stochastic_ojuice.combat import CombatModel, _sigmoid
from stochastic_ojuice.types import CombatParams


@pytest.fixture
def model():
    return CombatModel(CombatParams(k=1.5, delta=0.3, noise_sigma=0.0))


def test_sigmoid_bounds():
    x = np.linspace(-20, 20, 1000)
    s = _sigmoid(x)
    assert np.all(s > 0) and np.all(s < 1)


def test_monotonicity(model):
    """P strictly increases as S_att - S_def increases."""
    S_att = np.linspace(1, 10, 50)
    S_def = 5.0
    probs = model.success_probability(S_att, S_def)
    assert np.all(np.diff(probs) > 0), "P(success) must be strictly monotone in ΔS"


def test_bounds(model):
    S_att = np.random.default_rng(0).uniform(0, 20, 500)
    S_def = np.random.default_rng(1).uniform(0, 20, 500)
    p = model.success_probability(S_att, S_def)
    assert np.all(p > 0) and np.all(p < 1)


def test_defender_advantage(model):
    """With delta=0.3, P(ΔS=0) < 0.5 (defender advantage)."""
    p_equal = float(model.success_probability(5.0, 5.0))
    assert p_equal < 0.5


def test_symmetry_at_delta():
    """At S_att - S_def = delta, P = 0.5 (inflection point)."""
    params = CombatParams(k=1.5, delta=1.0, noise_sigma=0.0)
    m = CombatModel(params)
    p = float(m.success_probability(S_att=6.0, S_def=5.0))  # ΔS = 1.0 = delta
    assert abs(p - 0.5) < 1e-9


def test_log_likelihood_success_vs_failure():
    """
    Log-likelihood of success should be > failure for a strong attacker.
    """
    model = CombatModel(CombatParams(k=2.0, delta=0.0, noise_sigma=0.0))
    S_att = 10.0
    particles = np.array([3.0, 3.0, 3.0])

    log_lik_success = model.log_likelihood(True,  S_att, particles)
    log_lik_failure = model.log_likelihood(False, S_att, particles)
    assert np.all(log_lik_success > log_lik_failure)


def test_batch_resolve_shape():
    model = CombatModel(CombatParams())
    rng = np.random.default_rng(0)
    B = 200
    S_att = rng.uniform(2, 10, B)
    S_def = rng.uniform(2, 10, B)
    successes, sa, sd = model.resolve_batch(S_att, S_def, rng)
    assert successes.shape == (B,)
    assert sa.shape == (B,)
    assert sd.shape == (B,)


def test_troop_loss_on_failure():
    """Attacker always loses fixed cost; extra loss on failure."""
    model = CombatModel(CombatParams(
        k=0.01,          # nearly random
        delta=100.0,     # extreme disadvantage → almost always fail
        noise_sigma=0.0,
        attack_cost_frac=0.1,
        failure_drain_frac=0.2,
    ))
    rng = np.random.default_rng(99)
    S_att = 10.0
    S_def = 0.0
    # Even with S_def=0, delta is so large that P(success) ≈ 0
    # Run many trials to check at least some failures
    total_success = 0
    for _ in range(100):
        success, new_att, _ = model.resolve(S_att, S_def, rng)
        if not success:
            expected_remaining = S_att * (1 - 0.1 - 0.2)
            assert abs(new_att - expected_remaining) < 1e-9
        total_success += int(success)
    # Should rarely succeed
    assert total_success < 50, "Expected mostly failures with delta=100"
