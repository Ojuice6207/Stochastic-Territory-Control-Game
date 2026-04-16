"""
Core data structures for the Stochastic Territory Control Game.

All hot-path numerical state lives in struct-of-arrays (SoA) layout
inside GameState so NumPy vectorisation never touches Python objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------

UNOWNED: int = -1  # sentinel: node belongs to no player


# ---------------------------------------------------------------------------
# Parameter containers (frozen after construction)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeParams:
    """
    Per-node Ornstein-Uhlenbeck parameters.

    Each territory has its own (theta, mu, sigma) triple so the game
    designer can create "volatile" vs "stable" regions on the map.

    Attributes
    ----------
    theta : float
        Mean-reversion speed (1/time). Half-life = ln(2)/theta.
        Typical range: [0.05, 2.0].
    mu : float
        Long-run equilibrium territory value.
    sigma : float
        Instantaneous volatility (units: value / sqrt(time)).
    """
    theta: float
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.theta <= 0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")


@dataclass(frozen=True)
class CombatParams:
    """
    Logistic combat model hyper-parameters.

    P(success | S_att, S_def) = sigmoid( k*(S_att - S_def - delta) + eps )
    where eps ~ N(0, noise_sigma^2) is injected into the linear predictor.

    Attributes
    ----------
    k : float
        Steepness of the logistic curve.  High k ≈ deterministic (strong
        beats weak); low k ≈ random.  Typical range: [0.5, 5.0].
    delta : float
        Defender home-advantage bias.  Positive delta penalises the
        attacker: equal-strength attack (ΔS=0) has P < 0.5.
    noise_sigma : float
        Standard deviation of battlefield noise injected into η = k*(ΔS -
        delta) before the sigmoid.  0 = pure logistic.
    attack_cost_frac : float
        Fraction of attacker troops lost regardless of outcome.
    failure_drain_frac : float
        Additional fraction of attacker troops lost on a failed attack.
    """
    k: float = 1.5
    delta: float = 0.3
    noise_sigma: float = 0.2
    attack_cost_frac: float = 0.15
    failure_drain_frac: float = 0.25

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be > 0, got {self.k}")


# ---------------------------------------------------------------------------
# Central mutable game state (struct-of-arrays)
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Complete, authoritative game state.  All arrays are indexed by node_id.

    Stored in struct-of-arrays (SoA) layout so that stochastic updates,
    ownership checks, and EV scans are fully vectorised over N nodes.

    Parameters
    ----------
    n_nodes : int
        Total number of territory nodes.
    n_players : int
        Number of competing players.
    ou_params : List[NodeParams]
        Per-node OU parameters, length n_nodes.
    initial_values : np.ndarray, shape (n_nodes,)
        Starting territory values V_0.
    initial_strengths : np.ndarray, shape (n_nodes,)
        Starting troop strengths S_0.
    initial_owners : np.ndarray[int], shape (n_nodes,)
        Starting owners (-1 = neutral).
    rng : np.random.Generator
        Seeded RNG for reproducibility.
    """

    # Dimensions
    n_nodes: int
    n_players: int

    # Per-node OU parameter arrays (populated from NodeParams list)
    ou_theta: np.ndarray  # shape (n_nodes,)
    ou_mu: np.ndarray     # shape (n_nodes,)
    ou_sigma: np.ndarray  # shape (n_nodes,)

    # True (hidden) state arrays — only the environment reads these directly
    true_values: np.ndarray    # shape (n_nodes,), float64
    true_strengths: np.ndarray # shape (n_nodes,), float64
    owners: np.ndarray         # shape (n_nodes,), int32

    # Simulation clock
    t: float = 0.0
    step_count: int = 0

    # Seeded RNG (passed in so outer loop controls reproducibility)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng()
    )

    @classmethod
    def build(
        cls,
        ou_params: List[NodeParams],
        initial_values: np.ndarray,
        initial_strengths: np.ndarray,
        initial_owners: np.ndarray,
        n_players: int,
        rng: np.random.Generator,
    ) -> "GameState":
        """
        Factory that unpacks NodeParams into contiguous arrays.

        This is the only correct way to construct GameState — it
        guarantees array dtype and shape consistency.
        """
        n = len(ou_params)
        assert initial_values.shape == (n,), "values shape mismatch"
        assert initial_strengths.shape == (n,), "strengths shape mismatch"
        assert initial_owners.shape == (n,), "owners shape mismatch"

        theta = np.array([p.theta for p in ou_params], dtype=np.float64)
        mu    = np.array([p.mu    for p in ou_params], dtype=np.float64)
        sigma = np.array([p.sigma for p in ou_params], dtype=np.float64)

        return cls(
            n_nodes=n,
            n_players=n_players,
            ou_theta=theta,
            ou_mu=mu,
            ou_sigma=sigma,
            true_values=initial_values.astype(np.float64).copy(),
            true_strengths=initial_strengths.astype(np.float64).copy(),
            owners=initial_owners.astype(np.int32).copy(),
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def owned_by(self, player: int) -> np.ndarray:
        """Boolean mask of nodes owned by `player`."""
        return self.owners == player

    def total_value(self, player: int) -> float:
        """Sum of true territory values for `player`."""
        return float(self.true_values[self.owned_by(player)].sum())

    def __repr__(self) -> str:
        scores = {
            p: round(self.total_value(p), 2) for p in range(self.n_players)
        }
        return (
            f"GameState(t={self.t:.3f}, step={self.step_count}, "
            f"n_nodes={self.n_nodes}, scores={scores})"
        )


# ---------------------------------------------------------------------------
# Observation type returned to agents each step
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    What a single player observes at the start of each turn.

    Owned territory values/strengths are observed exactly.
    Neighbouring *enemy* territories yield noisy observations.
    Non-adjacent enemy territories are fully hidden (NaN).

    Attributes
    ----------
    player_id : int
    step : int
    noisy_values : np.ndarray, shape (n_nodes,)
        NaN where not observable.
    noisy_strengths : np.ndarray, shape (n_nodes,)
        NaN where not observable.
    visible_owners : np.ndarray[int], shape (n_nodes,)
        -2 where unknown.
    combat_result : Tuple[int, int, bool] | None
        (attacker_node, defender_node, success) from last action, or None.
    """
    player_id: int
    step: int
    noisy_values: np.ndarray
    noisy_strengths: np.ndarray
    visible_owners: np.ndarray
    combat_result: Tuple[int, int, bool] | None = None
