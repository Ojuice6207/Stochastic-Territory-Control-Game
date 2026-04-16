"""
Game Environment
================

The authoritative simulation engine.  Holds the *true* game state, runs
the OU stochastic updates on every step, resolves combat, and produces
per-player Observation objects.

Design principles
-----------------
- GameEnvironment is the *only* object that reads `GameState.true_*` arrays.
  Agents never access ground truth; they only see Observation objects.
- All per-turn stochastic updates are vectorised over N nodes in O(1) NumPy
  calls, so even 10,000-node maps run at acceptable speeds.
- The step() method follows the OpenAI Gym / PettingZoo interface contract
  (action → observation, reward, done, info) so this engine slots directly
  into RL training loops.

Graph topology
--------------
The map is a networkx.DiGraph G = (V, E).  Node integer IDs must be
contiguous [0, N-1] and must match the indices of GameState arrays.
Directed edges model the *valid attack directions* (e.g., coastlines may
only be attackable from one side).

Action format
-------------
An action is a dict with a single key:
  {"wait": None}
  {"attack": (from_node: int, to_node: int)}
  {"reinforce": (node: int, strength_delta: float)}   # future extension
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .types import GameState, Observation, CombatParams, UNOWNED
from .stochastic import OUProcess
from .combat import CombatModel


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def wait_action() -> Dict[str, Any]:
    return {"wait": None}

def attack_action(from_node: int, to_node: int) -> Dict[str, Any]:
    return {"attack": (from_node, to_node)}


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class GameEnvironment:
    """
    Stochastic Territory Control Game — authoritative simulation engine.

    Parameters
    ----------
    graph : nx.DiGraph
        Territory adjacency.  Node IDs must be integers in [0, N-1].
    state : GameState
        Initial (and ongoing) true state.
    combat_params : CombatParams
    dt : float
        Time increment per game step (same units as OU theta^{-1}).
    obs_value_noise_std : float
        σ_obs for value observations (added to true value on adjacency).
    obs_strength_noise_std : float
        σ_obs for strength observations (enemy adjacent nodes).
    max_steps : int
        Episode length before forced termination.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        state: GameState,
        combat_params: CombatParams,
        *,
        dt: float = 0.1,
        obs_value_noise_std: float = 0.5,
        obs_strength_noise_std: float = 1.0,
        max_steps: int = 500,
    ) -> None:
        self.graph   = graph
        self.state   = state
        self.dt      = dt
        self.obs_value_noise_std    = obs_value_noise_std
        self.obs_strength_noise_std = obs_strength_noise_std
        self.max_steps = max_steps

        # Build stochastic engine from GameState OU arrays
        self._ou = OUProcess(
            theta=state.ou_theta,
            mu=state.ou_mu,
            sigma=state.ou_sigma,
        )
        self._combat = CombatModel(combat_params)

        # Pre-compute adjacency as a sparse int array for fast neighbour lookups
        self._adj: Dict[int, List[int]] = {
            n: list(graph.successors(n)) for n in graph.nodes()
        }

        # History for analysis / replay
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface (Gym-compatible)
    # ------------------------------------------------------------------

    def reset(self) -> List[Observation]:
        """
        Reset the simulation clock (NOT the state values — use GameState.build
        to construct fresh initial conditions).

        Returns initial observations for all players.
        """
        self.state.t = 0.0
        self.state.step_count = 0
        self._history.clear()
        return self._make_observations(combat_results=None)

    def step(
        self,
        actions: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[Observation], Dict[int, float], bool, Dict[str, Any]]:
        """
        Advance the game by one time-step dt.

        Processing order per step:
        1. Stochastic value update (all nodes, vectorised).
        2. Resolve each player's action in a random order (avoids turn-order bias).
        3. Generate per-player observations.
        4. Compute per-player rewards (Δ total owned value).
        5. Check terminal conditions.

        Parameters
        ----------
        actions : dict mapping player_id → action dict

        Returns
        -------
        observations : List[Observation], one per player, sorted by player_id
        rewards      : Dict[player_id → float]
        done         : bool
        info         : dict with debug metadata
        """
        prev_scores = {
            p: self.state.total_value(p) for p in range(self.state.n_players)
        }

        # 1. Stochastic value update (vectorised, exact OU step)
        self.stochastic_update()

        # 2. Resolve actions in random order
        combat_results: Dict[int, Tuple[int, int, bool]] = {}
        player_order = list(actions.keys())
        self.state.rng.shuffle(player_order)

        for pid in player_order:
            action = actions.get(pid, wait_action())
            result = self._resolve_player_action(pid, action)
            if result is not None:
                combat_results[pid] = result

        # 3. Advance clocks
        self.state.t += self.dt
        self.state.step_count += 1

        # 4. Generate observations
        observations = self._make_observations(combat_results=combat_results)

        # 5. Rewards = change in total owned value
        rewards = {
            p: self.state.total_value(p) - prev_scores[p]
            for p in range(self.state.n_players)
        }

        # 6. Terminal check
        done = self._check_terminal()

        info = {
            "step": self.state.step_count,
            "t": self.state.t,
            "combat_results": combat_results,
            "owners": self.state.owners.copy(),
            "true_values": self.state.true_values.copy(),
        }
        self._history.append(info)

        return observations, rewards, done, info

    # ------------------------------------------------------------------
    # Stochastic update (vectorised OU step over all nodes)
    # ------------------------------------------------------------------

    def stochastic_update(self) -> None:
        """
        Apply one exact OU step to all territory values simultaneously.

        V_{t+dt}[i] = mu[i] + phi[i]*(V_t[i] - mu[i]) + sigma_dt[i]*Z[i]

        where Z[i] ~ N(0,1) are independent across nodes.
        This single NumPy call handles all N nodes with no Python loop.
        """
        self.state.true_values = self._ou.step(
            self.state.true_values,
            dt=self.dt,
            rng=self.state.rng,
        )

    # ------------------------------------------------------------------
    # Combat resolution
    # ------------------------------------------------------------------

    def resolve_combat(
        self,
        attacker_id: int,
        from_node: int,
        to_node: int,
    ) -> bool:
        """
        Resolve a single combat engagement between two adjacent nodes.

        Guards:
        - from_node must be owned by attacker_id.
        - to_node must be adjacent (edge exists in graph).
        - to_node must not be owned by attacker_id.
        - from_node must have positive strength.

        Side-effects on GameState:
        - On success: to_node.owner = attacker_id, to_node.strength = 0.
          Attacker loses fixed cost.
        - On failure: Attacker loses fixed + failure cost.

        Returns
        -------
        success : bool
        """
        s = self.state

        # Validity checks
        if s.owners[from_node] != attacker_id:
            raise ValueError(f"Player {attacker_id} does not own node {from_node}.")
        if to_node not in self._adj[from_node]:
            raise ValueError(f"No edge {from_node} → {to_node}.")
        if s.owners[to_node] == attacker_id:
            raise ValueError(f"Node {to_node} already owned by player {attacker_id}.")
        if s.true_strengths[from_node] <= 0:
            return False  # No troops to attack with

        success, new_att_str, new_def_str = self._combat.resolve(
            S_att=float(s.true_strengths[from_node]),
            S_def=float(s.true_strengths[to_node]),
            rng=s.rng,
        )

        # Apply state transitions
        s.true_strengths[from_node] = new_att_str
        if success:
            s.true_strengths[to_node] = new_att_str * 0.3  # move partial force in
            s.owners[to_node] = attacker_id
        else:
            s.true_strengths[to_node] = new_def_str

        return success

    # ------------------------------------------------------------------
    # Monte Carlo EV estimation  (for agent planning)
    # ------------------------------------------------------------------

    def monte_carlo_ev(
        self,
        from_node: int,
        to_node: int,
        S_att: float,
        S_def_mean: float,
        S_def_std: float,
        V_def_mean: float,
        n_scenarios: int = 2000,
        horizon: int = 20,
    ) -> Dict[str, float]:
        """
        Estimate the Expected Value of attacking `to_node` from `from_node`
        via Monte Carlo simulation over a planning horizon.

        This is used by planning agents when analytical EV is insufficient
        (e.g., multi-step chain attacks, overextension risk).

        Parameters
        ----------
        S_att       : float — attacker's current known strength
        S_def_mean  : float — posterior mean of defender strength
        S_def_std   : float — posterior std of defender strength
        V_def_mean  : float — Kalman posterior mean of territory value
        n_scenarios : int   — Monte Carlo sample count
        horizon     : int   — steps to simulate after capture

        Returns
        -------
        dict with keys: ev, p_success, ev_given_success, ev_given_failure,
                        value_std (uncertainty in the value estimate)
        """
        rng = self.state.rng  # use shared RNG for reproducibility

        # Sample defender strengths from posterior
        S_def_samples = np.maximum(
            0.1,
            rng.normal(S_def_mean, S_def_std, n_scenarios),
        )

        # Resolve combat for all scenarios simultaneously
        S_att_arr = np.full(n_scenarios, S_att)
        successes, _, _ = self._combat.resolve_batch(S_att_arr, S_def_samples, rng)

        p_success = float(successes.mean())

        # Simulate OU value evolution for the territory over the horizon
        # (only relevant if we capture it)
        V0_arr = np.full(n_scenarios, V_def_mean)  # start from current estimate
        if horizon > 0:
            paths = self._ou_single_node(to_node, V0_arr, horizon, rng)
            # Total discounted value collected if captured (simple sum for now)
            discount = np.exp(-0.05 * self.dt * np.arange(1, horizon + 1))
            terminal_values = (paths[:, 1:] * discount[np.newaxis, :]).sum(axis=1)
        else:
            terminal_values = V0_arr

        ev_given_success = float(terminal_values[successes].mean()) if successes.any() else 0.0
        ev_given_failure = 0.0  # no value if attack fails

        ev = p_success * ev_given_success + (1 - p_success) * ev_given_failure
        value_std = float(terminal_values[successes].std()) if successes.any() else 0.0

        return {
            "ev": ev,
            "p_success": p_success,
            "ev_given_success": ev_given_success,
            "ev_given_failure": ev_given_failure,
            "value_std": value_std,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_player_action(
        self,
        player_id: int,
        action: Dict[str, Any],
    ) -> Optional[Tuple[int, int, bool]]:
        """Execute one player's action.  Returns combat result tuple or None."""
        if "attack" in action:
            from_node, to_node = action["attack"]
            try:
                success = self.resolve_combat(player_id, from_node, to_node)
                return (from_node, to_node, success)
            except ValueError:
                return None  # invalid action → silently ignored
        # "wait" and "reinforce" (extension point) are no-ops here
        return None

    def _make_observations(
        self,
        combat_results: Optional[Dict[int, Tuple[int, int, bool]]],
    ) -> List[Observation]:
        """
        Construct one Observation per player.

        Visibility rules:
        - Owned nodes: exact value and strength.
        - Adjacent enemy/neutral nodes: noisy value and noisy strength.
        - Non-adjacent nodes: NaN (fully hidden).
        """
        s = self.state
        obs_list = []
        rng = s.rng

        for pid in range(s.n_players):
            noisy_values    = np.full(s.n_nodes, np.nan)
            noisy_strengths = np.full(s.n_nodes, np.nan)
            visible_owners  = np.full(s.n_nodes, -2, dtype=np.int32)  # -2 = unknown

            owned_mask  = s.owners == pid
            owned_nodes = np.where(owned_mask)[0]

            # Exact observations for owned nodes
            noisy_values[owned_mask]    = s.true_values[owned_mask]
            noisy_strengths[owned_mask] = s.true_strengths[owned_mask]
            visible_owners[owned_mask]  = pid

            # Noisy observations for neighbours of owned nodes
            for n in owned_nodes:
                for nb in self._adj[n]:
                    if s.owners[nb] != pid:
                        # Add observation noise
                        noisy_values[nb] = (
                            s.true_values[nb]
                            + rng.normal(0.0, self.obs_value_noise_std)
                        )
                        noisy_strengths[nb] = max(
                            0.0,
                            s.true_strengths[nb]
                            + rng.normal(0.0, self.obs_strength_noise_std),
                        )
                        visible_owners[nb] = s.owners[nb]

            cr = combat_results.get(pid) if combat_results else None
            obs_list.append(Observation(
                player_id=pid,
                step=s.step_count,
                noisy_values=noisy_values,
                noisy_strengths=noisy_strengths,
                visible_owners=visible_owners,
                combat_result=cr,
            ))

        return obs_list

    def _check_terminal(self) -> bool:
        """Episode ends when one player owns all nodes or max_steps reached."""
        if self.state.step_count >= self.max_steps:
            return True
        for pid in range(self.state.n_players):
            if np.all(self.state.owners == pid):
                return True
        return False

    def _ou_single_node(
        self,
        node_id: int,
        V0_batch: np.ndarray,
        n_steps: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate OU paths for a single node across B scenarios.

        Returns paths of shape (B, n_steps+1).
        """
        theta = float(self.state.ou_theta[node_id])
        mu    = float(self.state.ou_mu[node_id])
        sigma = float(self.state.ou_sigma[node_id])

        phi = np.exp(-theta * self.dt)
        Q   = (sigma**2 / (2*theta)) * (1 - phi**2) if theta > 1e-10 else sigma**2 * self.dt
        sd  = np.sqrt(Q)

        B = V0_batch.shape[0]
        paths = np.empty((B, n_steps + 1))
        paths[:, 0] = V0_batch
        for t in range(n_steps):
            z = rng.standard_normal(B)
            paths[:, t+1] = mu + phi * (paths[:, t] - mu) + sd * z
        return paths

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.state.n_nodes

    @property
    def n_players(self) -> int:
        return self.state.n_players

    def __repr__(self) -> str:
        return (
            f"GameEnvironment(nodes={self.n_nodes}, players={self.n_players}, "
            f"dt={self.dt}, step={self.state.step_count})"
        )
