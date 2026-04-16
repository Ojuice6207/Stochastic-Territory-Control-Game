"""
Player Agent: POMDP Belief State + EV-Based Decision Making
============================================================

PlayerAgent maintains a private *belief state* — a factorised
approximation to the POMDP posterior — and uses it to select actions
via an Expected Value (EV) heuristic.

Belief state components (per observable node):
  - KalmanBeliefNode    : Gaussian filter on territory value V_i
  - ParticleStrengthEst : Particle filter on enemy strength S_j

On each game step the agent:
  1. Receives an Observation from the environment.
  2. Updates beliefs (Kalman predict+update for values; particle
     predict+weight_update+resample for enemy strengths).
  3. Evaluates EV(attack i→j) for all adjacent enemy nodes.
  4. Selects the highest-EV action above a minimum threshold, else waits.

Expected Value formula
----------------------
For an attack from owned node i to enemy node j:

  EV(i→j) = P̂_ij · V̂_j  -  (1 - P̂_ij) · C_fail  -  C_fixed  -  λ·Π_ij

Where:
  P̂_ij    = sigmoid( k·(S_i - Ŝ_j_mean - δ) ) · noise_correction
  V̂_j     = Kalman posterior mean of territory j's value
  C_fail   = failure_drain_frac · S_i        (expected troop loss on failure)
  C_fixed  = attack_cost_frac · S_i          (always paid)
  Π_ij     = overextension penalty: expected value of losing node i
             if its defending troops drop below a safe threshold

The agent is risk-neutral by default (lambda_overext=0) but can be set
risk-averse or risk-seeking by tuning lambda_overext and explore_bonus.

This agent can serve as:
  - A strong rule-based baseline for RL agent evaluation.
  - A human-interpretable policy for debugging game balance.
  - The starting policy for RL fine-tuning (behaviour cloning init).
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .types import Observation, GameState, CombatParams, UNOWNED
from .combat import CombatModel
from .belief import KalmanBeliefNode, ParticleStrengthEstimator, NodeBelief


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """
    Hyper-parameters governing the agent's decision-making policy.

    Attributes
    ----------
    min_ev_to_attack : float
        Minimum EV (in value units) required to choose an attack over wait.
        Higher = more conservative.
    lambda_overext : float
        Weight on the overextension penalty.  0 = ignore risk of losing
        home node.  Positive = risk-averse.
    explore_bonus : float
        Bonus per unit of posterior std on territory value V_j.
        Positive = risk-seeking / exploration-focused (good for RL warm-up).
    n_particles : int
        Particle count for each enemy-strength estimator.
    safe_strength_frac : float
        If attacking would leave home node with fewer than
        safe_strength_frac * current_strength troops, apply overextension
        penalty.
    obs_value_noise_var : float
        Assumed observation noise variance for the Kalman filter (should
        match the environment's obs_value_noise_std^2).
    """
    min_ev_to_attack: float = 0.5
    lambda_overext: float = 0.3
    explore_bonus: float = 0.0
    n_particles: int = 500
    safe_strength_frac: float = 0.4
    obs_value_noise_var: float = 0.25   # (0.5)^2


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class PlayerAgent:
    """
    POMDP-based territory control agent.

    Parameters
    ----------
    player_id : int
    graph : nx.DiGraph
        Reference to the shared map topology (read-only).
    combat_params : CombatParams
    agent_config : AgentConfig
    ou_params_belief : dict mapping node_id → (theta, mu, sigma)
        The agent's *assumed* OU parameters for each node.  May differ
        from the true parameters (model mismatch → richer Bayesian inference
        is needed in a full system, but here we assume correct model).
    rng : np.random.Generator
    """

    def __init__(
        self,
        player_id: int,
        graph: nx.DiGraph,
        combat_params: CombatParams,
        agent_config: AgentConfig,
        ou_params_belief: Dict[int, Tuple[float, float, float]],
        rng: np.random.Generator,
    ) -> None:
        self.player_id = player_id
        self.graph     = graph
        self.config    = agent_config
        self.rng       = rng
        self._adj: Dict[int, List[int]] = {
            n: list(graph.successors(n)) for n in graph.nodes()
        }
        self._combat = CombatModel(combat_params)

        # Agent's internal belief about each node (initialised lazily)
        self._beliefs: Dict[int, NodeBelief] = {}
        self._ou_params = ou_params_belief   # {node_id: (theta, mu, sigma)}

        # Track what the agent thinks it owns
        self._owned_nodes: List[int] = []

        # Log of own strength per node (known exactly for owned nodes)
        self._own_strengths: Dict[int, float] = {}

        # Step counter (for belief freshness tracking)
        self._step: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def observe(
        self,
        obs: Observation,
        dt: float,
    ) -> None:
        """
        Ingest a new Observation and update all belief state components.

        Call this *before* act() on each game step.

        Parameters
        ----------
        obs : Observation
            Produced by GameEnvironment._make_observations().
        dt  : float
            Time elapsed this step (for Kalman/OU dynamics propagation).
        """
        self._step = obs.step
        self._owned_nodes = [
            i for i, owner in enumerate(obs.visible_owners)
            if owner == self.player_id
        ]
        # Update exact own strengths
        for n in self._owned_nodes:
            if not np.isnan(obs.noisy_strengths[n]):
                self._own_strengths[n] = float(obs.noisy_strengths[n])

        # Update beliefs for all observed nodes
        n_nodes = len(obs.noisy_values)
        for node_id in range(n_nodes):
            v_obs = obs.noisy_values[node_id]
            s_obs = obs.noisy_strengths[node_id]
            owner = int(obs.visible_owners[node_id])

            if np.isnan(v_obs):
                # Node not visible: predict-only (no measurement update)
                if node_id in self._beliefs:
                    self._beliefs[node_id].value_kf.predict(dt)
                    self._beliefs[node_id].strength_pf.predict(self.rng)
                continue

            # Ensure belief object exists
            belief = self._get_or_create_belief(node_id, v_obs, s_obs)

            # --- Kalman update for territory value ---
            belief.value_kf.step(dt, z=v_obs)

            # --- Particle filter update for enemy strength ---
            if owner != self.player_id and not np.isnan(s_obs):
                # Prediction step (strength can drift)
                belief.strength_pf.predict(self.rng)
                # If we have a direct (noisy) observation of strength,
                # build a Gaussian likelihood around it
                log_liks = self._noisy_strength_log_likelihood(
                    belief.strength_pf.particles, s_obs
                )
                belief.strength_pf.update(log_liks)
                belief.strength_pf.resample_if_needed(self.rng)

            belief.last_seen_step = self._step

        # Incorporate combat outcome into particle filters
        if obs.combat_result is not None:
            from_node, to_node, success = obs.combat_result
            self._update_beliefs_from_combat(from_node, to_node, success)

    def act(self) -> Dict[str, Any]:
        """
        Select an action based on current belief state.

        Returns
        -------
        action dict suitable for GameEnvironment.step().

        Decision procedure:
        1. For each owned node i, for each enemy-adjacent node j:
           Compute EV(i→j).
        2. Select (i*, j*) = argmax EV.
        3. If EV(i*, j*) > min_ev_to_attack → return attack(i*, j*)
           Else → return wait.
        """
        best_ev   = self.config.min_ev_to_attack  # minimum threshold
        best_from = None
        best_to   = None

        ev_table: List[Tuple[int, int, float]] = []

        for from_node in self._owned_nodes:
            S_att = self._own_strengths.get(from_node, 1.0)
            if S_att <= 0:
                continue

            for to_node in self._adj.get(from_node, []):
                # Skip own nodes
                if to_node in self._owned_nodes:
                    continue

                ev = self._compute_ev(from_node, to_node, S_att)
                ev_table.append((from_node, to_node, ev))

                if ev > best_ev:
                    best_ev   = ev
                    best_from = from_node
                    best_to   = to_node

        if best_from is not None:
            return {"attack": (best_from, best_to)}
        return {"wait": None}

    def ev_table(self) -> List[Tuple[int, int, float]]:
        """
        Return the full EV table for all adjacent attacks.
        Useful for visualisation and debugging agent decision-making.
        """
        table = []
        for from_node in self._owned_nodes:
            S_att = self._own_strengths.get(from_node, 1.0)
            for to_node in self._adj.get(from_node, []):
                if to_node not in self._owned_nodes:
                    ev = self._compute_ev(from_node, to_node, S_att)
                    table.append((from_node, to_node, round(ev, 4)))
        return sorted(table, key=lambda x: -x[2])

    def belief_summary(self) -> Dict[int, Dict[str, float]]:
        """
        Return a human-readable summary of all current beliefs.

        Returns dict: node_id → {value_mean, value_std, strength_mean, strength_std}
        """
        out = {}
        for nid, b in self._beliefs.items():
            out[nid] = {
                "value_mean":    round(b.value_mean(), 3),
                "value_std":     round(np.sqrt(max(b.value_var(), 0)), 3),
                "strength_mean": round(b.strength_mean(), 3),
                "strength_std":  round(b.strength_std(), 3),
                "last_seen":     b.last_seen_step,
            }
        return out

    # ------------------------------------------------------------------
    # EV computation
    # ------------------------------------------------------------------

    def _compute_ev(
        self,
        from_node: int,
        to_node: int,
        S_att: float,
    ) -> float:
        """
        EV(i→j) = P̂ · V̂_j  -  (1-P̂) · C_fail  -  C_fixed  -  λ · Π_ij
                + explore_bonus · σ(V̂_j)

        All quantities derived from the agent's current belief state.
        """
        # Posterior belief for target node
        belief_j = self._beliefs.get(to_node)
        V_j_mean = belief_j.value_mean() if belief_j else self._ou_params.get(to_node, (0.1, 5.0, 1.0))[1]
        V_j_std  = np.sqrt(belief_j.value_var()) if belief_j else 2.0
        S_j_mean = belief_j.strength_mean() if belief_j else 5.0

        # Combat success probability using posterior mean of enemy strength
        # (conservative: ignores uncertainty in S_j which would typically
        #  *increase* P̂ via Jensen's inequality through the S-shaped sigmoid)
        P_hat = float(self._combat.success_probability(
            S_att, S_j_mean, include_noise_variance=True
        ))

        # Costs
        C_fixed = self._combat.params.attack_cost_frac * S_att
        C_fail  = self._combat.params.failure_drain_frac * S_att

        # Overextension penalty
        overext = self._overextension_penalty(from_node, S_att)

        # Exploration bonus (optional: reward high-uncertainty targets)
        bonus = self.config.explore_bonus * V_j_std

        ev = (
            P_hat * V_j_mean
            - (1.0 - P_hat) * C_fail
            - C_fixed
            - self.config.lambda_overext * overext
            + bonus
        )
        return float(ev)

    def _overextension_penalty(
        self,
        from_node: int,
        S_att: float,
    ) -> float:
        """
        Expected value lost if from_node becomes dangerously weak after attack.

        Penalty = P(from_node recaptured) × V̂_from

        P(recapture) = max over adjacent enemy nodes k of
                       P(k wins against from_node after strength depletion).
        """
        safe_threshold = self.config.safe_strength_frac * S_att
        # Troops left in from_node after attacker moves out
        troops_remaining = S_att * (
            1.0 - self._combat.params.attack_cost_frac
        )

        if troops_remaining >= safe_threshold:
            return 0.0

        # Find maximum threat from adjacent enemies
        max_p_recapture = 0.0
        for nb in self._adj.get(from_node, []):
            if nb not in self._owned_nodes:
                nb_belief = self._beliefs.get(nb)
                S_nb = nb_belief.strength_mean() if nb_belief else 5.0
                p_rc = float(self._combat.success_probability(
                    S_nb, troops_remaining, include_noise_variance=True
                ))
                max_p_recapture = max(max_p_recapture, p_rc)

        belief_from = self._beliefs.get(from_node)
        V_from = belief_from.value_mean() if belief_from else 5.0
        return max_p_recapture * V_from

    # ------------------------------------------------------------------
    # Belief maintenance helpers
    # ------------------------------------------------------------------

    def _get_or_create_belief(
        self,
        node_id: int,
        v_obs: float,
        s_obs: float,
    ) -> NodeBelief:
        """Lazily initialise NodeBelief on first observation of a node."""
        if node_id not in self._beliefs:
            theta, mu, sigma = self._ou_params.get(node_id, (0.5, 5.0, 1.0))
            stationary_var   = sigma**2 / (2.0 * theta)

            kf = KalmanBeliefNode(
                theta=theta,
                mu=mu,
                sigma=sigma,
                obs_noise_var=self.config.obs_value_noise_var,
                m=float(v_obs) if not np.isnan(v_obs) else mu,
                P=stationary_var,
            )
            pf = ParticleStrengthEstimator(
                n_particles=self.config.n_particles,
                init_mean=float(s_obs) if not np.isnan(s_obs) else 5.0,
                init_std=2.0,
            )
            pf.reset(
                mean=float(s_obs) if not np.isnan(s_obs) else 5.0,
                std=2.0,
                rng=self.rng,
            )
            self._beliefs[node_id] = NodeBelief(
                node_id=node_id,
                value_kf=kf,
                strength_pf=pf,
                last_seen_step=self._step,
            )
        return self._beliefs[node_id]

    def _update_beliefs_from_combat(
        self,
        from_node: int,
        to_node: int,
        success: bool,
    ) -> None:
        """
        Update the particle filter for `to_node` using the binary
        combat outcome as a likelihood signal.

        This is the key Bayesian inference step:
        After observing Y ∈ {0,1}, the posterior over S_def changes because
        the outcome is informative about the defender's strength.
        """
        if to_node not in self._beliefs:
            return

        belief_j = self._beliefs[to_node]
        S_att = self._own_strengths.get(from_node, 5.0)

        log_liks = self._combat.log_likelihood(
            outcome=success,
            S_att=S_att,
            S_def_particles=belief_j.strength_pf.particles,
        )
        belief_j.strength_pf.update(log_liks)
        belief_j.strength_pf.resample_if_needed(self.rng)

    def _noisy_strength_log_likelihood(
        self,
        particles: np.ndarray,
        s_obs: float,
    ) -> np.ndarray:
        """
        Gaussian likelihood for a direct (noisy) strength observation.

        log P(s_obs | S^(k)) = -0.5 * (s_obs - S^(k))^2 / obs_var + const
        """
        obs_var = (self.config.obs_value_noise_var * 4.0)  # strength noise slightly higher
        return -0.5 * (s_obs - particles)**2 / obs_var
