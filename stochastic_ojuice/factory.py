"""
Factory helpers for common game configurations.

Provides ready-to-run setups so experiments can start with a single call
rather than 50 lines of boilerplate.  These are reference configurations —
copy and mutate them for custom map designs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from .types import NodeParams, CombatParams, GameState
from .environment import GameEnvironment
from .agent import PlayerAgent, AgentConfig


# ---------------------------------------------------------------------------
# Map generators
# ---------------------------------------------------------------------------

def make_grid_map(rows: int, cols: int) -> nx.DiGraph:
    """
    N×M grid graph with bidirectional edges.
    Node id = row*cols + col.
    """
    G = nx.grid_2d_graph(rows, cols)
    # Relabel to integer ids
    mapping = {(r, c): r * cols + c for r, c in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return nx.DiGraph(G)


def make_random_map(
    n_nodes: int,
    edge_prob: float = 0.15,
    seed: int = 42,
) -> nx.DiGraph:
    """
    Erdős-Rényi random map.  Re-sampled until the graph is strongly connected.
    """
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=int(rng.integers(1e6)), directed=True)
        if nx.is_strongly_connected(G):
            return G
    raise RuntimeError(
        f"Could not generate a strongly connected Erdős-Rényi graph "
        f"with n={n_nodes}, p={edge_prob} after 1000 attempts.  Increase p."
    )


def make_ring_map(n_nodes: int) -> nx.DiGraph:
    """Simple ring topology — each node attacks only its neighbours."""
    G = nx.cycle_graph(n_nodes, create_using=nx.DiGraph)
    # Make bidirectional
    G.add_edges_from([(v, u) for u, v in G.edges()])
    return G


# ---------------------------------------------------------------------------
# Default OU parameter presets
# ---------------------------------------------------------------------------

def homogeneous_ou_params(
    n_nodes: int,
    theta: float = 0.5,
    mu: float = 10.0,
    sigma: float = 2.0,
) -> List[NodeParams]:
    """All nodes share the same OU parameters."""
    return [NodeParams(theta=theta, mu=mu, sigma=sigma) for _ in range(n_nodes)]


def heterogeneous_ou_params(
    n_nodes: int,
    rng: np.random.Generator,
) -> List[NodeParams]:
    """
    Random OU parameters for each node — creates 'stable anchor' vs
    'volatile frontier' territory archetypes.
    """
    thetas = rng.uniform(0.1, 2.0, n_nodes)
    mus    = rng.uniform(5.0, 20.0, n_nodes)
    sigmas = rng.uniform(0.5, 4.0, n_nodes)
    return [NodeParams(theta=float(t), mu=float(m), sigma=float(s))
            for t, m, s in zip(thetas, mus, sigmas)]


# ---------------------------------------------------------------------------
# Full environment factory
# ---------------------------------------------------------------------------

def make_game(
    graph: nx.DiGraph,
    n_players: int = 2,
    ou_params: Optional[List[NodeParams]] = None,
    combat_params: Optional[CombatParams] = None,
    dt: float = 0.1,
    seed: int = 0,
    obs_value_noise_std: float = 0.5,
    obs_strength_noise_std: float = 1.0,
    max_steps: int = 500,
) -> Tuple[GameEnvironment, GameState]:
    """
    Construct a GameEnvironment with a balanced initial state.

    Initial conditions:
    - Territory values sampled from each node's OU stationary distribution.
    - Troop strengths uniform in [3, 8].
    - Territories divided as evenly as possible between players.

    Parameters
    ----------
    graph       : map topology (use make_grid_map / make_random_map etc.)
    n_players   : number of competing agents
    ou_params   : per-node OU params (default: homogeneous θ=0.5, μ=10, σ=2)
    combat_params : (default: CombatParams())
    dt          : time step
    seed        : RNG seed for full reproducibility
    ...

    Returns
    -------
    env   : GameEnvironment
    state : GameState  (same object as env.state)
    """
    rng    = np.random.default_rng(seed)
    n      = graph.number_of_nodes()

    ou     = ou_params    or homogeneous_ou_params(n)
    combat = combat_params or CombatParams()

    # Initial values from stationary OU distribution  N(mu, sigma^2/(2*theta))
    init_values = np.array([
        rng.normal(p.mu, np.sqrt(p.sigma**2 / (2*p.theta)))
        for p in ou
    ])

    init_strengths = rng.uniform(3.0, 8.0, n)

    # Divide nodes round-robin by player
    init_owners = np.full(n, -1, dtype=np.int32)
    nodes = np.arange(n)
    rng.shuffle(nodes)
    for idx, node in enumerate(nodes):
        init_owners[node] = idx % n_players

    state = GameState.build(
        ou_params=ou,
        initial_values=init_values,
        initial_strengths=init_strengths,
        initial_owners=init_owners,
        n_players=n_players,
        rng=rng,
    )

    env = GameEnvironment(
        graph=graph,
        state=state,
        combat_params=combat,
        dt=dt,
        obs_value_noise_std=obs_value_noise_std,
        obs_strength_noise_std=obs_strength_noise_std,
        max_steps=max_steps,
    )
    return env, state


def make_agents(
    env: GameEnvironment,
    agent_config: Optional[AgentConfig] = None,
    seed: int = 1,
) -> List[PlayerAgent]:
    """
    Construct one PlayerAgent per player, all sharing the same AgentConfig.

    Returns a list indexed by player_id.
    """
    cfg  = agent_config or AgentConfig()
    rng  = np.random.default_rng(seed)
    s    = env.state

    # Build the agent's assumed OU params (here: same as true — no model mis-spec)
    ou_params_belief: Dict[int, Tuple[float, float, float]] = {
        i: (float(s.ou_theta[i]), float(s.ou_mu[i]), float(s.ou_sigma[i]))
        for i in range(s.n_nodes)
    }

    agents = []
    for pid in range(env.n_players):
        agents.append(PlayerAgent(
            player_id=pid,
            graph=env.graph,
            combat_params=env._combat.params,
            agent_config=cfg,
            ou_params_belief=ou_params_belief,
            rng=np.random.default_rng(seed + pid),
        ))
    return agents
