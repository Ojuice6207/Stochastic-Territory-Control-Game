"""
Demo: Run a complete game episode with two EV-maximising agents.

Demonstrates:
  - Environment construction via factory helpers
  - The step() loop with observation / belief update / action cycle
  - Live printout of belief state vs true state (to show POMDP inference)
  - Episode summary stats

Run:  python run_demo.py
"""
import numpy as np
import networkx as nx

from stochastic_ojuice.factory import make_game, make_agents, make_grid_map
from stochastic_ojuice.types import CombatParams
from stochastic_ojuice.agent import AgentConfig


def main() -> None:
    # ------------------------------------------------------------------ setup
    ROWS, COLS = 4, 5        # 20-node grid map
    N_PLAYERS  = 2
    SEED       = 42
    MAX_STEPS  = 200
    VERBOSE    = True
    PRINT_EVERY = 10         # print state every N steps

    graph = make_grid_map(ROWS, COLS)

    combat_params = CombatParams(
        k=1.5,
        delta=0.3,
        noise_sigma=0.2,
        attack_cost_frac=0.10,
        failure_drain_frac=0.20,
    )

    agent_config = AgentConfig(
        min_ev_to_attack=0.3,
        lambda_overext=0.25,
        explore_bonus=0.1,
        n_particles=300,
    )

    env, state = make_game(
        graph=graph,
        n_players=N_PLAYERS,
        combat_params=combat_params,
        dt=0.1,
        seed=SEED,
        obs_value_noise_std=0.5,
        obs_strength_noise_std=1.0,
        max_steps=MAX_STEPS,
    )

    agents = make_agents(env, agent_config=agent_config, seed=SEED + 10)

    # ------------------------------------------------------------------ init
    observations = env.reset()
    for agent, obs in zip(agents, observations):
        agent.observe(obs, dt=env.dt)

    print(f"Starting episode | {ROWS}×{COLS} grid | {N_PLAYERS} players | max_steps={MAX_STEPS}")
    print("-" * 65)

    cumulative_rewards = {p: 0.0 for p in range(N_PLAYERS)}

    # ------------------------------------------------------------------ loop
    done = False
    while not done:
        # Each agent decides based on its current belief state
        actions = {agent.player_id: agent.act() for agent in agents}

        # Environment step
        observations, rewards, done, info = env.step(actions)

        # Accumulate rewards
        for pid, r in rewards.items():
            cumulative_rewards[pid] += r

        # Update agent beliefs with new observations
        for agent, obs in zip(agents, observations):
            agent.observe(obs, dt=env.dt)

        step = info["step"]

        # ---------------------------------------------------------- printing
        if VERBOSE and step % PRINT_EVERY == 0:
            true_vals  = info["true_values"]
            owners     = info["owners"]
            scores     = {p: round(state.total_value(p), 2) for p in range(N_PLAYERS)}
            n_owned    = {p: int((owners == p).sum()) for p in range(N_PLAYERS)}

            print(f"\nStep {step:4d} | t={info['t']:.2f}")
            print(f"  Scores   : {scores}")
            print(f"  Territories: {n_owned}")

            # Show one agent's belief accuracy for a sample of nodes
            agent0 = agents[0]
            print(f"\n  Agent 0 belief vs ground truth (sample nodes):")
            print(f"  {'node':>4} | {'true_V':>8} | {'belief_V':>8} | {'err':>6} | "
                  f"{'true_S':>8} | {'belief_S':>8} | {'owner':>5}")
            print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}")
            sample_nodes = list(range(min(8, state.n_nodes)))
            for n in sample_nodes:
                bsummary = agent0.belief_summary().get(n, {})
                v_mean   = bsummary.get("value_mean", float("nan"))
                s_mean   = bsummary.get("strength_mean", float("nan"))
                tv       = true_vals[n]
                ts       = state.true_strengths[n]
                err      = abs(v_mean - tv)
                own_str  = str(owners[n]) if owners[n] >= 0 else "N"
                print(f"  {n:>4} | {tv:>8.3f} | {v_mean:>8.3f} | {err:>6.3f} | "
                      f"{ts:>8.3f} | {s_mean:>8.3f} | {own_str:>5}")

            # Show top EV candidates for agent 0
            ev_tbl = agent0.ev_table()[:3]
            if ev_tbl:
                print(f"\n  Agent 0 top attack candidates (EV):")
                for from_n, to_n, ev in ev_tbl:
                    print(f"    node {from_n:>2} -> node {to_n:>2}  |  EV={ev:+.4f}")

    # ----------------------------------------------------------------- summary
    print("\n" + "=" * 65)
    print("Episode complete.")
    print(f"  Final step: {state.step_count}")
    print(f"  Cumulative rewards: {cumulative_rewards}")
    final_scores = {p: round(state.total_value(p), 2) for p in range(N_PLAYERS)}
    n_owned      = {p: int((state.owners == p).sum()) for p in range(N_PLAYERS)}
    print(f"  Final territory counts: {n_owned}")
    print(f"  Final value scores:     {final_scores}")
    winner = max(n_owned, key=n_owned.get)
    print(f"  Winner by territory count: Player {winner}")


if __name__ == "__main__":
    main()
