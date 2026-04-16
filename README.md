# Stochastic Territory Control Game

A territory control game built on graph theory, stochastic calculus, and
partially observable Markov decision processes (POMDPs).

---

## What Is This?

You control territories on a map (a graph). Each territory has a **value that
fluctuates over time** driven by a stochastic differential equation. You attack
adjacent enemy territories, but combat is probabilistic — strength advantage
matters, but you can still lose. You never see the enemy's true strength or the
true value of their territories. You have to *infer* them from noisy observations
and combat outcomes using Bayesian inference.

The core challenge: **when do you attack vs wait for a better moment?** That
tension — between the expected value of acting now and the option value of
waiting — is the entire game.

---

## The Three Big Ideas (Read These First)

Before touching any code, make sure you understand these conceptually.

### Idea 1: Territory Values Are Stochastic Processes

Territory values do not sit still. They evolve according to an
**Ornstein-Uhlenbeck (OU) process**:

```
dV_t = θ(μ - V_t) dt + σ dW_t
```

Plain English:
- `μ` is the territory's "fair value" (long-run average)
- `θ` controls how fast it snaps back toward `μ` after drifting away
- `σ` controls how wildly it fluctuates
- `dW_t` is pure random noise (a Wiener process increment)

**Why not Geometric Brownian Motion (used in Black-Scholes)?**
GBM lets values drift to zero or infinity. OU keeps them anchored near `μ`.
This creates timing windows: attack when `V_t >> μ` (spike), wait when
`V_t << μ` (trough). That is the strategic clock of the game.

The exact discrete-time update (no approximation error):
```
V_{t+dt} = μ + φ(V_t - μ) + σ_dt * Z
  where φ = exp(-θ * dt)
        σ_dt = sqrt( σ²/(2θ) * (1 - φ²) )
        Z ~ Normal(0, 1)
```

### Idea 2: Combat Is a Logistic Function with Noise

Attacking a territory does not guarantee capture. The probability of success is:

```
P(win) = sigmoid( k * (S_attacker - S_defender - δ) + ε )
  where sigmoid(x) = 1 / (1 + exp(-x))
        k     = steepness (high k → strong beats weak almost always)
        δ     = defender home advantage (positive = defender favoured at parity)
        ε     ~ Normal(0, σ_noise²)  (injected into the predictor, not onto P)
```

Why inject noise into the predictor instead of adding it to `P`?
Because `P + noise` can fall outside [0, 1]. Injecting `ε` into the linear
predictor before sigmoid keeps P bounded automatically, and gives us a
well-defined likelihood function for Bayesian updating.

### Idea 3: You Never See the Full State (POMDP)

This is the hard part — and the interesting part. The game is a
**Partially Observable Markov Decision Process**:

- You **know exactly** the values and troop strengths of territories you own.
- You see **noisy estimates** of adjacent enemy territories.
- You see **nothing** about non-adjacent enemy territories.
- After each attack you observe only win/loss — not the enemy's true strength.

Your agent must maintain a **belief state**: a probability distribution over
what the true state of the world might be, updated every turn via Bayes' rule.

Two filters handle this:
- **Kalman filter** — tracks territory values (Gaussian, exact for OU dynamics)
- **Particle filter** — tracks enemy troop strengths (non-Gaussian, needed because
  the logistic combat likelihood is not conjugate to any simple prior)

---

## Project Structure

```
stochastic-ojuice/
│
├── stochastic_ojuice/          # Core engine (the library)
│   ├── types.py                # Data structures: NodeParams, GameState, Observation
│   ├── stochastic.py           # OU process: exact vectorised simulation
│   ├── combat.py               # Logistic combat model + Bayesian likelihood
│   ├── belief.py               # Kalman filter (values) + Particle filter (strengths)
│   ├── environment.py          # GameEnvironment: step(), stochastic_update(), resolve_combat()
│   ├── agent.py                # PlayerAgent: belief tracking + EV-based decisions
│   └── factory.py              # Helper functions to build maps and games quickly
│
├── tests/                      # Unit tests (23 tests, all pass)
│   ├── test_stochastic.py      # OU moments, convergence, shapes
│   ├── test_combat.py          # Monotonicity, bounds, log-likelihood
│   └── test_belief.py          # Kalman convergence, particle filter inference
│
├── run_demo.py                 # Full game episode: 2 agents, 4x5 grid, 200 steps
└── requirements.txt            # numpy, scipy, networkx, pytest
```

---

## Step-by-Step Build Guide (For Friends Starting From Scratch)

Follow these phases in order. Each builds on the last. Do not skip ahead.

---

### Phase 1 — Math Foundation (Days 1–3)

Before writing a single line of code, get the math clear on paper.

**Step 1.1 — Understand the OU process**

Read the SDE: `dV_t = θ(μ - V_t) dt + σ dW_t`

Derive by hand (or verify):
- The conditional mean: `E[V_{t+dt} | V_t] = μ + exp(-θdt)(V_t - μ)`
- The conditional variance: `Var[V_{t+dt} | V_t] = σ²/(2θ) * (1 - exp(-2θdt))`
- The stationary distribution: `V_∞ ~ Normal(μ, σ²/(2θ))`
- The half-life of a deviation: `ln(2) / θ`

Test your intuition: if `θ = 0.5` and `V_0 = 20` but `μ = 10`, roughly when does
the value return to within 1 unit of `μ`? (Answer: about 3–4 time units.)

Key insight: this discretisation is **exact** (no Euler-Maruyama error), because
the OU SDE is linear. This matters when you run thousands of Monte Carlo steps.

**Step 1.2 — Understand the logistic combat function**

Write out:
```
P(win | S_att, S_def) = 1 / (1 + exp(-k * (S_att - S_def - δ)))
```

Verify:
- At `S_att - S_def = δ`: P = 0.5 (break-even point)
- As `S_att >> S_def`: P → 1
- As `S_att << S_def`: P → 0
- Derivative `∂P/∂S_att = k * P * (1 - P) > 0` always (strictly monotone)

Play with `k` and `δ` in a spreadsheet or Desmos. Understand what "defender
advantage δ = 0.3" means in practice (you need `S_att > S_def + 0.3` to be
at break-even).

**Step 1.3 — Understand the Kalman filter**

The Kalman filter tracks a hidden state `x_t` given noisy observations `z_t`.

For our OU value tracking:
```
State transition:   x_{t+1} = μ + φ*(x_t - μ) + process_noise
Observation model:  z_t = x_t + obs_noise
```

The filter maintains a Gaussian belief `(m, P)` — mean and variance:
```
Predict:  m_pred = μ + φ*(m - μ)
          P_pred = φ² * P + Q           (Q = process noise variance)

Update:   K = P_pred / (P_pred + R)     (R = obs noise variance)
          m_new = m_pred + K*(z - m_pred)
          P_new = (1 - K) * P_pred
```

`K` is the Kalman gain — how much to trust the new observation vs your prediction.
When `K → 1`, you trust the observation. When `K → 0`, you trust your prediction.

**Step 1.4 — Understand the Particle Filter**

When the likelihood is not Gaussian (like our logistic combat outcome), the
Kalman filter cannot be used. Particle filters represent the belief as a
cloud of weighted samples:

```
Particles: {S^(k), w^(k)}, k = 1..N

Predict:  Each S^(k) += small random drift   (enemy troops change between turns)
Update:   w^(k) *= P(combat_outcome | S_att, S^(k))
          Normalise weights to sum to 1
Resample: When effective N = 1/sum(w^2) < N/2, resample proportionally to weights
```

The key insight: after a *successful* attack against a weak enemy, the particles
with low `S_def` gain weight. The distribution shifts toward lower values.
After a *failed* attack, it shifts higher. Over many combats, the agent learns
the true enemy strength without ever observing it directly.

---

### Phase 2 — Core Engine (Days 4–7)

Now implement. Go in this exact order — each module depends only on the previous.

**Step 2.1 — `types.py`**: Define your data structures first.
- `NodeParams(theta, mu, sigma)` — frozen dataclass, OU parameters per node
- `CombatParams(k, delta, noise_sigma, attack_cost_frac, failure_drain_frac)`
- `GameState` — the authoritative true state, struct-of-arrays layout:
  - `true_values[N]`, `true_strengths[N]`, `owners[N]` as NumPy arrays
  - Never put per-node data in Python dicts — it kills vectorisation
- `Observation` — what one player sees: noisy arrays with NaN where hidden

**Step 2.2 — `stochastic.py`**: The OU engine.
- `OUProcess(theta, mu, sigma)` — takes arrays, not scalars
- `step(V, dt, rng)` — single step, handles both `(N,)` and `(B, N)` batch shapes
- `simulate_batch(V0, dt, n_steps, n_scenarios, rng)` — returns `(B, T+1, N)`
- Pre-draw all noise at once: `rng.standard_normal((B, T, N))` — one allocation
- Test: simulate 5000 paths for 2000 steps, check empirical mean ≈ `μ` and
  variance ≈ `σ²/(2θ)` within 5% tolerance

**Step 2.3 — `combat.py`**: The logistic combat model.
- `CombatModel(params)`
- `success_probability(S_att, S_def, include_noise_variance)` — vectorised
  - With `include_noise_variance=True`, use the Gaussian-logistic approximation:
    `P ≈ sigmoid(η / sqrt(1 + (π/√8)² * σ_ε²))` — avoids Monte Carlo integration
- `resolve(S_att, S_def, rng)` — single combat, returns `(success, new_S_att, new_S_def)`
- `resolve_batch(S_att_arr, S_def_arr, rng)` — vectorised version for Monte Carlo
- `log_likelihood(outcome, S_att, S_def_particles)` — for the particle filter weight update
- Test: verify `P(ΔS=δ) = 0.5`, `∂P/∂S_att > 0` everywhere, batch shapes correct

**Step 2.4 — `belief.py`**: The two filters.
- `KalmanBeliefNode(theta, mu, sigma, obs_noise_var, m, P)`
  - `predict(dt)` — time update only (call when node not observed)
  - `update(z)` — measurement update, returns Kalman gain
  - `step(dt, z=None)` — combined predict + optional update
  - Test: start at `m=5`, true value = 10, after 200 obs with low noise, `m ≈ 10`
- `ParticleStrengthEstimator(n_particles, init_mean, init_std, strength_drift_std)`
  - `reset(mean, std, rng)` — initialise particle cloud
  - `predict(rng)` — random walk drift on all particles
  - `update(log_likelihoods)` — log-space weight update (numerically stable)
  - `resample_if_needed(rng)` — systematic resampling when `N_eff < N/2`
  - Test: run 30 combat updates against `S_true=3`, starting from prior `mean=5`.
    Posterior mean should shift toward 3.

---

### Phase 3 — Game Environment (Days 8–10)

**Step 3.1 — `environment.py`**: The authoritative simulator.

`GameEnvironment(graph, state, combat_params, dt, ...)`

Implement these methods in order:

1. `stochastic_update()` — call `OUProcess.step()` on `state.true_values`. This
   is a single NumPy call. No Python loop over nodes.

2. `resolve_combat(attacker_id, from_node, to_node)` — validates the action,
   calls `CombatModel.resolve()`, updates `state.owners`, `state.true_strengths`.
   Side-effects only on `GameState`.

3. `_make_observations(combat_results)` — produce one `Observation` per player.
   Visibility rules:
   - Owned nodes → exact value and strength (no noise)
   - Adjacent enemy nodes → add Gaussian noise to true values
   - Non-adjacent → `NaN` (hidden)

4. `step(actions)` — the main game loop method:
   ```
   stochastic_update()
   for player in random_order(players):
       resolve their action
   make_observations()
   compute rewards (delta in total owned value)
   check terminal condition
   return (observations, rewards, done, info)
   ```

5. `monte_carlo_ev(from_node, to_node, ...)` — simulate `n_scenarios` paths of
   `n_steps` into the future to estimate the value of capturing a territory.

**Step 3.2 — `factory.py`**: Construction helpers.

- `make_grid_map(rows, cols)` — `nx.DiGraph` grid
- `make_random_map(n_nodes, edge_prob)` — Erdős-Rényi, resample until connected
- `make_game(graph, n_players, ...)` — one-call environment setup
- `make_agents(env, agent_config)` — one agent per player

---

### Phase 4 — Agent (Days 11–13)

**Step 4.1 — `agent.py`**: The POMDP agent.

`PlayerAgent(player_id, graph, combat_params, agent_config, ou_params_belief, rng)`

Implement in this order:

1. `observe(obs, dt)` — the belief update step:
   - For each visible node: call `KalmanBeliefNode.step(dt, z=noisy_value)`
   - For each visible enemy node: call `ParticleStrengthEstimator.predict()` +
     `update(noisy_strength_log_likelihood)`
   - For nodes not visible: `predict()` only (uncertainty grows while blind)
   - If `obs.combat_result` is not None: call `_update_beliefs_from_combat()`

2. `_update_beliefs_from_combat(from_node, to_node, success)`:
   - Compute `log_likelihoods = CombatModel.log_likelihood(success, S_att, particles)`
   - Call `ParticleStrengthEstimator.update(log_likelihoods)`
   - This is where the agent learns about enemy strength from combat outcomes

3. `_compute_ev(from_node, to_node, S_att)` — the Expected Value formula:
   ```
   P_hat = sigmoid(k * (S_att - S_def_posterior_mean - δ))
   EV = P_hat * V_j_mean
      - (1 - P_hat) * failure_drain_cost
      - fixed_attack_cost
      - λ * overextension_penalty
      + explore_bonus * sqrt(V_j_posterior_variance)
   ```

4. `act()` — scan all (from_node, to_node) pairs, pick max EV above threshold.
   Return `{"attack": (i, j)}` or `{"wait": None}`.

**Step 4.2 — Connect everything in `run_demo.py`**:
```python
env, state = make_game(graph, n_players=2)
agents = make_agents(env)

observations = env.reset()
for agent, obs in zip(agents, observations):
    agent.observe(obs, dt=env.dt)

done = False
while not done:
    actions = {agent.player_id: agent.act() for agent in agents}
    observations, rewards, done, info = env.step(actions)
    for agent, obs in zip(agents, observations):
        agent.observe(obs, dt=env.dt)
```

---

### Phase 5 — Testing (Days 14–15)

Write tests **before** you trust any output. Key things to test:

| Test | What to Check |
|---|---|
| OU stationary moments | After 2000 steps, empirical mean ≈ `μ`, variance ≈ `σ²/(2θ)` within 5% |
| OU conditional mean | `E[V_{t+dt}\|V_t]` formula matches simulation average |
| Logistic monotonicity | `P(S_att+ε, S_def) > P(S_att, S_def)` for all ε > 0 |
| Logistic bounds | `0 < P < 1` always |
| Defender advantage | `P(S_att=S_def) < 0.5` when `δ > 0` |
| Kalman convergence | After 200 obs, mean error < 1.0 from truth |
| Particle inference | After 30 combat updates, posterior mean moves toward true strength |
| Particle resampling | Resampling preserves the weighted mean |
| Combat batch shapes | Vectorised batch output matches expected shapes |

Run with:
```bash
py -3.12 -m pytest tests/ -v
```

---

### Phase 6 — Extensions (Days 16+)

Once the core works, the natural next steps are:

**Reinforcement Learning**
The `step()` interface already follows the Gym contract. Plug in:
- **PPO** (Proximal Policy Optimisation) — good first RL algorithm
- Use the `Observation` as the agent's input feature vector
- Replace `PlayerAgent.act()` with a neural network policy

The belief state (Kalman mean/variance + particle mean/std per node) is a
compact, fixed-size feature vector — directly usable as RL input.

**Monte Carlo Tree Search (MCTS)**
Use `monte_carlo_ev()` for deeper lookahead. MCTS + UCB1 exploration can
plan multi-step attack chains rather than one-step greedy EV.

**Heterogeneous Maps**
Give different nodes different `(θ, μ, σ)` parameters via `heterogeneous_ou_params()`.
This creates "stable anchor" territories (high θ, low σ) vs
"volatile frontier" territories (low θ, high σ) — forcing agents to reason
about different risk profiles across the map.

**Model Mismatch**
The agent currently assumes it knows the true OU parameters.
Give it wrong beliefs (`θ_believed ≠ θ_true`) and measure how much it degrades.
This is a classic POMDP robustness experiment.

**Multi-Agent Equilibria**
With 3+ players, pure EV maximisation breaks down. Agents should reason about
blocking, coalition formation, and letting two enemies weaken each other.
This is where game theory (Nash equilibria, Shapley values) enters.

---

## Running the Code

### Install dependencies

```bash
pip install numpy scipy networkx pytest
```

### Run the demo game

```bash
py -3.12 run_demo.py
```

This runs a 200-step, 2-player game on a 4×5 grid. Every 10 steps it prints:
- True territory values vs the agent's Kalman belief
- True enemy strengths vs the agent's particle filter posterior
- Top 3 attack candidates by EV

### Run the tests

```bash
py -3.12 -m pytest tests/ -v
```

Expected: 23 passed.

---

## Key Parameters and What They Do

| Parameter | Location | Effect |
|---|---|---|
| `theta` | `NodeParams` | Reversion speed. Low = slow drift, large timing windows. High = snaps back fast. |
| `mu` | `NodeParams` | Long-run territory value. Sets the strategic anchor. |
| `sigma` | `NodeParams` | Volatility. High = wild swings, more timing opportunities. |
| `k` | `CombatParams` | Combat sharpness. High = deterministic (strong always wins). Low = random. |
| `delta` | `CombatParams` | Defender advantage. 0 = symmetric. 0.3 = mild home advantage. |
| `noise_sigma` | `CombatParams` | Battlefield chaos. Adds uncertainty beyond strength difference. |
| `attack_cost_frac` | `CombatParams` | Always-paid attack cost. Discourages spamming attacks. |
| `failure_drain_frac` | `CombatParams` | Extra loss on failure. Makes failed attacks costly. |
| `min_ev_to_attack` | `AgentConfig` | Attack threshold. Lower = more aggressive agent. |
| `lambda_overext` | `AgentConfig` | Overextension penalty weight. Higher = more cautious about leaving home weak. |
| `explore_bonus` | `AgentConfig` | Reward for uncertain territory values. Makes agent explore unknown regions. |
| `n_particles` | `AgentConfig` | Particle filter resolution. 200 = fast, 2000 = accurate. |

---

## Mathematical Reference

### OU Stationary Distribution
```
V_∞ ~ Normal( μ,  σ²/(2θ) )
Half-life of deviation = ln(2) / θ
```

### Exact OU Transition
```
V_{t+dt} | V_t ~ Normal(  μ + φ(V_t - μ),   σ²/(2θ)(1 - φ²)  )
where φ = exp(-θ * dt)
```

### Logistic Combat Probability
```
P(win) = sigmoid( k*(S_att - S_def - δ) )
       = 1 / (1 + exp( -k*(S_att - S_def - δ) ))
Marginalised over noise ε ~ N(0, σ_ε²):
P(win) ≈ sigmoid( k*(ΔS - δ) / sqrt(1 + (πσ_ε/√8)²) )
```

### Kalman Filter (OU Value Belief)
```
Predict:  m' = μ + φ(m - μ)          P' = φ²P + Q
Update:   K  = P'/(P' + R)            m  = m' + K(z - m')    P = (1-K)P'
where Q = σ²/(2θ)(1-φ²),   R = observation noise variance
```

### Particle Filter Weight Update (Enemy Strength)
```
log w^(k) += log P(combat_outcome | S_att, S^(k))
Normalise: w^(k) /= sum(w)
Resample when N_eff = 1/sum(w²) < N/2
```

### Expected Value of Attack
```
EV(i→j) = P̂ · V̂_j  -  (1-P̂) · C_fail  -  C_fixed  -  λ · Π_ij  +  α · σ(V̂_j)

where:
  P̂       = logistic combat probability using posterior mean of S_j
  V̂_j     = Kalman posterior mean of territory value
  C_fail   = failure_drain_frac * S_i
  C_fixed  = attack_cost_frac * S_i
  Π_ij     = P(node i recaptured) * V̂_i     (overextension penalty)
  α · σ   = exploration bonus (optional)
```

---

## Dependencies

- `numpy` — all numerical computation, vectorised simulation
- `scipy` — normal distribution CDF (for credible intervals)
- `networkx` — graph topology (adjacency, path finding)
- `pytest` — testing

No deep learning frameworks required for the base engine.
PyTorch or JAX can be added later for RL agents.

---

## Credits

Architecture: Senior Quant Researcher / Algorithmic Game Theorist design session.
Math: Ornstein-Uhlenbeck (1930), Kalman (1960), Gordon et al. particle filter (1993),
      Mackay Gaussian-logistic approximation (1992).
