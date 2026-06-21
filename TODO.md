# SI2-Breakout — Improvements TODO

Checkbox TODO for the DQN/PPO research track. Tackle top-to-bottom, one task at a
time. Each training task = a ~1M-step run + a 30-episode greedy eval.

## Working principles (read once)
- [ ] Keep **two axes separate**: *Axis A = algorithm/architecture* runs with the env
  FIXED at current obs+reward (keeps DQN/PPO/A2C comparable). *Axis B = env design*
  (obs feature, reward) runs only on the **best algorithm from Axis A**.
- [ ] **One toggle per run** (implement variants as flags so each is ablatable).
- [ ] **Unique `--tag` per run** → `models/<tag>/` + `runs/<tag>/` (current script
  hardcodes `run_name` and would overwrite — see Prerequisites).
- [ ] **Same `--seed`** across compared variants (ideally 2–3 seeds for headline claims).
- [ ] **Judge with `evaluate.py` (≥30 greedy episodes)**, not the shaped-reward TB curve.

## Phase 0 — Baselines (DONE)
- [x] Gym env, train.py, rl_agent.py, evaluate.py, TB game-score logging
- [x] DQN @1M (median peak 218) and PPO @1M (median 288, current best)
- [ ] Back up both baselines → `models/baseline_dqn.zip`, `models/baseline_ppo.zip`

## Prerequisites
- [x] `uv add sb3-contrib`; verify `from sb3_contrib import RecurrentPPO`
- [ ] Add `train.py` flags needed now: `--tag` (output folder → no overwrite) and
  `--ent-coef` (Phase 1 PPO tuning).
- [ ] Reuse existing `make_stacked_env`, `GameScoreCallback`, `EvalCallback`, `CallbackList`

## Phase 1 — PPO quick tuning (cheap, high-ROI)
- [ ] Run PPO `--ent-coef 0.01 --tag ppo_tuned`
      (fixes the `ent_coef=0` premature-convergence plateau)
- [ ] Eval `ppo_tuned` vs PPO baseline; keep if better

## Phase 2 — Advanced DQN (hand-implemented) → separate file `agents/dqn_variants.py`
- [ ] Make `agents/dqn_variants.py` a SELF-CONTAINED training script: its own `main()`
      + flags `--double --dueling --per --n-step --tag`, reusing `make_stacked_env`,
      `GameScoreCallback`, `EvalCallback` from train.py. Keeps train.py free of DQN-variant code.
- [ ] **2a. Double DQN** — subclass SB3 `DQN`; in the loss, select next action with
      online `q_net`, evaluate with `q_net_target` (~15-line override). Run + eval.
- [ ] **2b. Dueling network** — custom Q-net: shared trunk → V(s) + A(s,a),
      `Q = V + (A - mean A)`; wire via `policy_kwargs`. Combine with 2a. Run + eval.
- [ ] **2c. Prioritized Experience Replay** — custom `ReplayBuffer` subclass
      (proportional/sum-tree + IS weights), pass via `replay_buffer_class`. Run + eval.
- [ ] **2d. (near-free) n-step returns** — `--n-step 3` using SB3 `NStepReplayBuffer`. Run + eval.
- [ ] Build the DQN-variants comparison (vanilla vs +Double vs +Double+Dueling vs +PER vs +n-step)

## Phase 3 — RecurrentPPO (LSTM)
- [ ] Add `recurrentppo` to the algo map (sb3_contrib); try `--n-stack 1` (recurrence
      replaces frame-stacking)
- [ ] **Deployment state-threading**: carry LSTM hidden state across `deliberate()`
      calls, reset on `game_over`; thread state through `predict()` in
      `agents/rl_agent.py` AND `agents/evaluate.py`
- [ ] Run 1M + eval; verify recurrent greedy scores look sane

## Phase 4 — Environment design (Axis B, on Axis-A winner)
- [ ] **4a. Paddle–ball x-distance feature** — add `(ball_x - paddle_center)/width`
      to `build_observation` (behind a flag to preserve v1); bumps `OBS_DIM`;
      rl_agent/evaluate update automatically (they reuse `build_observation`). Retrain + eval.
- [ ] **4b. Event-based reward** — in `calculate_reward`, replace dense alignment with
      one-off **+paddle-hit / −miss** (hit = ball_vy flips +→− near `paddle_y`; miss =
      life lost). A/B vs current reward.
- [ ] **4c. Graded brick rewards** — weight by row (top worth more); rows fixed by
      brick index (0–4 top / 5–10 mid / 11–15 bottom). Run + eval.
- [ ] **4d. `align_coef=0` A/B** — simplest reward ablation (train with `--align-coef 0`)

## Phase 5 — A2C (breadth, last)
- [ ] Add `a2c` (core SB3) to the algo map; 1M run + eval (third datapoint for report)

## Phase 6 — Consolidate + report
- [ ] Run `evaluate.py` (≥30 greedy eps) on every saved `best_model.zip`
- [ ] Build one **comparison table** (variant × mean/median/max peak score, boards
      cleared, clear rate)
- [ ] Write README/report: architecture (obs, frame-stacking, networks), reward design
      + ablations, TB curves, comparison table, DQN-vs-PPO discussion

## Files this will touch
- [ ] **Create** `agents/dqn_variants.py` (Double, Dueling, PER + its own training `main()`)
- [ ] **Modify** `agents/train.py` (general flags `--tag/--ent-coef/--lr-schedule`, lr schedule, algo map for recurrentppo/a2c)
- [ ] **Modify** `agents/environment.py` (`build_observation` 4a; `calculate_reward` 4b/4c — all flag-gated)
- [ ] **Modify** `agents/rl_agent.py` & `agents/evaluate.py` (extend `ALGOS`; RecurrentPPO state)
- [ ] **Modify** `pyproject.toml` (`sb3-contrib`)

## Per-phase verification
- [ ] `check_env` passes after any env change (`uv run python -m agents.environment`)
- [ ] Smoke run (`--timesteps 3000 --tag smoke_x`) trains & saves without touching other runs
- [ ] `evaluate.py --model models/<tag>/best_model.zip --episodes 30` vs baseline
- [ ] Spot-watch headline model on the viewer (`agents.rl_agent`)
