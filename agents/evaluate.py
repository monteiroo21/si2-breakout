import argparse

import numpy as np
from stable_baselines3 import DQN, PPO
from sb3_contrib import RecurrentPPO

from agents.train import make_stacked_env

ALGOS = {"dqn": DQN, "ppo": PPO, "recurrentppo": RecurrentPPO}
CLEAR_SCORE = 148  # 16 bricks * 3 + 100 clear bonus = a full board cleared


def run_episodes(model, venv, n_episodes: int):
    peaks, clears, survivals = [], [], []
    obs = venv.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    cur_peak, cur_steps = 0, 0
    while len(peaks) < n_episodes:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        obs, _, dones, infos = venv.step(action)
        episode_starts = dones
        cur_peak = max(cur_peak, infos[0]["score"])  # peak before any death-rollback
        cur_steps += 1
        if dones[0]:
            peaks.append(cur_peak)
            clears.append(infos[0].get("boards_cleared", 0))
            survivals.append(cur_steps)
            cur_peak, cur_steps = 0, 0
    return np.array(peaks), np.array(clears), np.array(survivals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy game-score evaluation")
    parser.add_argument("--algo", choices=["dqn", "ppo", "recurrentppo"], default="dqn")
    parser.add_argument("--model", default=None,
                        help="model zip path (default: models/<algo>_breakout/best_model.zip)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model or f"models/{args.algo}_breakout/best_model.zip"
    # max_steps high -> no truncation; episodes end on game-over like the server.
    venv = make_stacked_env(1, args.n_stack, args.seed, {"max_steps": 100_000})
    model = ALGOS[args.algo].load(model_path)

    peaks, clears, survivals = run_episodes(model, venv, args.episodes)

    print(f"\n=== {args.algo.upper()}  |  {model_path}  |  {args.episodes} greedy episodes ===")
    print(f"peak score    : mean={peaks.mean():6.1f}  std={peaks.std():6.1f}  "
          f"min={peaks.min():.0f}  median={np.median(peaks):.0f}  max={peaks.max():.0f}")
    print(f"boards cleared: mean={clears.mean():.2f}   max={int(clears.max())}")
    print(f"survival steps: mean={survivals.mean():.0f}   max={int(survivals.max())}")
    print(f"cleared >=1 board: {int((peaks >= CLEAR_SCORE).sum())}/{args.episodes}")
    print(f"sorted peaks  : {sorted(int(p) for p in peaks)}")


if __name__ == "__main__":
    main()
