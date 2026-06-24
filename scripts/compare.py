import argparse
from collections import deque

import numpy as np
import torch

from agents.environment import OBS_DIM, BreakoutEnv
from agents.dqn_play_agent import device, load_model  # config-aware loader (plain or dueling)

CLEAR_SCORE = 148  # 16 bricks * 3 + 100 clear bonus = one full board

# (label, checkpoint) — these are the default --tag folders from each training script.
VARIANTS = [
    ("vanilla", "models/vanilla_dqn/best_model.pt"),
    ("double",  "models/double_dqn/best_model.pt"),
    ("dueling", "models/dueling_dqn/best_model.pt"),
    ("per",     "models/per_dqn/best_model.pt"),
    ("nstep",   "models/nstep_dqn/best_model.pt"),
]


def run_episodes(net, n_stack, n_episodes, seed, max_steps=100_000):
    env = BreakoutEnv(max_steps=max_steps)
    peaks, clears = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)  # same seed sequence for every variant -> fair
        frames = deque((np.zeros(OBS_DIM, dtype=np.float32) for _ in range(n_stack)), maxlen=n_stack)
        frames.append(obs)
        cur_peak, info, done = 0, {}, False
        while not done:
            stacked = np.concatenate(list(frames)).astype(np.float32)
            with torch.no_grad():
                t = torch.as_tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(net(t).argmax(dim=1).item())
            obs, _, terminated, truncated, info = env.step(action)
            frames.append(obs)
            cur_peak = max(cur_peak, info["score"])
            done = terminated or truncated
        peaks.append(cur_peak)
        clears.append(info.get("boards_cleared", 0))
    return np.array(peaks), np.array(clears)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the from-scratch DQN variants side by side")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"\n{args.episodes} greedy episodes each, identical seeds (peak game score)\n")
    header = f"{'variant':9s} {'mean':>7s} {'median':>7s} {'max':>6s} {'std':>7s} {'boards':>7s} {'clear%':>7s}"
    print(header)
    print("-" * len(header))

    for name, path in VARIANTS:
        try:
            net, cfg = load_model(path)
        except FileNotFoundError:
            print(f"{name:9s}   (no checkpoint at {path} — train it first)")
            continue
        peaks, clears = run_episodes(net, cfg["n_stack"], args.episodes, args.seed)
        clear_rate = 100.0 * float((peaks >= CLEAR_SCORE).mean())
        print(f"{name:9s} {peaks.mean():7.1f} {np.median(peaks):7.0f} {peaks.max():6.0f} "
              f"{peaks.std():7.1f} {clears.mean():7.2f} {clear_rate:6.0f}%")

    print("\nreference baselines: SB3 DQN median 218 | SB3 PPO median 288")


if __name__ == "__main__":
    main()
