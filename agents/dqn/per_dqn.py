import argparse
import os
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from agents.environment import OBS_DIM, BreakoutEnv

MODELS_DIR = "models"
TB_DIR = "runs"
CLEAR_SCORE = 148
GAMMA = 0.99
NET_ARCH = (256, 256)
TRAIN_EVAL_EPISODES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, in_dim: int, n_actions: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def act(self, stacked: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.as_tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
            return int(self.forward(t).argmax(dim=1).item())

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.alpha = alpha          # how strongly to prioritize (0 = uniform)
        self.capacity = capacity
        self.buffer: list = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0  # new samples get max priority
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[: self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)   # IS correction
        weights /= weights.max()

        state, action, reward, next_state, done = zip(*samples)
        return (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
            indices,
            np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, priorities) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self) -> int:
        return len(self.buffer)


def new_stack(n_stack: int) -> deque:
    return deque((np.zeros(OBS_DIM, dtype=np.float32) for _ in range(n_stack)), maxlen=n_stack)


def stack_obs(frames: deque, obs: np.ndarray) -> np.ndarray:
    frames.append(obs)
    return np.concatenate(list(frames)).astype(np.float32)

def linear_epsilon(step: int, total: int, frac: float = 0.2, start: float = 1.0, end: float = 0.05) -> float:
    progress = step / max(int(frac * total), 1)
    return end if progress >= 1.0 else start + progress * (end - start)

def beta_by_step(step: int, total: int, beta_start: float = 0.4) -> float:
    # IS correction is annealed from beta_start up to 1.0 over training.
    return min(1.0, beta_start + step * (1.0 - beta_start) / max(total, 1))

def compute_td_loss(online, target, optimizer, buffer, batch_size, beta) -> float:
    states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size, beta)

    states_t = torch.as_tensor(states, device=device)
    next_states_t = torch.as_tensor(next_states, device=device)
    actions_t = torch.as_tensor(actions, device=device).unsqueeze(1)
    rewards_t = torch.as_tensor(rewards, device=device)
    dones_t = torch.as_tensor(dones, device=device)
    weights_t = torch.as_tensor(weights, device=device)

    q_value = online(states_t).gather(1, actions_t).squeeze(1)
    with torch.no_grad():
        next_q = target(next_states_t).max(dim=1)[0]
        target_q = rewards_t + GAMMA * next_q * (1.0 - dones_t)

    td_error = q_value - target_q
    loss = (F.smooth_l1_loss(q_value, target_q, reduction="none") * weights_t).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()

    buffer.update_priorities(indices, td_error.detach().abs().cpu().numpy() + 1e-5)
    return float(loss.item())

def evaluate_model(net, n_stack, n_episodes, seed=0, max_steps=100_000):
    env = BreakoutEnv(max_steps=max_steps)
    net.eval()
    peaks, clears, survivals = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        frames = new_stack(n_stack)
        stacked = stack_obs(frames, obs)
        cur_peak, cur_steps, info, done = 0, 0, {}, False
        while not done:
            with torch.no_grad():
                t = torch.as_tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(net(t).argmax(dim=1).item())
            obs, _, terminated, truncated, info = env.step(action)
            stacked = stack_obs(frames, obs)
            cur_peak = max(cur_peak, info["score"])
            cur_steps += 1
            done = terminated or truncated
        peaks.append(cur_peak)
        clears.append(info.get("boards_cleared", 0))
        survivals.append(cur_steps)
    net.train()
    return np.array(peaks), np.array(clears), np.array(survivals)


def print_eval(tag, model_path, episodes, peaks, clears, survivals):
    print(f"\n=== {tag}  |  {model_path}  |  {episodes} greedy episodes ===")
    print(f"peak score    : mean={peaks.mean():6.1f}  std={peaks.std():6.1f}  "
          f"min={peaks.min():.0f}  median={np.median(peaks):.0f}  max={peaks.max():.0f}")
    print(f"boards cleared: mean={clears.mean():.2f}   max={int(clears.max())}")
    print(f"survival steps: mean={survivals.mean():.0f}   max={int(survivals.max())}")
    print(f"cleared >=1 board: {int((peaks >= CLEAR_SCORE).sum())}/{episodes}")
    print(f"sorted peaks  : {sorted(int(p) for p in peaks)}")

def save_checkpoint(net, n_stack, n_actions, path) -> None:
    torch.save({
        "state_dict": net.state_dict(),
        "config": {"dueling": False, "n_stack": int(n_stack), "obs_dim": int(OBS_DIM),
                   "n_actions": int(n_actions)},
    }, path)

def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    net = QNetwork(cfg["obs_dim"] * cfg["n_stack"], cfg["n_actions"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, cfg

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(args) -> None:
    set_seed(args.seed)
    out_dir = os.path.join(MODELS_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(TB_DIR, args.tag))

    env = BreakoutEnv()
    n_actions = int(env.action_space.n)
    in_dim = OBS_DIM * args.n_stack

    online = QNetwork(in_dim, n_actions).to(device)
    target = QNetwork(in_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    buffer = PrioritizedReplayBuffer(args.buffer_size)

    print(f"[{args.tag}] PER DQN  device={device}  steps={args.timesteps}")

    frames = new_stack(args.n_stack)
    obs, _ = env.reset(seed=args.seed)
    stacked = stack_obs(frames, obs)
    best_eval, losses = -np.inf, []

    for step in range(1, args.timesteps + 1):
        epsilon = linear_epsilon(step, args.timesteps)
        action = online.act(stacked, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_stacked = stack_obs(frames, next_obs)

        buffer.push(stacked, action, reward, next_stacked, float(done))
        stacked = next_stacked

        if done:
            writer.add_scalar("game/peak_score", info.get("episode_peak_score", 0), step)
            writer.add_scalar("game/boards_cleared", info.get("boards_cleared", 0), step)
            frames = new_stack(args.n_stack)
            obs, _ = env.reset()
            stacked = stack_obs(frames, obs)

        if len(buffer) > args.learning_starts and step % args.train_freq == 0:
            beta = beta_by_step(step, args.timesteps)
            losses.append(compute_td_loss(online, target, optimizer, buffer, args.batch_size, beta))

        if step % args.target_update == 0:
            target.load_state_dict(online.state_dict())

        if step % 1000 == 0 and losses:
            writer.add_scalar("train/loss", float(np.mean(losses)), step)
            writer.add_scalar("train/epsilon", epsilon, step)
            writer.add_scalar("train/beta", beta_by_step(step, args.timesteps), step)
            losses = []

        if step % args.eval_freq == 0:
            peaks, clears, _ = evaluate_model(online, args.n_stack, TRAIN_EVAL_EPISODES, seed=args.seed + 1000)
            mean_peak = float(peaks.mean())
            writer.add_scalar("eval/mean_peak", mean_peak, step)
            writer.add_scalar("eval/boards_cleared", float(clears.mean()), step)
            print(f"[{args.tag}] step {step}: eval mean_peak={mean_peak:.1f} "
                  f"median={np.median(peaks):.0f} max={peaks.max():.0f}")
            if mean_peak > best_eval:
                best_eval = mean_peak
                save_checkpoint(online, args.n_stack, n_actions, os.path.join(out_dir, "best_model.pt"))

    save_checkpoint(online, args.n_stack, n_actions, os.path.join(out_dir, "final_model.pt"))
    writer.close()
    print(f"[{args.tag}] done. best eval mean_peak={best_eval:.1f} -> {out_dir}/best_model.pt")

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", default="scratch_per", help="output folder under models/ and runs/")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--eval-only", action="store_true", help="evaluate a saved checkpoint, no training")
    parser.add_argument("--model", default=None, help="checkpoint .pt for --eval-only")
    parser.add_argument("--episodes", type=int, default=30, help="episodes for --eval-only")

def main() -> None:
    parser = argparse.ArgumentParser(description="Prioritized Experience Replay DQN from scratch")
    add_common_args(parser)
    args = parser.parse_args()

    if args.eval_only:
        if not args.model:
            parser.error("--eval-only requires --model <path>")
        set_seed(args.seed)
        net, cfg = load_model(args.model)
        peaks, clears, survivals = evaluate_model(net, cfg["n_stack"], args.episodes, seed=args.seed)
        print_eval("per", args.model, args.episodes, peaks, clears, survivals)
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
