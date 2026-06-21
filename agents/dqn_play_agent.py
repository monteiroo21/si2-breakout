import argparse
import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from agents.environment import ACTIONS, OBS_DIM, build_observation

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


class DuelingQNetwork(nn.Module):
    def __init__(self, in_dim: int, n_actions: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        feat_dim, head_dim = 256, 256

        self.feature = nn.Sequential(
            nn.Linear(in_dim, feat_dim), 
            nn.ReLU()
        )
        
        self.value = nn.Sequential(
            nn.Linear(feat_dim, head_dim), 
            nn.ReLU(), 
            nn.Linear(head_dim, 1)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(feat_dim, head_dim), 
            nn.ReLU(), 
            nn.Linear(head_dim, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        advantage = self.advantage(f)
        return self.value(f) + advantage - advantage.mean(dim=1, keepdim=True)


def load_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cls = DuelingQNetwork if cfg["dueling"] else QNetwork
    net = cls(cfg["obs_dim"] * cfg["n_stack"], cfg["n_actions"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, cfg


class ScratchRLAgent(BaseAgent):
    def __init__(
        self,
        model_path: str,
        n_stack: Optional[int] = None,
        deterministic: bool = True,
        server_uri: str = "ws://localhost:8765/ws",
    ) -> None:
        super().__init__(server_uri=server_uri)
        self.net, cfg = load_model(model_path)
        self.n_stack = n_stack or cfg["n_stack"]
        self.deterministic = deterministic
        self._reset_stack()
        logging.info(f"Loaded scratch DQN model from {model_path} "
                     f"(dueling={cfg['dueling']}, n_stack={self.n_stack})")

    def _reset_stack(self) -> None:
        # Mirror VecFrameStack / rl_agent.py: zero-filled start, newest frame last.
        self.frames = deque(
            (np.zeros(OBS_DIM, dtype=np.float32) for _ in range(self.n_stack)),
            maxlen=self.n_stack,
        )

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        state = self.current_state
        if not state or state.get("game_over"):
            self._reset_stack()  # restarted game begins with a clean stack
            return None

        self.frames.append(build_observation(state))
        obs = np.concatenate(list(self.frames)).astype(np.float32)

        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = self.net(t)
            if self.deterministic:
                action = int(q_values.argmax(dim=1).item())
            else:
                probs = torch.softmax(q_values, dim=1)
                action = int(torch.multinomial(probs, 1).item())

        direction = ACTIONS[action]
        if direction is None:
            return None  # STAY
        return {"action": "move", "direction": direction}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a from-scratch DQN agent on Breakout")
    parser.add_argument("--model", required=True, help="checkpoint .pt path")
    parser.add_argument("--n-stack", type=int, default=None, help="override stack size (default: from checkpoint)")
    parser.add_argument("--stochastic", action="store_true", help="sample instead of greedy")
    parser.add_argument("--uri", default="ws://localhost:8765/ws")
    args = parser.parse_args()

    agent = ScratchRLAgent(
        model_path=args.model,
        n_stack=args.n_stack,
        deterministic=not args.stochastic,
        server_uri=args.uri,
    )
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
