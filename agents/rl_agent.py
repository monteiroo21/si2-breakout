import argparse
import asyncio
import logging
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3 import A2C, DQN, PPO

from agents.base_agent import BaseAgent
from agents.environment import ACTIONS, OBS_DIM, build_observation

ALGOS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}


class RLAgent(BaseAgent):
    def __init__(
        self,
        model_path: str,
        algo: str = "dqn",
        n_stack: int = 4,
        deterministic: bool = True,
        server_uri: str = "ws://localhost:8765/ws",
    ) -> None:
        super().__init__(server_uri=server_uri)
        self.model = ALGOS[algo].load(model_path)
        self.n_stack = n_stack
        self.deterministic = deterministic
        self._reset_stack()
        logging.info(f"Loaded {algo.upper()} model from {model_path}")

    def _reset_stack(self) -> None:
        # Mirror VecFrameStack's zero-filled start (newest frame goes last).
        self.frames = deque(
            (np.zeros(OBS_DIM, dtype=np.float32) for _ in range(self.n_stack)),
            maxlen=self.n_stack,
        )

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        state = self.current_state
        if not state or state.get("game_over"):
            self._reset_stack()  # so a restarted game begins with a clean stack
            return None

        # Same observation + same stacking order as training.
        self.frames.append(build_observation(state))
        obs = np.concatenate(list(self.frames)).astype(np.float32)

        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        direction = ACTIONS[int(action)]
        if direction is None:
            return None  # STAY
        return {"action": "move", "direction": direction}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained RL agent on Breakout")
    parser.add_argument("--model", default=None,
                        help="model zip path (default: models/<algo>_breakout/best_model.zip)")
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c"], default="dqn")
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true", help="sample instead of greedy")
    parser.add_argument("--uri", default="ws://localhost:8765/ws")
    args = parser.parse_args()
    model_path = args.model or f"models/{args.algo}_breakout/best_model.zip"

    agent = RLAgent(
        model_path=model_path,
        algo=args.algo,
        n_stack=args.n_stack,
        deterministic=not args.stochastic,
        server_uri=args.uri,
    )
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
