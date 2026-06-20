import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from server.logic import Breakout

# Fixed brick layout: 16 bricks at fixed indices
NUM_BRICKS = 16
OBS_DIM = 3 + NUM_BRICKS  # ball_x, ball_y, paddle_x  +  one active-flag per brick
ACTIONS: Dict[int, Optional[str]] = {0: "WEST", 1: None, 2: "EAST"}


def build_observation(state: Dict[str, Any]) -> np.ndarray:
    width = state["width"]
    height = state["height"]

    ball_x = state["ball_x"] / width
    ball_y = state["ball_y"] / height
    paddle_x = state["paddle_x"] / width

    bricks = np.zeros(NUM_BRICKS, dtype=np.float32)
    for b in state["bricks"]:
        idx = b["index"]
        if 0 <= idx < NUM_BRICKS:
            bricks[idx] = 1.0

    obs = np.concatenate(
        [np.array([ball_x, ball_y, paddle_x], dtype=np.float32), bricks]
    )
    return np.clip(obs, 0.0, 1.0).astype(np.float32)


class BreakoutEnv(gym.Env):
    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        max_steps: int = 3000,
        brick_reward: float = 1.0,
        clear_reward: float = 20.0,
        death_penalty: float = 20.0,
        align_coef: float = 0.02,
        step_reward: float = 0.0,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.brick_reward = brick_reward
        self.clear_reward = clear_reward
        self.death_penalty = death_penalty
        self.align_coef = align_coef
        self.step_reward = step_reward

        self.game = Breakout()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._steps = 0
        self._prev_active = NUM_BRICKS
        self._prev_lives = self.game.lives

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.game.reset_game()
        self._steps = 0

        state = self.game.get_state()
        self._prev_active = len(state["bricks"])
        self._prev_lives = state["lives"]
        return build_observation(state), {}
    
    def calculate_reward(self, state: Dict[str, Any]) -> float:
        active = len(state["bricks"])
        lives = state["lives"]

        reward = self.step_reward

        delta = self._prev_active - active
        if delta > 0:
            reward += self.brick_reward * delta

        if active == 0 and self._prev_active > 0:
            reward += self.clear_reward

        if lives < self._prev_lives:
            reward -= self.death_penalty

        if self.align_coef and self.game.ball_vy > 0.0:
            paddle_center = state["paddle_x"] + state["paddle_width"] / 2.0
            err = abs(state["ball_x"] - paddle_center) / state["width"]
            reward += self.align_coef * (1.0 - 2.0 * err)

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        direction = ACTIONS[int(action)]
        if direction is not None:
            self.game.move_paddle(direction)
        self.game.update(self.dt)
        self._steps += 1

        state = self.game.get_state()
        obs = build_observation(state)
        active = len(state["bricks"])
        lives = state["lives"]

        reward = self.calculate_reward(state)

        self._prev_active = active
        self._prev_lives = lives

        terminated = bool(state["game_over"])
        truncated = self._steps >= self.max_steps
        info = {"score": state["score"], "lives": lives, "active_bricks": active}
        return obs, float(reward), terminated, truncated, info
