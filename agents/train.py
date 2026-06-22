import argparse
import os
from typing import Any, Dict

from stable_baselines3 import DQN, PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

from agents.environment import BreakoutEnv

MODELS_DIR = "models"
TB_DIR = "runs"

class GameScoreCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_peak_score" in info:
                self.logger.record_mean("game/peak_score", float(info["episode_peak_score"]))
                self.logger.record_mean("game/boards_cleared", float(info["boards_cleared"]))
        return True


def make_stacked_env(
    n_envs: int, n_stack: int, seed: int, env_kwargs: Dict[str, Any]
) -> VecEnv:
    env = make_vec_env(BreakoutEnv, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    return VecFrameStack(env, n_stack=n_stack)


def build_model(algo: str, venv: VecEnv, seed: int, learning_starts: int):
    common = dict(
        policy="MlpPolicy",
        env=venv,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=TB_DIR,
        seed=seed,
        verbose=1,
    )
    if algo == "dqn":
        return DQN(
            learning_rate=2.5e-4,
            buffer_size=100_000,
            learning_starts=learning_starts,
            batch_size=64,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.2,   # anneal epsilon over first 20% of training
            exploration_final_eps=0.05,
            **common,
        )
    if algo == "ppo":
        return PPO(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gae_lambda=0.95,
            ent_coef=0.0,
            **common,
        )
    if algo == "recurrentppo":
        return RecurrentPPO(
            learning_rate=3e-4,
            n_steps=128,
            batch_size=128,
            gae_lambda=0.95,
            ent_coef=0.0,
            **{**common, "policy": "MlpLstmPolicy"},
        )
    raise ValueError(f"unknown algo: {algo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent on Breakout")
    parser.add_argument("--algo", choices=["dqn", "ppo", "recurrentppo"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-stack", type=int, default=4, help="frames to stack")
    parser.add_argument("--n-envs", type=int, default=1, help="parallel envs (PPO benefits from >1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-starts", type=int, default=5_000, help="DQN: steps before learning")
    parser.add_argument("--eval-freq", type=int, default=25_000, help="steps between evaluations")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)
    run_name = f"{args.algo}_breakout"
    env_kwargs: Dict[str, Any] = {}

    train_env = make_stacked_env(args.n_envs, args.n_stack, args.seed, env_kwargs)
    eval_env = make_stacked_env(1, args.n_stack, args.seed + 1000, env_kwargs)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, run_name),
        log_path=os.path.join(MODELS_DIR, run_name),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model = build_model(args.algo, train_env, args.seed, args.learning_starts)
    callbacks = CallbackList([eval_cb, GameScoreCallback()])

    print(f"Training {args.algo.upper()} for {args.timesteps} steps "
          f"(n_stack={args.n_stack}, n_envs={args.n_envs})...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=run_name)

    final_path = os.path.join(MODELS_DIR, f"{run_name}_final")
    model.save(final_path)
    print(f"Done. Final model -> {final_path}.zip | "
          f"best model -> {os.path.join(MODELS_DIR, run_name, 'best_model.zip')}")


if __name__ == "__main__":
    main()
