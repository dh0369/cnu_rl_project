import os
import time

import gymnasium as gym
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from callback import SaveOnBestTrainingRewardCallback
from utils import SEED, PONG_ENV_ID, set_global_seeds


def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func


def train(
    env_id: str = PONG_ENV_ID,
    log_base_dir: str = "logs",
    model_base_dir: str = "models",
    model_name: str | None = None,
    total_timesteps: int = 1_000_000,
):
    set_global_seeds(SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, log_base_dir, env_id)
    model_dir = os.path.join(script_dir, model_base_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    n_envs = 4
    env = make_vec_env(env_id, n_envs=n_envs)
    env = VecMonitor(env, log_dir)

    if model_name is None:
        model_name = env_id + "_PPO"

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256], vf=[256])],
        ortho_init=False,
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=linear_schedule(2.5e-4),
        n_steps=128,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=4,
        ent_coef=0.01,
        clip_range=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
        seed=SEED,
        device="auto",
    )

    callback = SaveOnBestTrainingRewardCallback(check_freq=10_000, log_dir=log_dir)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=4,
        tb_log_name="PPO_Pong",
        reset_num_timesteps=True,
        progress_bar=True,
    )

    save_path = os.path.join(model_dir, model_name)
    model.save(save_path)
    env.close()


def run(
    env_id: str = PONG_ENV_ID,
    model_base_dir: str = "models",
    model_name: str | None = None,
    n_episodes: int = 5,
):
    env = gym.make(env_id, render_mode="human")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name is None:
        model_name = env_id + "_PPO"
    model_path = os.path.join(script_dir, model_base_dir, model_name)
    model = PPO.load(model_path, env)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0

        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            time.sleep(1 / 60)

            if terminated or truncated:
                print(f"[PPO] Episode {episode + 1}: Total reward = {episode_reward:.2f}")
                time.sleep(1.0)
                break

    env.close()


if __name__ == "__main__":
    env_id = PONG_ENV_ID
    train(env_id)
    run(env_id)
