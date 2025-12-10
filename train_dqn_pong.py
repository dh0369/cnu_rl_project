import os
import time

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import DQN

from callback import SaveOnBestTrainingRewardCallback
from utils import SEED, PONG_ENV_ID, set_global_seeds


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

    env = make_vec_env(env_id, n_envs=1)
    env = VecMonitor(env, log_dir)

    if model_name is None:
        model_name = env_id + "_DQN"

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=2.5e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        max_grad_norm=10,
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=[256]),
        verbose=1,
        seed=SEED,
        device="auto",
    )

    callback = SaveOnBestTrainingRewardCallback(check_freq=10_000, log_dir=log_dir)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=4,
        tb_log_name="DQN_Pong",
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
        model_name = env_id + "_DQN"
    model_path = os.path.join(script_dir, model_base_dir, model_name)
    model = DQN.load(model_path, env)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0

        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(1 / 60)
            episode_reward += reward
            if terminated or truncated:
                print(f"[DQN] Episode {episode + 1}: Total reward = {episode_reward}")
                time.sleep(1.0)
                break

    env.close()


if __name__ == "__main__":
    env_id = PONG_ENV_ID
    train(env_id)
    run(env_id)
