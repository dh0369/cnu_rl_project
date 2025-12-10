import os
import random
from typing import Optional

import gymnasium as gym
import numpy as np
import ale_py
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

gym.register_envs(ale_py)

SEED = 42

PONG_ENV_ID = "ALE/Pong-v5"


def set_global_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_pong_env(render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(PONG_ENV_ID, render_mode=render_mode)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    return env


def make_pong_vec_env(n_envs: int, log_dir: Optional[str] = None) -> VecMonitor:
    env = make_vec_env(
        PONG_ENV_ID,
        n_envs=n_envs,
    )

    if log_dir is not None:
        env = VecMonitor(env, log_dir)
    else:
        env = VecMonitor(env)
    return env
