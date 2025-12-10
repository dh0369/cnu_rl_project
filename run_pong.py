import time
import os
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN

from utils import PONG_ENV_ID


ALGOS = ["ppo", "a2c", "dqn"]


def eval_model(
    algo: str = "ppo",
    env_id: str = PONG_ENV_ID,
    model_base_dir: str = "models",
    n_episodes: int = 3,
):
    assert algo in ALGOS, f"algo must be one of {ALGOS}"

    env = gym.make(env_id, render_mode="human")

    if algo == "ppo":
        ModelClass = PPO
        suffix = "_PPO"
    elif algo == "a2c":
        ModelClass = A2C
        suffix = "_A2C"
    else:
        ModelClass = DQN
        suffix = "_DQN"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_base_dir, env_id + suffix)
    print(f"Loading model from: {model_path}")
    model = ModelClass.load(model_path, env=env)

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_rew = 0.0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            env.render()
            time.sleep(1 / 60)

            if terminated or truncated:
                print(f"[{algo.upper()}] Episode {ep + 1}: reward = {ep_rew:.2f}")
                time.sleep(1.0)
                break

    env.close()


if __name__ == "__main__":
    eval_model(algo="ppo")
    # eval_model(algo="a2c")
    # eval_model(algo="dqn")
