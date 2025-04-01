import gymnasium as gym

from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb

from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=1)

wandb.init(
    project="cartpole-tech-zoo",
    sync_tensorboard=True,
)

model = PPO("MlpPolicy", vec_env, device="cpu", verbose=1, tensorboard_log='logs')
model.learn(total_timesteps=100000, log_interval=4, callback=WandbCallback())
model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

env = gym.make("CartPole-v1")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
