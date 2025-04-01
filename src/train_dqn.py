import gymnasium as gym

from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback
import wandb

wandb.init(
    project="cartpole-tech-zoo",
    sync_tensorboard=True,
)

env = gym.make("CartPole-v1", render_mode="human")

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log='logs')
model.learn(total_timesteps=10000, log_interval=4, callback=WandbCallback())
model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
