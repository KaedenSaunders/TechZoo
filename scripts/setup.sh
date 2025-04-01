export PROJECT_PATH=/home/ks679318/TechZoo

cd $PROJECT_PATH
source .venv/bin/activate
uv pip install stable_baselines3
uv pip install gymnasium[classic-control]
uv pip install wandb
uv pip install tensorboard
