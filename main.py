import os
import yaml
from model.ppo_model import create_ppo_model
from gym_mtsim.envs.mt_env import MtEnv
from stable_baselines3.common.env_util import make_vec_env

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def create_env(config):
    return MtEnv(
        symbols=config['env']['symbols'],
        timeframes=config['env']['timeframes'],
        window_size=config['env']['window_size'],
        max_leverage=config['env']['max_leverage']
    )

if __name__ == "__main__":
    config = load_config()
    venv = make_vec_env(lambda: create_env(config), n_envs=1)
    model = create_ppo_model(venv, config)
    
    model.learn(total_timesteps=1_500_000)
    model.save("ppo_transformer_mtsim_final")