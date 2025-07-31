import os
import yaml
from model.ppo_model import create_ppo_model
from gym_mtsim.envs.mt_env import MtEnv
from stable_baselines3.common.env_util import make_vec_env

def load_config():
    # Try possible config file locations
    config_paths = [
        "config/config.yaml",  # Local development path
        "/content/ryan-rl-trader/config/config.yaml",  # Google Colab path
        os.path.join(os.path.dirname(__file__), "config/config.yaml")  # Relative to script
    ]
    
    for path in config_paths:
        try:
            with open(path) as f:
                print(f"Successfully loaded config from: {path}")
                return yaml.safe_load(f)
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError(
        f"Could not find config.yaml in any of these locations:\n"
        f"{config_paths}\n"
        f"Please ensure the config file exists in one of these paths."
    )

def create_env(config):
    return MtEnv(
        symbols=config['env']['symbols'],
        timeframes=config['env']['timeframes'],
        window_size=config['env']['window_size'],
        max_leverage=config['env']['max_leverage'],
        reward_scaling=config['env'].get('reward_scaling', 1.0),
        risk_adjusted_reward=config['env'].get('risk_adjusted_reward', True)
    )

if __name__ == "__main__":
    try:
        config = load_config()
        venv = make_vec_env(lambda: create_env(config), n_envs=1)
        model = create_ppo_model(venv, config)
        
        model.learn(total_timesteps=1_500_000, progress_bar=True, verbose=1)
        model.save("ppo_transformer_mtsim_final")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise