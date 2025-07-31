import os
import yaml
import numpy as np
from typing import Dict, Any
from model.ppo_model import create_ppo_model
from gym_mtsim.envs.mt_env import MtEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from model.smart_coach import SmartCoach

class TrainingMonitor(BaseCallback):
    def __init__(self, config: Dict[str, Any], coach: SmartCoach, verbose=0):
        super().__init__(verbose)
        self.coach = coach
        self.config = config
        self.episode_rewards = []
        self.current_episode_rewards = []
        
    def _on_step(self) -> bool:
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_rewards.append(reward)
        
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(np.sum(self.current_episode_rewards))
            self.current_episode_rewards = []
            
            if len(self.episode_rewards) % self.config['coach'].get('consult_frequency', 10) == 0:
                self.coach.analyze_performance(
                    episode_rewards=self.episode_rewards,
                    current_config=self.config,
                    model=self.model
                )
        
        return True

def load_config():
    config_paths = [
        "config/config.yaml",
        "/content/ryan-rl-trader/config/config.yaml",
        os.path.join(os.path.dirname(__file__), "config/config.yaml")
    ]
    
    for path in config_paths:
        try:
            with open(path) as f:
                print(f"Successfully loaded config from: {path}")
                return yaml.safe_load(f)
        except FileNotFoundError:
            continue
    
    raise FileNotFoundError(f"Could not find config.yaml in any of these locations:\n{config_paths}")

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
        
      
        coach = SmartCoach(config)
        
        
        model.learn(
            total_timesteps=1_500_000,
            progress_bar=True,
            verbose=1,
            callback=TrainingMonitor(config, coach)
        )
        
        model.save("ppo_transformer_mtsim_final")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
