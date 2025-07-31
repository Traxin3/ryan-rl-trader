import os
import yaml
from model.ppo_model import create_ppo_model
from gym_mtsim.envs.mt_env import MtEnv
from stable_baselines3.common.env_util import make_vec_env
from model.smart_coach import SmartCoach
from stable_baselines3.common.callbacks import BaseCallback

class CoachCallback(BaseCallback):
    def __init__(self, coach, config):
        super().__init__()
        self.coach = coach
        self.config = config
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            self.episode_rewards.append(self.locals['episode']['r'])
            
            if len(self.episode_rewards) % self.config['coach']['consult_frequency'] == 0:
                self.coach.analyze_and_adjust(
                    model=self.model,
                    recent_rewards=self.episode_rewards[-self.config['coach']['consult_frequency']:]
                )
        return True

def load_config():
    config_paths = [
        "config/config.yaml",
        os.path.join(os.path.dirname(__file__), "config/config.yaml")
    ]
    for path in config_paths:
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Config not found in: {config_paths}")

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
    
    coach = SmartCoach(config)
    
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=CoachCallback(coach, config),
        progress_bar=True
    )
    
    model.save("trained_model")
