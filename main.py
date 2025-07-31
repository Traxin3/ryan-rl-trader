import platform

# Detect if MetaTrader5 is available (for Colab/cloud compatibility)
MT5_AVAILABLE = False
if platform.system() == "Windows":
    try:
        import MetaTrader5 as mt5
        MT5_AVAILABLE = True
    except ImportError:
        print("MetaTrader5 not installed. Will use local cache or skip MT5 features.")
else:
    print("MetaTrader5 is only available on Windows. Running in cloud/offline mode.")
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
import torch.nn as nn
import gymnasium as gym
from gym_mtsim.envs.mt_env import MtEnv, TensorboardMetricsCallback

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        features_dim = d_model
        super().__init__(observation_space, features_dim)
        window_size, feature_dim = observation_space['features'].shape
        self.d_model = d_model
        self.input_proj = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, obs):
        x = obs['features']  # shape: (batch, window, feature_dim)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # (batch, d_model, window)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        return x

class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(d_model=64, nhead=4, num_layers=2, dropout=0.1),
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )

def create_env():
    return MtEnv(
        symbols=['EURUSD'],
        timeframes=[1, 5, 15],
        window_size=30,
        symbol_max_orders=2,
        reward_scaling=1.0,
        max_leverage=5.0,
        risk_adjusted_reward=True,
        early_close_bonus=0.1,
        diversification_bonus=0.05
    )



if __name__ == "__main__":
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    venv = make_vec_env(create_env, n_envs=1)
    env = venv  # Use venv for evaluation and rendering as well
    
    metrics_callback = TensorboardMetricsCallback()
    callback = CallbackList([metrics_callback])
    
    model = PPO(
        policy=TransformerPolicy,
        env=venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device='auto',
    )
    
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    
    try:
        model.learn(
            total_timesteps=1_500_000,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False
        )
        model.save("ppo_transformer_mtsim_final")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save("ppo_transformer_mtsim_interrupted")
    
        print("\n=== Final Evaluation ===")
    model = PPO.load("ppo_transformer_mtsim_final", env=venv)
    obs = venv.reset()
    rewards = []
    steps = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)  # Changed from 5 to 4 outputs
        rewards.append(reward)
        steps += 1
        
        if done or steps >= 10000:
            break

    print(f"Evaluation completed. Total reward: {sum(rewards):.2f} over {steps} steps")
    print(f"Average reward per step: {np.mean(rewards):.4f}")
    
    print("\n=== Performance Metrics ===")
    print(f"Win Rate: {info['win_rate']:.2%}")
    print(f"Avg Trade Profit: {info['avg_trade_profit']:.4f}")
    print(f"Max Drawdown: {info['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {info['sharpe_ratio']:.2f}")
    print(f"Avg Holding Time: {info['avg_holding_time']:.2f} steps")
    
    try:
        venv.render(mode='advanced_figure')
    except Exception as e:
        print(f"Rendering failed: {str(e)}")
