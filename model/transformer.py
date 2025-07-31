import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__(observation_space, d_model)
        window_size, feature_dim = observation_space['features'].shape
        self.d_model = d_model
        self.input_proj = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, obs):
        x = obs['features']
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)