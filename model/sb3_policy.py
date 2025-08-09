import torch
import torch.nn as nn
from typing import Dict, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from .transformer import TransformerFeatureExtractor


class SB3TransformerExtractor(BaseFeaturesExtractor):
    """SB3 FeaturesExtractor that wraps our TransformerFeatureExtractor and fuses
    liquidity and orders signals when present in the observation dict.

    Expects observation_space to be a gymnasium.spaces.Dict with keys including
    'features' (sequence [T, F]) and optionally 'liquidity' ([T, L]) and 'orders' (flat vector),
    plus account metrics 'balance','equity','margin'.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 256,
        transformer_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim)

        transformer_config = transformer_config or {}
        d_model = transformer_config.get("d_model", features_dim)

        self.transformer = TransformerFeatureExtractor(
            observation_space,
            config={
                "d_model": d_model,
                "nhead": transformer_config.get("nhead", 8),
                "num_layers": transformer_config.get("num_layers", 4),
                "dropout": transformer_config.get("dropout", 0.1),
                "scales": transformer_config.get("scales", [1, 2, 4]),
            },
        )

        self.use_liquidity = (
            isinstance(observation_space, spaces.Dict)
            and "liquidity" in observation_space.spaces
        )
        if self.use_liquidity:
            liq_space = observation_space.spaces["liquidity"]
            assert isinstance(liq_space, spaces.Box) and len(liq_space.shape) == 2
            liq_in = liq_space.shape[1]
            self.liquidity_mlp = nn.Sequential(
                nn.Linear(liq_in * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.liquidity_mlp = None

        self.use_orders = (
            isinstance(observation_space, spaces.Dict)
            and "orders" in observation_space.spaces
        )
        if self.use_orders:
            ord_space = observation_space.spaces["orders"]
            assert isinstance(ord_space, spaces.Box) and len(ord_space.shape) == 1
            ord_in = ord_space.shape[0]
            self.orders_mlp = nn.Sequential(
                nn.Linear(ord_in, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.orders_mlp = None

        concat_dim = d_model
        if self.use_liquidity:
            concat_dim += d_model
        if self.use_orders:
            concat_dim += d_model
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )
        self._features_dim = features_dim

    def forward(self, observations):
        if isinstance(observations, dict):
            obs = {k: v.float() for k, v in observations.items()}
        else:
            obs = observations.float()

        x = self.transformer(obs)
        feats = [x]

        if self.use_liquidity and "liquidity" in obs:
            liq = obs["liquidity"]  # [B, T, L] or [T, L]
            if liq.dim() == 2:
                liq = liq.unsqueeze(0)
            mean_pool = liq.mean(dim=1)
            max_pool, _ = liq.max(dim=1)
            liq_vec = torch.cat([mean_pool, max_pool], dim=-1)
            feats.append(self.liquidity_mlp(liq_vec))

        if self.use_orders and "orders" in obs:
            ords = obs["orders"]
            if ords.dim() > 2:
                ords = ords.view(ords.size(0), -1)
            feats.append(self.orders_mlp(ords))

        fused = torch.cat(feats, dim=-1)
        return self.fusion(fused)
