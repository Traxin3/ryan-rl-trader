
import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
from .transformer import TransformerFeatureExtractor


class TradingValueHead(nn.Module):
    def __init__(self, features_dim, net_arch=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = features_dim
        for dim in net_arch:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*layers)
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.constant_(layer.bias, 0)
    def forward(self, features):
        return self.value_net(features)


class TradingPolicyHead(nn.Module):
    def __init__(self, features_dim, action_dim, net_arch=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = features_dim
        for dim in net_arch:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        self.shared_net = nn.Sequential(*layers)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.1)
        nn.init.constant_(self.mean_head.bias, 0)
        nn.init.constant_(self.log_std_head.weight, -2.0)
        nn.init.constant_(self.log_std_head.bias, -2.0)
    def forward(self, features):
        shared_features = self.shared_net(features)
        mean = torch.tanh(self.mean_head(shared_features))
        log_std = torch.clamp(self.log_std_head(shared_features), -5, 1)
        return mean, log_std


from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch

class TransformerRLlibModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        custom_cfg = model_config.get("custom_model_config", {})
        features_dim = custom_cfg.get("features_dim", 256)
        transformer_cfg = custom_cfg.get("transformer", {})
        self.feature_extractor = TransformerFeatureExtractor(obs_space, config=transformer_cfg)
        self.action_net = TradingPolicyHead(features_dim, num_outputs)
        self.value_net = TradingValueHead(features_dim)
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if isinstance(obs, dict):
            obs_tensor = torch.cat([v.float() for v in obs.values()], dim=-1)
        else:
            obs_tensor = obs.float()
        features = self.feature_extractor(obs_tensor)
        self._features = features
        mean, log_std = self.action_net(features)
        return mean, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.value_net(self._features).squeeze(-1)

ModelCatalog.register_custom_model("transformer_trading_model", TransformerRLlibModel)
