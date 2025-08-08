import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.utils.annotations import override
from .transformer import TransformerFeatureExtractor

class TransformerRLModule(TorchRLModule):
    def __init__(self, config: RLModuleConfig):
        super().__init__(config)
        transformer_cfg = config.model_config_dict.get('transformer', {}) if hasattr(config, 'model_config_dict') and config.model_config_dict else {}
        obs_space = config.observation_space
        act_space = config.action_space

        price_cfg = {**transformer_cfg, 'scales': transformer_cfg.get('price_scales', transformer_cfg.get('scales', [1,2,4]))}
        liq_cfg = {**transformer_cfg, 'scales': transformer_cfg.get('liq_scales', [1])}

        if hasattr(obs_space, 'spaces') and 'features' in obs_space.spaces:
            features_space = obs_space.spaces['features']
        else:
            features_space = obs_space
        self.feature_extractor = TransformerFeatureExtractor(features_space, config=price_cfg)

        self.has_liquidity = hasattr(obs_space, 'spaces') and ('liquidity' in obs_space.spaces)
        features_dim = transformer_cfg.get('d_model', 256)
        if self.has_liquidity:
            liquidity_space = obs_space.spaces['liquidity']
            self.liquidity_extractor = TransformerFeatureExtractor(liquidity_space, config=liq_cfg)
            self.cross_attn = nn.MultiheadAttention(features_dim, transformer_cfg.get('nhead', 8), batch_first=True)
            self.fusion = nn.Sequential(
                nn.Linear(features_dim * 2, features_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

        # Orders context branch
        self.has_orders = hasattr(obs_space, 'spaces') and ('orders' in obs_space.spaces)
        self.orders_ctx_dim = 64
        if self.has_orders:
            orders_dim = int(obs_space.spaces['orders'].shape[0])
            self.orders_context = nn.Sequential(
                nn.Linear(orders_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(128, self.orders_ctx_dim),
                nn.LayerNorm(self.orders_ctx_dim),
            )
            self.orders_fusion = nn.Sequential(
                nn.Linear(features_dim + self.orders_ctx_dim, features_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

        self.use_time_tokens = transformer_cfg.get('use_time_tokens', True)
        if self.use_time_tokens:
            self.time_tokens = nn.Parameter(torch.zeros(2, features_dim))
            nn.init.normal_(self.time_tokens, std=0.02)

        if hasattr(act_space, 'n'):
            self.is_discrete = True
            self.action_dim = act_space.n
        else:
            self.is_discrete = False
            self.action_dim = act_space.shape[0]

        if self.is_discrete:
            self.policy_head = nn.Sequential(
                nn.Linear(features_dim, features_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(features_dim // 2, self.action_dim),
            )
        else:
            self.policy_backbone = nn.Sequential(
                nn.Linear(features_dim, features_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.mean_head = nn.Linear(features_dim // 2, self.action_dim)
            self.log_std_head = nn.Linear(features_dim // 2, self.action_dim)

        self.value_head = nn.Sequential(
            nn.Linear(features_dim, features_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim // 2, 1),
        )

    def _append_time_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_time_tokens:
            return x
        B = x.size(0)
        tokens = self.time_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, 2, D]
        x_seq = torch.cat([x.unsqueeze(1), tokens], dim=1)  # [B, 3, D]
        return torch.mean(x_seq, dim=1)

    def _encode(self, batch):
        feats = self.feature_extractor(batch['obs'])
        fused = feats
        if self.has_liquidity:
            liq = batch['obs'].get('liquidity') if isinstance(batch['obs'], dict) else None
            if liq is not None:
                liq_enc = self.liquidity_extractor(liq)
                q = liq_enc.unsqueeze(1)
                k = v = feats.unsqueeze(1)
                attn_out, _ = self.cross_attn(q, k, v)
                cross = attn_out.squeeze(1)
                fused = self.fusion(torch.cat([feats, cross], dim=-1))
        
        if self.has_orders and isinstance(batch['obs'], dict) and ('orders' in batch['obs']):
            orders_vec = batch['obs']['orders']
            orders_ctx = self.orders_context(orders_vec)
            fused = self.orders_fusion(torch.cat([fused, orders_ctx], dim=-1))
        return self._append_time_tokens(fused)

    @override(TorchRLModule)
    def _forward_inference(self, batch):
        features = self._encode(batch)
        if self.is_discrete:
            logits = self.policy_head(features)
        else:
            h = self.policy_backbone(features)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h)
            logits = torch.cat([mean, log_std], dim=-1)
        return {"action_dist_inputs": logits}

    @override(TorchRLModule)
    def _forward_exploration(self, batch):
        return self._forward_inference(batch)

    @override(TorchRLModule)
    def _forward_train(self, batch):
        features = self._encode(batch)
        if self.is_discrete:
            logits = self.policy_head(features)
        else:
            h = self.policy_backbone(features)
            mean = self.mean_head(h)
            log_std = self.log_std_head(h)
            logits = torch.cat([mean, log_std], dim=-1)
        value = self.value_head(features)
        return {"action_dist_inputs": logits, "values": value.squeeze(-1)}

   
    def get_inference_action_dist_cls(self):
        try:
            # Prefer Diagonal Gaussian for broad compatibility (has from_logits in older RLlib)
            from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian, TorchCategorical
            return TorchCategorical if self.is_discrete else TorchDiagGaussian
        except Exception:
            try:
                from ray.rllib.models.torch.torch_action_dist import TorchCategorical
                return TorchCategorical
            except Exception:
                # If imports fail entirely, raise to surface the issue
                raise

    def get_exploration_action_dist_cls(self):
        return self.get_inference_action_dist_cls()

    @override(TorchRLModule)
    def forward_inference(self, batch):
        return self._forward_inference(batch)

    @override(TorchRLModule)
    def forward_exploration(self, batch):
        return self._forward_exploration(batch)

    @override(TorchRLModule)
    def forward_train(self, batch):
        return self._forward_train(batch)
