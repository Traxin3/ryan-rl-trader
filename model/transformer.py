import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiScaleAttention(nn.Module):
    """Multi-scale attention for different trading timeframes"""

    def __init__(self, d_model, nhead, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead, batch_first=True) for _ in scales]
        )
        self.fusion = nn.Linear(d_model * len(scales), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        outputs = []

        for i, (scale, attn) in enumerate(zip(self.scales, self.attentions)):
            if scale == 1:
                attended, _ = attn(x, x, x)
                outputs.append(attended)
            else:
                pooled_len = seq_len // scale
                if pooled_len > 0:
                    pooled = F.adaptive_avg_pool1d(
                        x.transpose(1, 2), pooled_len
                    ).transpose(1, 2)
                    attended, _ = attn(pooled, pooled, pooled)
                    upsampled = F.interpolate(
                        attended.transpose(1, 2),
                        size=seq_len,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                    outputs.append(upsampled)
                else:
                    outputs.append(x)

        fused = self.fusion(torch.cat(outputs, dim=-1))
        return self.norm(fused + x)


class TradingTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, scales=None):
        super().__init__()
        self.multi_scale_attn = MultiScaleAttention(
            d_model, nhead, scales=scales or [1, 2, 4]
        )
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x2 = self.multi_scale_attn(x)
        x = self.norm1(x + x2)

        x2, _ = self.self_attn(x, x, x)
        x = self.norm2(x + self.dropout(x2))

        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout(x2))

        return x


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, observation_space, config=None):
        super().__init__()
        if config is None:
            config = {}
        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        num_layers = config.get("num_layers", 4)
        dropout = config.get("dropout", 0.1)
        scales = config.get("scales", [1, 2, 4])

        if (
            hasattr(observation_space, "spaces")
            and "features" in observation_space.spaces
        ):
            window_size, feature_dim = observation_space.spaces["features"].shape
        elif hasattr(observation_space, "shape") and len(observation_space.shape) == 2:
            window_size, feature_dim = observation_space.shape
        elif hasattr(observation_space, "shape") and len(observation_space.shape) == 1:
            total_features = observation_space.shape[0]
            window_size = 50  # Default window size
            feature_dim = total_features // window_size
        else:
            window_size, feature_dim = 50, 256

        self.d_model = d_model
        self.window_size = window_size
        self.expected_feature_dim = int(feature_dim)
        self._warned_feat_dim_mismatch = False

        self.input_proj = nn.Sequential(
            nn.Linear(self.expected_feature_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_encoding = PositionalEncoding(d_model, window_size)

        self.transformer_blocks = nn.ModuleList(
            [
                TradingTransformerBlock(
                    d_model, nhead, d_model * 4, dropout, scales=scales
                )
                for _ in range(num_layers)
            ]
        )

        self.market_context = nn.Sequential(
            nn.Linear(3, 32),  # balance, equity, margin
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.attention_pool = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))

        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # 3 pooling strategies
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.final_proj = nn.Sequential(
            nn.Linear(d_model + 64, d_model),  # sequence + market features
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.sequence_only_proj = nn.Sequential(
            nn.Linear(d_model, d_model),  # only sequence features
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, obs):
        if isinstance(obs, dict) and "features" in obs:
            x = obs["features"]
        else:
            x = obs

        if x.dim() == 2:
            batch_size = x.shape[0]
            total_features = x.shape[1]

            if total_features % self.window_size != 0:
                target_size = (
                    total_features // self.window_size + 1
                ) * self.window_size
                if total_features < target_size:
                    padding = target_size - total_features
                    x = F.pad(x, (0, padding))
                    total_features = target_size

            feature_dim = total_features // self.window_size
            x = x.view(batch_size, self.window_size, feature_dim)
        elif x.dim() == 1:
            total_features = x.shape[0]

            if total_features % self.window_size != 0:
                target_size = (
                    total_features // self.window_size + 1
                ) * self.window_size
                if total_features < target_size:
                    padding = target_size - total_features
                    x = F.pad(x, (0, padding))
                    total_features = target_size

            feature_dim = total_features // self.window_size
            x = x.view(1, self.window_size, feature_dim)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(
                f"Unsupported tensor dimension: {x.dim()}. Expected 1D, 2D, or 3D tensor."
            )

        if x.size(-1) != self.expected_feature_dim:
            diff = x.size(-1) - self.expected_feature_dim
            if not self._warned_feat_dim_mismatch:
                try:
                    print(
                        f"[TransformerFeatureExtractor] Warning: feature dim mismatch runtime={x.size(-1)} expected={self.expected_feature_dim}. {'Trimming' if diff>0 else 'Padding'} to match."
                    )
                except Exception:
                    pass
                self._warned_feat_dim_mismatch = True
            if diff > 0:
                x = x[..., : self.expected_feature_dim]
            else:
                pad_width = -diff
                pad = x.new_zeros(x.size(0), x.size(1), pad_width)
                x = torch.cat([x, pad], dim=-1)

        x = self.input_proj(x)

        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x_t = x.transpose(1, 2)  # [batch, d_model, seq_len]

        avg_pooled = self.global_pool(x_t).squeeze(-1)

        max_pooled = self.max_pool(x_t).squeeze(-1)

        attention_weights = self.attention_pool(x)  # [batch, seq_len, 1]
        attention_pooled = torch.sum(x * attention_weights, dim=1)  # [batch, d_model]

        pooled = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)
        sequence_features = self.output_proj(pooled)

        if isinstance(obs, dict) and all(
            key in obs for key in ["balance", "equity", "margin"]
        ):
            market_state = torch.stack(
                [
                    (
                        obs["balance"].squeeze(-1)
                        if obs["balance"].dim() > 1
                        else obs["balance"]
                    ),
                    (
                        obs["equity"].squeeze(-1)
                        if obs["equity"].dim() > 1
                        else obs["equity"]
                    ),
                    (
                        obs["margin"].squeeze(-1)
                        if obs["margin"].dim() > 1
                        else obs["margin"]
                    ),
                ],
                dim=1,
            )
            market_features = self.market_context(market_state)
            combined_features = torch.cat([sequence_features, market_features], dim=1)
            final_features = self.final_proj(combined_features)
        else:
            final_features = self.sequence_only_proj(sequence_features)

        return final_features
