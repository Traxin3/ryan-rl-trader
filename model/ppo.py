from __future__ import annotations
import os
import platform
import inspect
from typing import Dict, Any, Callable, List

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium import spaces

from .sb3_policy import SB3TransformerExtractor
from gym_mtsim.envs.mt_env import MtEnv


def _filter_env_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Filter dict to only kwargs supported by MtEnv.__init__."""
    valid_args = inspect.signature(MtEnv.__init__).parameters
    return {k: v for k, v in cfg.items() if k in valid_args}


def make_vec_envs(env_cfg: Dict[str, Any], n_envs: int = 1, use_subproc: bool = False):
    """Create a vectorized environment for SB3 based on config.
    On Windows, subproc is disabled regardless of flag (use DummyVecEnv).
    """
    is_windows = platform.system().lower() == "windows"

    def factory() -> Callable[[], MtEnv]:
        def _init():
            return MtEnv(**_filter_env_config(env_cfg))

        return _init

    if n_envs <= 1 or use_subproc is False or is_windows:
        return DummyVecEnv([factory() for _ in range(max(1, n_envs))])
    else:
        return SubprocVecEnv([factory() for _ in range(n_envs)])


def build_policy_kwargs(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build SB3 policy_kwargs using the custom Transformer features extractor.
    Respects model.features_dim and model.transformer.* settings.
    """
    features_dim = int(model_cfg.get("features_dim", 256))
    transformer_cfg = model_cfg.get("transformer", {})

    return dict(
        features_extractor_class=SB3TransformerExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            transformer_config=transformer_cfg,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
