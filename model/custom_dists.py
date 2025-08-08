import torch
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian

class CompatDiagGaussian(TorchDiagGaussian):
    @classmethod
    def from_logits(cls, logits):
        return cls(logits, None)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        import numpy as np
        return int(np.prod(action_space.shape)) * 2


def register_custom_action_dists():
    try:
        from ray.rllib.models import ModelCatalog
        ModelCatalog.register_custom_action_dist("compat_diag_gaussian", CompatDiagGaussian)
        return True
    except Exception:
        return False
