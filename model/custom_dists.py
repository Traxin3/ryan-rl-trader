import torch
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian

# A compatibility wrapper adding from_logits expected by some RLlib policy paths
class CompatDiagGaussian(TorchDiagGaussian):
    @classmethod
    def from_logits(cls, logits):
        # TorchDiagGaussian usually accepts (inputs, model) in ctor; model can be None
        return cls(logits, None)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        # Mean and log_std for each action dim
        import numpy as np
        return int(np.prod(action_space.shape)) * 2


def register_custom_action_dists():
    try:
        from ray.rllib.models import ModelCatalog
        ModelCatalog.register_custom_action_dist("compat_diag_gaussian", CompatDiagGaussian)
        return True
    except Exception:
        return False
