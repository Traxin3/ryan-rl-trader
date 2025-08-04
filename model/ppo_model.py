from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from .transformer import TransformerFeatureExtractor

class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(
                d_model=256,        # Increased model size for better feature extraction
                nhead=8,            # More attention heads for multi-aspect analysis
                num_layers=3,       # Deeper network for complex patterns
                dropout=0.1
            ),
            net_arch=dict(
                pi=[512, 256, 128], # Larger policy network
                vf=[512, 256, 128]  # Larger value network
            )
        )

def create_ppo_model(env, config):
    return PPO(
        policy=TransformerPolicy,
        env=env,
        **config['ppo'],
        device='auto',
        verbose=1,
        tensorboard_log="./tensorboard_logs/"  # Enable tensorboard logging
    )
