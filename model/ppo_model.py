from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from .transformer import TransformerFeatureExtractor

class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(d_model=128, nhead=4, num_layers=2, dropout=0.1),
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )

def create_ppo_model(env, config):
    return PPO(
        policy=TransformerPolicy,
        env=env,
        **config['ppo'],
        device='auto',
        verbose=1,
    )