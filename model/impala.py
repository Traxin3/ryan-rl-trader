from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

def get_impala_config(env_config, model_config, training_config):
    """Build and return an IMPALAConfig.
    Parameters:
      - env_config: dict for environment
      - model_config: dict for model (e.g., transformer settings)
      - training_config: dict of hyperparams (lr, batch sizes, etc.)
    """
    from model.transformer_rl_module import TransformerRLModule
    return (
        ImpalaConfig()
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .training(
            train_batch_size=training_config.get("batch_size", 512),
            lr=training_config.get("learning_rate", 5e-4),
            minibatch_size=training_config.get("minibatch_size", 100),
            _enable_learner_api=True,
        )
        .resources(num_gpus=training_config.get("num_gpus", 1))
        .rollouts(
            num_rollout_workers=training_config.get("num_rollout_workers", 2),
            rollout_fragment_length=training_config.get("rollout_fragment_length", 50),
        )
        .rl_module(
            _enable_rl_module_api=True,
            rl_module_spec=SingleAgentRLModuleSpec(
                module_class=TransformerRLModule,
                model_config_dict=model_config.get("custom_model_config", {}),
            )
        )
    )
