from ray.rllib.algorithms.ppo import PPOConfig

def get_ppo_config(env_config, model_config, ppo_config):
    return (
        PPOConfig()
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=1)
        .training(
            model=model_config,
            train_batch_size=ppo_config.get("batch_size", 512),
            gamma=ppo_config.get("gamma", 0.999),
            lambda_=ppo_config.get("gae_lambda", 0.98),
            clip_param=ppo_config.get("clip_range", 0.1),
            vf_clip_param=ppo_config.get("clip_range_vf", 0.2),
            entropy_coeff=ppo_config.get("ent_coef", 0.001),
            grad_clip=ppo_config.get("max_grad_norm", 1.0),
            vf_loss_coeff=ppo_config.get("vf_coef", 0.8),
        )
        .resources(num_gpus=0)
    )
