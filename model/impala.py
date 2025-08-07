from ray.rllib.algorithms.impala import ImpalaConfig

def get_impala_config(env_config, model_config, ppo_config):
    return (
        ImpalaConfig()
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=1)
        .training(
            model=model_config,
            train_batch_size=ppo_config.get("batch_size", 512),
        )
        .resources(num_gpus=0)
    )
