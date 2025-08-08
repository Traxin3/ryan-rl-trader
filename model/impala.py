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
    try:
        from model.custom_dists import register_custom_action_dists
        register_custom_action_dists()
    except Exception:
        pass

    num_rollout_workers = training_config.get("num_rollout_workers", 2)
    rollout_fragment_length = training_config.get("rollout_fragment_length", 50)
    num_envs_per_worker = training_config.get("num_envs_per_worker", 1)
    compress_observations = training_config.get("compress_observations", False)
    learner_queue_size = training_config.get("learner_queue_size", 16)
    min_time_s_per_iteration = training_config.get("min_time_s_per_iteration", 10)

    train_batch_size = training_config.get("batch_size", 512)
    lr = training_config.get("learning_rate", 5e-4)
    raw_minibatch = training_config.get("minibatch_size", "auto")
    if isinstance(raw_minibatch, str) and raw_minibatch != "auto":
        try:
            raw_minibatch = int(raw_minibatch)
        except Exception:
            raw_minibatch = "auto"
    if isinstance(raw_minibatch, int):
        if raw_minibatch % rollout_fragment_length != 0 or raw_minibatch > train_batch_size:
            max_mb = max(rollout_fragment_length, (train_batch_size // rollout_fragment_length) * rollout_fragment_length)
            minibatch_size = max_mb
        else:
            minibatch_size = raw_minibatch
    else:
        minibatch_size = "auto"

    num_gpus = training_config.get("num_gpus", 1)

    algo = (
        ImpalaConfig()
        .environment(env="TradingEnv", env_config=env_config)
        .framework("torch")
        .training(
            train_batch_size=train_batch_size,
            lr=lr,
            minibatch_size=minibatch_size,
            _enable_learner_api=True,
            model={"custom_action_dist": "compat_diag_gaussian"},
            learner_queue_size=learner_queue_size,
        )
        .resources(num_gpus=num_gpus)
        .rollouts(
            num_rollout_workers=num_rollout_workers,
            rollout_fragment_length=rollout_fragment_length,
            num_envs_per_worker=num_envs_per_worker,
            compress_observations=compress_observations,
        )
        .rl_module(
            _enable_rl_module_api=True,
            rl_module_spec=SingleAgentRLModuleSpec(
                module_class=TransformerRLModule,
                model_config_dict=model_config.get("custom_model_config", {}),
            )
        )
        .reporting(
            min_time_s_per_iteration=min_time_s_per_iteration,
        )
    )
    return algo
