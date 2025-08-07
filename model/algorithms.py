from .ppo import get_ppo_config
from .impala import get_impala_config

def get_algorithm_class(algo_name):
    if algo_name.lower() == "ppo":
        return get_ppo_config
    elif algo_name.lower() == "impala":
        return get_impala_config
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
