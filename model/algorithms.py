# Deprecated: Ray IMPALA configuration removed during SB3 migration.

def get_algorithm_class(algo_name):
    raise ValueError("Ray/RLlib algorithms are no longer supported. Use Stable-Baselines3 via main.py run_training().")
