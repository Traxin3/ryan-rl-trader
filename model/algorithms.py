from .impala import get_impala_config


def get_algorithm_class(algo_name):
    name = algo_name.lower()
    if name == "impala":
        return get_impala_config
    raise ValueError(f"Unknown or unsupported algorithm: {algo_name}. Only 'impala' is supported.")
