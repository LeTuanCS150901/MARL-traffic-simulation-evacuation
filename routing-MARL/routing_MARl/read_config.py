import os
import yaml

# Global config dictionary to be accessed from any module
CONFIG = None

def read_config(config_filename: str = None) -> dict:
    """
    Reads the deep RL configuration YAML file and returns a dictionary.

    Args:
        config_filename (str): Filename of YAML config in ../config/.
                               Defaults to 'deep_rl.yaml'.

    Returns:
        dict: Configuration parameters.
    """
    if config_filename is None:
        config_filename = 'toynet.yaml'  # default config file name
    else:
        config_filename = config_filename + '.yaml'

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', config_filename)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # set the config so other modules can get this config by calling get_config() function
    set_config(config)

    return config


def set_config(config: dict):
    """
    Set the global configuration dictionary.

    Args:
        config (dict): Configuration dictionary to set globally.
    """
    global CONFIG
    CONFIG = config


def get_config() -> dict:
    """
    Get the global configuration dictionary.

    Returns:
        dict: Global configuration dictionary.
    """
    global CONFIG
    if CONFIG is None:
        raise RuntimeError("Config has not been set. Please call set_config(config) first.")
    return CONFIG