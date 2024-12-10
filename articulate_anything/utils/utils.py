import sys
from collections import defaultdict
import subprocess
from typing import List, Dict, Any
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import torch
import os
import random
import logging
import numpy as np
import json
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy


def seed_everything(seed: int, torch_deterministic=False) -> None:
    logging.info(f"Setting seed {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)


def join_path(*args):
    return os.path.join(*args)


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def string_to_file(string: str, filename: str) -> None:
    with open(filename, 'w') as file:
        file.write(string)


def file_to_string(filename: str) -> str:
    with open(filename, 'r') as file:
        return file.read()


def create_task_config(cfg: DictConfig, task_name) -> DictConfig:
    task_config = deepcopy(cfg)
    task_config.out_dir = join_path(task_config.out_dir, task_name)
    return task_config


def load_config(config_path="../../conf", config_name="config"):
    """
    Load and merge Hydra configuration.

    :param config_path: Path to the config directory
    :param config_name: Name of the main config file (without .yaml extension)
    :return: Merged configuration object
    """
    # Initialize Hydra
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path)

    # Compose the configuration
    cfg = compose(config_name=config_name)

    return cfg


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def config_to_command(cfg: DictConfig, script_path: str, conda_env: str = "articulate-anything-clean") -> List[str]:
    """
    Convert a configuration to a command-line command, flattening nested structures.

    Args:
    cfg (DictConfig): The configuration to convert.
    script_path (str): The path to the Python script to run.
    conda_env (str): The name of the Conda environment to use.

    Returns:
    List[str]: The command as a list of strings.
    """
    # Convert the configuration to a flat dictionary
    flat_cfg = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # Convert the flat dictionary to command-line arguments
    cmd_args = [f"{k}={v}" for k, v in flat_cfg.items() if v is not None]
    return make_cmd(script_path, conda_env, cmd_args)


def make_cmd(script_path: str, conda_env: str = "articulate-anything-clean",
             cmd_args=[]):
    # Construct the command
    command = [
        "conda", "run", "-n", conda_env,
        "python", script_path
    ] + cmd_args

    return command


def run_subprocess(command: List[str], env=None) -> None:
    """
    Run a command as a subprocess.

    Args:
    command (List[str]): The command to run as a list of strings.

    Raises:
    subprocess.CalledProcessError: If the command fails.
    """
    # convert all element in command to string
    command = [str(c) for c in command]
    if env is None:
        env = os.environ.copy()
    try:
        subprocess.run(command, check=True, env=env)

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error: {e}")


class Steps:
    def __init__(self):
        self.steps = defaultdict(dict)
        self.order = []

    def add_step(self, name: str, result: Any):
        self.steps[name] = result
        self.order.append(name)

    def __getitem__(self, name):
        return self.steps[name]

    def __iter__(self):
        for name in self.order:
            yield name, self.steps[name]

    def __str__(self):
        return str(self.steps)

    def __repr__(self):
        return str(self.steps)


class HideOutput(object):
    # https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''

    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)
