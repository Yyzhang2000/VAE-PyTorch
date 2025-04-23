import torch
import numpy as np
import os
import random

import logging
import yaml


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_tensorboard_writer(log_dir):
    """Set the TensorBoard writer to log info in `log_dir`.

    Example:
    ```
    writer = set_tensorboard_writer("logs")
    writer.add_scalar("Loss/train", loss.item(), step)
    ```

    Args:
        log_dir: (string) where to log
    """
    from torch.utils.tensorboard import SummaryWriter  # type: ignore

    writer = SummaryWriter(log_dir=log_dir)
    return writer


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def get_device():
    """
    Get the device to be used for training.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Silicon GPU (MPS).")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")

    return device


def config_to_dict(config):
    from dataclasses import asdict, is_dataclass

    def convert(value):
        if is_dataclass(value):
            return convert(asdict(value))  # type: ignore
        elif isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        else:
            return value

    return convert(config)


def save_config(train_config, model_config):
    """
    Save the config to a YAML file.
    """
    config_dict = {
        "train_config": config_to_dict(train_config),
        "model_config": config_to_dict(model_config),
    }
    with open(os.path.join(train_config.model_dir, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f)


def load_config(config_path):
    """
    Load the config from a YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
