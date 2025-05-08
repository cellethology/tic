# tic/utils/logging.py
import logging
from typing import Final, Dict
import os
import yaml

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with sane defaults."""
    fmt: Final = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    return logging.getLogger(name)


def save_experiment_config(config: Dict, experiment_dir: str, filename: str = "params.yaml") -> None:
    os.makedirs(experiment_dir, exist_ok=True)
    path = os.path.join(experiment_dir, filename)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[✓] Saved experiment parameters to: {path}")