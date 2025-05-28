"""tic.codex.download
====================

Download and prepare Murine CODEX spatial datasets (UPMC, Charville, DFCI).

This module handles fetching raw ZIP archives from Zenodo, extracting them,
and tidying up directory hierarchies for downstream loading.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal, Union

from ...constant import DEFAULT_DATACACHE_DIR
from ...data.io import download_file, extract_zip

# Supported CODEX datasets and their download URLs
CODEX_DATASETS: dict[str, dict[str, str]] = {
    "upmc": {
        "url": "https://zenodo.org/records/13179600/files/upmc_raw_data.zip?download=1",
        "zip_name": "upmc_raw_data.zip",
    },
    "charville": {
        "url": "https://zenodo.org/records/13179600/files/charville_raw_data.zip?download=1",
        "zip_name": "charville_raw_data.zip",
    },
    "dfci": {
        "url": "https://zenodo.org/records/13179600/files/dfci_raw_data.zip?download=1",
        "zip_name": "dfci_raw_data.zip",
    },
}

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

PathLike = Union[str, Path]


def ensure_codex_dataset(
    dataset: Literal["upmc", "charville", "dfci"],
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
) -> Path:
    """
    Download (if needed) and unpack the CODEX dataset into a cache directory.

    Parameters
    ----------
    dataset
        One of 'upmc', 'charville', or 'dfci'.
    cache_dir
        Directory where datasets are cached.

    Returns
    -------
    Path
        Root directory containing the dataset files.

    Raises
    ------
    KeyError
        If `dataset` is not one of the supported keys.
    RuntimeError
        If download or extraction fails.
    """
    if dataset not in CODEX_DATASETS:
        raise KeyError(f"Unsupported CODEX dataset: {dataset}")

    root = Path(cache_dir).expanduser().resolve() / f"codex_{dataset}"
    root.mkdir(parents=True, exist_ok=True)

    cfg = CODEX_DATASETS[dataset]
    zip_path = root / cfg["zip_name"]

    # 1) Download ZIP if missing
    download_file(cfg["url"], zip_path)

    # 2) Extract contents (no cleanup by default)
    extract_zip(zip_path, root, cleanup=False)

    # 3) Flatten nested raw_data/ directory if present
    raw_dir = root / "raw_data"
    if raw_dir.is_dir():
        logger.info("Flattening nested directory: %s", raw_dir)
        for child in raw_dir.iterdir():
            child.rename(root / child.name)
        shutil.rmtree(raw_dir)
        logger.info("Removed nested directory: %s", raw_dir)

    return root


def download_codex_dataset(
    dataset: Literal["upmc", "charville", "dfci"],
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
) -> None:
    """
    Convenience wrapper to ensure a CODEX dataset is fetched and ready.

    This is equivalent to calling `ensure_codex_dataset` and ignoring its return value.
    """
    ensure_codex_dataset(dataset, cache_dir)
    logger.info("CODEX dataset '%s' is available at %s", dataset, cache_dir)
