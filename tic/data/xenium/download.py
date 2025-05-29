"""tic.xenium.download
====================

Download and unpack 10x Genomics Xenium spatial transcriptomics datasets.

Provides:
- XENIUM_DATASETS metadata registry
- ensure_xenium_dataset: fetch & extract archives
- download_xenium_dataset: simple wrapper
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Union

from ...constant import DEFAULT_DATACACHE_DIR
from ...data.io import download_file, extract_zip

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Registry of public Xenium datasets
XENIUM_DATASETS: dict[str, dict[str, str]] = {
    "xenium_ffpe_human_breast": {
        "zip": (
            "https://cf.10xgenomics.com/samples/xenium/1.0.2/"
            "Xenium_V1_FFPE_Human_Breast_ILC/"
            "Xenium_V1_FFPE_Human_Breast_ILC_outs.zip"
        ),
        "expr": "cell_feature_matrix.h5",
        "cells": "cells.csv.gz",
    },
    "xenium_pancreas_cancer": {
        "zip": (
            "https://cf.10xgenomics.com/samples/xenium/1.6.0/"
            "Xenium_V1_hPancreas_Cancer_Add_on_FFPE/"
            "Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"
        ),
        "extra": "Xenium_V1_hPancreas_Cancer_Add_on_FFPE_cell_groups.csv",
        "expr": "cell_feature_matrix.h5",
        "cells": "cells.csv.gz",
    },
    "xenium_kidney_cancer": {
        "zip": (
            "https://cf.10xgenomics.com/samples/xenium/1.5.0/"
            "Xenium_V1_hKidney_cancer_section/"
            "Xenium_V1_hKidney_cancer_section_outs.zip"
        ),
        "expr": "cell_feature_matrix.h5",
        "cells": "cells.csv.gz",
    },
    "xenium_invasive_lung_cancer": {
        "zip": (
            "https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/1.3.0/"
            "Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE/"
            "Xenium_Preview_Human_Lung_Cancer_With_Add_on_2_FFPE_outs.zip"
        ),
        "expr": "cell_feature_matrix.h5",
        "cells": "cells.csv.gz",
    },
}

PathLike = Union[str, Path]


def ensure_xenium_dataset(
    name: Literal["xenium_ffpe_human_breast", "xenium_pancreas_cancer"],
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
    *,
    force: bool = False,
) -> Path:
    """
    Download (if needed) and extract a Xenium dataset into cache_dir/name.

    Parameters
    ----------
    name
        Key in XENIUM_DATASETS registry.
    cache_dir
        Base directory for caching datasets.
    force
        If True, re-download and re-extract even if present.

    Returns
    -------
    Path
        Directory containing extracted dataset files.

    Raises
    ------
    KeyError
        If name is not registered.
    RuntimeError
        If download or extraction fails.
    """
    if name not in XENIUM_DATASETS:
        raise KeyError(f"Unknown Xenium dataset: '{name}'")
    
    # First check if the dataset is already in the cache directory
    root = Path(cache_dir).expanduser().resolve() / name
    if root.exists() and any(root.iterdir()) and not force:
        logger.info("Xenium dataset '%s' already exists at %s", name, root)
        return root

    # If not, download and extract the dataset
    cfg = XENIUM_DATASETS[name]
    root.mkdir(parents=True, exist_ok=True)

    # Download ZIP archive
    zip_url = cfg["zip"]
    zip_path = root / Path(zip_url).name
    if force or not zip_path.exists():
        download_file(zip_url, zip_path)

    # Extract all contents
    if force or not (root / cfg["expr"]).exists():
        extract_zip(zip_path, root, cleanup=False)
        # Flatten nested *_outs directory
        for sub in root.iterdir():
            if sub.is_dir() and sub.name.endswith("_outs"):
                for child in sub.iterdir():
                    child.rename(root / child.name)
                sub.rmdir()
                break

    # Download optional extra table (cell_groups)
    if extra := cfg.get("extra"):
        base_url = "/".join(zip_url.split("/")[:-1]) 
        extra_url = f"{base_url}/{extra}"
        extra_path = root / extra
        if force or not extra_path.exists():
            download_file(extra_url, extra_path)

    logger.info("Xenium dataset '%s' ready at %s", name, root)
    return root


def download_xenium_dataset(
    name: Literal["xenium_ffpe_human_breast", "xenium_pancreas_cancer"],
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
    *,
    force: bool = False,
) -> None:
    """
    Convenience wrapper to ensure the dataset is downloaded and extracted.

    Equivalent to calling ensure_xenium_dataset and discarding its return.
    """
    ensure_xenium_dataset(name, cache_dir, force=force)
    logger.info("Download step complete for Xenium '%s'", name)
