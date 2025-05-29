"""tic.data.io
============

Low-level file download and archive extraction utilities for the `tic` package.

This module centralizes all interactions with remote URLs and local cache directories,
ensuring consistent logging, error handling, and path management.

Functions
---------
- download_file: Fetch a URL to a local path, skipping if already present.
- extract_zip: Unpack a ZIP archive to a directory, with optional cleanup.
- remove_cache: Remove the entire cache directory tree.
- list_datasets: Enumerate top-level dataset directories in the cache.
- remove_dataset: Delete a single dataset folder.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Union, Iterable

import requests
from tqdm import tqdm

from ..constant import DEFAULT_DATACACHE_DIR

# Initialize module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


PathLike = Union[str, Path]


def download_file(url: str, dest: PathLike) -> None:
    """
    Robust file downloader with custom headers and tqdm progress bar.

    Parameters
    ----------
    url : str
        URL of the file to download.
    dest : PathLike
        Path to save the downloaded file.
    """
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"Skipping download; file exists: {dest_path}")
        return

    headers = {
        "User-Agent": "Wget/1.21.1"
    }

    print(f"Downloading {url} → {dest_path}")

    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        print(f"Download complete: {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise RuntimeError(f"Download failed: {e}") from e


def extract_zip(
    archive: PathLike,
    out_dir: PathLike,
    cleanup: bool = True,
    members: Iterable[str] | None = None
) -> None:
    """
    Extract a ZIP archive to a target directory.

    Parameters
    ----------
    archive : Union[str, Path]
        Path to the .zip archive.
    out_dir : Union[str, Path]
        Directory where contents will be extracted.
    cleanup : bool, optional
        If True, delete the archive after successful extraction.
    members : Iterable[str], optional
        Specific members (file names) to extract; if None, extract all.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist.
    RuntimeError
        If extraction fails.
    """
    archive_path = Path(archive).expanduser().resolve()
    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        logger.error("Archive not found: %s", archive_path)
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    logger.info("Extracting %s → %s", archive_path.name, target_dir)
    try:
        with zipfile.ZipFile(archive_path, 'r') as zf:
            if members:
                zf.extractall(path=target_dir, members=members)
            else:
                zf.extractall(path=target_dir)
        logger.info("Extraction complete: %s", target_dir)
        if cleanup:
            archive_path.unlink(missing_ok=True)
            logger.info("Deleted archive: %s", archive_path)
    except Exception as e:
        logger.error("Failed to extract %s: %s", archive_path, e)
        raise RuntimeError(f"Extraction failed: {e}") from e


def remove_cache(cache_dir: PathLike = DEFAULT_DATACACHE_DIR) -> None:
    """
    Recursively delete the cache directory tree.

    Parameters
    ----------
    cache_dir : Union[str, Path]
        Root directory to remove.
    """
    path = Path(cache_dir).expanduser().resolve()
    if path.exists():
        shutil.rmtree(path)
        logger.info("Removed cache directory: %s", path)
    else:
        logger.warning("Cache directory not found: %s", path)


def list_datasets(cache_dir: PathLike = DEFAULT_DATACACHE_DIR) -> list[str]:
    """
    List all top-level dataset directories in the cache.

    Parameters
    ----------
    cache_dir : Union[str, Path]
        Directory containing cached datasets.

    Returns
    -------
    List[str]
        Sorted list of dataset directory names.
    """
    path = Path(cache_dir).expanduser().resolve()
    if not path.exists():
        logger.warning("Cache directory not found: %s", path)
        return []

    return sorted(
        entry.name
        for entry in path.iterdir()
        if entry.is_dir()
    )


def remove_dataset(dataset: str, cache_dir: PathLike = DEFAULT_DATACACHE_DIR) -> None:
    """
    Delete a single dataset folder from the cache.

    Parameters
    ----------
    dataset : str
        Name of the dataset directory to remove.
    cache_dir : Union[str, Path]
        Root cache directory.
    """
    ds_path = Path(cache_dir).expanduser().resolve() / dataset
    if ds_path.exists():
        shutil.rmtree(ds_path)
        logger.info("Removed dataset: %s", ds_path)
    else:
        logger.warning("Dataset not found: %s", ds_path)
