import os
import shutil
from typing import Literal
import zipfile

import wget

from ..constant import DEFAULT_DATACACHE_DIR

# ── Utilities ────────────────────────────────────────────────────────────────
def download_file(url: str, save_path: str):
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}, skipping download.")
        return
    print(f"Downloading from {url} to {save_path} using wget ...")
    wget.download(url, save_path, bar=wget.bar_adaptive)  # or bar_thermometer
    print("\nDownload complete.")

def remove_cache(cache_dir: str = DEFAULT_DATACACHE_DIR):
    shutil.rmtree(cache_dir)

def list_datasets(cache_dir: str = DEFAULT_DATACACHE_DIR):
    return os.listdir(cache_dir)

def remove_dataset(dataset_name: str, cache_dir: str = DEFAULT_DATACACHE_DIR):
    os.remove(os.path.join(cache_dir, dataset_name))

# ── Codex UPMC ────────────────────────────────────────────────────────────────
CODEX_DATASETS = {
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

def download_codex_dataset(dataset: str | Literal["upmc", "charville", "dfci"], cache_dir: str = DEFAULT_DATACACHE_DIR):
    """
    Download and prepare Codex dataset (upmc, charville, or dfci) from Zenodo.

    Parameters
    ----------
    dataset : str
        Dataset name, one of "upmc", "charville", or "dfci".
    cache_dir : str
        Cache directory to store the dataset.
    """
    if dataset not in CODEX_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose from {list(CODEX_DATASETS.keys())}")

    dataset_info = CODEX_DATASETS[dataset]
    dataset_dir = os.path.join(cache_dir, f"codex_{dataset}")
    os.makedirs(dataset_dir, exist_ok=True)

    zip_path = os.path.join(dataset_dir, dataset_info["zip_name"])
    url = dataset_info["url"]

    if not os.path.exists(zip_path):
        print(f"Downloading Codex {dataset.capitalize()} dataset to {zip_path} ...")
        wget.download(url, zip_path)
        print("\nDownload complete.")
    else:
        print(f"Found existing zip file at {zip_path}, skipping download.")

    sample_file = os.path.join(dataset_dir, "1000.cell_data.csv")
    if not os.path.exists(sample_file):
        print("Unzipping dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("Unzip complete.")

        # Handle inner raw_data directory
        raw_data_path = os.path.join(dataset_dir, "raw_data")
        if os.path.isdir(raw_data_path):
            for fname in os.listdir(raw_data_path):
                shutil.move(os.path.join(raw_data_path, fname), dataset_dir)
            shutil.rmtree(raw_data_path)
            print("Moved files from raw_data and cleaned up directory.")
    else:
        print("Data already unzipped.")

    print(f"Codex {dataset.capitalize()} data is available in: {dataset_dir}")

# ── Xenium ────────────────────────────────────────────────────────────────────
def download_xenium_pancreas_cancer_data(cache_dir: str = DEFAULT_DATACACHE_DIR):
    '''
    Download Xenium Pancreas Cancer dataset from 10x Genomics -> ~.cache/tic/xenium_pancreas_cancer
    '''
    xenium_dir = os.path.join(cache_dir, "xenium_pancreas_cancer")
    os.makedirs(xenium_dir, exist_ok=True)

    base_url = "https://cf.10xgenomics.com/samples/xenium/1.6.0/Xenium_V1_hPancreas_Cancer_Add_on_FFPE"
    cell_groups_url = f"{base_url}/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_cell_groups.csv"
    outs_zip_url = f"{base_url}/Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs.zip"

    cell_groups_path = os.path.join(xenium_dir, "Xenium_V1_hPancreas_Cancer_cell_groups.csv")
    outs_zip_path = os.path.join(xenium_dir, "Xenium_V1_hPancreas_Cancer_outs.zip")

    download_file(cell_groups_url, cell_groups_path)
    download_file(outs_zip_url, outs_zip_path)

    expected_file = os.path.join(xenium_dir, "cells.csv.gz")
    if not os.path.exists(expected_file):
        print("Unzipping expression data...")
        with zipfile.ZipFile(outs_zip_path, 'r') as zip_ref:
            zip_ref.extractall(xenium_dir)
        print("Unzip complete.")
    else:
        print("Expression data already unzipped.")

    print(f"All data available in: {xenium_dir}")

# ── Xenium Colorectal Cancer ────────────────────────────────────────────────────
# TODO: find data source
def download_xenium_colorectal_cancer_data(cache_dir: str = DEFAULT_DATACACHE_DIR):
    xenium_dir = os.path.join(cache_dir, "xenium_colorectal_cancer")
    os.makedirs(xenium_dir, exist_ok=True)

    raise NotImplementedError("Xenium colorectal cancer data is not available yet.")
    