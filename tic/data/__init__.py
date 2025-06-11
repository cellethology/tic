# tic/data/__init__.py

"""
tic.data
========

High-level interface for downloading, extracting, validating and loading
spatial transcriptomics datasets (Xenium, CODEX) and low-level I/O utilities.
"""

from .io import (
    download_file,
    extract_zip,
    remove_cache,
    list_datasets,
    remove_dataset,
)
from .utils import (
    check_spatial_anndata,
    check_EMT_genes,
    get_cell_types,
    get_biomarkers,
)
from .xenium.download import ensure_xenium_dataset, download_xenium_dataset
from .xenium.loader import load_xenium_dataset
from .codex.download import ensure_codex_dataset, download_codex_dataset
from .codex.loader import list_regions, load_region
from .bgi.loader import load_bgi_dataset

__all__ = [
    # low-level I/O
    "download_file",
    "extract_zip",
    "remove_cache",
    "list_datasets",
    "remove_dataset",
    # validation / metadata utils
    "check_spatial_anndata",
    "check_EMT_genes",
    "get_cell_types",
    "get_biomarkers",
    # Xenium
    "ensure_xenium_dataset",
    "download_xenium_dataset",
    "load_xenium_dataset",
    # CODEX
    "ensure_codex_dataset",
    "download_codex_dataset",
    "list_regions",
    "load_region",
    # BGI
    "load_bgi_dataset",
]