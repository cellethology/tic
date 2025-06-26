# file: tic/features/recipes.py
"""Built-in recipe definitions and helpers."""
from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = ["get_recipe"]


_BUILTIN: Dict[str, Dict[str, Dict[str, Any]]] = {
    # cell‑level only
    "cell_default": {
        "centre_gene": {},
        "centre_gene_hvg": {},
    },
    # full micro‑environment default
    "tme_default": {
        "centre_gene": {},
        "composition": {"normalize": True},
        "centre_gene_comp": {},
        "neighbor_gene_sum": {},
        "same_type_gene_sum": {},
        "same_type_gene_average": {},
        "celltype_gene_count": {},

    },
    "geometry_features": {
        "geometry_basic": {},
        "fourier_features": {},
    },
}


def get_recipe(name_or_dict: str | Mapping[str, Mapping[str, Any]]):  # noqa: D401
    """Return a *deep copy* of recipe dict."""
    import copy

    if isinstance(name_or_dict, str):
        if name_or_dict not in _BUILTIN:
            raise KeyError(f"Unknown recipe '{name_or_dict}'. Built-ins: {list(_BUILTIN)}")
        return copy.deepcopy(_BUILTIN[name_or_dict])
    # assume dict
    return copy.deepcopy(name_or_dict)