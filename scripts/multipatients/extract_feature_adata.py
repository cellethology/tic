#!/usr/bin/env python
"""Extract TIC features from multiple spatial regions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm

from tic.annotation.api import annotate_emt_state
from tic.data.codex.loader import list_regions, load_region
from tic.graph.utils import estimate_radius
from tic.wrappers.feature import FeatureWrapper
from utils.dataset.codex_upmc import (
    UPMC_EPITHELIAL_GENES,
    UPMC_MESENCHYMAL_GENES,
)
from tic.utils.logging import get_logger

###############################################################################
# --------------------------- Helper functions ------------------------------ #
###############################################################################

LOGGER = get_logger(__name__)

def tumour_types(adata: AnnData) -> List[str]:
    """Return unique tumour cell-type labels (case-insensitive)."""
    return [ct for ct in adata.obs["cell_type"].unique() if "tumor" in str(ct).lower()]


def mesenchymal_fraction(adata: AnnData, label_key: str = "emt_label") -> float:
    counts = adata.obs[label_key].value_counts(normalize=True)
    return float(counts.get("Mesenchymal", 0.0))


def concat_adatas(adatas: list[AnnData]) -> AnnData:
    if not adatas:
        raise ValueError("No AnnData objects were generated – perhaps your filters are too strict?")
    base_uns = {k: v for k, v in adatas[0].uns.items() if k != "X_predictors_meta"}
    combined = sc.concat(adatas, join="outer", label="region_id", keys=[a.uns["region_id"] for a in adatas])
    combined.uns |= base_uns  # merge dicts (3.9+)
    return combined


###############################################################################
# --------------------------- Core pipeline --------------------------------- #
###############################################################################


def build_features_for_regions(
    dataset: str,
    regions: list[str],
    epi_genes: list[str],
    mes_genes: list[str],
    *,
    min_m_fraction: float = 0.2,
    max_m_fraction: float = 0.8,
    tumour_labels: list[str] | None = None,
    
) -> Tuple[AnnData, pd.DataFrame]:
    """Return concatenated feature AnnData and a DataFrame of mesenchymal fractions."""
    adatas: list[AnnData] = []
    m_report: dict[str, float] = {}

    for rid in tqdm(regions, desc=f"[{dataset.upper()}] processing"):
        adata_full = load_region(dataset=dataset, region_id=rid)
        
        # Determine tumour labels for this region
        raw_labels = tumour_labels or tumour_types(adata_full)
        available_labels = set(adata_full.obs["cell_type"].unique())
        valid_labels = [l for l in raw_labels if l in available_labels]
        missing = set(raw_labels) - available_labels

        if missing:
            LOGGER.warning("Region %s: tumour labels not found and will be skipped: %s", rid, missing)
        if not valid_labels:
            LOGGER.warning("Region %s skipped (no valid tumour labels found)", rid)
            continue

        # Subset tumour cells only – speeds up downstream steps.
        tum_adata = adata_full[adata_full.obs["cell_type"].isin(valid_labels)].copy()
        if tum_adata.n_obs < 10:
            LOGGER.warning("Region %s skipped (fewer than 10 tumour cells)", rid)
            continue

        tum_adata = annotate_emt_state(
            tum_adata,
            epithelial_genes=epi_genes,
            mesenchymal_genes=mes_genes,
            cluster_on="scores",
            normalize_scores=True,
            general_score=True,
        )

        m_frac = mesenchymal_fraction(tum_adata)
        m_report[rid] = m_frac
        if m_frac < min_m_fraction:
            LOGGER.info("Region %s skipped (Mesenchymal %% < %.0f)", rid, min_m_fraction * 100)
            continue
        if m_frac > max_m_fraction:
            LOGGER.info("Region %s skipped (Mesenchymal %% > %.0f)", rid, max_m_fraction * 100)
            continue

        radius = estimate_radius(adata_full)
        fw = FeatureWrapper(
            recipe="tme_default",
            centre_types=valid_labels,  # <-- Only valid tumour labels
            graph_params={"method": "voronoi"},
            subgraph_params={"strategy": "radius", "radius": radius},
        )
        fea = fw.fit(adata_full)
        fea.uns["region_id"] = rid
        # for each cell , we will add the region_id to the cell_type
        fea.obs["region_id"] = rid
        adatas.append(fea)

    return concat_adatas(adatas), pd.DataFrame.from_dict(m_report, orient="index", columns=["mesenchymal_fraction"])

###############################################################################
# -------------------------- CLI entry-point -------------------------------- #
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TIC features from multiple regions.")
    parser.add_argument("--dataset", default="dfci", choices= ['upmc', 'charville', 'dfci'], help="Dataset identifier (e.g. 'upmc')")
    parser.add_argument("--out_dir", type=Path, default=Path("results/features"))
    parser.add_argument("--max_regions", type=int)
    parser.add_argument("--min_m_frac", type=float, default=0.2, help="Mesenchymal fraction threshold")
    parser.add_argument("--max_m_frac", type=float, default=0.8, help="Mesenchymal fraction threshold")
    parser.add_argument("--show_hist", action="store_true", help="Display histogram interactively")
    args = parser.parse_args()

    regions_all = list_regions(dataset=args.dataset.lower())
    if args.max_regions:
        regions_all = regions_all[: args.max_regions]

    epi_genes = UPMC_EPITHELIAL_GENES  # extend with other datasets as needed
    mes_genes = UPMC_MESENCHYMAL_GENES

    fea_adata, m_df = build_features_for_regions(
        dataset=args.dataset,
        regions=regions_all,
        epi_genes=epi_genes,
        mes_genes=mes_genes,
        min_m_fraction=args.min_m_frac,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"feature_adata_{args.dataset}"
    fea_adata.write_h5ad(prefix.with_suffix(".h5ad"))
    m_df.to_csv(prefix.with_name(prefix.name + "_m_fraction.csv"))

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(m_df["mesenchymal_fraction"], bins=20, edgecolor="black")
    plt.title("Mesenchymal fraction across regions")
    plt.xlabel("Fraction")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(prefix.with_name(prefix.name + "_m_fraction_hist.png"))
    if args.show_hist:
        plt.show()
    plt.close()

    LOGGER.info("All done - results written to %s", out_dir)


if __name__ == "__main__":
    main()