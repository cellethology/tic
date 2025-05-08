import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tic.constant import EMT_TF, EPITHELIAL_GENES, MESENCHYMAL_GENES

def _label_gene_type(gene: str, e_genes: list, m_genes: list, tf_genes: list) -> str:
    if gene in e_genes:
        return "Epithelial"
    elif gene in m_genes:
        return "Mesenchymal"
    elif gene in tf_genes:
        return "EMT TF"
    return "Other"

def _filter_emts_and_top(df: pd.DataFrame, key: str, top_n: int | None, emt_genes: list) -> pd.DataFrame:
    emt_df = df[df["biomarker"].isin(emt_genes)]

    if top_n is not None:
        top_df = df.reindex(df[key].abs().sort_values(ascending=False).index).head(top_n)
        combined_df = pd.concat([top_df, emt_df], axis=0).drop_duplicates(subset="biomarker")
    else:
        combined_df = emt_df

    return combined_df.copy()

def plot_monotonicity_metrics_bar(
    metrics: pd.DataFrame,
    biomarker_key: str = "biomarker",
    mono_correlation_key: str = "mono_correlation",
    title: str = "Biomarkers by Monotonicity Correlation",
    show_top_n: int | None = None,
    palette: dict | None = None,
    save_path: str | None = None,
    epithelial_genes: list | None = None,
    mesenchymal_genes: list | None = None,
    emt_tf_genes: list | None = None,
) -> plt.Figure:
    """
    Plot the monotonicity metrics with EMT markers highlighted.
    Sort the biomarkers by the absolute value of the monotonicity correlation and highlight the EMT markers.
    """
    # Use default if not provided
    epithelial_genes = list(epithelial_genes or EPITHELIAL_GENES)
    mesenchymal_genes = list(mesenchymal_genes or MESENCHYMAL_GENES)
    emt_tf_genes = list(emt_tf_genes or EMT_TF)
    emt_genes = epithelial_genes + mesenchymal_genes + emt_tf_genes

    df = metrics.copy()
    df["gene_type"] = df[biomarker_key].apply(
        lambda g: _label_gene_type(g, epithelial_genes, mesenchymal_genes, emt_tf_genes)
    )

    # 选出 top n + emt genes
    df = _filter_emts_and_top(df, mono_correlation_key, show_top_n, emt_genes)

    # 重新按照 abs(mono_correlation) 排序
    df = df.reindex(df[mono_correlation_key].abs().sort_values(ascending=False).index)

    if palette is None:
        palette = {
            "Epithelial": "#6ABAAB",
            "Mesenchymal": "#E9716A",
            "EMT TF": "#FFD700",
            "Other": "#AAB7B8"
        }

    plt.figure(figsize=(max(10, len(df) * 0.45), 6))
    bar = sns.barplot(
        data=df,
        x=biomarker_key,
        y=mono_correlation_key,
        hue="gene_type",
        dodge=False,
        palette=palette,
        edgecolor="black"
    )

    # Annotate bar values
    for p in bar.patches:
        height = p.get_height()
        if abs(height) > 0.05:
            bar.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2, height),
                         ha='center', va='bottom' if height > 0 else 'top',
                         fontsize=9, color='black', rotation=0)

    plt.title(title, fontsize=14)
    plt.ylabel("Monotonicity Correlation", fontsize=12)
    plt.xlabel("Biomarker", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Gene Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    return plt.gcf()

def plot_trend_metrics_bar(
    metrics: pd.DataFrame,
    biomarker_key: str = "biomarker",
    trend_statistic_key: str = "trend_statistic",
    title: str = "Biomarkers by Trend Statistic",
    save_path: str | None = None,
    show_top_n: int | None = None,
    epithelial_genes: list | None = None,
    mesenchymal_genes: list | None = None,
    emt_tf_genes: list | None = None,
) -> plt.Figure:
    """
    Plot the trend metrics with EMT markers highlighted.
    """
    epithelial_genes = epithelial_genes or EPITHELIAL_GENES
    mesenchymal_genes = mesenchymal_genes or MESENCHYMAL_GENES
    emt_tf_genes = emt_tf_genes or EMT_TF
    emt_genes = epithelial_genes + mesenchymal_genes + emt_tf_genes

    df = metrics.copy()
    df["gene_type"] = df[biomarker_key].apply(
        lambda g: _label_gene_type(g, epithelial_genes, mesenchymal_genes, emt_tf_genes)
    )
    df = _filter_emts_and_top(df, trend_statistic_key, show_top_n, emt_genes)
    df = df.sort_values(by=trend_statistic_key, ascending=False)

    palette = {"Epithelial": "#4daf4a", "Mesenchymal": "#e41a1c", "Other": "#377eb8"}

    plt.figure(figsize=(max(8, len(df) * 0.4), 6))
    sns.barplot(
        data=df,
        x=biomarker_key,
        y=trend_statistic_key,
        hue="gene_type",
        dodge=False,
        palette=palette,
    )
    plt.title(title)
    plt.ylabel("Trend Statistic")
    plt.xlabel("Biomarker")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Gene Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()