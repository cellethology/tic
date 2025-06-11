from tic.data import load_bgi_dataset
from tic.constant import EPITHELIAL_GENES, MESENCHYMAL_GENES, EMT_TF
from tic.metrics import compute_variability
from tic.plotting import plot_variability

B03425E1_adata = load_bgi_dataset("B03425E1", data_dir="/Users/zhangjiahao/Dataset/BGI/h5ad",
                     normalize=None, log=False, min_total_counts_ratio=None)

C03628C1_adata = load_bgi_dataset("C03628C1", data_dir="/Users/zhangjiahao/Dataset/BGI/h5ad",
                     normalize=None, log=False, min_total_counts_ratio=None)

gene_categories = {
    "Epithelial": EPITHELIAL_GENES,
    "Mesenchymal": MESENCHYMAL_GENES,
    "EMT Transcription Factors": EMT_TF,
}

B03425E1_variability = compute_variability(B03425E1_adata,metrics=['cv','nonzero_ratio'])
C03628C1_variability = compute_variability(C03628C1_adata,metrics=['cv','nonzero_ratio'])
B03425E1_variability.to_csv("tutorial/example_out/variablity/B03425E1_variability.csv")
C03628C1_variability.to_csv("tutorial/example_out/variablity/C03628C1_variability.csv")

for metric in ['cv', 'nonzero_ratio']:
    ax = plot_variability(
        variability={
            "B03425E1": B03425E1_variability,
            "C03628C1": C03628C1_variability,
        },
        metric=metric,
        gene_categories=gene_categories,
        title=f"{metric} across BGI datasets",
        annotate_emt=True,
    )
    fig = ax.get_figure()
    fig.savefig(f"tutorial/example_out/variablity/images/bgi_{metric}.png")