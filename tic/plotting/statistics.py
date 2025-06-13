import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_cell_type_pie_nature(counts: pd.Series, save_path=None, top_n=10, title='Cell Type Composition', legend_title='Cell Types'):
    """
    Nature-style donut pie chart for cell-type composition.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
    })

    if top_n is not None and len(counts) > top_n:
        top_counts = counts[:top_n]
        other_count = counts[top_n:].sum()
        sizes = np.append(top_counts.values, other_count)
        labels = top_counts.index.tolist() + ['Others']
    else:
        sizes = counts.values
        labels = counts.index.tolist()

    fig, ax = plt.subplots(figsize=(8, 6.5))  # 更大的画布
    cmap = plt.get_cmap('Pastel1')
    colors = cmap(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.75,
        textprops={'fontsize': 9},
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}
    )
    ax.add_artist(plt.Circle((0, 0), 0.5, color='white'))  # 调整中空半径

    # 更紧凑的图例
    ax.legend(
        wedges, labels,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        fontsize=8,
        title_fontsize=9,
        borderpad=0.4,
        labelspacing=0.3
    )

    ax.set_title(title, pad=15)
    ax.set(aspect="equal")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()