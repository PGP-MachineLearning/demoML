import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd


def plot_split_distribution(train_df, val_df, test_df,
                            labels=("Entrenamiento", "Validación", "Prueba"),
                            colors=("#c7e9c0", "#9ecae1", "#fb6a4a"),
                            figsize=(10, 1.5),
                            outpath="datos/plots/data_split_distribution.png"):  


    counts = (len(train_df), len(val_df), len(test_df))
    counts = [int(c) for c in counts]
    total = sum(counts)

    fracs = [c / total for c in counts]
    fig, ax = plt.subplots(figsize=figsize)

    left = 0
    for frac, label, color, cnt in zip(fracs, labels, colors, counts):
        ax.barh(0, width=cnt, left=left, height=1.2, color=color, edgecolor='k', linewidth=0.8)

        pct_text = f"{100*frac:.2f}%\n{label}"
        center = left + cnt / 2
        ax.text(center, 0, pct_text, ha='center', va='center', fontsize=10)
        left += cnt

    ax.set_xlim(0, total)
    ax.set_ylim(-0.6, 0.4)
    ax.set_yticks([])
    ax.set_xlabel('Cantidad de datos')

    ticks = [0, counts[0], counts[0] + counts[1], total]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(',', '.')))

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    outdir = os.path.dirname(outpath)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)

DfSplits = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

def stratified_split(df, test_size=0.25, val_size=0.10, random_state=42) -> DfSplits:
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(test_size * len(df))
    n_val = int(val_size * len(df))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    plot_split_distribution(df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx])
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


if __name__ == "__main__":
    import pandas as pd
    demo_path = "/home/moya/TP-IAA/datos/Resilience_CleanOnly_v1_PREPROCESSED_v2.csv"
    print(demo_path)
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        train, val, test = stratified_split(df)
        print('Saved plot to datos/plots/data_split_distribution.png')
    else:
        print(f"Demo file {demo_path} not found. Please provide your own DataFrame to test the plotting function.")
  