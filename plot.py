import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")
colors = {"tabpfn": "#e4572e", "xgboost": "#4c78a8", "logreg": "#999999", "llm": "#59a14f"}

for metric, label in [("auc", "ROC-AUC"), ("top10_hit", "hit rate in top 10% of leads")]:
    names = list(df.dataset.unique())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), sharey=False)
    for ax, name in zip(axes, names):
        d = df[df.dataset == name]
        for m, g in d.groupby("model"):
            s = g.groupby("n_train")[metric].agg(["mean", "std"])
            ax.plot(s.index, s["mean"], marker="o", ms=4, label=m, color=colors.get(m))
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.12, color=colors.get(m))
        ax.set_xscale("log")
        ax.set_xticks(sorted(d.n_train.unique()))
        ax.set_xticklabels(sorted(d.n_train.unique()))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("labelled rows")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(label)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/{metric}.png", dpi=160)
