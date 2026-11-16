import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tp = pd.read_csv("results/tabpfn.csv")
runs = {a: [json.load(open(f)) for f in sorted(glob.glob(f"results/{a}/*.json"))] for a in ["cold", "warm"]}

# --- incumbent over time, averaged over all datasets and seeds
grid = np.logspace(-1, np.log10(40), 60)
fig, ax = plt.subplots(figsize=(6, 4))
for arm, col in [("cold", "#4c78a8"), ("warm", "#e4572e")]:
    curves = []
    for r in runs[arm]:
        secs = np.array([t["secs"] for t in r["traj"]])
        test = np.array([t["test"] for t in r["traj"]])
        idx = np.searchsorted(secs, grid, side="right") - 1
        # nothing evaluated yet counts as guessing
        curves.append(np.where(idx >= 0, test[np.clip(idx, 0, None)], 0.5))
    ax.plot(grid, np.mean(curves, axis=0), label=f"{arm} search", color=col)
ax.scatter([tp.secs.mean()], [tp.test.mean()], color="black", zorder=3, label="tabpfn (one fit)")
ax.axhline(tp.test.mean(), color="black", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_xlabel("seconds")
ax.set_ylabel("test AUC, mean over 35 datasets")
ax.set_ylim(0.8, 0.93)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("results/race.png", dpi=160)

# --- per dataset: tabpfn minus the better of the two searches after 40 configs
rows = []
for r in runs["cold"]:
    rows.append({"did": r["did"], "name": r["name"], "seed": r["seed"], "cold": r["traj"][-1]["test"], **r["meta"]})
d = pd.DataFrame(rows)
w = pd.DataFrame([{"did": r["did"], "seed": r["seed"], "warm": r["traj"][-1]["test"]} for r in runs["warm"]])
d = d.merge(w).merge(tp[["did", "seed", "test"]].rename(columns={"test": "tabpfn"}))
d["gap"] = d.tabpfn - d[["cold", "warm"]].max(axis=1)
per = d.groupby("name").mean(numeric_only=True).sort_values("gap")
per.to_csv("results/per_dataset.csv")

fig, ax = plt.subplots(figsize=(6, 7))
ax.barh(per.index, per.gap, color=np.where(per.gap > 0, "#222222", "#e4572e"))
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("tabpfn AUC minus best search AUC")
ax.tick_params(axis="y", labelsize=7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/gap.png", dpi=160)

# --- which dataset properties go with a bigger gap
feats = ["log_n", "log_d", "n_per_d", "n_classes", "class_entropy", "minority", "frac_cat", "frac_missing", "mean_skew", "stump", "tree", "nn1"]
per["best_search"] = per[["cold", "warm"]].max(axis=1)
corr = per[feats + ["best_search"]].corrwith(per.gap, method="spearman").sort_values()
print(corr.round(2).to_string())
print()
print("gap by how hard the dataset is for the search:")
per["band"] = pd.cut(per.best_search, [0, 0.85, 0.95, 0.99, 1.0], labels=["<0.85", "0.85-0.95", "0.95-0.99", ">0.99"])
print(per.groupby("band", observed=True).gap.agg(["mean", "count"]).round(3).to_string())
print()
print("categorical share:")
per["cat"] = np.where(per.frac_cat > 0.5, "mostly categorical", "mostly numeric")
print(per.groupby("cat").gap.agg(["mean", "count"]).round(3).to_string())
print("\nthesis datasets:")
print(per.loc[["kc2", "credit-g"], ["cold", "warm", "tabpfn", "gap"]].round(3).to_string())

# --- is any of this more than noise? paired tests over the 35 datasets (seeds averaged first)
from scipy.stats import wilcoxon

print("\nwilcoxon signed-rank, paired by dataset:")
for a, b in [("tabpfn", "cold"), ("tabpfn", "warm"), ("tabpfn", "best_search")]:
    diff = per[a] - per[b]
    print(f"  {a} vs {b}: median diff {diff.median():.4f}, {int((diff > 0).sum())} of {len(diff)} positive, p = {wilcoxon(per[a], per[b]).pvalue:.2g}")

# warm vs cold at the same number of configs
for k in [3, 5, 10, 20, 40]:
    c = pd.DataFrame([{"did": r["did"], "seed": r["seed"], "cold": r["traj"][k - 1]["test"]} for r in runs["cold"]])
    w = pd.DataFrame([{"did": r["did"], "seed": r["seed"], "warm": r["traj"][k - 1]["test"]} for r in runs["warm"]])
    m = c.merge(w).groupby("did")[["cold", "warm"]].mean()
    diff = m.warm - m.cold
    p = wilcoxon(m.warm, m.cold).pvalue
    print(f"  warm vs cold after {k:2d} configs: mean diff {diff.mean():+.4f}, {int((diff > 0).sum())} of {len(diff)} positive, p = {p:.2g}")
