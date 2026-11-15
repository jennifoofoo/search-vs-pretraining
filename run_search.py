import json
import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from datasets import IDS, load, split
from metafeatures import metafeatures
from search import run_search

N_ITER = 40
SEEDS = [0, 1, 2]
K = 5          # neighbours used for the warm start
TOP = 2        # configs taken from each neighbour


def cold(did, seed):
    out = f"results/cold/{did}_{seed}.json"
    if os.path.exists(out):
        return
    name, X, y, cats = load(did)
    Xtr, ytr, Xte, yte = split(X, y, seed)
    traj = run_search(Xtr, ytr, Xte, yte, cats, N_ITER, seed)
    mf = metafeatures(Xtr, ytr, cats)
    json.dump({"did": did, "name": name, "seed": seed, "meta": mf, "traj": traj}, open(out, "w"))
    print("cold", name, seed, round(traj[-1]["test"], 3), flush=True)


def neighbours(did, seed):
    runs = {d: json.load(open(f"results/cold/{d}_{seed}.json")) for d in IDS}
    mf = pd.DataFrame({d: r["meta"] for d, r in runs.items()}).T.astype(float)
    z = (mf - mf.mean()) / (mf.std() + 1e-9)
    dist = ((z - z.loc[did]) ** 2).sum(axis=1).drop(did).sort_values()
    return list(dist.index[:K]), runs


def warm(did, seed):
    out = f"results/warm/{did}_{seed}.json"
    if os.path.exists(out):
        return
    nb, runs = neighbours(did, seed)
    # best configs of the most similar datasets go first, the target dataset itself is never used
    first = []
    for j in range(TOP):
        for d in nb:
            best = sorted(runs[d]["traj"], key=lambda r: -r["cv"])
            first.append((best[j]["algo"], best[j]["params"]))
    name, X, y, cats = load(did)
    Xtr, ytr, Xte, yte = split(X, y, seed)
    traj = run_search(Xtr, ytr, Xte, yte, cats, N_ITER, seed + 100, first=first)
    json.dump({"did": did, "name": name, "seed": seed, "neighbours": nb, "traj": traj}, open(out, "w"))
    print("warm", name, seed, round(traj[-1]["test"], 3), flush=True)


if __name__ == "__main__":
    phase = sys.argv[1]
    os.makedirs(f"results/{phase}", exist_ok=True)
    fn = cold if phase == "cold" else warm
    Parallel(n_jobs=4)(delayed(fn)(d, s) for d in IDS for s in SEEDS)
