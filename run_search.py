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
PHASE = "cold"
SEARCH = run_search
SEEDS = [0, 1, 2]
K = 5          # neighbours used for the warm start
TOP = 2        # configs taken from each neighbour


def cold(did, seed):
    out = f"results/{PHASE}/{did}_{seed}.json"
    if os.path.exists(out):
        return
    name, X, y, cats = load(did)
    Xtr, ytr, Xte, yte = split(X, y, seed)
    traj = SEARCH(Xtr, ytr, Xte, yte, cats, N_ITER, seed)
    mf = metafeatures(Xtr, ytr, cats)
    json.dump({"did": did, "name": name, "seed": seed, "meta": mf, "traj": traj}, open(out, "w"))
    print(PHASE, name, seed, round(traj[-1]["test"], 3), flush=True)


def neighbours(did, seed):
    runs = {d: json.load(open(f"results/{PHASE.replace('warm', 'cold')}/{d}_{seed}.json")) for d in IDS}
    mf = pd.DataFrame({d: r["meta"] for d, r in runs.items()}).T.astype(float)
    z = (mf - mf.mean()) / (mf.std() + 1e-9)
    dist = ((z - z.loc[did]) ** 2).sum(axis=1).drop(did).sort_values()
    return list(dist.index[:K]), runs


def warm(did, seed):
    out = f"results/{PHASE}/{did}_{seed}.json"
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
    traj = SEARCH(Xtr, ytr, Xte, yte, cats, N_ITER, seed + 100, first=first)
    json.dump({"did": did, "name": name, "seed": seed, "neighbours": nb, "traj": traj}, open(out, "w"))
    print(PHASE, name, seed, round(traj[-1]["test"], 3), flush=True)


if __name__ == "__main__":
    # python run_search.py cold | warm            random search, 40 configs
    # python run_search.py smac_cold | smac_warm  smac, 200 configs
    # python run_search.py cold200                random search, 200 configs
    PHASE = sys.argv[1]
    if PHASE.startswith("smac"):
        from smac_search import run_smac as SEARCH
        N_ITER = 200
    else:
        SEARCH = run_search
        if PHASE.endswith("200"):
            N_ITER = 200
    os.makedirs(f"results/{PHASE}", exist_ok=True)
    fn = warm if "warm" in PHASE else cold
    Parallel(n_jobs=5)(delayed(fn)(d, s) for d in IDS for s in SEEDS)
