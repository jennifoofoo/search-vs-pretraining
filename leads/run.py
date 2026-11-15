import sys
import time
import pandas as pd
from dotenv import load_dotenv

from data import DATASETS, load, split
from metrics import auc, top_decile_hit_rate, lift
from models import MODELS

load_dotenv()

SIZES = [50, 100, 250, 500, 1000]
SEEDS = range(5)
OUT = "results/results.csv"


def main(models):
    rows = []
    for name in DATASETS:
        X, y = load(name)
        print(name, X.shape, "pos rate", round(y.mean(), 3))
        for n in SIZES:
            for seed in SEEDS:
                Xtr, ytr, Xte, yte = split(X, y, n, seed)
                for m in models:
                    t = time.time()
                    p = MODELS[m](Xtr, ytr, Xte)
                    rows.append({
                        "dataset": name, "n_train": n, "seed": seed, "model": m,
                        "auc": auc(yte, p),
                        "top10_hit": top_decile_hit_rate(yte, p),
                        "lift": lift(yte, p),
                        "secs": time.time() - t,
                    })
            df = pd.DataFrame(rows)
            print(n, df[(df.dataset == name) & (df.n_train == n)].groupby("model").auc.mean().round(3).to_dict())
    return pd.DataFrame(rows)


if __name__ == "__main__":
    models = sys.argv[1:] or list(MODELS)
    df = main(models)
    # keep old rows of models we didnt rerun
    try:
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old.model.isin(models)], df])
    except FileNotFoundError:
        pass
    df.to_csv(OUT, index=False)
