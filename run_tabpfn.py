import time
import pandas as pd
from dotenv import load_dotenv
from tabpfn import TabPFNClassifier

from datasets import IDS, load, split
from search import auc

load_dotenv()

# own script bc torch and xgboost dont like each other in one process on my mac
rows = []
for did in IDS:
    name, X, y, cats = load(did)
    for seed in [0, 1, 2]:
        Xtr, ytr, Xte, yte = split(X, y, seed)
        t = time.time()
        m = TabPFNClassifier().fit(Xtr, ytr)
        test = auc(yte, m.predict_proba(Xte))
        rows.append({"did": did, "name": name, "seed": seed, "test": test, "secs": time.time() - t})
        print(name, seed, round(test, 3), round(rows[-1]["secs"], 1), flush=True)
    pd.DataFrame(rows).to_csv("results/tabpfn.csv", index=False)
