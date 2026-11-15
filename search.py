import time
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from space import sample, build


def auc(y, proba):
    if proba.shape[1] == 2:
        return roc_auc_score(y, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr", labels=np.arange(proba.shape[1]))


def cv_score(model, X, y, seed):
    scores = []
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        scores.append(auc(y.iloc[va], m.predict_proba(X.iloc[va])))
    return float(np.mean(scores))


def run_search(Xtr, ytr, Xte, yte, cat_cols, n_iter, seed, first=()):
    """Random search over algorithm + hyperparameters.
    `first` are configs to try before the random ones (thats the warm start)."""
    rng = np.random.default_rng(seed)
    cols = list(Xtr.columns)
    best_cv, best_test = -1, None
    t0 = time.time()
    traj = []
    for i in range(n_iter):
        algo, p = first[i] if i < len(first) else sample(rng)
        model = build(algo, p, cat_cols, cols)
        try:
            cv = cv_score(model, Xtr, ytr, seed)
        except ValueError:
            cv = 0.0   # e.g. knn with more neighbors than rows in a fold
        if cv > best_cv:
            best_cv = cv
            best_test = auc(yte, model.fit(Xtr, ytr).predict_proba(Xte))
        traj.append({"i": i, "secs": time.time() - t0, "algo": algo, "params": p,
                     "cv": cv, "best_cv": best_cv, "test": best_test})
    return traj
