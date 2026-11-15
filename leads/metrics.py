import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y, p):
    return roc_auc_score(y, p)


def top_decile_hit_rate(y, p):
    # of the 10% highest scored leads, how many actually converted
    y = np.asarray(y)
    k = max(1, len(y) // 10)
    top = np.argsort(-np.asarray(p))[:k]
    return y[top].mean()


def lift(y, p):
    return top_decile_hit_rate(y, p) / np.mean(y)
