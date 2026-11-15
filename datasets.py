import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split

# small classification datasets from the OpenML-CC18 suite
# (500-6000 rows, <= 60 features, <= 10 classes)
IDS = [23381, 1063, 6332, 40994, 1510, 1480, 11, 29, 15, 188, 1464, 37, 469, 54, 50, 31,
       1494, 1068, 1462, 1049, 23, 1050, 40975, 40982, 22, 18, 1067, 40984, 3, 38, 44,
       40983, 40701, 1489, 1497]

MAX_TRAIN = 2000


def load(did):
    d = openml.datasets.get_dataset(did)
    X, y, is_cat, _ = d.get_data(target=d.default_target_attribute)
    X = X.copy()
    cat_cols = []
    for c, cat in zip(X.columns, is_cat):
        if cat or not pd.api.types.is_numeric_dtype(X[c]):
            codes = X[c].astype("category").cat.codes.astype(float)
            X[c] = codes.replace(-1, np.nan)
            cat_cols.append(c)
        else:
            X[c] = X[c].astype(float)
    y = pd.Series(pd.factorize(y)[0])
    return d.name, X.reset_index(drop=True), y, cat_cols


def split(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    if len(Xtr) > MAX_TRAIN:
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=MAX_TRAIN, stratify=ytr, random_state=seed)
    return Xtr, ytr, Xte, yte
