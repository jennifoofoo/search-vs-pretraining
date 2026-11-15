import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def metafeatures(X, y, cat_cols):
    n, d = X.shape
    freq = y.value_counts(normalize=True).values
    Xf = X.fillna(X.median()).fillna(0)
    num = X.drop(columns=cat_cols)
    f = {
        "log_n": np.log10(n),
        "log_d": np.log10(d),
        "n_per_d": np.log10(n / d),
        "n_classes": len(freq),
        "class_entropy": float(-(freq * np.log2(freq)).sum() / np.log2(len(freq))),
        "minority": float(freq.min()),
        "frac_cat": len(cat_cols) / d,
        "frac_missing": float(X.isna().mean().mean()),
        "mean_skew": float(num.skew().abs().mean()) if num.shape[1] else 0.0,
    }
    # landmarkers: how far do two very cheap models get
    f["stump"] = cross_val_score(DecisionTreeClassifier(max_depth=1), Xf, y, cv=3).mean()
    f["tree"] = cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=0), Xf, y, cv=3).mean()
    f["nn1"] = cross_val_score(KNeighborsClassifier(1), Xf, y, cv=3).mean()
    return f
