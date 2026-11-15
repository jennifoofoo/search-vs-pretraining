import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ALGOS = ["logreg", "knn", "rf", "extratrees", "xgboost"]


def sample(rng):
    algo = str(rng.choice(ALGOS))
    if algo == "logreg":
        p = {"C": float(10 ** rng.uniform(-3, 2))}
    elif algo == "knn":
        p = {"n_neighbors": int(rng.integers(1, 40)), "weights": str(rng.choice(["uniform", "distance"]))}
    elif algo in ("rf", "extratrees"):
        p = {
            "n_estimators": int(rng.choice([100, 200, 400])),
            "max_features": float(rng.uniform(0.1, 1.0)),
            "min_samples_leaf": int(rng.integers(1, 15)),
        }
    else:
        p = {
            "n_estimators": int(rng.choice([100, 200, 400])),
            "learning_rate": float(10 ** rng.uniform(-2.3, -0.5)),
            "max_depth": int(rng.integers(2, 10)),
            "subsample": float(rng.uniform(0.5, 1.0)),
            "colsample_bytree": float(rng.uniform(0.4, 1.0)),
        }
    return algo, p


def build(algo, p, cat_cols, all_cols):
    num_cols = [c for c in all_cols if c not in cat_cols]
    if algo in ("logreg", "knn"):
        # these two need scaling and one-hot, the trees are fine with codes
        pre = make_column_transformer(
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            (make_pipeline(SimpleImputer(strategy="most_frequent"),
                           OneHotEncoder(handle_unknown="ignore", max_categories=20)), cat_cols),
        )
        clf = LogisticRegression(max_iter=2000, **p) if algo == "logreg" else KNeighborsClassifier(**p)
        return make_pipeline(pre, clf)
    if algo == "rf":
        clf = RandomForestClassifier(n_jobs=1, random_state=0, **p)
    elif algo == "extratrees":
        clf = ExtraTreesClassifier(n_jobs=1, random_state=0, **p)
    else:
        return XGBClassifier(n_jobs=1, random_state=0, **p)
    return make_pipeline(SimpleImputer(strategy="median"), clf)
