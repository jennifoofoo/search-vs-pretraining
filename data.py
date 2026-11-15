import numpy as np
import pandas as pd
import openml

# openml ids, all "did this person convert / leave" type problems
DATASETS = {
    "bank-marketing": 1461,
    "online-shoppers": 45060,
    "telco-churn": 42178,
    "caravan-insurance": 298,
}

TEST_SIZE = 2000

BANK_COLS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
             "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]


def load(name):
    d = openml.datasets.get_dataset(DATASETS[name])
    X, y, _, _ = d.get_data(target=d.default_target_attribute)
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("string", "str"):
            X[c] = X[c].astype("category")
    # telco has TotalCharges as strings w/ blanks
    if name == "telco-churn" and "TotalCharges" in X:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"].astype(str), errors="coerce")

    # call duration is only known after the call, so it leaks the answer
    if name == "bank-marketing":
        X.columns = BANK_COLS
        X = X.drop(columns=["duration"])

    # positive = the rare class (bought / churned)
    counts = y.value_counts()
    y = (y == counts.idxmin()).astype(int)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def split(X, y, n_train, seed):
    # test set is always the same, only the train sample changes w/ seed
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    test, pool = idx[:TEST_SIZE], idx[TEST_SIZE:]

    rng = np.random.default_rng(seed)
    for _ in range(50):
        tr = rng.choice(pool, size=n_train, replace=False)
        if y.iloc[tr].sum() >= 2:
            break
    return X.iloc[tr], y.iloc[tr], X.iloc[test], y.iloc[test]


def to_numeric(X_train, X_test):
    # ordinal codes for the tree / tabpfn models
    a, b = X_train.copy(), X_test.copy()
    for c in a.columns:
        if str(a[c].dtype) == "category":
            cats = a[c].cat.categories
            a[c] = a[c].cat.codes.astype(float)
            b[c] = pd.Categorical(b[c], categories=cats).codes.astype(float)
    return a.replace(-1, np.nan), b.replace(-1, np.nan)
