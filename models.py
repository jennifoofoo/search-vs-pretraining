import numpy as np
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import to_numeric


def logreg(X_train, y_train, X_test):
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat = OneHotEncoder(handle_unknown="ignore")
    pre = make_column_transformer(
        (num, make_column_selector(dtype_include="number")),
        (cat, make_column_selector(dtype_include="category")),
    )
    m = make_pipeline(pre, LogisticRegression(max_iter=1000))
    m.fit(X_train, y_train)
    return m.predict_proba(X_test)[:, 1]


def xgb(X_train, y_train, X_test):
    # imported here bc xgboost + torch in one process segfaults on my mac (two libomp)
    from xgboost import XGBClassifier
    a, b = to_numeric(X_train, X_test)
    m = XGBClassifier(n_jobs=4)
    m.fit(a, y_train)
    return m.predict_proba(b)[:, 1]


def tabpfn(X_train, y_train, X_test):
    from tabpfn import TabPFNClassifier
    a, b = to_numeric(X_train, X_test)
    m = TabPFNClassifier()
    m.fit(a, y_train)
    return m.predict_proba(b)[:, 1]


MODELS = {"logreg": logreg, "xgboost": xgb, "tabpfn": tabpfn}
