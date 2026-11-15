import numpy as np
import pandas as pd

from data import split
from metrics import top_decile_hit_rate, lift


def fake(n=5000):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((rng.random(n) < 0.1).astype(int))
    return X, y


def test_test_set_stays_the_same():
    X, y = fake()
    _, _, te1, _ = split(X, y, 50, seed=0)
    _, _, te2, _ = split(X, y, 500, seed=3)
    assert list(te1.index) == list(te2.index)


def test_no_leak():
    X, y = fake()
    tr, _, te, _ = split(X, y, 500, seed=2)
    assert not set(tr.index) & set(te.index)
    assert len(tr) == 500


def test_train_has_positives():
    X, y = fake()
    for s in range(10):
        _, ytr, _, _ = split(X, y, 50, seed=s)
        assert ytr.sum() >= 2


def test_top_decile():
    y = np.array([1] * 10 + [0] * 90)
    perfect = np.linspace(1, 0, 100)
    assert top_decile_hit_rate(y, perfect) == 1.0
    assert lift(y, perfect) == 10
    assert top_decile_hit_rate(y, -perfect) == 0.0
