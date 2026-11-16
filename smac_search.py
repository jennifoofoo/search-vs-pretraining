import logging
import tempfile
import time
import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, EqualsCondition, Float, InCondition, Integer
from smac import HyperparameterOptimizationFacade as HPO, Scenario

from search import auc, cv_score
from space import build

logging.getLogger("smac").setLevel(logging.ERROR)


def make_space():
    # same ranges as space.py, just written down for smac
    cs = ConfigurationSpace()
    algo = Categorical("algo", ["logreg", "knn", "rf", "extratrees", "xgboost"])
    C = Float("C", (1e-3, 1e2), log=True)
    k = Integer("n_neighbors", (1, 39))
    weights = Categorical("weights", ["uniform", "distance"])
    t_n = Categorical("t_n_estimators", [100, 200, 400])
    t_feat = Float("max_features", (0.1, 1.0))
    t_leaf = Integer("min_samples_leaf", (1, 14))
    x_n = Categorical("x_n_estimators", [100, 200, 400])
    lr = Float("learning_rate", (10 ** -2.3, 10 ** -0.5), log=True)
    depth = Integer("max_depth", (2, 9))
    sub = Float("subsample", (0.5, 1.0))
    col = Float("colsample_bytree", (0.4, 1.0))
    cs.add([algo, C, k, weights, t_n, t_feat, t_leaf, x_n, lr, depth, sub, col])
    cs.add([EqualsCondition(C, algo, "logreg"), EqualsCondition(k, algo, "knn"), EqualsCondition(weights, algo, "knn")])
    cs.add([InCondition(h, algo, ["rf", "extratrees"]) for h in (t_n, t_feat, t_leaf)])
    cs.add([EqualsCondition(h, algo, "xgboost") for h in (x_n, lr, depth, sub, col)])
    return cs


def to_params(cfg):
    d = dict(cfg)
    algo = str(d.pop("algo"))
    # smac hands back numpy types, json doesnt like those
    return algo, {k.replace("t_n_", "n_").replace("x_n_", "n_"): (v.item() if hasattr(v, "item") else v)
                  for k, v in d.items()}


def to_config(cs, algo, p):
    pre = {"rf": "t_", "extratrees": "t_", "xgboost": "x_"}.get(algo, "")
    vals = {"algo": algo}
    for k, v in p.items():
        vals[pre + k if k == "n_estimators" else k] = v
    return Configuration(cs, values=vals)


def run_smac(Xtr, ytr, Xte, yte, cat_cols, n_iter, seed, first=()):
    cs = make_space()
    cols = list(Xtr.columns)
    state = {"best_cv": -1, "best_test": None}
    traj = []
    t0 = time.time()

    def target(cfg, seed=0):
        algo, p = to_params(cfg)
        model = build(algo, p, cat_cols, cols)
        try:
            cv = cv_score(model, Xtr, ytr, seed)
        except ValueError:
            cv = 0.0
        if cv > state["best_cv"]:
            state["best_cv"] = cv
            state["best_test"] = auc(yte, model.fit(Xtr, ytr).predict_proba(Xte))
        traj.append({"i": len(traj), "secs": time.time() - t0, "algo": algo, "params": p,
                     "cv": cv, "best_cv": state["best_cv"], "test": state["best_test"]})
        return 1 - cv

    with tempfile.TemporaryDirectory() as tmp:
        sc = Scenario(cs, deterministic=True, n_trials=n_iter, seed=seed, output_directory=tmp)
        if first:
            # warm start: the transferred configs replace smacs own initial design
            init = HPO.get_initial_design(sc, n_configs=0, additional_configs=[to_config(cs, a, p) for a, p in first])
        else:
            init = HPO.get_initial_design(sc, n_configs=10)
        HPO(sc, target, initial_design=init, overwrite=True, logging_level=logging.ERROR).optimize()
    return traj
