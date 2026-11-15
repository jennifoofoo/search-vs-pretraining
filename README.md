# search vs pretraining

My bachelor thesis was about making AutoML faster with meta-learning: instead of
starting the search from scratch, start from what worked on similar datasets
([code here](https://github.com/jennifoofoo/automl-workbench)).
TabPFN moves the meta-learning into pretraining and skips the search completely.
I wanted to see both ideas side by side on small tabular datasets.

Work in progress, started on 15 Nov 2026. Numbers below are from the first full run.

## What I compare

- **cold search**: random search over 5 algorithms (logistic regression, kNN, random
  forest, extra trees, XGBoost) and their hyperparameters. 40 configs, 3-fold CV.
- **warm search**: the same search, but the first 10 configs are the best ones from the
  5 most similar datasets. Similarity is computed on a handful of simple meta-features
  and the target dataset is always left out. This is the core idea of my thesis,
  rebuilt in a small version (the original runs inside a larger AutoML system).
- **TabPFN**: one fit, no tuning.

35 small classification datasets from OpenML-CC18 (500 to 6000 rows, up to 60 features),
3 seeds each, 70/30 split, training set capped at 2000 rows. Metric is ROC AUC on the
test split (one-vs-rest for multiclass).

## First results

| | mean test AUC | time per dataset |
|---|---|---|
| cold search, after 40 configs | 0.904 | ~24 s |
| warm search, after 40 configs | 0.904 | ~25 s |
| TabPFN, one fit | 0.916 | ~8 s (CPU) |

- TabPFN is ahead on 26 of 35 datasets, 8 are ties (within 0.002 AUC), the search wins 1.
- The warm start helps early: after 3 to 10 configs it is ahead of the cold search in
  about 55% of runs (cold ahead in about 30%). After 40 configs the difference is gone.
  So it gets you there faster, not further, which is what I expected from the thesis.

## What this does not show

- 40 random configs is a small search. A real AutoML system with hours of budget and
  ensembling would get closer.
- The search ran with 4 processes in parallel, so the timings are rough.
- CC18 is a standard benchmark that is also used a lot in TabPFN research, which may
  favour TabPFN.

## Lead scoring

`leads/` is the experiment I started with: four public "did this person buy / churn"
datasets with only 50 to 1000 labelled rows, TabPFN against logistic regression and
XGBoost. I built an account scoring pipeline at work where we never had enough history
for a model, so I wanted to know how little data is enough. Plots are in `leads/results/`.

## Run it

```
uv sync
uv run python run_search.py cold
uv run python run_search.py warm     # needs the cold runs
uv run python run_tabpfn.py          # needs TABPFN_TOKEN in .env, see .env.example
```

## Next

Curves of accuracy over time, and a closer look at which kinds of datasets TabPFN
wins on most clearly.
