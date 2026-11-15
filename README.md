# search vs pretraining

work in progress, started 15 Nov 2026

In my bachelor thesis ([code](https://github.com/jennifoofoo/automl-workbench)) I tried to
make AutoML search faster by starting from configs that worked on similar datasets.
TabPFN doesn't search at all, it just predicts. I used SMAC back then, which comes from
Frank Hutter's group, and so does TabPFN. So I wanted to know what's left of my approach
once the search is gone.

Also a lot of teams just have a table with a few hundred rows and nobody who tunes
models. If something works well without tuning, that's what they'll use.

## setup

35 small classification datasets from OpenML-CC18 (500 to 6000 rows), 3 seeds, 70/30 split,
train capped at 2000 rows, ROC AUC on the test part.

- cold: random search over logreg, knn, random forest, extra trees, xgboost. 40 configs, 3-fold CV
- warm: same, but the first 10 configs come from the 5 most similar datasets (a few simple
  meta-features, target dataset left out). That's the core idea of my thesis, rebuilt in a small version
- tabpfn: one fit, no tuning

## first numbers

| | test AUC (mean) | time per dataset |
|---|---|---|
| cold, 40 configs | 0.904 | ~24 s |
| warm, 40 configs | 0.904 | ~25 s |
| tabpfn | 0.916 | ~8 s on cpu |

TabPFN is ahead on 26 of 35 datasets, 8 are basically ties, the search wins on one (dresses-sales).

Warm start is ahead of cold in about 55% of runs after 3 to 10 configs, after 40 there is no
difference anymore. So faster but not better, roughly what I saw in the thesis too.

Keep in mind: 40 random configs is a small search, a real AutoML run with ensembling would
get closer. Timings are rough because the search ran 4 processes in parallel. And CC18 gets
used a lot in TabPFN research itself.

## leads/

What I started with. Four public buy/churn datasets, only 50 to 1000 labelled rows,
tabpfn vs logreg vs xgboost. Plots are in `leads/results`.

## run

```
uv sync
uv run python run_search.py cold
uv run python run_search.py warm   # needs the cold runs
uv run python run_tabpfn.py        # needs TABPFN_TOKEN in .env
```

## todo

- curves over time instead of just the end result
- which datasets tabpfn wins on, and why
- kc2 and credit-g were in my thesis experiments, compare those directly
