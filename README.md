# NFL Game Prediction Project

Predicts the winner (home-win probability) of NFL games from team box-score history (2002–2025),
with data validation, DVC data/pipeline versioning, MLflow experiment tracking and model registry,
and a Streamlit dashboard.

## Quick start

```bash
pip install -r requirements.txt

python validate.py        # data quality checks → processed/validation\_report.json
python features.py        # feature engineering → processed/features\_\*.parquet
python train.py           # train + compare candidates, log to MLflow, register the best model
python backtest.py        # walk-forward, season-by-season accuracy vs Elo and home-team baselines
python predict.py         # predict every game in data/upcoming\_games.csv (uses the registry champion)
python predict.py --away Bills --home Dolphins          # one-off matchup
streamlit run dashboard.py
```

Or run the whole pipeline with DVC: `dvc repro` (only re-runs stages whose inputs changed; `dvc dag` shows the graph).

## Experiment tracking \& model registry (MLflow)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
```

* **Experiment `nfl-game-predictor`**: each `train.py` run is a parent run with one nested run per
candidate model (hyperparameters + validation log loss / Brier / AUC / accuracy). The parent run
logs the data version (SHA-256), season splits, test metrics vs. baselines, and artifacts
(validation report, model comparison, feature importance, test predictions).
* **Registered model `nfl-game-winner`**: every run registers a new version with a description and
tags (selected model, test metrics, data hash). The version with the best test log loss gets the
`champion` alias; a worse one gets `challenger`. `predict.py` loads `models:/nfl-game-winner@champion`
and falls back to `models/best\_model.joblib` if the registry is unavailable.

## Data versioning (DVC)

\## Data versioning (DVC)

`dvc.yaml` defines the pipeline stages validate → features → train → backtest / predict.

`dvc.lock` records the exact hash of every input (including the raw CSV) and output, so any

result can be traced back to the data and code that produced it, and `dvc repro` re-runs only

the stages whose inputs changed. The same data hash is logged on every MLflow run and registered

model version. The raw CSV stays in git so teammates get it on clone; moving it to a DVC remote

is a planned next step.Files

&#x20;   

|File|Role|
|-|-|
|`config.py`|paths, season splits, Elo settings, MLflow settings (all overridable via env vars / `.env`)|
|`data.py`|loads the CSV and reads `data/upcoming\_games.csv` (batch ingestion)|
|`validate.py`|18 schema, completeness, consistency and range checks; writes `processed/validation\_report.json` and computes the data version hash|
|`features.py`|team-game rows, leak-free rolling form (3/8-game + EWM), rest days, Elo, game-level matrix; saves Parquet|
|`train.py`|candidate models (logistic regression, random forest, hist gradient boosting, XGBoost), chosen on validation seasons by log loss, scored once on test seasons, refit on all seasons, logged + registered in MLflow|
|`backtest.py`|trains on seasons before S, predicts S, for every season|
|`predict.py`|predictions for upcoming games / a single matchup|
|`dashboard.py`|Streamlit cards for `models/predictions\_upcoming.csv`|
|`dvc.yaml`|reproducible pipeline definition|

## Season splits (config.py)

train 2003–2019 · validation 2020–2022 · test 2023–2025. 2002 is used only to warm up form and Elo.

## Adding games to predict

Add rows to `data/upcoming\_games.csv`:

```
season,week,date,time\_et,neutral,away,home
2026,3,2026-09-27,1:00 PM,False,Bills,Dolphins
```

Team names must match the CSV's nicknames (`49ers`, `Commanders`, `Raiders`, …).
Form and Elo come from each team's latest completed game in the history CSV, so append new
results to the history CSV as the season goes on to keep predictions current.

