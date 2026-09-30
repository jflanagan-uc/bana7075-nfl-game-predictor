# NFL Game Prediction Project

An end-to-end machine learning project for predicting NFL game winners and home-team win probabilities from historical team box-score data.

The project includes data validation, leakage-safe feature engineering, chronological model selection, walk-forward backtesting, MLflow experiment tracking and model registry, DVC pipeline reproducibility, batch or one-off predictions, and a Streamlit dashboard.

## Project overview

Historical NFL games from 2002-2025 are transformed into pre-game team-form features. The model predicts the probability that the home team wins while ensuring that only information available before kickoff is used.

The modeling workflow is chronological:

- **2002:** warm-up season for rolling features and Elo
- **2003-2019:** training
- **2020-2022:** validation and model selection
- **2023-2025:** held-out test set
- **2026:** default season for upcoming-game predictions

Candidate models include logistic regression, random forest, histogram gradient boosting, and XGBoost. Tree-based models are probability-calibrated, and the best candidate is selected using validation log loss.

## Repository structure

| Path | Purpose |
| --- | --- |
| `config.py` | Project paths, season splits, Elo settings, MLflow configuration, and environment-variable overrides |
| `data.py` | Loads and normalizes historical and upcoming game data |
| `validate.py` | Runs schema, completeness, consistency, range, and upcoming-game validation checks |
| `features.py` | Builds leakage-safe rolling form, rest, matchup, and Elo features |
| `train.py` | Trains candidate models, selects the best model, evaluates the test set, and registers the production model |
| `backtest.py` | Performs season-by-season walk-forward evaluation |
| `predict.py` | Generates batch or single-matchup predictions |
| `dashboard.py` | Streamlit dashboard for generated upcoming-game predictions |
| `get_logos.py` | Optional helper for downloading team logos |
| `dvc.yaml` | Reproducible validate → features → train → backtest/predict pipeline |
| `data/` | Historical NFL data and upcoming games |
| `processed/` | Validation and engineered-feature outputs |
| `models/` | Trained-model outputs, metrics, reports, backtests, and predictions |

## Requirements

Python 3.10+ is recommended.

The main dependencies are pandas, NumPy, scikit-learn, MLflow, DVC, Streamlit, PyArrow, matplotlib, joblib, and XGBoost. Install the exact project dependencies from `requirements.txt`.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/jflanagan-uc/bana7075-nfl-game-predictor.git
cd bana7075-nfl-game-predictor
```

If you are contributing through a fork, clone your fork instead.

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No external data download is required for the default workflow because the historical and upcoming-game CSV files are included in the repository.

## Running the project

### Option 1: Run the full pipeline with DVC

The simplest way to reproduce the complete workflow is:

```bash
dvc repro
```

The DVC pipeline runs:

```text
validate
   ↓
features
   ↓
train
  ↙   ↘
backtest predict
```

DVC only reruns stages whose dependencies have changed.

To inspect the pipeline:

```bash
dvc dag
```

The repository currently does not configure a DVC remote. The raw source data is stored in Git, while DVC tracks pipeline dependencies and generated artifacts through `dvc.yaml` and `dvc.lock`.

### Option 2: Run each stage manually

Run the stages in this order:

```bash
python validate.py
python features.py
python train.py
python backtest.py
python predict.py
```

#### 1. Validate data

```bash
python validate.py
```

Validation checks the historical dataset and upcoming-game file for required columns, missing values, duplicate games, valid weeks and dates, team consistency, score ranges, statistical consistency, and other data-quality issues.

The report is written to:

```text
processed/validation_report.json
```

Critical validation failures stop training.

#### 2. Build features

```bash
python features.py
```

Feature engineering creates pre-game information using only games completed before each kickoff. Features include:

- rolling 3-game and 8-game team form
- exponentially weighted team form
- offensive and defensive efficiency
- rest days
- home/away differences
- offense-versus-defense matchup features
- Elo ratings and home-field advantage
- postseason and early-season indicators

Outputs:

```text
processed/features_team_games.parquet
processed/features_games.parquet
```

#### 3. Train and evaluate models

```bash
python train.py
```

Training evaluates multiple configurations of:

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting
- XGBoost

The best candidate is selected on the 2020-2022 validation seasons using log loss. It is then refit on training + validation data and evaluated once on the held-out 2023-2025 test seasons.

The selected model is finally refit on all completed modeling seasons for production predictions.

Important outputs include:

```text
models/best_model.joblib
models/best_model_features.txt
models/metrics.json
models/model_selection_valid.csv
models/test_predictions.csv
models/feature_importance.csv
models/feature_importance.png
```

### 4. Run the walk-forward backtest

```bash
python backtest.py
```

For each backtest season, the model trains only on earlier seasons and predicts the current season. Results are compared with an Elo-only baseline and a home-team baseline.

Output:

```text
models/backtest_by_season.csv
```

### 5. Generate predictions

For every game in `data/upcoming_games.csv`:

```bash
python predict.py
```

Predictions are printed to the terminal and saved to:

```text
models/predictions_upcoming.csv
```

You can also predict a single matchup without editing the CSV:

```bash
python predict.py --away Bills --home Dolphins --date 2026-09-27 --season 2026 --week 3
```

For a neutral-site game:

```bash
python predict.py --away TeamA --home TeamB --date 2026-09-27 --season 2026 --week 3 --neutral
```

Team names must match the nicknames used by the historical dataset, such as `Bills`, `Dolphins`, `Chiefs`, or `49ers`.

`predict.py` first attempts to load the MLflow registry model with the `champion` alias. If the registry cannot be loaded, it falls back to `models/best_model.joblib`.

Run `python train.py` at least once before making predictions if a trained model is not already available locally.

## Upcoming-games file

Batch predictions are driven by `data/upcoming_games.csv`.

Required columns:

```text
season,week,date,away,home
```

Optional columns:

```text
time_et,neutral
```

Example:

```csv
season,week,date,time_et,neutral,away,home
2026,3,2026-09-27,1:00 PM,False,Bills,Dolphins
```

Kickoff times are interpreted as US/Eastern.

Rolling form, rest, and Elo for upcoming games come from each team's most recent completed games in the historical dataset. To keep predictions current during a season, append newly completed game results to the historical CSV before rebuilding predictions.

## Streamlit dashboard

Generate batch predictions first:

```bash
python predict.py
```

Then start the dashboard:

```bash
streamlit run dashboard.py
```

The dashboard reads `models/predictions_upcoming.csv`, groups games by week, and displays predicted win probabilities, picks, confidence, kickoff times, and team logos when available.

## MLflow experiment tracking and model registry

Training uses a local SQLite MLflow backend by default:

```text
sqlite:///mlflow.db
```

Start the MLflow UI with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000`.

The default experiment is:

```text
nfl-game-predictor
```

The registered model is:

```text
nfl-game-winner
```

Each training execution creates a parent MLflow run and nested runs for candidate models. Runs capture hyperparameters, validation metrics, test metrics, the data SHA-256 version, season splits, model reports, feature importance, and prediction artifacts.

New registered model versions are compared with the current registry champion using test log loss. A better or equal version receives the `champion` alias; otherwise it receives the `challenger` alias.

## Configuration

Defaults are defined in `config.py`. Settings can be overridden with environment variables or a local `.env` file.

Examples:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
CURRENT_SEASON=2026
BACKTEST_START=2008
ELO_K=20
ELO_HFA=48
```

Useful path overrides include:

```text
NFL_DATA_CSV
UPCOMING_CSV
PROCESSED_DATA_PATH
MODELS_PATH
LOGS_PATH
LOGOS_DIR
```

Season split variables can also be overridden:

```text
TRAIN_SEASONS_START
TRAIN_SEASONS_END
VALID_SEASONS_START
VALID_SEASONS_END
TEST_SEASONS_START
TEST_SEASONS_END
```

## Model evaluation

The project reports several complementary metrics:

- **Log loss** — primary model-selection metric; rewards well-calibrated probabilities
- **Brier score** — probability calibration/error
- **ROC AUC** — ranking ability
- **Accuracy** — winner classification at a 0.50 probability threshold

The held-out test model is also compared against an Elo-only probability baseline and the historical home-team rate.

Tied games are excluded from model training and evaluation.

## Leakage prevention

The feature pipeline is designed so that a game's own result or box-score statistics cannot enter its prediction features.

Rolling statistics use only completed games strictly before kickoff. Elo values are pre-game ratings, and chronological train/validation/test splits prevent future seasons from entering earlier model fitting.

## Optional team logos

The dashboard uses PNG files in `logos/` when they are available. To redownload them, install the optional packages:

```bash
pip install nfl_data_py requests
python get_logos.py
```

Logo downloading is not required for training or prediction.

## Typical development workflow

After changing source data or feature/model code:

```bash
dvc repro
```

For a targeted workflow:

```bash
python validate.py
python features.py
python train.py
python predict.py
streamlit run dashboard.py
```

Use MLflow to compare experiments and `models/metrics.json`, `models/model_selection_valid.csv`, and `models/backtest_by_season.csv` to review model performance.
