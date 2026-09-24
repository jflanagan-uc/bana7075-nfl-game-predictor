# NFL Game Prediction Project (CSV edition)

Predicts NFL game winners from team box-score history (2002–2025). This version runs entirely from `data/nfl_team_stats_2002-2025.csv`

## Quick start
```bash
pip install -r requirements.txt
python features.py        # optional: writes processed/features_*.csv for inspection
python train.py           # select + evaluate model, save models/best_model.joblib
python backtest.py        # walk-forward, season-by-season accuracy vs Elo and home-team baselines
python predict.py         # predict every game in data/upcoming_games.csv
python predict.py --away Bills --home Dolphins          # one-off matchup
streamlit run dashboard.py
```

## Files
| File | Role |
|---|---|
| `config.py` | paths, season splits, Elo settings (all overridable via env vars / `.env`) |
| `data.py` | loads the CSV and reads `data/upcoming_games.csv` |
| `features.py` | team-game rows, leak-free rolling form (3/8-game + EWM), rest days, Elo, game-level matrix |
| `train.py` | candidate models (logistic regression, random forest, hist gradient boosting, XGBoost if installed), chosen on validation seasons, scored once on test seasons, then refit on all seasons |
| `backtest.py` | trains on seasons before S, predicts S, for every season |
| `predict.py` | predictions for upcoming games / a single matchup |
| `dashboard.py` | Streamlit cards for `models/predictions_upcoming.csv` |

## Season splits (config.py)
train 2003–2019 · validation 2020–2022 · test 2023–2025. 2002 is used only to warm up form and Elo.

## Adding games to predict
Add rows to `data/upcoming_games.csv`:
```
season,week,date,time_et,neutral,away,home
2026,3,2026-09-27,1:00 PM,False,Bills,Dolphins
```
Team names must match the CSV's nicknames (`49ers`, `Commanders`, `Raiders`, …).
Form and Elo come from each team's latest completed game in the history CSV, so append new
results to the history CSV as the season goes on to keep predictions current.


