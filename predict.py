"""Predict upcoming games.

Two ways to use it:
  python predict.py                         # every game in data/upcoming_games.csv
  python predict.py --away Bills --home Dolphins [--date 2026-09-27] [--neutral]

Upcoming games are appended to the historical CSV and run through the same feature
pipeline, so each team's form, rest and Elo come from its most recent completed games.
"""
import argparse
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd

from config import MODELS_PATH, CURRENT_SEASON, MLFLOW_TRACKING_URI, REGISTERED_MODEL
from data import _normalize, load_upcoming
from features import build_features
from utils import get_logger

logger = get_logger("predict")


def load_model():
    """Champion model from the MLflow registry; falls back to models/best_model.joblib."""
    with open(os.path.join(MODELS_PATH, "best_model_features.txt")) as f:
        features = [line.strip() for line in f if line.strip()]
    try:
        import mlflow.sklearn
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        uri = f"models:/{REGISTERED_MODEL}@champion"
        model = mlflow.sklearn.load_model(uri)
        logger.info(f"Loaded {uri} from the MLflow registry")
        return model, features
    except Exception as e:
        logger.warning(f"Registry load failed ({type(e).__name__}); using local joblib file")
    path = os.path.join(MODELS_PATH, "best_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model at {path}. Run train.py first.")
    return joblib.load(path), features


def cli_matchup(args) -> pd.DataFrame:
    return _normalize(pd.DataFrame([{
        "season": args.season, "week": args.week, "date": args.date,
        "time_et": "1:00 PM", "neutral": args.neutral, "away": args.away, "home": args.home,
    }]), completed=False)


def predict(upcoming: pd.DataFrame) -> pd.DataFrame:
    model, features = load_model()
    _, matrix = build_features(include_future=True, upcoming=upcoming)
    games = matrix[~matrix["completed"]].copy()
    if games.empty:
        return games

    unknown = set(games["home_team"]) | set(games["away_team"])
    unknown -= set(matrix.loc[matrix["completed"], "home_team"])
    if unknown:
        raise ValueError(f"Unknown team name(s): {sorted(unknown)}. Use CSV nicknames, e.g. 'Chiefs', '49ers'.")

    X = games.reindex(columns=features).astype(float).fillna(0.0)
    p_home = model.predict_proba(X)[:, 1]

    out = games[["game_id", "season", "week", "home_team", "away_team", "kickoff_ts", "neutral"]].copy()
    out["home_win_prob"] = p_home
    out["away_win_prob"] = 1 - p_home
    out["elo_home_win_prob"] = games["elo_prob_home"].to_numpy()
    out["pick"] = np.where(p_home >= 0.5, out["home_team"], out["away_team"])
    out["confidence"] = np.abs(p_home - 0.5) * 2  # 0 = coin flip, 1 = certain
    return out.sort_values(["kickoff_ts", "game_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home"); ap.add_argument("--away")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--week", default="1")
    ap.add_argument("--neutral", action="store_true")
    args = ap.parse_args()

    if args.home or args.away:
        if not (args.home and args.away):
            ap.error("--home and --away must be given together")
        upcoming = cli_matchup(args)
    else:
        upcoming = load_upcoming()
        if upcoming.empty:
            print("No upcoming games found. Add rows to data/upcoming_games.csv or use --home/--away.")
            return

    out = predict(upcoming)
    print("\n=== Predictions ===")
    for _, r in out.iterrows():
        print(f"{r['kickoff_ts']:%a %b %d %I:%M %p} | {r['away_team']:>10} {r['away_win_prob']:6.1%}  @  "
              f"{r['home_team']:<10} {r['home_win_prob']:6.1%} | Pick: {r['pick']:<10} "
              f"(conf {r['confidence']:.2f}, Elo says home {r['elo_home_win_prob']:.1%})")

    if not (args.home or args.away):
        path = os.path.join(MODELS_PATH, "predictions_upcoming.csv")
        out.to_csv(path, index=False)
        print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
