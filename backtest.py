"""Walk-forward backtest: for each season S, train on every season before S and
predict S — the same situation the model faces in real use. Compares against an
Elo-only baseline and a 'pick the home team' baseline."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from config import MODELS_PATH, TRAIN_SEASONS_START, TEST_SEASONS_END, BACKTEST_START
from train import load_matrix, metrics, xy
from utils import get_logger

logger = get_logger("backtest")


def main():
    template = joblib.load(os.path.join(MODELS_PATH, "best_model.joblib"))
    with open(os.path.join(MODELS_PATH, "best_model_features.txt")) as f:
        cols = [line.strip() for line in f if line.strip()]

    m = load_matrix()
    results = []
    for yr in range(max(BACKTEST_START, TRAIN_SEASONS_START + 3), TEST_SEASONS_END + 1):
        past, cur = m[m["season"] < yr], m[m["season"] == yr]
        if cur.empty:
            continue
        Xp, yp = xy(past, cols)
        Xc, yc = xy(cur, cols)
        p = clone(template).fit(Xp, yp).predict_proba(Xc)[:, 1]

        row = {"season": yr}
        for tag, probs in (("model", p), ("elo", cur["elo_prob_home"].to_numpy())):
            for k, v in metrics(yc, probs).items():
                row[k if k == "games" else f"{tag}_{k}"] = v
        row["home_pick_acc"] = float(yc.mean())
        results.append(row)
        logger.info(f"{yr}: model acc={row['model_accuracy']:.3f} logloss={row['model_logloss']:.4f} | "
                    f"elo acc={row['elo_accuracy']:.3f} | home acc={row['home_pick_acc']:.3f}")

    out = pd.DataFrame(results)
    w = out["games"]
    summary = {"season": "ALL", "games": int(w.sum())}
    for c in out.columns.drop(["season", "games"]):
        summary[c] = float(np.average(out[c], weights=w))
    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)

    path = os.path.join(MODELS_PATH, "backtest_by_season.csv")
    out.to_csv(path, index=False)
    logger.info(f"Overall: model acc={summary['model_accuracy']:.3f}, elo acc={summary['elo_accuracy']:.3f}, "
                f"home acc={summary['home_pick_acc']:.3f} → {path}")


if __name__ == "__main__":
    main()
