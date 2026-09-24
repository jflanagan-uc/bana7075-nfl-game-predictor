"""Train and select the game-outcome model.

Split (all chronological, set in config.py):
    train  → fit each candidate model
    valid  → pick the best candidate by log loss
    test   → one honest evaluation of the winner (refit on train+valid)
Then the winner is refit on every completed season and saved for predict.py.
"""
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (MODELS_PATH, SEED, TRAIN_SEASONS_START, TRAIN_SEASONS_END,
                    VALID_SEASONS_START, VALID_SEASONS_END, TEST_SEASONS_START, TEST_SEASONS_END)
from features import build_features, model_feature_columns
from utils import get_logger

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is optional
    XGBClassifier = None

logger = get_logger("train")


def candidate_models():
    """(name, unfitted estimator) pairs. Small grids — the signal in NFL games is modest."""
    cands = []
    for C in (0.003, 0.01, 0.03, 0.1, 1.0):
        cands.append((f"logreg_C{C}", make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=5000))))
    for depth in (4, 6):
        cands.append((f"rf_d{depth}", RandomForestClassifier(
            n_estimators=400, max_depth=depth, min_samples_leaf=25,
            n_jobs=-1, random_state=SEED)))
    for lr, depth, iters in ((0.03, 2, 300), (0.03, 3, 300), (0.05, 2, 200)):
        cands.append((f"hgb_lr{lr}_d{depth}_n{iters}", HistGradientBoostingClassifier(
            learning_rate=lr, max_depth=depth, max_iter=iters, l2_regularization=1.0,
            min_samples_leaf=40, random_state=SEED)))
    if XGBClassifier is not None:
        for depth in (2, 3):
            cands.append((f"xgb_d{depth}", XGBClassifier(
                n_estimators=300, max_depth=depth, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.7, eval_metric="logloss", random_state=SEED)))
    return cands


def finalize(estimator):
    """Tree models get sigmoid calibration; logistic regression is already calibrated."""
    if hasattr(estimator, "steps"):  # the scaled logistic-regression pipeline
        return estimator
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=5)


def metrics(y, p):
    return {
        "games": int(len(y)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }


def load_matrix():
    _, m = build_features(TRAIN_SEASONS_START, TEST_SEASONS_END, include_future=False)
    m = m[m["completed"] & m["home_win"].isin([0.0, 1.0])].copy()  # ties dropped from training/eval
    m["home_win"] = m["home_win"].astype(int)
    return m


def split(m, start, end):
    return m[(m["season"] >= start) & (m["season"] <= end)]


def xy(df, cols):
    return df[cols].astype(float).fillna(0.0), df["home_win"].to_numpy()


def save_importance(model, X, y, cols, outdir):
    """Permutation importance (works for every model type) on held-out data."""
    r = permutation_importance(model, X, y, scoring="neg_log_loss", n_repeats=3,
                               random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": cols, "importance": r.importances_mean})
             .sort_values("importance", ascending=False))
    imp.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)

    top = imp.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 features (permutation importance, log-loss increase)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=130)
    plt.close()


def main():
    Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
    m = load_matrix()
    cols = model_feature_columns(m)

    train = split(m, TRAIN_SEASONS_START, TRAIN_SEASONS_END)
    valid = split(m, VALID_SEASONS_START, VALID_SEASONS_END)
    test = split(m, TEST_SEASONS_START, TEST_SEASONS_END)
    logger.info(f"{len(cols)} features | train {len(train)} | valid {len(valid)} | test {len(test)} games")

    Xtr, ytr = xy(train, cols)
    Xva, yva = xy(valid, cols)

    # ---- Model selection on the validation seasons ----
    rows = []
    best = (np.inf, None, None)
    for name, est in candidate_models():
        fitted = finalize(clone(est)).fit(Xtr, ytr)
        res = metrics(yva, fitted.predict_proba(Xva)[:, 1])
        rows.append({"model": name, **res})
        logger.info(f"  {name:28s} valid logloss={res['logloss']:.4f} acc={res['accuracy']:.3f}")
        if res["logloss"] < best[0]:
            best = (res["logloss"], name, est)

    # Baselines on validation for context
    rows.append({"model": "baseline_elo_only", **metrics(yva, valid["elo_prob_home"].to_numpy())})
    rows.append({"model": "baseline_home_rate",
                 **metrics(yva, np.full(len(yva), ytr.mean()))})
    pd.DataFrame(rows).to_csv(os.path.join(MODELS_PATH, "model_selection_valid.csv"), index=False)

    _, best_name, best_est = best
    logger.info(f"Selected: {best_name}")

    # ---- Honest test: refit on train+valid, score the untouched test seasons ----
    trval = pd.concat([train, valid])
    Xtv, ytv = xy(trval, cols)
    Xte, yte = xy(test, cols)
    model_tv = finalize(clone(best_est)).fit(Xtv, ytv)
    p_test = model_tv.predict_proba(Xte)[:, 1]

    report = {
        "selected_model": best_name,
        "n_features": len(cols),
        "test_seasons": f"{TEST_SEASONS_START}-{TEST_SEASONS_END}",
        "test_model": metrics(yte, p_test),
        "test_baseline_elo_only": metrics(yte, test["elo_prob_home"].to_numpy()),
        "test_baseline_home_rate": metrics(yte, np.full(len(yte), ytv.mean())),
    }
    for k in ("test_model", "test_baseline_elo_only", "test_baseline_home_rate"):
        r = report[k]
        logger.info(f"{k:26s} logloss={r['logloss']:.4f} brier={r['brier']:.4f} "
                    f"auc={r['auc']:.3f} acc={r['accuracy']:.3f}")

    test_out = test[["game_id", "season", "week", "home_team", "away_team", "home_win", "elo_prob_home"]].copy()
    test_out["model_prob_home"] = p_test
    test_out.to_csv(os.path.join(MODELS_PATH, "test_predictions.csv"), index=False)

    save_importance(model_tv, Xte, yte, cols, MODELS_PATH)

    # ---- Production model: refit on every completed season ----
    Xall, yall = xy(m, cols)
    final = finalize(clone(best_est)).fit(Xall, yall)
    joblib.dump(final, os.path.join(MODELS_PATH, "best_model.joblib"))
    with open(os.path.join(MODELS_PATH, "best_model_features.txt"), "w") as f:
        f.write("\n".join(cols) + "\n")
    with open(os.path.join(MODELS_PATH, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved production model (trained on {len(m)} games, "
                f"{int(m['season'].min())}-{int(m['season'].max())}) → {MODELS_PATH}")


if __name__ == "__main__":
    main()
