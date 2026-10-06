"""Train and select the game-outcome model.

Selection uses a walk-forward backtest instead of one validation split:
    tuning folds → for each season S from CV_START to VALID_SEASONS_END, fit every
                   candidate on all seasons before S and predict S; pick the
                   candidate with the lowest games-weighted log loss across folds
    test         → one honest evaluation of the winner (refit on every pre-test
                   season), on test seasons that no fold ever touched
Then the winner is refit on every completed season and saved for predict.py.

Folds stop at VALID_SEASONS_END so the test seasons never influence model choice.
Set the first predicted season with CV_START (env var or config.py; default 2008).
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

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from config import (MODELS_PATH, SEED, TRAIN_SEASONS_START, TRAIN_SEASONS_END,
                    VALID_SEASONS_START, VALID_SEASONS_END, TEST_SEASONS_START, TEST_SEASONS_END,
                    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, REGISTERED_MODEL,
                    PROCESSED_DATA_PATH)
from features import build_features, model_feature_columns
from utils import get_logger
from validate import run_validation

try:  # first season predicted by the tuning folds
    from config import CV_START
except ImportError:
    CV_START = int(os.getenv("CV_START", "2008"))

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


def model_params(est):
    """Hyperparameters worth logging (scalars only), for any candidate type."""
    inner = est.steps[-1][1] if hasattr(est, "steps") else est
    return {k: v for k, v in inner.get_params().items()
            if isinstance(v, (int, float, str, bool)) or v is None}


# --------------------------------------------------------------------------
# Walk-forward cross-validation
# --------------------------------------------------------------------------
def walk_forward_folds(m, start, end):
    """(season, all earlier seasons, that season) for each season in start..end."""
    for yr in range(start, end + 1):
        past, cur = m[m["season"] < yr], m[m["season"] == yr]
        if not past.empty and not cur.empty:
            yield yr, past, cur


def summarize_folds(per_season: pd.DataFrame) -> dict:
    """Games-weighted average of each metric across folds, plus season-to-season spread."""
    w = per_season["games"]
    out = {k: float(np.average(per_season[k], weights=w))
           for k in ("logloss", "brier", "auc", "accuracy")}
    out["logloss_std"] = float(per_season["logloss"].std(ddof=1)) if len(per_season) > 1 else 0.0
    out["games"] = int(w.sum())
    out["folds"] = int(len(per_season))
    return out


def cv_score(est, m, cols, start, end):
    """Walk-forward score for one candidate. Returns (per-season DataFrame, summary)."""
    rows = []
    for yr, past, cur in walk_forward_folds(m, start, end):
        Xp, yp = xy(past, cols)
        Xc, yc = xy(cur, cols)
        p = finalize(clone(est)).fit(Xp, yp).predict_proba(Xc)[:, 1]
        rows.append({"season": yr, **metrics(yc, p)})
    per_season = pd.DataFrame(rows)
    return per_season, summarize_folds(per_season)


def cv_baselines(m, start, end):
    """Elo-only and home-rate baselines scored on exactly the same folds."""
    elo_rows, home_rows = [], []
    for yr, past, cur in walk_forward_folds(m, start, end):
        yc = cur["home_win"].to_numpy()
        elo_rows.append({"season": yr, **metrics(yc, cur["elo_prob_home"].to_numpy())})
        home_rows.append({"season": yr, **metrics(yc, np.full(len(yc), past["home_win"].mean()))})
    return summarize_folds(pd.DataFrame(elo_rows)), summarize_folds(pd.DataFrame(home_rows))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
def register_and_promote(model_info, report, data_sha):
    """Document the new registry version, then point the 'champion' alias at it
    if it beats the current champion's test log loss."""
    client = MlflowClient()
    version = model_info.registered_model_version
    t = report["test_model"]
    cv = report["cv_selected"]
    client.update_model_version(
        REGISTERED_MODEL, version,
        description=(f"{report['selected_model']} ({report['n_features']} features). "
                     f"Selected by walk-forward CV log loss over {report['cv_seasons']} "
                     f"({cv['folds']} folds, {cv['games']:,} games; cv logloss={cv['logloss']:.4f} "
                     f"±{cv['logloss_std']:.4f}). "
                     f"Test {report['test_seasons']}: acc={t['accuracy']:.3f}, logloss={t['logloss']:.4f}, "
                     f"AUC={t['auc']:.3f}. Production refit on all seasons "
                     f"{TRAIN_SEASONS_START}-{TEST_SEASONS_END}. Data sha256 {data_sha[:12]}."))
    for k, v in {"selected_model": report["selected_model"], "test_logloss": f"{t['logloss']:.5f}",
                 "test_accuracy": f"{t['accuracy']:.4f}", "cv_logloss": f"{cv['logloss']:.5f}",
                 "cv_seasons": report["cv_seasons"], "data_sha256": data_sha}.items():
        client.set_model_version_tag(REGISTERED_MODEL, version, k, v)

    try:
        champ = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
        champ_ll = float(champ.tags.get("test_logloss", "inf"))
    except Exception:  # no champion yet
        champ, champ_ll = None, float("inf")
    if t["logloss"] <= champ_ll:
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
        logger.info(f"Registered {REGISTERED_MODEL} v{version} → alias 'champion'")
    else:
        client.set_registered_model_alias(REGISTERED_MODEL, "challenger", version)
        logger.info(f"Registered {REGISTERED_MODEL} v{version} → 'challenger' "
                    f"(champion v{champ.version} still better: {champ_ll:.4f})")


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
    validation = run_validation()          # stops here if the raw data fails a critical check
    data_sha = validation["data_sha256"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="train_select_register") as parent:
        _train(data_sha, parent)


def _train(data_sha, parent):
    mlflow.log_artifact(os.path.join(PROCESSED_DATA_PATH, "validation_report.json"),
                        artifact_path="data")
    m = load_matrix()
    cols = model_feature_columns(m)

    if not TRAIN_SEASONS_START < CV_START <= VALID_SEASONS_END:
        raise ValueError(f"CV_START={CV_START} must be after {TRAIN_SEASONS_START} "
                         f"and no later than {VALID_SEASONS_END}")

    # Everything before the test seasons is available for tuning; the test seasons never are.
    tune = split(m, TRAIN_SEASONS_START, VALID_SEASONS_END)
    test = split(m, TEST_SEASONS_START, TEST_SEASONS_END)
    cv_seasons = f"{CV_START}-{VALID_SEASONS_END}"
    n_folds = VALID_SEASONS_END - CV_START + 1
    logger.info(f"{len(cols)} features | tuning {len(tune)} games, {n_folds} walk-forward folds "
                f"({cv_seasons}) | test {len(test)} games")
    mlflow.set_tags({"stage": "model_selection", "selection": "walk_forward_cv",
                     "data_sha256": data_sha})
    mlflow.log_params({
        "data_version": data_sha[:12], "n_features": len(cols),
        "tuning_seasons": f"{TRAIN_SEASONS_START}-{VALID_SEASONS_END}",
        "cv_seasons": cv_seasons, "cv_folds": n_folds,
        "test_seasons": f"{TEST_SEASONS_START}-{TEST_SEASONS_END}",
        "n_tune": len(tune), "n_test": len(test), "seed": SEED,
    })

    # ---- Model selection: walk-forward CV, one nested MLflow run per candidate ----
    rows, fold_rows = [], []
    best = (np.inf, None, None, None)
    for name, est in candidate_models():
        with mlflow.start_run(run_name=name, nested=True):
            per_season, res = cv_score(est, tune, cols, CV_START, VALID_SEASONS_END)
            mlflow.set_tags({"candidate": name, "family": name.split("_")[0], "data_sha256": data_sha})
            mlflow.log_params(model_params(est))
            for _, r in per_season.iterrows():   # per-season line charts in the MLflow UI
                mlflow.log_metrics({"fold_logloss": r["logloss"], "fold_accuracy": r["accuracy"]},
                                   step=int(r["season"]))
            mlflow.log_metrics({f"cv_{k}": v for k, v in res.items() if k not in ("games", "folds")})
        rows.append({"model": name, **res})
        fold_rows.append(per_season.assign(model=name))
        logger.info(f"  {name:28s} cv logloss={res['logloss']:.4f} (±{res['logloss_std']:.4f}) "
                    f"acc={res['accuracy']:.3f}")
        if res["logloss"] < best[0]:
            best = (res["logloss"], name, est, res)

    # Baselines on the same folds for context
    base_elo_cv, base_home_cv = cv_baselines(tune, CV_START, VALID_SEASONS_END)
    rows.append({"model": "baseline_elo_only", **base_elo_cv})
    rows.append({"model": "baseline_home_rate", **base_home_cv})
    selection = pd.DataFrame(rows).sort_values("logloss")
    # File name kept so existing DVC outputs and reports still find it
    selection.to_csv(os.path.join(MODELS_PATH, "model_selection_valid.csv"), index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(
        os.path.join(MODELS_PATH, "model_selection_cv_by_season.csv"), index=False)

    _, best_name, best_est, best_cv = best
    logger.info(f"Selected: {best_name} (cv logloss {best_cv['logloss']:.4f} vs "
                f"Elo {base_elo_cv['logloss']:.4f}, home rate {base_home_cv['logloss']:.4f})")
    mlflow.log_param("selected_model", best_name)
    mlflow.log_params({f"best__{k}": v for k, v in model_params(best_est).items()})
    mlflow.log_metrics({
        "cv_selected_logloss": best_cv["logloss"], "cv_selected_accuracy": best_cv["accuracy"],
        "cv_selected_logloss_std": best_cv["logloss_std"],
        "cv_elo_logloss": base_elo_cv["logloss"], "cv_elo_accuracy": base_elo_cv["accuracy"],
        "cv_home_rate_logloss": base_home_cv["logloss"], "cv_home_rate_accuracy": base_home_cv["accuracy"],
    })

    # ---- Honest test: refit on every pre-test season, score the untouched test seasons ----
    Xtv, ytv = xy(tune, cols)
    Xte, yte = xy(test, cols)
    model_tv = finalize(clone(best_est)).fit(Xtv, ytv)
    p_test = model_tv.predict_proba(Xte)[:, 1]

    report = {
        "selected_model": best_name,
        "n_features": len(cols),
        "selection": "walk_forward_cv",
        "cv_seasons": cv_seasons,
        "cv_selected": best_cv,
        "cv_baseline_elo_only": base_elo_cv,
        "cv_baseline_home_rate": base_home_cv,
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

    # ---- MLflow: metrics, artifacts, and a versioned entry in the model registry ----
    for k in ("test_model", "test_baseline_elo_only", "test_baseline_home_rate"):
        mlflow.log_metrics({f"{k}_{mk}": mv for mk, mv in report[k].items() if mk != "games"})
    for fname in ("metrics.json", "model_selection_valid.csv", "model_selection_cv_by_season.csv",
                  "test_predictions.csv", "feature_importance.csv", "feature_importance.png",
                  "best_model_features.txt"):
        mlflow.log_artifact(os.path.join(MODELS_PATH, fname), artifact_path="reports")

    model_info = mlflow.sklearn.log_model(
        final, name="model", registered_model_name=REGISTERED_MODEL,
        signature=infer_signature(Xall.head(50), final.predict_proba(Xall.head(50))[:, 1]),
        input_example=Xall.head(3),
        serialization_format="cloudpickle")  # our own trained model, same trust level as joblib
    register_and_promote(model_info, report, data_sha)

    logger.info(f"Saved production model (trained on {len(m)} games, "
                f"{int(m['season'].min())}-{int(m['season'].max())}) → {MODELS_PATH}")


if __name__ == "__main__":
    main()