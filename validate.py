"""Data validation for the raw game CSV and the upcoming-games CSV.

Run on its own:   python validate.py
Also runs automatically at the start of train.py (results are logged to MLflow).

Every check returns pass/fail plus a count of offending rows. Critical failures
stop the pipeline; warnings are logged but let it continue.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

from config import DATA_CSV, UPCOMING_CSV, PROCESSED_DATA_PATH
from data import BOX_STATS, POSTSEASON_ORDER
from utils import get_logger

logger = get_logger("validate")

REQUIRED_COLS = (["season", "week", "date", "time_et", "neutral", "away", "home",
                  "score_away", "score_home"]
                 + [f"{s}_{side}" for s in BOX_STATS for side in ("home", "away")])
VALID_WEEKS = {str(w) for w in range(1, 19)} | set(POSTSEASON_ORDER)
# (made, attempted) pairs where made can never exceed attempted
MADE_VS_ATT = [("pass_comp", "pass_att"), ("third_down_comp", "third_down_att"),
               ("fourth_down_comp", "fourth_down_att"), ("redzone_comp", "redzone_att")]


def file_sha256(path: str) -> str:
    """Content hash of a file: a data version ID that changes whenever the data changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(results, name, n_bad, critical=True, detail=""):
    results.append({"check": name, "passed": bool(n_bad == 0), "failed_rows": int(n_bad),
                    "severity": "critical" if critical else "warning", "detail": detail})


def validate_history(path: str = DATA_CSV) -> list[dict]:
    df = pd.read_csv(path)
    r = []

    missing = sorted(set(REQUIRED_COLS) - set(df.columns))
    _check(r, "required_columns_present", len(missing), detail=", ".join(missing))
    if missing:
        return r  # nothing else can be checked reliably

    _check(r, "no_missing_values_in_key_columns",
           df[["season", "week", "date", "home", "away", "score_home", "score_away"]].isna().any(axis=1).sum())
    _check(r, "no_duplicate_games", df.duplicated(subset=["season", "week", "home", "away"]).sum())
    _check(r, "valid_week_labels", (~df["week"].astype(str).str.strip().isin(VALID_WEEKS)).sum())
    _check(r, "dates_parse", pd.to_datetime(df["date"], errors="coerce").isna().sum())
    _check(r, "team_not_playing_itself", (df["home"] == df["away"]).sum())
    _check(r, "scores_non_negative", ((df["score_home"] < 0) | (df["score_away"] < 0)).sum())

    for made, att in MADE_VS_ATT:
        bad = sum((df[f"{made}_{s}"] > df[f"{att}_{s}"]).sum() for s in ("home", "away"))
        _check(r, f"{made}_le_{att}", bad)

    poss = pd.concat([df["possession_home"], df["possession_away"]]).astype(str)
    _check(r, "possession_is_mm_ss", (~poss.str.match(r"^\d{1,2}:\d{2}$")).sum())

    n_teams = pd.concat([df["home"], df["away"]]).nunique()
    _check(r, "exactly_32_teams", 0 if n_teams == 32 else 1, detail=f"found {n_teams}")

    per_season = df.groupby("season").size()
    _check(r, "season_game_counts_plausible", ((per_season < 250) | (per_season > 290)).sum(),
           critical=False, detail=per_season.to_dict().__str__())
    return r


def validate_upcoming(path: str = UPCOMING_CSV, known_teams: set | None = None) -> list[dict]:
    r = []
    if not os.path.exists(path):
        _check(r, "upcoming_file_exists", 1, critical=False)
        return r
    up = pd.read_csv(path)
    missing = sorted({"season", "week", "date", "away", "home"} - set(up.columns))
    _check(r, "upcoming_required_columns", len(missing), detail=", ".join(missing))
    if missing:
        return r
    _check(r, "upcoming_dates_parse", pd.to_datetime(up["date"], errors="coerce").isna().sum())
    _check(r, "upcoming_no_duplicates", up.duplicated(subset=["season", "week", "home", "away"]).sum())
    if known_teams:
        unknown = (~up["home"].isin(known_teams)) | (~up["away"].isin(known_teams))
        _check(r, "upcoming_team_names_known", unknown.sum(),
               detail=str(sorted(set(up.loc[unknown, "home"]) | set(up.loc[unknown, "away"]))))
    return r


def run_validation(raise_on_fail: bool = True) -> dict:
    hist = pd.read_csv(DATA_CSV, usecols=["home", "away"])
    teams = set(hist["home"]) | set(hist["away"])
    results = validate_history() + validate_upcoming(known_teams=teams)

    report = {
        "data_csv": os.path.basename(DATA_CSV),
        "data_sha256": file_sha256(DATA_CSV),
        "n_checks": len(results),
        "n_failed_critical": sum(1 for c in results if not c["passed"] and c["severity"] == "critical"),
        "n_warnings": sum(1 for c in results if not c["passed"] and c["severity"] == "warning"),
        "checks": results,
    }
    Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(PROCESSED_DATA_PATH, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    for c in results:
        status = "PASS" if c["passed"] else ("FAIL" if c["severity"] == "critical" else "WARN")
        logger.info(f"[{status}] {c['check']}" + (f" ({c['failed_rows']} rows)" if not c["passed"] else ""))
    logger.info(f"{report['n_checks']} checks, {report['n_failed_critical']} critical failures, "
                f"{report['n_warnings']} warnings | data version {report['data_sha256'][:12]}")

    if raise_on_fail and report["n_failed_critical"]:
        raise ValueError("Data validation failed; see processed/validation_report.json")
    return report


if __name__ == "__main__":
    try:
        run_validation()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
