"""Data access layer. Replaces db.py: reads games from the NFL team-stats CSV
(and an optional CSV of upcoming games) instead of Postgres.

Output schema of load_games() — one row per game:
    game_id, season, week, week_label, season_type, kickoff_ts, neutral,
    home_team, away_team, home_score, away_score, home_win, completed,
    plus every box-score stat as <stat>_home / <stat>_away
"""
import os

import numpy as np
import pandas as pd

from config import DATA_CSV, UPCOMING_CSV, GAME_TYPES

POSTSEASON_ORDER = {"Wildcard": 1, "Division": 2, "Conference": 3, "Superbowl": 4}

# Box-score stats in the CSV (each exists as <name>_home and <name>_away)
BOX_STATS = [
    "first_downs", "first_downs_from_passing", "first_downs_from_rushing",
    "first_downs_from_penalty", "third_down_comp", "third_down_att",
    "fourth_down_comp", "fourth_down_att", "plays", "drives", "yards",
    "pass_comp", "pass_att", "pass_yards", "sacks_num", "sacks_yards",
    "rush_att", "rush_yards", "pen_num", "pen_yards", "redzone_comp",
    "redzone_att", "fumbles", "interceptions", "def_st_td", "possession",
]


def _mmss_to_seconds(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.split(":", expand=True)
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def _normalize(raw: pd.DataFrame, completed: bool) -> pd.DataFrame:
    df = raw.copy()
    df["week_label"] = df["week"].astype(str).str.strip()
    is_post = df["week_label"].isin(POSTSEASON_ORDER.keys())
    df["season_type"] = np.where(is_post, "POST", "REG")

    # Numeric week: regular-season weeks as-is, playoff rounds continue after the last REG week
    reg_week = pd.to_numeric(df["week_label"].where(~is_post), errors="coerce")
    max_reg = reg_week.groupby(df["season"]).transform("max")
    max_reg = max_reg.fillna(pd.Series(np.where(df["season"] >= 2021, 18, 17), index=df.index))
    df["week"] = reg_week.fillna(max_reg + df["week_label"].map(POSTSEASON_ORDER)).astype(int)

    time_et = df["time_et"].fillna("1:00 PM").astype(str) if "time_et" in df else "1:00 PM"
    # Times in the source are US/Eastern; kept as naive Eastern timestamps.
    df["kickoff_ts"] = pd.to_datetime(df["date"].astype(str) + " " + time_et, format="%Y-%m-%d %I:%M %p")

    if "neutral" not in df:
        df["neutral"] = False
    df["neutral"] = df["neutral"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])

    df = df.rename(columns={"home": "home_team", "away": "away_team",
                            "score_home": "home_score", "score_away": "away_score"})
    df["game_id"] = (df["season"].astype(str) + "_" + df["week"].astype(str).str.zfill(2)
                     + "_" + df["away_team"] + "_" + df["home_team"])

    if completed:
        for side in ("home", "away"):
            df[f"possession_{side}"] = _mmss_to_seconds(df[f"possession_{side}"])
        df["home_win"] = np.select(
            [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
            [1.0, 0.0], default=0.5)  # 0.5 = tie
    else:
        df["home_score"] = np.nan
        df["away_score"] = np.nan
        df["home_win"] = np.nan
        for stat in BOX_STATS:
            for side in ("home", "away"):
                df[f"{stat}_{side}"] = np.nan

    df["completed"] = completed
    keep = ["game_id", "season", "week", "week_label", "season_type", "kickoff_ts", "neutral",
            "home_team", "away_team", "home_score", "away_score", "home_win", "completed"]
    stat_cols = [f"{s}_{side}" for s in BOX_STATS for side in ("home", "away")]
    return df[keep + stat_cols]


def load_upcoming(path: str = UPCOMING_CSV) -> pd.DataFrame:
    """Games to predict. Required columns: season, week, date, away, home.
    Optional: time_et (e.g. '1:00 PM'), neutral (True/False)."""
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    up = pd.read_csv(path)
    if up.empty:
        return pd.DataFrame()
    missing = {"season", "week", "date", "away", "home"} - set(up.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return _normalize(up, completed=False)


def load_games(season_start=None, season_end=None, include_future=False,
               game_types=GAME_TYPES, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """Completed games from the CSV (+ upcoming games when include_future=True)."""
    games = _normalize(pd.read_csv(DATA_CSV), completed=True)

    if include_future:
        up = load_upcoming() if upcoming is None else upcoming
        if not up.empty:
            # don't duplicate a game that already has a result
            up = up[~up["game_id"].isin(games["game_id"])]
            games = pd.concat([games, up], ignore_index=True)

    if season_start is not None:
        games = games[games["season"] >= season_start]
    if season_end is not None:
        games = games[(games["season"] <= season_end) | (~games["completed"])]
    if game_types:
        games = games[games["season_type"].isin(game_types) | (~games["completed"])]

    return games.sort_values(["kickoff_ts", "game_id"]).reset_index(drop=True)
