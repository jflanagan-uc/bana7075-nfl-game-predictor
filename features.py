"""Feature engineering from the box-score CSV.

Pipeline
  1. games (one row per game)            -> data.load_games
  2. team-game rows (two per game)       -> off_* = the team's own box score,
                                            def_* = what its defense allowed
  3. pre-game form features              -> rolling 3/8-game means + EWM, built ONLY
                                            from games strictly before kickoff
  4. Elo ratings + rest days             -> sequential, pre-game values only
  5. game-level matrix (one row per game) -> home-minus-away differences + Elo + context

Nothing about the game being predicted (its score or box score) ever enters
its own feature row.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (PROCESSED_DATA_PATH, TRAIN_SEASONS_START, TEST_SEASONS_END,
                    ELO_K, ELO_HFA, ELO_SEASON_REVERT, ROLL_WINDOWS, EWM_HALFLIFE)
from data import load_games
from utils import get_logger, safe_div

logger = get_logger("features")

# Per-game stats computed for each side. off_<stat> = team's offense, def_<stat> = allowed.
SIDE_STATS = [
    "points", "yards", "plays", "ypp", "first_downs", "ny_a", "comp_pct", "sack_rate",
    "rush_ypc", "pass_rate", "third_pct", "turnovers", "pts_per_drive",
    "yds_per_drive", "pen_yards", "top_sec",
]

# Everything the model is NOT allowed to see (same-game results / identifiers)
ID_COLS = ["game_id", "season", "week", "week_label", "season_type", "kickoff_ts",
           "home_team", "away_team", "completed"]
TARGET = "home_win"


# --------------------------------------------------------------------------
# 1-2. Team-game rows
# --------------------------------------------------------------------------
def _side_stats(g: pd.DataFrame, s: str) -> pd.DataFrame:
    """Offensive box-score stats for side s ('home' or 'away') of each game."""
    c = lambda name: g[f"{name}_{s}"].astype(float)
    score = g[f"{s}_score"].astype(float)
    dropbacks = c("pass_att") + c("sacks_num")
    return pd.DataFrame({
        "points": score,
        "yards": c("yards"),
        "plays": c("plays"),
        "ypp": safe_div(c("yards"), c("plays")),
        "first_downs": c("first_downs"),
        "ny_a": safe_div(c("pass_yards"), dropbacks),           # pass_yards in the CSV is net of sacks
        "comp_pct": safe_div(c("pass_comp"), c("pass_att")),
        "sack_rate": safe_div(c("sacks_num"), dropbacks),
        "rush_ypc": safe_div(c("rush_yards"), c("rush_att")),
        "pass_rate": safe_div(dropbacks, dropbacks + c("rush_att")),
        "third_pct": safe_div(c("third_down_comp"), c("third_down_att")),
        "turnovers": c("fumbles") + c("interceptions"),
        "pts_per_drive": safe_div(score, c("drives")),
        "yds_per_drive": safe_div(c("yards"), c("drives")),
        "pen_yards": c("pen_yards"),
        "top_sec": c("possession"),
    }, index=g.index)


def build_team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["game_id", "season", "week", "week_label", "season_type",
                 "kickoff_ts", "neutral", "completed"]
    stats = {s: _side_stats(games, s) for s in ("home", "away")}

    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        tg = games[base_cols].copy()
        tg["team"] = games[f"{side}_team"]
        tg["opp"] = games[f"{opp}_team"]
        tg["is_home"] = int(side == "home")
        own = stats[side].add_prefix("off_")
        allowed = stats[opp].add_prefix("def_")
        tg = pd.concat([tg, own, allowed], axis=1)
        tg["margin"] = tg["off_points"] - tg["def_points"]
        tg["team_win"] = np.select([tg["margin"] > 0, tg["margin"] < 0], [1.0, 0.0], 0.5)
        tg.loc[~tg["completed"], "team_win"] = np.nan
        rows.append(tg)

    tg = pd.concat(rows, ignore_index=True)
    return tg.sort_values(["kickoff_ts", "game_id", "is_home"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Pre-game rolling form
# --------------------------------------------------------------------------
def form_columns() -> list[str]:
    base = [f"off_{s}" for s in SIDE_STATS] + [f"def_{s}" for s in SIDE_STATS] + ["margin", "team_win"]
    cols = []
    for b in base:
        cols += [f"{b}_r{w}" for w in ROLL_WINDOWS] + [f"{b}_ewm"]
    return cols


def add_rolling_form(tg: pd.DataFrame) -> pd.DataFrame:
    """For every row, attach each team's form entering that game.

    Form is computed on completed games only ("state after game N"), then attached
    to each row with an as-of join on the *previous* completed game. That handles
    upcoming games (and several upcoming games per team) without any leakage.
    """
    base = [f"off_{s}" for s in SIDE_STATS] + [f"def_{s}" for s in SIDE_STATS] + ["margin", "team_win"]
    done = tg[tg["completed"]].sort_values(["team", "kickoff_ts"]).copy()
    grp = done.groupby("team", sort=False)

    cols = {}
    for b in base:
        for w in ROLL_WINDOWS:
            cols[f"{b}_r{w}"] = grp[b].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        cols[f"{b}_ewm"] = grp[b].transform(lambda s: s.ewm(halflife=EWM_HALFLIFE, ignore_na=True).mean())
    cols["games_played"] = grp.cumcount() + 1
    state = pd.concat([done[["team", "kickoff_ts"]], pd.DataFrame(cols)], axis=1)

    left = tg.sort_values("kickoff_ts").reset_index()
    state = state.sort_values("kickoff_ts")
    merged = pd.merge_asof(left, state, on="kickoff_ts", by="team",
                           allow_exact_matches=False, direction="backward")
    merged["games_played"] = merged["games_played"].fillna(0)
    return merged.set_index("index").sort_index()


def add_rest_days(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["team", "kickoff_ts"]).copy()
    prev = tg.groupby("team")["kickoff_ts"].shift(1)
    rest = (tg["kickoff_ts"].dt.normalize() - prev.dt.normalize()).dt.days
    tg["rest_days"] = rest.where(rest <= 21, 14).fillna(14).clip(upper=14)  # offseason/bye -> 14
    return tg.sort_index()


# --------------------------------------------------------------------------
# 4. Elo
# --------------------------------------------------------------------------
def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """538-style Elo with margin-of-victory multiplier and offseason regression.
    Returns pre-game ratings per game (upcoming games get current ratings, no update)."""
    ratings, last_season, out = {}, {}, []
    for g in games.sort_values(["kickoff_ts", "game_id"]).itertuples(index=False):
        for t in (g.home_team, g.away_team):
            if t not in ratings:
                ratings[t], last_season[t] = 1500.0, g.season
            elif last_season[t] != g.season:
                ratings[t] = ratings[t] * (1 - ELO_SEASON_REVERT) + 1505 * ELO_SEASON_REVERT
                last_season[t] = g.season

        rh, ra = ratings[g.home_team], ratings[g.away_team]
        hfa = 0.0 if g.neutral else ELO_HFA
        exp_home = 1.0 / (1.0 + 10 ** (-(rh + hfa - ra) / 400))
        out.append((g.game_id, rh, ra, exp_home))

        if g.completed:
            margin = g.home_score - g.away_score
            result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            if margin == 0:
                mult = 1.0
            else:
                winner_diff = (rh + hfa - ra) if margin > 0 else (ra - rh - hfa)
                mult = np.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            shift = ELO_K * mult * (result - exp_home)
            ratings[g.home_team] = rh + shift
            ratings[g.away_team] = ra - shift

    return pd.DataFrame(out, columns=["game_id", "elo_home_pre", "elo_away_pre", "elo_prob_home"])


# --------------------------------------------------------------------------
# 5. Game-level model matrix
# --------------------------------------------------------------------------
def build_game_matrix(tg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    feat = form_columns() + ["games_played", "rest_days"]
    home = tg[tg["is_home"] == 1].set_index("game_id")[feat]
    away = tg[tg["is_home"] == 0].set_index("game_id")[feat]

    # A few direct offense-vs-defense matchup features (home O vs away D, and vice versa)
    matchup = {}
    for stat in ("ypp", "ny_a", "rush_ypc", "pts_per_drive", "turnovers"):
        matchup[f"match_home_off_{stat}"] = home[f"off_{stat}_ewm"] - away[f"def_{stat}_ewm"]
        matchup[f"match_away_off_{stat}"] = away[f"off_{stat}_ewm"] - home[f"def_{stat}_ewm"]
    diffs = pd.concat([(home - away).add_prefix("diff_"), pd.DataFrame(matchup)], axis=1)

    out = games.set_index("game_id")[
        ["season", "week", "week_label", "season_type", "kickoff_ts", "neutral",
         "home_team", "away_team", "home_score", "away_score", "home_win", "completed"]
    ].join(diffs)
    out = out.join(compute_elo(games).set_index("game_id")).copy()
    out["home_field"] = (~out["neutral"].astype(bool)).astype(int)
    out["elo_diff"] = out["elo_home_pre"] - out["elo_away_pre"]
    out["is_postseason"] = (out["season_type"] == "POST").astype(int)
    out["early_season"] = (out["week"] <= 4).astype(int)  # form is still mostly last season
    return out.reset_index()


def model_feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Columns the model trains on: pre-game only, numeric."""
    exclude = set(ID_COLS) | {TARGET, "home_score", "away_score", "neutral",
                              "elo_home_pre", "elo_away_pre"}
    return [c for c in matrix.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(matrix[c])]


def build_features(season_start=None, season_end=None, include_future=False, upcoming=None):
    """Returns (team_games, game_matrix).

    History always starts at the first season in the CSV so rolling form and Elo are
    warmed up; season_start/season_end only filter the returned rows."""
    logger.info("Loading games from CSV…")
    games = load_games(include_future=include_future, upcoming=upcoming)

    logger.info("Building team-game rows…")
    tg = build_team_game_rows(games)
    logger.info("Adding rest days and rolling form…")
    tg = add_rest_days(tg)
    tg = add_rolling_form(tg)

    logger.info("Building game-level matrix + Elo…")
    matrix = build_game_matrix(tg, games)

    def _filter(df):
        keep = pd.Series(True, index=df.index)
        if season_start is not None:
            keep &= (df["season"] >= season_start) | ~df["completed"]
        if season_end is not None:
            keep &= (df["season"] <= season_end) | ~df["completed"]
        return df[keep].reset_index(drop=True)

    return _filter(tg), _filter(matrix)


def main():
    Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
    tg, matrix = build_features(TRAIN_SEASONS_START - 1, TEST_SEASONS_END, include_future=False)
    tg.to_csv(os.path.join(PROCESSED_DATA_PATH, "features_team_games.csv"), index=False)
    matrix.to_csv(os.path.join(PROCESSED_DATA_PATH, "features_games.csv"), index=False)
    logger.info(f"Wrote {len(tg):,} team-game rows and {len(matrix):,} game rows "
                f"({len(model_feature_columns(matrix))} model features) → {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
