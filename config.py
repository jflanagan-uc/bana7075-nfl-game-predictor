"""Project settings. Every value can be overridden with an environment variable (or a .env file)."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(env, default):
    return os.path.abspath(os.getenv(env, os.path.join(BASE_DIR, default)))


# ---- Data sources ----
DATA_CSV     = _path("NFL_DATA_CSV", "data/nfl_team_stats_2002-2025.csv")
UPCOMING_CSV = _path("UPCOMING_CSV", "data/upcoming_games.csv")   # games you want predicted

# ---- Output folders ----
PROCESSED_DATA_PATH = _path("PROCESSED_DATA_PATH", "processed")
MODELS_PATH         = _path("MODELS_PATH", "models")
LOGS_PATH           = _path("LOGS_PATH", "logs")
LOGOS_DIR           = _path("LOGOS_DIR", "logos")
MODELS_DIR          = MODELS_PATH  # kept for older scripts that imported MODELS_DIR

# ---- Game filters ----
# REG = weeks 1-18, POST = Wildcard/Division/Conference/Superbowl
GAME_TYPES = [t.strip().upper() for t in os.getenv("GAME_TYPES", "REG,POST").split(",")]

# ---- Season splits (strictly chronological, no overlap) ----
# 2002 is used only as history for the rolling/Elo features, not as training rows.
TRAIN_SEASONS_START = int(os.getenv("TRAIN_SEASONS_START", "2003"))
TRAIN_SEASONS_END   = int(os.getenv("TRAIN_SEASONS_END", "2019"))
VALID_SEASONS_START = int(os.getenv("VALID_SEASONS_START", "2020"))
VALID_SEASONS_END   = int(os.getenv("VALID_SEASONS_END", "2022"))
TEST_SEASONS_START  = int(os.getenv("TEST_SEASONS_START", "2023"))
TEST_SEASONS_END    = int(os.getenv("TEST_SEASONS_END", "2025"))
BACKTEST_START      = int(os.getenv("BACKTEST_START", "2008"))

CURRENT_SEASON = int(os.getenv("CURRENT_SEASON", "2026"))

# ---- Elo settings (538-style) ----
ELO_K            = float(os.getenv("ELO_K", "20"))
ELO_HFA          = float(os.getenv("ELO_HFA", "48"))      # home-field advantage in Elo points
ELO_SEASON_REVERT = float(os.getenv("ELO_SEASON_REVERT", "0.33"))  # pull toward the mean each offseason

# ---- Rolling-form settings ----
ROLL_WINDOWS = (3, 8)
EWM_HALFLIFE = 5

SEED = int(os.getenv("SEED", "42"))
