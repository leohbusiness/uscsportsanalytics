"""Project paths, column mapping and model settings.

Edit COLUMN_MAP once the official play-by-play file has been inspected: keys are the
pipeline's standard column names, values are the column names in the raw file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
# Derived bench-stint table committed for the hosted dashboard (no raw play-by-play).
APP_DATA = ROOT / "app" / "data" / "bench_stints.parquet"

# Season analyzed (team decision: one season only, per the competition rules).
SEASON_FILE = "2025-2026_combined_stats.csv"

# Standard schema used by every downstream module: {standard name: raw column in the
# official competition file}. `clock_seconds` and `phase` are derived in load.standardize.
COLUMN_MAP = {
    "game_id": "game_id",
    "date": "date",
    "data_set": "data_set",
    "period": "period",
    "remaining_time": "remaining_time",  # "H:MM:SS" left in the period
    "home_team": "home_team",
    "away_team": "away_team",
    "home_score": "home_score",  # score after the event
    "away_score": "away_score",
    "event_type": "event_type",
    "team": "team",
    "player": "player",  # "personId/Name" in the raw file
    "possession": "possession",
    "play_id": "play_id",
    # Players on court after each event (5 per side), by name.
    **{f"home_p{i}": f"h{i}" for i in range(1, 6)},
    **{f"away_p{i}": f"a{i}" for i in range(1, 6)},
}

REQUIRED = [
    "game_id", "period", "clock_seconds", "home_team", "away_team",
    "home_score", "away_score",
]
HOME_LINEUP = [f"home_p{i}" for i in range(1, 6)]
AWAY_LINEUP = [f"away_p{i}" for i in range(1, 6)]

REGULATION_PERIOD_SECONDS = 12 * 60
OVERTIME_PERIOD_SECONDS = 5 * 60
REGULATION_PERIODS = 4

# Season phases (from data_set) used for the model and the dashboard.
PHASES = {
    "NBA 2025-2026 Regular Season": "Regular season",
    "NBA 2025 In-Season Tournament": "Regular season",  # IST games count in the standings
    "NBA 2026 Play-In": "Play-in",
    "NBA 2026 Playoffs": "Playoffs",
}

# Stints whose entry leverage is above this multiple of the league average count as "high leverage".
HIGH_LEVERAGE_THRESHOLD = 1.5
# "True reserve": started at most this share of the games he played.
RESERVE_MAX_START_RATE = 0.5
# Empirical-Bayes shrinkage: pseudo-stints of zero WPA added when ranking players in a situation.
SHRINKAGE_STINTS = 10
