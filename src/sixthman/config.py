"""Project paths, column mapping and model settings.

Edit COLUMN_MAP once the official play-by-play file has been inspected: keys are the
pipeline's standard column names, values are the column names in the raw file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Standard schema used by every downstream module.
# TODO: point the values at the real column names in the provided dataset.
COLUMN_MAP = {
    "game_id": "game_id",
    "period": "period",
    "clock_seconds": "clock_seconds",  # seconds remaining in the period
    "home_team": "home_team",
    "away_team": "away_team",
    "home_score": "home_score",  # score after the event
    "away_score": "away_score",
    "event_type": "event_type",
    "team": "team",
    "player": "player",
    # Players on court for each event (5 per side).
    **{f"home_p{i}": f"home_p{i}" for i in range(1, 6)},
    **{f"away_p{i}": f"away_p{i}" for i in range(1, 6)},
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

# Stints whose entry leverage is above this multiple of the league average count as "high leverage".
HIGH_LEVERAGE_THRESHOLD = 1.5
# Empirical-Bayes shrinkage: pseudo-stints of zero WPA added when ranking players in a situation.
SHRINKAGE_STINTS = 5
