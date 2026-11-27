"""Load raw play-by-play, map it to the standard schema and add game-state columns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def read_raw(path: Path | None = None) -> pd.DataFrame:
    """Read one CSV/Parquet file, or every such file in data/raw/."""
    paths = [Path(path)] if path else sorted(
        p for p in config.RAW_DIR.iterdir() if p.suffix in {".csv", ".parquet"}
    )
    if not paths:
        raise FileNotFoundError(
            f"No CSV/Parquet files in {config.RAW_DIR}. Download the play-by-play data "
            "there, or run `python -m sixthman.sample_data` for synthetic dev data."
        )
    frames = [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to the standard schema and validate required fields."""
    inverse = {raw_name: std for std, raw_name in config.COLUMN_MAP.items() if raw_name in raw.columns}
    df = raw.rename(columns=inverse)
    missing = [c for c in config.REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after mapping: {missing}. Update config.COLUMN_MAP.")
    return df


def period_length(period: pd.Series) -> pd.Series:
    return np.where(period <= config.REGULATION_PERIODS,
                    config.REGULATION_PERIOD_SECONDS, config.OVERTIME_PERIOD_SECONDS)


def add_game_state(df: pd.DataFrame) -> pd.DataFrame:
    """Add event order, seconds remaining in regulation/OT, score margin and final outcome."""
    df = df.copy()
    df["event_idx"] = np.arange(len(df))  # preserve source order within the file
    df = df.sort_values(["game_id", "period", "clock_seconds", "event_idx"],
                        ascending=[True, True, False, True]).reset_index(drop=True)
    df["seq"] = df.groupby("game_id").cumcount()

    later_reg_periods = (config.REGULATION_PERIODS - df["period"]).clip(lower=0)
    df["seconds_remaining"] = df["clock_seconds"] + later_reg_periods * config.REGULATION_PERIOD_SECONDS
    df["is_overtime"] = df["period"] > config.REGULATION_PERIODS

    df["margin_home"] = df["home_score"] - df["away_score"]
    final = df.groupby("game_id")["margin_home"].last()
    df["home_win"] = df["game_id"].map(final > 0).astype(int)
    return df


def load_events(path: Path | None = None) -> pd.DataFrame:
    return add_game_state(standardize(read_raw(path)))
