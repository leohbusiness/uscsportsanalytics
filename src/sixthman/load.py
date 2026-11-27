"""Load raw play-by-play, map it to the standard schema and add game-state columns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _default_paths() -> list[Path]:
    """The configured season file (CSV or zipped CSV) if present, else any data file in data/raw/."""
    for name in (config.SEASON_FILE, config.SEASON_FILE + ".zip"):
        if (config.RAW_DIR / name).exists():
            return [config.RAW_DIR / name]
    return sorted(p for p in config.RAW_DIR.iterdir()
                  if p.suffix in {".csv", ".parquet"} or p.name.endswith(".csv.zip"))


def read_raw(path: Path | None = None) -> pd.DataFrame:
    paths = [Path(path)] if path else _default_paths()
    if not paths:
        raise FileNotFoundError(
            f"No play-by-play file in {config.RAW_DIR}. Put {config.SEASON_FILE}(.zip) there, "
            "or run `python -m sixthman.sample_data` for synthetic dev data."
        )
    frames = [pd.read_parquet(p) if p.suffix == ".parquet"
              else pd.read_csv(p, dtype={"game_id": str}, low_memory=False) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _clock_to_seconds(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.split(":", expand=True).astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to the standard schema, derive clock/phase, validate."""
    inverse = {raw_name: std for std, raw_name in config.COLUMN_MAP.items()
               if raw_name in raw.columns and std not in raw.columns}
    df = raw.rename(columns=inverse)
    if "clock_seconds" not in df.columns and "remaining_time" in df.columns:
        df["clock_seconds"] = _clock_to_seconds(df["remaining_time"])
    if "player" in df.columns:  # "1630578/Alperen Sengun" -> "Alperen Sengun"
        df["player"] = df["player"].astype("string").str.split("/", n=1).str[-1]
    df["phase"] = df["data_set"].map(config.PHASES).fillna("Regular season") \
        if "data_set" in df.columns else "Regular season"
    missing = [c for c in config.REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after mapping: {missing}. Update config.COLUMN_MAP.")
    return df


def add_game_state(df: pd.DataFrame) -> pd.DataFrame:
    """Add event order, seconds remaining in regulation, score margin and final outcome."""
    df = df.copy()
    df["event_idx"] = np.arange(len(df))  # source order breaks ties at the same clock
    df = df.sort_values(["game_id", "period", "clock_seconds", "event_idx"],
                        ascending=[True, True, False, True]).reset_index(drop=True)
    # Scores can be blank on non-scoring rows in some feeds: carry them forward.
    df[["home_score", "away_score"]] = df.groupby("game_id")[["home_score", "away_score"]].ffill().fillna(0)

    later_reg_periods = (config.REGULATION_PERIODS - df["period"]).clip(lower=0)
    df["seconds_remaining"] = df["clock_seconds"] + later_reg_periods * config.REGULATION_PERIOD_SECONDS
    df["is_overtime"] = df["period"] > config.REGULATION_PERIODS
    df["margin_home"] = df["home_score"] - df["away_score"]

    final = df.groupby("game_id")["margin_home"].last()
    decided = final[final != 0].index  # drop the rare game whose feed ends tied (incomplete)
    df = df[df["game_id"].isin(decided)].reset_index(drop=True)
    df["home_win"] = df["game_id"].map(final > 0).astype(int)
    df["seq"] = df.groupby("game_id").cumcount()
    return df


def load_events(path: Path | None = None) -> pd.DataFrame:
    return add_game_state(standardize(read_raw(path)))
