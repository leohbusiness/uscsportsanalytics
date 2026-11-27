"""End-to-end pipeline: raw play-by-play -> WP model -> stints -> processed outputs.

Usage:
    python -m sixthman.pipeline                # every CSV/Parquet in data/raw/
    python -m sixthman.pipeline --input FILE   # one specific file
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .load import load_events
from .expected import add_expected_wpa
from .stints import build_stints
from .win_prob import WinProbModel, annotate, cross_validate


def run(input_path: Path | None = None, out_dir: Path = config.PROCESSED_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(input_path)
    print(f"Loaded {len(events):,} events from {events['game_id'].nunique():,} games")

    cv = cross_validate(events)
    print(f"WP model CV Brier {cv['brier']:.4f} (constant baseline {cv['brier_baseline']:.4f})")

    wp = WinProbModel.fit(events)
    events = annotate(events, wp)
    stints = build_stints(events)
    stints, exp_info = add_expected_wpa(stints)
    cv["expected_wpa_model"] = exp_info
    print(f"Expected-WPA model R^2 {exp_info['r2']:.3f}")
    print(f"Built {len(stints):,} stints ({(~stints['is_starter']).sum():,} bench)")

    events.to_parquet(out_dir / "events_wp.parquet", index=False)
    export_dashboard_data(stints)
    stints.to_parquet(out_dir / "stints.parquet", index=False)
    (out_dir / "wp_model_eval.json").write_text(json.dumps(cv, indent=2))
    return cv


def export_dashboard_data(stints) -> None:
    """Small derived table the hosted dashboard reads (bench stints only, no raw play-by-play)."""
    config.APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    starts = stints.groupby(["player", "game_id"])["is_starter"].any().groupby("player").mean()
    stints = stints.assign(player_start_rate=stints["player"].map(starts))
    stints[~stints["is_starter"]].drop(columns=["is_starter"]).to_parquet(config.APP_DATA, index=False)
    print(f"Wrote dashboard data to {config.APP_DATA}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=config.PROCESSED_DIR)
    args = parser.parse_args()
    run(args.input, args.out)


if __name__ == "__main__":
    main()
