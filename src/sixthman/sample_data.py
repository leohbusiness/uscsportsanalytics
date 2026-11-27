"""Write a small SYNTHETIC play-by-play file in the standard schema for development.

Not real NBA data: use only to exercise the pipeline and dashboard before the official
dataset is mapped. Never report results from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def roster(team: str) -> list[str]:
    return [f"{team} Player {i}" for i in range(1, 11)]  # 1-5 starters, 6-10 bench


def simulate_game(game_id: int, home: str, away: str, rng: np.random.Generator) -> list[dict]:
    lineups = {home: roster(home)[:5], away: roster(away)[:5]}
    benches = {home: roster(home)[5:], away: roster(away)[5:]}
    skill = {p: rng.normal(0, 0.01) for t in (home, away) for p in roster(t)}
    score = {home: 0, away: 0}
    rows = []
    for period in range(1, 5):
        clock = 720
        while clock > 0:
            clock = max(0, clock - int(rng.integers(8, 25)))
            if rng.random() < 0.08:  # substitution
                team = home if rng.random() < 0.5 else away
                i, j = rng.integers(0, 5), rng.integers(0, 5)
                lineups[team][i], benches[team][j] = benches[team][j], lineups[team][i]
                event, actor = "substitution", lineups[team][i]
            else:
                edge = sum(skill[p] for p in lineups[home]) - sum(skill[p] for p in lineups[away])
                team = home if rng.random() < 0.5 + edge else away
                made = rng.random() < 0.5
                pts = int(rng.choice([2, 3], p=[0.65, 0.35])) if made else 0
                score[team] += pts
                event, actor = ("made_shot" if made else "missed_shot"), str(rng.choice(lineups[team]))
            row = {"game_id": game_id, "period": period, "clock_seconds": clock,
                   "home_team": home, "away_team": away,
                   "home_score": score[home], "away_score": score[away],
                   "event_type": event, "team": team, "player": actor}
            row.update({f"home_p{i+1}": p for i, p in enumerate(lineups[home])})
            row.update({f"away_p{i+1}": p for i, p in enumerate(lineups[away])})
            rows.append(row)
    if score[home] == score[away]:  # break ties so every game has a winner
        rows[-1]["home_score"] += 1
    return rows


def main(n_games: int = 120, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        rows += simulate_game(g + 1, str(home), str(away), rng)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RAW_DIR / "sample_pbp.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows):,} synthetic events to {out}")


if __name__ == "__main__":
    main()
