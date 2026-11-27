"""Smoke tests on a tiny synthetic season: the pipeline runs and core invariants hold."""
import numpy as np
import pandas as pd

from sixthman import load, sample_data
from sixthman.metrics import bench_stints, rank_players
from sixthman.stints import build_stints
from sixthman.win_prob import WinProbModel, annotate


def _events(n_games=20):
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_games):
        home, away = rng.choice(sample_data.TEAMS, size=2, replace=False)
        rows += sample_data.simulate_game(g + 1, str(home), str(away), rng)
    return load.add_game_state(load.standardize(pd.DataFrame(rows)))


def test_win_prob_is_monotonic_in_margin():
    wp = WinProbModel.fit(_events())
    p = wp.predict(np.array([-10, 0, 10]), np.array([300, 300, 300]))
    assert p[0] < p[1] < p[2]


def test_leverage_higher_late_in_close_games():
    wp = WinProbModel.fit(_events())
    assert wp.leverage([0], [60])[0] > wp.leverage([0], [2400])[0]


def test_team_stint_wpa_sums_to_game_swing():
    events = _events()
    events = annotate(events, WinProbModel.fit(events))
    stints = build_stints(events)
    # Five players are always on court, so a team's stint WPA sums to 5x its whole-game WPA.
    per_team = stints.groupby(["game_id", "side"])["wpa"].sum()
    assert np.allclose(per_team.groupby("game_id").sum(), 0, atol=1e-9)


def test_rank_players_returns_bench_only():
    events = _events()
    events = annotate(events, WinProbModel.fit(events))
    ranked = rank_players(bench_stints(build_stints(events)))
    assert not ranked["player"].str.endswith(tuple(f" Player {i}" for i in range(1, 6))).any()
    assert ranked["shrunk_wpa_per_stint"].is_monotonic_decreasing


def test_standardize_official_format():
    raw = pd.DataFrame({
        "game_id": ["0022500001"] * 2, "data_set": ["NBA 2026 Playoffs"] * 2, "period": [1, 1],
        "remaining_time": ["0:12:00", "0:11:36"], "home_team": ["OKC"] * 2, "away_team": ["HOU"] * 2,
        "home_score": [0, 2], "away_score": [0, 0], "player": [None, "1631096/Chet Holmgren"],
        **{f"h{i}": [f"H{i}"] * 2 for i in range(1, 6)}, **{f"a{i}": [f"A{i}"] * 2 for i in range(1, 6)},
    })
    df = load.standardize(raw)
    assert df["clock_seconds"].tolist() == [720, 696]
    assert df["player"].iloc[1] == "Chet Holmgren"
    assert df["phase"].iloc[0] == "Playoffs"
    assert df["home_p1"].iloc[0] == "H1" and df["away_p5"].iloc[0] == "A5"
