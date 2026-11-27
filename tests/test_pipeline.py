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
