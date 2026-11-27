"""Win-probability model fit on the provided season only (no external WP data).

Baseline: logistic regression on score margin scaled by time remaining, the standard
"margin / sqrt(time)" form. Validated with game-grouped cross-validation so no game's
events appear in both train and test folds.

Leverage index (Tango-style): how much one point swings win probability in the current
state, relative to the season-average swing. LI = 1 is an average moment; LI = 3 means a
point is worth three times as much as usual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold


def features(margin_home: np.ndarray, seconds_remaining: np.ndarray) -> np.ndarray:
    margin = np.asarray(margin_home, dtype=float)
    minutes = np.asarray(seconds_remaining, dtype=float) / 60.0
    scaled = margin / np.sqrt(minutes + 1.0)
    return np.column_stack([margin, scaled, minutes / 48.0])


@dataclass
class WinProbModel:
    model: LogisticRegression
    mean_point_swing: float = 1.0

    @classmethod
    def fit(cls, events: pd.DataFrame) -> "WinProbModel":
        reg = events[~events["is_overtime"]]
        X = features(reg["margin_home"], reg["seconds_remaining"])
        model = LogisticRegression(C=1.0, max_iter=1000).fit(X, reg["home_win"])
        wp = cls(model)
        wp.mean_point_swing = float(np.mean(wp.point_swing(reg["margin_home"], reg["seconds_remaining"])))
        return wp

    def predict(self, margin_home, seconds_remaining) -> np.ndarray:
        return self.model.predict_proba(features(margin_home, seconds_remaining))[:, 1]

    def point_swing(self, margin_home, seconds_remaining) -> np.ndarray:
        m = np.asarray(margin_home, dtype=float)
        return (self.predict(m + 1, seconds_remaining) - self.predict(m - 1, seconds_remaining)) / 2

    def leverage(self, margin_home, seconds_remaining) -> np.ndarray:
        return self.point_swing(margin_home, seconds_remaining) / self.mean_point_swing


def cross_validate(events: pd.DataFrame, n_splits: int = 5) -> dict:
    """Game-grouped CV of the WP model. Returns Brier score and log loss vs. a constant baseline."""
    reg = events[~events["is_overtime"]].reset_index(drop=True)
    n_splits = min(n_splits, reg["game_id"].nunique())
    preds = np.zeros(len(reg))
    for train, test in GroupKFold(n_splits=n_splits).split(reg, groups=reg["game_id"]):
        preds[test] = WinProbModel.fit(reg.iloc[train]).predict(
            reg.loc[test, "margin_home"], reg.loc[test, "seconds_remaining"])
    y = reg["home_win"].to_numpy()
    base = np.full(len(y), y.mean())
    return {
        "brier": float(brier_score_loss(y, preds)),
        "brier_baseline": float(brier_score_loss(y, base)),
        "log_loss": float(log_loss(y, preds)),
        "log_loss_baseline": float(log_loss(y, base)),
        "n_events": int(len(y)),
        "n_games": int(reg["game_id"].nunique()),
    }


def annotate(events: pd.DataFrame, wp: WinProbModel) -> pd.DataFrame:
    """Add home win probability after each event, before it, and leverage at that moment."""
    df = events.copy()
    secs = df["seconds_remaining"].where(~df["is_overtime"], df["clock_seconds"])
    df["wp_home"] = wp.predict(df["margin_home"], secs)
    # Final event of each game: WP collapses to the actual result.
    last = df.groupby("game_id")["seq"].transform("max") == df["seq"]
    df.loc[last, "wp_home"] = df.loc[last, "home_win"].astype(float)

    pregame = float(wp.predict([0], [48 * 60])[0])
    df["wp_home_before"] = df.groupby("game_id")["wp_home"].shift(1).fillna(pregame)
    df["margin_home_before"] = df.groupby("game_id")["margin_home"].shift(1).fillna(0)
    same_period = df.groupby(["game_id", "period"])["clock_seconds"]
    period_len = np.where(df["period"] <= 4, 720, 300)
    df["clock_before"] = same_period.shift(1).fillna(pd.Series(period_len, index=df.index))
    df["leverage"] = wp.leverage(df["margin_home_before"], secs)
    return df
