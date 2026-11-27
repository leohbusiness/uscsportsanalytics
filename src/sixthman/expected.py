"""Expected WPA for a stint given who else is on the floor, and WPA over expected.

Raw on-court WPA rewards a reserve for playing next to great teammates against weak
opponents. We fit, over all stints of the season (starters included), a linear model of
stint WPA on the strength gap between the player's four teammates and the five opponents,
scaled by stint length and game leverage. The residual, WPA over expected (WPAOE), is what
the player's stints produced beyond what that context predicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _design(st: pd.DataFrame) -> np.ndarray:
    minutes = st["duration_seconds"].to_numpy() / 60
    gap = (st["teammates_net48"] - st["opp_lineup_net48"]).to_numpy() / 48  # points per minute
    lev = st["entry_leverage"].to_numpy()
    # Points expected from the context gap, and the same weighted by leverage (WP per point).
    return np.column_stack([minutes * gap, minutes * gap * lev])


def add_expected_wpa(st: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    st = st.copy()
    X, y = _design(st), st["wpa"].to_numpy()
    model = LinearRegression().fit(X, y)
    st["expected_wpa"] = model.predict(X)
    st["wpa_over_expected"] = y - st["expected_wpa"]
    info = {"coef": model.coef_.tolist(), "intercept": float(model.intercept_),
            "r2": float(model.score(X, y)), "n_stints": int(len(y))}
    return st, info
