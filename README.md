# The Sixth Man Advantage

**USC Sports Data Analytics Competition, MSBA Internal Selection Round (Fall 2026)**

An interactive tool that answers one question for a coach: **"Who do I bring in right now?"**
Given the game situation (score margin, time left, who the opponent has on the floor, how much
the moment matters), it ranks a team's bench players by the win probability they have added in
comparable situations over the season.

**Data:** the 2025-26 NBA play-by-play file from the competition Google Drive (one season, per
the rules). **Headline results:** [docs/findings.md](docs/findings.md).

## Repository layout

```
app/streamlit_app.py      Dashboard (Deliverable A)
app/data/                 Derived bench-stint table the hosted dashboard reads
src/sixthman/
  config.py               Paths, raw-to-standard column mapping, model settings
  load.py                 Read raw PBP, standardize columns, add game state (time left, margin, result)
  win_prob.py             Win-probability model fit on this season + leverage index
  stints.py               Starter/bench split, on-court stints, entry context, stint WPA
  metrics.py              Situation buckets and shrunk player rankings
  expected.py             Expected WPA from teammate/opponent strength -> WPA over expected
  pipeline.py             CLI: raw data -> data/processed/*.parquet + app/data/bench_stints.parquet
  report.py               Writes docs/findings.md (headline rankings for the deck)
  sample_data.py          Synthetic dev data generator (never used for reported results)
tests/                    Smoke tests and invariants
data/raw/                 Official PBP files go here (git-ignored)
data/processed/           Pipeline outputs (git-ignored)
docs/AI_DISCLOSURE.md     How AI tools were used
```

## How to run

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 1. Data: copy 2025-2026_combined_stats.csv.zip from the competition Google Drive into data/raw/
#    (zipped or unzipped both work). Column mapping lives in src/sixthman/config.py.

# 2. Build processed tables (~1 min): WP model, stints, expected WPA, dashboard table
python -m sixthman.pipeline
python -m sixthman.report        # optional: regenerate docs/findings.md

# 3. Launch the dashboard
streamlit run app/streamlit_app.py

# Tests
pytest -q
```

## Methodology

### Data boundary
Only the 2025-26 play-by-play file provided by the organizers is used (762k events, 1,322
games including In-Season Tournament, Play-In and Playoffs). No external data, scraping, or
third-party win-probability models; everything, including the win-probability model, is
estimated from this season. Raw data is git-ignored. The only data committed is
`app/data/bench_stints.parquet`, a derived table the pipeline produces so the hosted
dashboard can run without the raw file.

### 1. Game state
For every event: seconds left in regulation (OT handled per period), score margin before and
after the event, and the game's final result. The file lists all ten players on court after
every event (`h1-h5`, `a1-a5`), so no lineup reconstruction is needed.

### 2. Win-probability model (fit on this season)
Logistic regression of `home_win` on margin, margin / sqrt(minutes left + 1), and time left.
Validated with **game-grouped 5-fold cross-validation** (no game in both train and test):
Brier 0.162 vs 0.247 for a constant baseline; log loss 0.477 vs 0.687.

### 3. Leverage index
How much one point moves win probability in the current state, relative to the season
average: `LI = [WP(margin+1) - WP(margin-1)] / 2 / mean swing`. LI = 1 is an average moment; a
tie game in the final two minutes is several times higher.

### 4. Bench stints
Starters are the five players on court on each team's first event of the game; everyone else is
a bench player for that game. A **stint** is a run of consecutive events in one period with the
player on court (128k stints, 58k by bench players). For each stint: entry margin, time, leverage,
the opposing lineup, duration, plus-minus, and **win probability added (WPA)** = team WP when he
checks out minus team WP when he checks in. "True reserves" are players who started at most half
of the games they played; the dashboard shows only them by default.

### 5. Context adjustment: WPA over expected
Raw on-court WPA credits a reserve for playing next to stars against weak units. Each player gets
a season on-court net rating (plus-minus per 48, shrunk with 250 prior minutes). For every stint
we compute the gap between his four teammates' and the five opponents' average rating, and fit
`WPA ~ minutes x gap + minutes x gap x leverage` over all 128k stints. **WPA over expected** is
the residual. The same opponent rating defines the dashboard's "opponent lineup" filter
(terciles: weaker / average / stronger).

### 6. The metric: situational WPA over expected per stint, shrunk
For any situation a coach selects (score, time, opponent unit, leverage), a player's score is
`sum(WPA over expected) / (stints + k)`, equivalent to adding k zero-value stints, with
k = 10 everywhere (dashboard and `docs/findings.md`; set in `config.SHRINKAGE_STINTS`). This shrinks
small samples toward replacement level. The chart shows about one standard error as whiskers.

### Known limitations (to address or defend)
- On-court WPA is shared by all ten players. The teammate/opponent adjustment is a linear
  first-order correction, not a full ridge (RAPM-style) decomposition; a player's own minutes
  partly feed his teammates' ratings.
- Single stints are noisy (the context model's R^2 is about 0.04), which is why we shrink, show
  error bars and enforce minimum stints.
- Stints are cut at period boundaries. The WP model has no possession or team-strength term.

## Deploying the dashboard
Streamlit Community Cloud (free): sign in at share.streamlit.io with GitHub, click **Create
app**, pick this repo, branch `main`, main file `app/streamlit_app.py`. It installs
`requirements.txt` and reads `app/data/bench_stints.parquet`, so no raw data upload is needed.

## AI tool disclosure
See [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md).

## Team
**Athletic Intelligence:** Minzy Sung, Leo Hwang, Jace Kang, Vincent Huang.
