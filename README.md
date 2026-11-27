# The Sixth Man Advantage

**USC Sports Data Analytics Competition, MSBA Internal Selection Round (Fall 2026)**

An interactive tool that answers one question for a coach: **"Who do I bring in right now?"**
Given the game situation (score margin, time left, who the opponent has on the floor, how much
the moment matters), it ranks a team's bench players by the win probability they have added in
comparable situations over the season.

> Status: project scaffold. The pipeline and dashboard run end to end on synthetic data; the
> column mapping still needs to be pointed at the official play-by-play file.

## Repository layout

```
app/streamlit_app.py      Dashboard (Deliverable A)
src/sixthman/
  config.py               Paths, raw-to-standard column mapping, model settings
  load.py                 Read raw PBP, standardize columns, add game state (time left, margin, result)
  win_prob.py             Win-probability model fit on this season + leverage index
  stints.py               Starter/bench split, on-court stints, entry context, stint WPA
  metrics.py              Situation buckets and shrunk player rankings
  pipeline.py             CLI: raw data -> data/processed/*.parquet
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

# 1. Data: put the official play-by-play file(s) from the competition Google Drive in data/raw/
#    and map their column names in src/sixthman/config.py (COLUMN_MAP).
#    For development only, generate synthetic data instead:
python -m sixthman.sample_data

# 2. Build processed tables (WP model, stints)
python -m sixthman.pipeline

# 3. Launch the dashboard
streamlit run app/streamlit_app.py

# Tests
pytest -q
```

## Methodology

### Data boundary
Only the one season of play-by-play data provided by the organizers is used. No external data,
scraping, or third-party win-probability models. Everything, including the win-probability
model, is estimated from this season.

### 1. Game state
For every event we compute seconds remaining in regulation, the home score margin, and the
game's final result.

### 2. Win-probability model (fit on this season)
Logistic regression of `home_win` on margin, margin / sqrt(minutes left + 1), and time left.
It is validated with **game-grouped 5-fold cross-validation** (no game appears in both train and
test), and Brier score and log loss are reported against a constant baseline in
`data/processed/wp_model_eval.json`. Planned upgrades: possession indicator, team-strength term
estimated in-season, calibration plot.

### 3. Leverage index
How much a single point moves win probability in the current state, divided by the season
average: `LI = [WP(margin+1) - WP(margin-1)] / 2 / mean swing`. LI = 1 is an average moment;
a tie game in the last two minutes is several times higher.

### 4. Bench stints
Starters are the five players on court on each team's first event of the game; everyone else is
a bench player for that game. A **stint** is a run of consecutive events in one period with the
player on court. For each stint we record entry margin, time left, entry leverage, the opposing
lineup and how many opposing starters are on the floor, duration, plus-minus, and
**win probability added (WPA)** = team WP at exit minus team WP at entry.

### 5. The metric: situational WPA per stint, shrunk
WPA already weights every point by how much it mattered, which is why it is preferred over raw
plus-minus. For any situation a coach selects, a player's score is

`sum(stint WPA) / (stints + k)`

which is equivalent to adding k zero-WPA stints (k = 5 by default), shrinking small samples
toward replacement level so a player with two lucky stints doesn't top the list.

### Known limitations (to address or defend)
- On-court WPA is shared by all ten players; it is not yet adjusted for teammates or opponent
  quality beyond the "opposing starters on floor" proxy. A ridge-regularized (RAPM-style)
  adjustment is the main candidate upgrade.
- Stints are cut at period boundaries.
- The WP model has no possession or team-strength term yet.

## AI tool disclosure
See [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md).

## Team
_Add team member names here._
