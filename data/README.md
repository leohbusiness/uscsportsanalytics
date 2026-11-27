# Data

Only the **official one-season NBA play-by-play data** provided by the competition organizers
(Google Drive: "Material for MSBA 1st Round Case Competition") may be used. No scraping or
blending with other sources.

- `data/raw/` – put the downloaded play-by-play file(s) here (CSV or Parquet). Git-ignored.
- `data/processed/` – outputs written by `python -m sixthman.pipeline`. Git-ignored.

Raw column names are mapped to the pipeline's standard schema in `src/sixthman/config.py`
(`COLUMN_MAP`). Update that mapping once the real file is inspected.

Until the real data is in place, `python -m sixthman.sample_data` writes a small synthetic
file to `data/raw/sample_pbp.csv` so the pipeline and dashboard can be developed end to end.
The synthetic file is for development only and must not appear in any reported result.
