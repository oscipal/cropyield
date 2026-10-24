# Rules

- NEVER open, read, head, or otherwise inspect the contents of the data files in `/scratch2/oscipal/swissyield/` (rasters, NetCDF, CSV, JSON metadata). Only `README.txt` and `MANIFEST.json` there may be read. The directory/file structure (names, sizes) may be listed. Learn about the data content through the analysis scripts in `scripts/swissyield/` and their written reports.
- Keep the visualizer (`viz/`) current with the results. It reads each run's `metrics.json` / `preds_*.parquet` live (refresh every 2 min), so do not export extra result JSONs for it; instead, whenever a new model / run config is added or results change:
  - register the config in `MODEL_FAMILIES` in `viz/data.py` (otherwise it lands under "Other" on the Runs / Predictions pages);
  - for a model meant for the summary, add it to `MODELS` and `DESCRIPTIONS` in `scripts/LSTM/10_summary.py` (the homepage results tables call its `load()` / `overview()` directly), and add new summary tables as functions there that both the report and `viz/app.py` use, never as a second copy of the logic in `viz/`;
  - rerun `python scripts/LSTM/10_summary.py` so `docs/results_summary.md` matches the homepage.
- When new results come in (a run or screening finishes), write them into the repo in the same turn, without being asked:
  - `docs/results_summary.md`: add the model to `scripts/LSTM/10_summary.py` (`MODELS`, `DESCRIPTIONS`, notes / "tried and not kept" text) and rerun it; never edit the .md by hand.
  - Full test tables of an experiment: `docs/results_<topic>.md` via `scripts/LSTM/03_report.py --configs ... --out docs/results_<topic>.md`, with a header saying what is new, on which split suite, and the caveats. `docs/results_lstm.md` is the frozen record of the old split suite: never overwrite it (`03_report.py` writes there by default, so always pass `--out`).
  - Val-only screenings (model choice): `docs/<topic>_screening.md`; test scores are never used for choices.
  - Update the status / results section of the plan in `.claude/plans/<topic>.md`, and list every new results doc and script in `README.md` (the `Results:` line, the `docs/` layout and the commands).
  - Report the numbers in the reply with their caveats (single held-out sets, few test fields per subset).
