# cropyield

Pixel-level crop yield prediction on the YieldSAT dataset (Germany). The current stage inspects the preprocessed data and trains a first model that is directly comparable to a value reported in the YieldSAT paper: an LSTM on Sentinel-2 only, for rapeseed (GER-R), under the paper's CV10 protocol.

Data is not part of this repository. All data paths are configured in `configs/paths.yaml`.

## Repository layout

```
configs/
  paths.yaml          data locations and work directory (the only place paths are set)
  ger_r_lstm.yaml     experiment config: bands, seed, splits, model, training, W&B
src/
  config.py           loads paths.yaml
  data.py             lazy NetCDF access, categorical decoding, block-wise feature reader
  inspect_data.py     helpers for the data inspection notebook
  splits.py           save/load CV splits as zarr
  preprocess.py       per-fold band normalisation
  models.py           LSTM regressor
  metrics.py          field- and pixel-level R² / RMSE, per fold and pooled
notebooks/
  00_inspect.ipynb    data inspection, writes docs/data_overview.md
scripts/
  01_extract_ger_r.py extract rapeseed pixels to X.npy + meta.parquet
  02_make_splits.py   CV10 splits -> splits/ger_r_cv10.zarr
  03_train.py         train LSTM or baseline B0 per fold
  04_report.py        write docs/results_step1.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then edit `configs/paths.yaml`:

| key | meaning |
|---|---|
| `data.preprocessed` | YieldSAT preprocessed (xarray/NetCDF) directory |
| `data.raw` | YieldSAT raw / flexible-format directory |
| `data.germany_nc` | German merged NetCDF file |
| `work_dir` | where derived files (`.npy`, parquet, predictions) are written; must be outside the data folders |

Training logs to Weights & Biases. Run `wandb login` once, set `wandb.mode` in `configs/ger_r_lstm.yaml` to `offline` or `disabled`, or pass `--no-wandb`.

## Usage

### 1. Inspect the data

```bash
jupyter lab notebooks/00_inspect.ipynb
```

or headless:

```bash
jupyter nbconvert --to notebook --execute notebooks/00_inspect.ipynb \
  --output 00_inspect_executed.ipynb --ExecutePreprocessor.timeout=-1
```

The notebook lists both data directories, checks archive extraction, opens the German NetCDF lazily, decodes the categorical variables, and reports field/pixel counts, target distributions, S2 NaN fractions, timestep dates, seeding-date types and suspicious soil/temperature values. It writes `docs/data_overview.md` and figures to `docs/figures/`.

### 2. Train and evaluate (GER-R, LSTM, CV10)

```bash
python scripts/01_extract_ger_r.py          # --crop-label <label> if the crop name is not matched automatically
python scripts/02_make_splits.py
python scripts/03_train.py --model b0       # constant mean baseline
python scripts/03_train.py --model lstm     # --folds 0 1 to run a subset, --device cpu|cuda
python scripts/04_report.py
```

All scripts accept `--config` (default `configs/ger_r_lstm.yaml`).

Outputs:

- `<work_dir>/ger_r/X.npy` (pixels × 24 timesteps × 12 bands), `meta.parquet`, `info.json`
- `splits/ger_r_cv10.zarr`
- `<work_dir>/ger_r/preds/<experiment>/fold<k>.parquet` with test predictions per fold
- `docs/results_step1.md` and `docs/figures/step1_field_scatter.png`

## Protocol

- **Splits:** `StratifiedGroupKFold(n_splits=10)`, grouped by field (`field_shared_name`), stratified by farm, fixed seed. Per fold, 10 % of the training fields form an inner validation set (grouped by field) used only for early stopping; the test fold is never used for model selection.
- **Split file:** a zarr store with one entry per field: `farm`, `test_fold` (fold in which the field is tested) and `inner_val` (fold × field, boolean).
- **Preprocessing:** per-band mean/std from the training pixels of each fold, then NaN → 0. This differs from the official tutorial, which fills raw values with −1 without normalisation.
- **Model:** 1-layer LSTM, hidden size 64, last hidden state, linear head; MSE, Adam (lr 1e-3), batch 1024, at most 50 epochs, early stopping with patience 5.
- **Baseline B0:** mean of the training pixels as a constant prediction.
- **Metrics:** R² and RMSE at pixel level and at field level (mean prediction vs. mean target per field-year), reported as mean ± std over folds (as in the paper) and pooled over all test predictions.

## Notes

- The German NetCDF is about 7 GB float32. It is opened without loading, and features are read in contiguous pixel blocks, so memory use stays bounded.
- `.gitignore` excludes data and derived arrays (`*.nc`, `*.npy`, `*.parquet`, …). Do not commit data.
