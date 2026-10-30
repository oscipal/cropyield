# Crop yield prediction on YieldSAT

Pixel-level crop yield prediction on the [YieldSAT](https://yieldsat.github.io/) dataset (Miranda et al., CVPR 2026;
paper in `docs/Yieldsat.pdf`): 2,173 fields in Argentina, Brazil, Uruguay and Germany (corn, rapeseed, soybean,
wheat, 2016–2024), 12.4 M labelled 10 m pixels, each with a 24-step Sentinel-2 time series plus weather, soil and
topography.

The repository contains

1. a **split suite**: 24 train/val/test splits that hold out whole years (`loyo`) or whole regions (`loro`), for
   Germany, South America and the world, per crop and for all crops pooled (`docs/splits.md`);
2. the **paper's LSTM**, replicated and trained once per split, with Sentinel-2 only and with the additional
   modalities (S2 + ADM);
3. **LSTM v2**, the paper LSTM with extra inputs, per-crop target scaling, field-balanced training,
   regularisation and a 5-member Deep Ensemble.

Results: `docs/results_summary.md` (all models, short), full tables in `docs/results_convlstm.md` (current split suite), `docs/results_inputs_quick.md` (quick input test with the new loader) and `docs/results_lstm.md` (old suite, before Switzerland); World variants (base + one change each, val and test) in `docs/world_variants_screening.md` and their test tables in `docs/results_world_variants.md`; also on the visualizer's homepage.

## Setup

```bash
source /scratch2/oscipal/cropyield/bin/activate   # Python 3.13 venv
pip install -r requirements.txt
```

All paths are set in `configs/paths.yaml` (data, work directory, split store). The YieldSAT data itself
(preprocessed NetCDF per country and raw files) is not in the repository. Derived files go to `work_dir`
(`/scratch2/oscipal/yieldsat_work`), never into the repository.

Training logs to Weights & Biases (project `yieldsat`, one group per config). Use `--no-wandb` or set
`wandb.mode: disabled` in the config to switch it off.

## Layout

```
configs/
  paths.yaml              data, work_dir and split store locations
  lstm_s2.yaml            paper LSTM, Sentinel-2 only
  lstm_s2_adm.yaml        paper LSTM, S2 + weather, soil, topography (input fusion)
  lstm_v2_s2.yaml         LSTM v2, Sentinel-2 only
  inputs_quick/           small pixel LSTM via src/loader.py: quick_s2 (S2 only), quick_all (+ weather, soil, terrain)
  loyo_conv/              LOYO screening of the ConvLSTM (time axis, LAI, ERA5-Land soil water; .claude/plans/loyo_conv.md)
  world_variants/         base model (convlstm_base, World LOYO) + one change each: edge, 1d, wxdaily, sm, smanom,
                          gdd, scenes (.claude/plans/world_variants.md)
  field_image_s2.yaml     field-image model (temporal encoder + U-Net over the whole field)
scripts/
  05_build_field_table.py one row per field-year (physical fields, regions)
  06_make_split_suite.py  builds the 24 splits
  07_split_report.py      writes docs/splits.md
  check_loader.py         src/loader.py against the old input pipeline and the source files
  check_lai.py            src/lai.py against the reference SNAP LAI script, plausibility, angle sensitivity
  loyo_diagnosis.py       read-only diagnosis of the LOYO val errors and the data behind them -> docs/loyo_diagnosis.md
  loyo_eval.py            val-only LOYO evaluation: peak-NDVI baseline, new-fields-only score, paired bootstrap CIs -> docs/loyo_eval.md
  features_doc.py         writes docs/features.md from src/features.py
  LSTM/
    01_extract.py         NetCDF -> memmaps for training (once)
    02_train.py           paper LSTM, one model per split
    03_report.py          writes docs/results_lstm.md
    04_train_v2.py        LSTM v2, one ensemble per split
    17_baseline_peak_ndvi.py  peak-NDVI baseline (loyo_eval.py's OLS per country x crop) as run peak_ndvi, 6 pooled splits, val + test
    lstm_lib.py           paper LSTM model, band groups, normalisation
    lstm_v2.py            v2 features, target scaling, model, field sampling
    07_train_field_image.py  field-image model, one ensemble per split (inputs and outputs as 04)
    field_image.py        field images from the pixel memmaps, crops, tiles, encoder + U-Net
    check_field_image.py  data alignment and shape checks (--gpu: memory of a training step)
    convlstm.py           ConvLSTM model (gates = convolutions over the pixel's neighbourhood; optional late GRU branch)
    check_convlstm.py     ConvLSTM shape / gradient checks
    11_build_s2_grid.py   Sentinel-2 cubes on 30 / 15 / 10-day windows from the raw scenes -> lstm/data_w<days>/
    12_scene_availability.py  per field x window: scenes, valid share, cause of gaps -> docs/scene_availability.md
    13_build_weather_grid.py  ERA5 weather per field on the S2 time grids (degC, mm, GDD, ...) -> lstm/<grid>/weather.npy
    14_build_terrain.py   elevation, slope, aspect, curvature recomputed from the raw DEM -> lstm/terrain.npy
    15_download_era5land.py  ERA5-Land soil water + weather per 0.1 deg cell, 1995-2024, from the CDS -> era5land/
    16_build_era5land_grid.py  ERA5-Land on the S2 time grids, with anomalies vs 1995-2015 -> lstm/<grid>/era5land.npy
                          (--grids daily: one value per day, days 210-719 -> lstm/daily/, for model.daily_features)
    19_build_s2_scenes.py all usable S2 scenes per field-year, no windows (>= 5 % clear) -> lstm/scenes/
    20_build_gdd_grid.py  S2 cube on a thermal-time axis (17 windows of equal GDD, fixed start per hemisphere x crop)
                          -> lstm/data_gdd/ (grid "gdd")
    21_patchiness.py      a run's errors vs the patchiness of the yield map (holes, edge share) -> docs/patchiness_<run>.md
    paper_lstm_results.yaml   the paper's LSTM results (appendix Tab. 13–18)
src/
  config.py               loads configs/paths.yaml
  loader.py               load(): pick features and time grid, get arrays; fit_scaler, YieldDataset (torch)
  features.py             registry of the loader's features (groups, units, sources, known issues)
  soil.py                 SoilGrids soil in physical units, depth selection
  lai.py                  leaf area index from S2 reflectances (SNAP / S2ToolBox biophysical network)
  scenes.py               scene-sequence inputs (every usable S2 scene a step; padded batches, augmentation)
  data.py                 opening the NetCDF files, decoded pixel metadata
  metrics.py              field- and pixel-level R² and RMSE
  splits.py               load_split()
  season.py               sowing / harvest dates of a field-year on the cube's day axis
  splitting.py            split construction and scoring (used by 06/07)
viz/                      browser visualizer (Streamlit): fields + inputs, runs, predictions, splits, results explorer,
                          configure run
                          (writes configs/custom/<name>.yaml for 04_train_v2.py; never starts training)
  run.sh                  starts it on 127.0.0.1:8501 (builds viz/build_index.py outputs on first start)
docs/
  results_summary.md      short comparison of all models (scripts/LSTM/10_summary.py)
  results_convlstm.md     full tables on the current split suite, ConvLSTM and 10-day S2 series
  results_lstm.md         full tables on the old split suite (before Switzerland), compared with the paper
  convlstm_screening.md   val-only screening of the ConvLSTM variants
  results_inputs_quick.md full tables of the quick input test (S2 only vs. + weather, soil, terrain)
  inputs_quick_screening.md  val-only screening of the quick input test
  loyo_conv_screening.md  val-only LOYO screening of the ConvLSTM (time axis, LAI, ERA5-Land soil water)
  loyo_diagnosis.md       where the LOYO error comes from (level / scale / ranking), year structure, S2 coverage, sowing dates, seen vs new fields, edge pixels
  loyo_literature.md      how the literature handles each problem of loyo_diagnosis.md, candidate fixes with references
  world_vs_regional_screening.md  ConvLSTM LOYO: one World model vs separate Europe / South America models (val, South America)
  world_inputs_screening.md       World LOYO: + weather, + soil/terrain, edge loss mask vs loyo_ref and the early ConvLSTMs (val)
  world_variants_screening.md     World LOYO: base ConvLSTM without augmentation + one change per variant (val and test)
  results_world_variants.md       test tables of the world variants (03_report.py)
  patchiness_wv_base.md           error vs patchiness of the yield map (holes, edges) for wv_base (scripts/LSTM/21_patchiness.py)
  loyo_eval.md            val scores vs a peak-NDVI baseline, on new fields only, with paired bootstrap CIs
  features.md             every loader feature: unit, source, known issues, coverage per country
  scene_availability.md   Sentinel-2 scene availability per window (scripts/LSTM/12_scene_availability.py)
  splits.md               dataset statistics and every split
  literature_review.md    best practices from the two reviews in docs/ and ~45 papers they cite (models, inputs,
                          validation) and what they suggest for this project
  Yieldsat.pdf            the YieldSAT paper
  agronomy-14-02264.pdf, PIIS2405844024168673.pdf   the two reviews (Wang et al. 2024; Jabed & Azmi Murad 2024)
```

## Visualizer

```
bash viz/run.sh                                  # on the server (setsid nohup ... & to keep it running)
ssh -L 8501:localhost:8501 pf-pc37.ethz.ch       # on your laptop, then open http://localhost:8501
```
It reads only derived files in `work_dir` (memmaps sliced per field) and binds to localhost only: the data holds
exact field locations, do not expose or host it.

## Pipeline

Run from the repository root with the venv active.

### 1. Split suite (already built)

```bash
python scripts/05_build_field_table.py     # -> work_dir/fields.parquet
python scripts/06_make_split_suite.py      # -> split_suite (zarr)
python scripts/07_split_report.py          # -> docs/splits.md
```

The unit of a split is the field-year; pixels of a field are never split. A region is a farm, merged with other
farms that share a field or have fields within 10 km, so in `loro` every physical field (the same land across
years) stays on one side and held-out regions are more than 10 km from any training field. `docs/splits.md`
explains how the splits are chosen. Load one with

```python
from src.splits import load_split
s = load_split("/scratch2/oscipal/yieldsat_splits.zarr", "south_america", "loro", "soybean")
s["train_fields"], s["val_fields"], s["test_fields"], s["attrs"]
```

### 2. Extract the data for training (once, ~20 min)

```bash
python scripts/LSTM/01_extract.py
```

Writes `work_dir/lstm/data` (~23 GB): the 16 bands that change over time (12 S2 + 4 weather) as
pixels × 24 × 16, the 104 static bands (soil, topography, coordinates) once per pixel, and pixel metadata. Bands are
matched by name, since their order differs between the country files.

### 2b. Inputs on the 30 / 15 / 10-day grids and the loader (once, already built)

```bash
python scripts/LSTM/11_build_s2_grid.py --window-days 10   # also 15, 30 -> work_dir/lstm/data_w<days>/
python scripts/LSTM/13_build_weather_grid.py               # weather on every grid
python scripts/LSTM/14_build_terrain.py                    # -> work_dir/lstm/terrain.npy (~30 s)
python scripts/check_loader.py                             # loader checks
python scripts/features_doc.py                             # -> docs/features.md
```

The delivered weather bands are sums between selected scenes (K, m) and the delivered slope / curvature / TWI are
inconsistent between countries, so these are rebuilt from the raw files; `docs/features.md` lists every feature
with unit, source and known issues. Load any combination on any grid with

```python
from src.loader import load, fit_scaler, YieldDataset
d = load("world/loyo/all", role="train", grid="10d",
         features=["s2", "indices", "observed", "weather:tmean_c,prec_mm", "terrain", "soil:0-30cm", "crop"])
d.dyn, d.static, d.y, d.meta, d.dyn_names, d.static_names, d.units, d.days
ds = YieldDataset(d, fit_scaler(d))   # torch Dataset, z-scored, NaN -> 0
```

Options: `align="sowing"|"harvest"` with `season=(first, last)` days (e.g. inputs up to two months before
harvest), `patch=k` for any odd k, pixel filters `edge_buffer_m`, `min_observed`, `countries` / `crops` / `years`,
`pixels_per_field`; table in `docs/features.md`. `04_train_v2.py` uses the loader for configs with
`inputs: {features: [...], grid: 10d}` (optional `align`, `season`, and `train_filters` for the training pixels
only; quote specs with commas in YAML); configs with `inputs.dynamic` keep the old input path
(`work_dir/lstm/data`, 24 × 30 days).

### 3. Train and evaluate

```bash
python scripts/LSTM/02_train.py --config configs/lstm_s2.yaml        # paper LSTM, S2       (~3 h)
python scripts/LSTM/02_train.py --config configs/lstm_s2_adm.yaml    # paper LSTM, S2 + ADM (~3 h)
python scripts/LSTM/04_train_v2.py --config configs/lstm_v2_s2.yaml  # LSTM v2              (~4 h)
python scripts/LSTM/07_train_field_image.py --config configs/field_image_s2.yaml   # field-image model (not run yet)
python scripts/LSTM/17_baseline_peak_ndvi.py                        # peak-NDVI baseline run, pooled splits (~10 min)
python scripts/LSTM/04_train_v2.py --config configs/world_variants/base.yaml   # world variants: base (+ edge, 1d,
                                     # wxdaily, sm, smanom, gdd, scenes), World LOYO, 5 members (~3 h each, scenes ~10 h)
python scripts/LSTM/03_report.py --out docs/results_<topic>.md       # full tables of the given --configs
python scripts/LSTM/10_summary.py                                    # -> docs/results_summary.md
```

`03_report.py` reports the configs given with `--configs` (default: paper LSTMs, v2, baselines); add
`configs/field_image_s2.yaml` there to include the field-image model.

Runtimes are for all 24 splits on one RTX 2080 Ti. Each script trains every split of the suite (or only those
given with `--splits germany/loyo/rapeseed ...`) and skips splits that already have results, so an interrupted run
continues where it stopped (`--overwrite` retrains). For long runs, start them detached so they survive a
disconnect, e.g.

```bash
setsid nohup bash -c "source /scratch2/oscipal/cropyield/bin/activate && \
  python scripts/LSTM/04_train_v2.py --config configs/lstm_v2_s2.yaml >> /scratch2/oscipal/yieldsat_work/logs/v2.log 2>&1" \
  < /dev/null > /dev/null 2>&1 &
```

Outputs per split in `work_dir/lstm/runs/<config>/<scope>__<method>__<crop>/`: test and val predictions per pixel
(`preds_*.parquet`), training history, `metrics.json` (overall and per country × crop) and model weights.

## Models

**Paper LSTM** (paper ref. [36], Pathak et al., IGARSS 2023): 2 stacked LSTM layers with 128 hidden units, last
hidden state, FC 128 → BatchNorm → ReLU → FC 1; MSE, Adam lr 1e-3, batch 1024, at most 50 epochs, early stopping
after 8 epochs without improvement on the split's val set. Every pixel is an independent sample. Not given in the
paper and chosen here: z-score per band with the training pixels' statistics, missing time steps set to 0 after
that; the ADM set is weather + soil (8 properties × 6 depths, no uncertainty layers) + 5 topography bands.

**LSTM v2** adds, each switchable in `configs/lstm_v2_s2.yaml`:

- NDVI, NDRE, NDMI per time step;
- a channel marking observed time steps;
- crop and country as one-hot inputs;
- the yield z-scored per crop × country (predictions converted back to t/ha);
- field-balanced training (at most 500 pixels per field per epoch) and early stopping on the field-averaged val error
  (`train.select_on: field_mse`, the default: pixel MSE in t/ha² averaged per field), or, with `select_on: field_z`
  (from 2026-10-30), on the field-mean error divided by the country × crop training SD, averaged per group and then
  over groups with ≥ 5 val fields (≈ 1 − field R² per group); both are logged per epoch;
- dropout 0.2 and weight decay 1e-4;
- a Deep Ensemble of 5 models with different seeds, averaged.

## Evaluation

Every model is evaluated on its split's test set, at pixel level and at field level (mean prediction vs. mean
measured yield per field-year), overall and per country × crop. The per country × crop values are the
meaningful ones: R² on pooled crops is inflated by the yield differences between crops.

The paper's held-out year and region results are averages over many folds of per country × crop models, ours come
from single held-out sets, so the comparison in `docs/results_lstm.md` is approximate. In short: the replicated
paper LSTM is below the paper (median field R² 0.04 vs. 0.39 over the same subsets), the additional modalities do
not help it on average, and LSTM v2 is the best variant (median 0.20, better than the baseline in 43 of 52 subsets,
lower RMSE than the paper on average).
