# SwissYield analysis scripts

Never open the data in `/scratch2/oscipal/swissyield/` by hand (see repo `CLAUDE.md`); these scripts are the way in.
`./run_all.sh` (venv activated inside) writes everything to `<work_dir>/swissyield/` (`/scratch2/oscipal/yieldsat_work/swissyield/`):

- `profile.md` - full description of the dataset: file tree, NetCDF variables/bands/time axis/categoricals, raw metadata schema,
  grids, CRS, extent, areas, terrain, scenes, SCL shares, weather columns, yield stats, relations. **Read this first.**
- `tables/` (CSV per field-year etc.), `figures/` (histograms vs YieldSAT, profiles), `report/` (sections), `cache/`.

| script | content |
|---|---|
| `swiss_lib.py` | loaders for Swiss and YieldSAT (memmaps from scripts/LSTM/01_extract.py), samples, field aggregates, plotting |
| `00_file_tree.py` | folder structure, counts, sizes |
| `01_inventory.py` | NetCDF schema, bands, time axis, labels, field-year table |
| `04_raw_profile.py` | raw metadata, rasters, extent, scenes, weather |
| `02_target_hist.py` | yield histograms Swiss vs YieldSAT |
| `03_feature_hist.py` | input histograms (S2, NDVI, weather, soil, terrain) + shift table |
| `05_timeseries_relations.py` | seasonal profiles, NDVI-yield correlations, static vs yield |
| `99_assemble_report.py` | joins sections into profile.md |
