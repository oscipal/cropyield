"""Configure a new training run: start from an existing model config, choose the inputs (src/loader.py), the training
pixels, the model size and the training settings, check them on a few pixels, and save a config for
scripts/LSTM/04_train_v2.py. The page only writes the config file; it never starts training."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

import data
from src import features as F
from src import soil
from src.config import REPO_ROOT

st.set_page_config(page_title="Configure run", page_icon="🛠️", layout="wide")
st.title("Configure a new run")
st.caption("Start from an existing model, change what you want, check it, save. Saved configs go to "
           "`configs/custom/` and are trained with `scripts/LSTM/04_train_v2.py` (command shown after saving). "
           "Feature details: `docs/features.md`.")

OUT_DIR = REPO_ROOT / "configs" / "custom"
WINDOW = {"30d": 30, "15d": 15, "10d": 10}
LEGACY_GRID = {"data": "30d", "data_w30": "30d", "data_w15": "15d", "data_w10": "10d"}
SEASON_RANGE = {"calendar": (210, 720), "sowing": (-150, 510), "harvest": (-510, 60)}
SEASON_DEFAULT = {"calendar": (210, 720), "sowing": (0, 360), "harvest": (-360, 0)}       # = src/loader.py
DEPTH_STOPS = [0, 5, 15, 30, 60, 100, 200]
MODEL_PARAMS = {  # type -> [(key, label, default, kind)]
    "lstm": [("hidden_size", "LSTM hidden size", 128, "int"), ("num_layers", "LSTM layers", 2, "int"),
             ("head_size", "Head size", 128, "int"), ("dropout", "Dropout", 0.2, "float")],
    "patch": [("conv_channels", "CNN channels per step", 32, "int"), ("hidden_size", "LSTM hidden size", 128, "int"),
              ("num_layers", "LSTM layers", 2, "int"), ("head_size", "Head size", 128, "int"),
              ("dropout", "Dropout", 0.2, "float")],
    "convlstm": [("hidden_channels", "ConvLSTM channels", 32, "int"), ("num_layers", "ConvLSTM layers", 1, "int"),
                 ("kernel_size", "Conv kernel (odd)", 3, "odd"), ("head_size", "Head size", 128, "int"),
                 ("dropout", "Dropout", 0.2, "float")],
}
MODEL_LABEL = {"lstm": "Pixel LSTM (LSTM v2)", "patch": "Patch LSTM (CNN per step + LSTM)",
               "convlstm": "ConvLSTM (convolutional gates)"}


# ---------------------------------------------------------------- base configs

def model_type(cfg: dict) -> str | None:
    """lstm / patch / convlstm for configs scripts/LSTM/04_train_v2.py trains, else None."""
    m, inputs = cfg.get("model") or {}, cfg.get("inputs") or {}
    if "ensemble" not in cfg or not ("features" in inputs or "dynamic" in inputs):
        return None
    t = m.get("type", "lstm")
    if t in ("patch", "convlstm"):
        return t
    return "lstm" if t == "lstm" and set(m) <= {"type", "hidden_size", "num_layers", "head_size", "dropout"} else None


@st.cache_data(ttl=60)
def base_configs() -> dict[str, str]:
    """name -> path of every config 04_train_v2.py can train (main configs first, then screens, then custom)."""
    out = {}
    files = sorted((REPO_ROOT / "configs").glob("*.yaml")) + sorted((REPO_ROOT / "configs").glob("*/*.yaml"))
    for p in files:
        try:
            cfg = yaml.safe_load(p.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(cfg, dict) and "name" in cfg and model_type(cfg):
            out[cfg["name"]] = str(p)
    return out


def loader_inputs(cfg: dict) -> tuple[dict, list[str]]:
    """The base config's inputs in loader form (features, grid, align, season, train_filters) + notes on what
    changes for old-style configs."""
    inputs = cfg.get("inputs") or {}
    if "features" in inputs:
        return {k: copy.deepcopy(inputs.get(k)) for k in ("features", "grid", "align", "season", "train_filters")}, []
    notes, feats = [], []
    dyn = inputs.get("dynamic") or []
    if "s2" in dyn:
        feats.append("s2")
    if any(b in dyn for b in ("weather", "temp_mean", "total_prec")):
        feats.append("weather")
        notes.append("weather: the old bands (sums between scenes, K and m) become the fixed-window weather "
                     "(°C, mm, GDD, ...).")
    if inputs.get("indices"):
        feats.append("indices:" + ",".join(inputs["indices"]) if len(inputs["indices"]) < 3 else "indices")
    if inputs.get("mask_channel"):
        feats.append("observed")
    static = inputs.get("static") or []
    if "dem" in static:
        feats.append("terrain")
        notes.append("terrain: the delivered slope / aspect / curvature / TWI become the rebuilt terrain features.")
    if "soil" in static:
        feats.append("soil")
        notes.append("soil: the 48 raw SoilGrids layers become 8 properties at 0–30 cm in physical units.")
    if "coords" in static:
        feats.append("coords")
    feats += [c for c in inputs.get("categorical") or [] if c in F.CATEGORICAL]
    old = cfg.get("data_dir", "data")
    grid = LEGACY_GRID.get(old, "30d")
    if old == "data":
        notes.append("time grid: the delivered 24-step cube is not offered by the loader; its 30-day rebuild "
                     "(`30d`, steps 7–23 of the cube, the earlier steps are empty) is used instead.")
    return {"features": feats, "grid": grid, "align": None, "season": None, "train_filters": None}, notes


def parse_feats(specs: list[str]) -> dict[str, str | None]:
    """group -> argument (None = whole group) of a feature list."""
    out = {}
    for s in specs:
        g, _, arg = s.partition(":")
        out[g] = arg or None
    return out


def pick(label: str, options: list[str], default: list[str], key: str,
         docs: dict[str, tuple[str, str]] | None = None) -> list[str]:
    """Multi-choice where every option stays visible (pills, selected = highlighted), in the options' order;
    with docs a small table of meaning and unit below."""
    sel = st.pills(f"{label} — click to switch on / off", options, selection_mode="multi",
                   default=[o for o in default if o in options], key=key) or []
    sel = [o for o in options if o in sel]
    st.caption(f"{len(sel)} of {len(options)} selected")
    if docs:
        with st.expander("What these are"):
            st.markdown("\n".join(["| feature | meaning | unit |", "|---|---|---|"]
                                  + [f"| `{o}` | {docs[o][0]} | {docs[o][1]} |" for o in options]))
    return sel


configs = base_configs()
if not configs:
    st.error("No trainable config found in configs/.")
    st.stop()
MAIN = {"lstm_v2_s2": "LSTM", "convlstm_s2": "ConvLSTM", "lstm_patch_v3_s2": "Patch LSTM"}  # one per model type
custom = [n for n, p in configs.items() if "/custom/" in p]
c1, c2 = st.columns([3, 1])
show_all = c2.toggle("Show older and screening variants", False, key="cfg_show_all",
                     help="Variants of the three models that only differ in settings this page can change.")
main = [n for n in MAIN if n in configs]
order = main + custom + ([n for n in configs if n not in main and n not in custom] if show_all else [])
default = "lstm_patch_v3_s2" if "lstm_patch_v3_s2" in order else order[0]
base_name = c1.selectbox("Start from model", order, index=order.index(default),
                         format_func=lambda n: MAIN.get(n) or (f"{n}  ·  {data.family_of(n)}"
                                                               + ("  · screen" if "_screen" in configs[n] else "")
                                                               + ("  · custom" if "/custom/" in configs[n] else "")))
base = yaml.safe_load(Path(configs[base_name]).read_text())
mtype = model_type(base)
info = data.config_info(base_name)
st.markdown(f"**{MODEL_LABEL[mtype]}** — `{Path(configs[base_name]).relative_to(REPO_ROOT)}`. "
            f"{info['desc'] or info['header'] or ''}")
inp0, notes = loader_inputs(base)
if notes:
    st.info("This model uses the old input format; it is converted to the loader:\n\n- " + "\n- ".join(notes))
K = base_name  # widget keys per base model: switching the base resets the form to its values
p0 = parse_feats(inp0["features"])
summary_box = st.container(border=True)   # filled at the end of the page, once every choice is known

tab_in, tab_px, tab_model, tab_train, tab_save = st.tabs(
    ["1 · Inputs", "2 · Training pixels", "3 · Model", "4 · Training & splits", "5 · Check & save"])

# ---------------------------------------------------------------- 1 inputs
with tab_in:
    c1, c2, c3 = st.columns([1, 1, 2])
    grid = c1.radio("Time grid", list(WINDOW), index=list(WINDOW).index(inp0["grid"] or "10d"), horizontal=True,
                    key=f"{K}_grid", help="Window length; best clear Sentinel-2 scene per window.")
    wd = WINDOW[grid]
    align = c2.radio("Time origin", ["calendar", "sowing", "harvest"],
                     index=["calendar", "sowing", "harvest"].index(inp0["align"] or "calendar"), horizontal=True,
                     key=f"{K}_align", help="calendar: day 0 = 1 Jan of the year before harvest; sowing / harvest: "
                                            "step 0 = the window with that date (exact to one window).")
    lo, hi = SEASON_RANGE[align]
    s0 = tuple(inp0["season"]) if inp0["season"] and (inp0["align"] or "calendar") == align else SEASON_DEFAULT[align]
    season = c3.slider("Season window (days from the origin)", lo, hi, s0, step=wd, key=f"{K}_season_{align}",
                       help="Steps whose start day lies in [first, last). E.g. origin harvest, (-360, -60): inputs "
                            "end two months before harvest.")
    n_steps = len(range(-(-season[0] // wd), -(-season[1] // wd)))
    c3.caption(f"{n_steps} steps of {wd} days")

    st.subheader("Per time step")
    feats: list[str] = []
    a, b = st.columns(2)
    with a:
        if st.checkbox("Sentinel-2 bands", "s2" in p0, key=f"{K}_s2"):
            sel = pick("Bands", F.S2_BANDS, (p0.get("s2") or ",".join(F.S2_BANDS)).split(","), f"{K}_s2_p")
            feats += [("s2" if len(sel) == len(F.S2_BANDS) else "s2:" + ",".join(sel))] if sel else []
        if st.checkbox("Spectral indices", "indices" in p0, key=f"{K}_ind"):
            sel = pick("Indices", list(F.INDICES), (p0.get("indices") or ",".join(F.INDICES)).split(","),
                       f"{K}_ind_p", {i: (F.INDEX_DOCS[i], "-1..1") for i in F.INDICES})
            feats += [("indices" if len(sel) == len(F.INDICES) else "indices:" + ",".join(sel))] if sel else []
        if st.checkbox("Observed flag", "observed" in p0, key=f"{K}_obs",
                       help="1 where the pixel has a clear observation; recommended whenever S2 is used."):
            feats.append("observed")
    with b:
        if st.checkbox("Weather (per field)", "weather" in p0, key=f"{K}_w"):
            arg = p0.get("weather")
            dflt = list(F.WEATHER_UNITS) if arg == "all" else (arg.split(",") if arg else F.WEATHER_DEFAULT)
            sel = pick("Weather features", list(F.WEATHER_UNITS), dflt, f"{K}_w_p",
                       {w: (F.WEATHER_DOCS[w], F.WEATHER_UNITS[w]) for w in F.WEATHER_UNITS})
            if set(sel) & {"rh_mean", "srad", "vpd_kpa"}:
                st.warning("rh_mean, srad and vpd_kpa are missing for all Brazilian and most Argentine fields.")
            feats += [("weather" if sel == F.WEATHER_DEFAULT else "weather:" + ",".join(sel))] if sel else []

    st.subheader("Per pixel")
    a, b = st.columns(2)
    with a:
        if st.checkbox("Terrain", "terrain" in p0, key=f"{K}_t"):
            sel = pick("Terrain features", list(F.TERRAIN_UNITS),
                       (p0.get("terrain") or ",".join(F.TERRAIN_UNITS)).split(","), f"{K}_t_p",
                       {k: (F.TERRAIN_DOCS[k], F.TERRAIN_UNITS[k]) for k in F.TERRAIN_UNITS})
            feats += [("terrain" if len(sel) == len(F.TERRAIN_UNITS) else "terrain:" + ",".join(sel))] if sel else []
        if st.checkbox("Location (coords)", "coords" in p0, key=f"{K}_c"):
            st.warning("Coordinates identify the region: under LORO splits they leak the held-out region.")
            feats.append("coords")
        cats = [c for c in F.CATEGORICAL if st.checkbox(f"{c.capitalize()} (one-hot)", c in p0, key=f"{K}_{c}")]
    with b:
        if st.checkbox("Soil (SoilGrids, 250 m)", "soil" in p0, key=f"{K}_soil"):
            sp_props, sp_depth = soil.parse(p0.get("soil"))
            sel = pick("Properties", list(soil.PROPS), sp_props, f"{K}_soil_p",
                       {s: (soil.PROPS[s][2], soil.PROPS[s][1]) for s in soil.PROPS})
            depth = st.select_slider("Depth (cm)", DEPTH_STOPS, (sp_depth if sp_depth[0] in DEPTH_STOPS
                                                                 and sp_depth[1] in DEPTH_STOPS else (0, 30)),
                                     key=f"{K}_soil_d")
            if sel:
                d = f"{depth[0]}-{depth[1]}cm"
                feats.append(f"soil:{d}" if len(sel) == len(soil.PROPS) else f"soil:{','.join(sel)}@{d}")
    feats += cats
    if not any(f.partition(":")[0] in F.DYNAMIC_GROUPS for f in feats):
        st.error("Choose at least one per-time-step feature: the models are sequence models.")

# ---------------------------------------------------------------- 2 training pixels
with tab_px:
    st.caption("Applied to the training pixels only, so val and test stay comparable between runs.")
    tf0 = inp0.get("train_filters") or {}
    ft = data.fields_table()
    c1, c2 = st.columns(2)
    edge = c1.slider("Drop pixels within this distance of the field edge (m)", 0, 60, int(tf0.get("edge_buffer_m", 0)),
                     step=10, key=f"{K}_edge", help="10 = the outermost ring. Holes in the yield map count as edge. "
                                                    "The Swiss Perich farm is already buffered by 20 m.")
    minobs = c2.number_input("Minimum clear S2 observations per pixel (on the chosen steps)", 0, 51,
                             int(tf0.get("min_observed", 0)), key=f"{K}_minobs")
    c1, c2, c3 = st.columns(3)
    countries = c1.multiselect("Only countries (empty = all)", sorted(ft["country"].unique()),
                               tf0.get("countries") or [], key=f"{K}_countries")
    crops = c2.multiselect("Only crops (empty = all)", sorted(ft["crop"].unique()), tf0.get("crops") or [],
                           key=f"{K}_crops")
    years = c3.multiselect("Only years (empty = all)", sorted(int(y) for y in ft["year"].unique()),
                           [int(y) for y in tf0.get("years") or []], key=f"{K}_years")
    train_filters = {k: v for k, v in {"edge_buffer_m": edge, "min_observed": int(minobs), "countries": countries,
                                       "crops": crops, "years": years}.items() if v}

# ---------------------------------------------------------------- 3 model
with tab_model:
    m0 = base.get("model") or {}
    st.markdown(f"**{MODEL_LABEL[mtype]}** (the model type comes from the base config)")
    model = {"type": mtype} if mtype != "lstm" else {}
    cols = st.columns(3)
    if mtype in ("patch", "convlstm"):
        model["patch_size"] = cols[0].selectbox("Patch size (pixels, odd)", list(range(1, 16, 2)),
                                                index=list(range(1, 16, 2)).index(m0.get("patch_size", 5)),
                                                key=f"{K}_patch", help="k × k neighbourhood of 10 m pixels.")
    for i, (key, label, dflt, kind) in enumerate(MODEL_PARAMS[mtype], start=1):
        col = cols[i % 3]
        v = m0.get(key, dflt)
        if kind == "float":
            model[key] = col.number_input(label, 0.0, 0.9, float(v), 0.05, key=f"{K}_m_{key}")
        elif kind == "odd":
            model[key] = col.selectbox(label, [1, 3, 5, 7], index=[1, 3, 5, 7].index(v), key=f"{K}_m_{key}")
        else:
            model[key] = int(col.number_input(label, 1, 1024, int(v), key=f"{K}_m_{key}"))

# ---------------------------------------------------------------- 4 training & splits
with tab_train:
    t0 = base.get("train") or {}
    aug0 = t0.get("augment") or {}
    c1, c2, c3, c4 = st.columns(4)
    train = {
        "lr": c1.number_input("Learning rate", 1e-5, 1e-1, float(t0.get("lr", 1e-3)), format="%.0e", key=f"{K}_lr"),
        "weight_decay": c2.number_input("Weight decay", 0.0, 1e-1, float(t0.get("weight_decay", 1e-4)),
                                        format="%.0e", key=f"{K}_wd"),
        "batch_size": int(c3.selectbox("Batch size", [256, 512, 1024, 2048, 4096],
                                       index=[256, 512, 1024, 2048, 4096].index(t0.get("batch_size", 1024)),
                                       key=f"{K}_bs")),
        "pixels_per_field": int(c4.number_input("Pixels per field and epoch", 10, 100000,
                                                int(t0.get("pixels_per_field") or 500), key=f"{K}_ppf")),
    }
    c1, c2, c3, c4 = st.columns(4)
    train["max_epochs"] = int(c1.number_input("Max epochs", 1, 500, int(t0.get("max_epochs", 100)), key=f"{K}_ep"))
    train["patience"] = int(c2.number_input("Early-stopping patience", 1, 100, int(t0.get("patience", 10)),
                                            key=f"{K}_pat"))
    members = int(c3.number_input("Ensemble members", 1, 10, int((base.get("ensemble") or {}).get("n_members", 5)),
                                  key=f"{K}_mem"))
    scale_by = c4.multiselect("Scale yield per", ["crop", "country"],
                              (base.get("target") or {}).get("standardize_by", ["crop", "country"]), key=f"{K}_tgt")
    train.update({"eval_batch_size": int(t0.get("eval_batch_size", 4096)),
                  "gpu_data_max_gb": float(t0.get("gpu_data_max_gb", 2.0))})
    st.markdown("**Augmentation**")
    c1, c2, c3 = st.columns(3)
    shift_days = c1.slider("Random time shift (± days)", 0, 60, int(aug0.get("time_shift", 0)) *
                           (WINDOW.get(inp0["grid"] or "10d", 10)), step=wd, key=f"{K}_shift_{grid}",
                           help="Shifts every training series by a random number of whole windows.")
    drop = c2.number_input("Step dropout probability", 0.0, 0.5, float(aug0.get("step_dropout", 0.0)), 0.05,
                           key=f"{K}_drop")
    noise = c3.number_input("Spectral noise (std, z-units)", 0.0, 0.5, float(aug0.get("spectral_noise", 0.0)), 0.05,
                            key=f"{K}_noise")
    augment = {k: v for k, v in {"time_shift": shift_days // wd, "step_dropout": float(drop),
                                 "spectral_noise": float(noise)}.items() if v}
    if augment:
        train["augment"] = augment

    st.markdown("**Splits**")
    groups = data.split_groups()
    all_splits = [f"{s}/{m}/{c}" for s, m, c in groups[["scope", "method", "crop"]].itertuples(index=False)]
    pooled = [s for s in all_splits if s.endswith("/all") and not s.startswith("germany/")]
    s_base = base.get("splits")
    s_dflt = all_splits if s_base == "all" else [s for s in (s_base or []) if s in all_splits]
    preset = st.radio("Quick choice", ["as base model", "pooled all-crop (6)", "europe + south america (4)",
                                       "one split (quick test)"], horizontal=True, key=f"{K}_preset")
    s_pick = {"as base model": s_dflt, "pooled all-crop (6)": pooled,
              "europe + south america (4)": [s for s in pooled if not s.startswith("world/")],
              "one split (quick test)": ["europe/loyo/all"]}[preset]
    splits = st.multiselect("Splits", all_splits, s_pick, key=f"{K}_splits_{preset}")

# ---------------------------------------------------------------- 5 check & save
with tab_save:
    inputs = {"features": feats, "grid": grid}
    if align != "calendar":
        inputs["align"] = align
    if tuple(season) != SEASON_DEFAULT[align]:
        inputs["season"] = [int(season[0]), int(season[1])]
    if train_filters:
        inputs["train_filters"] = train_filters
    name = st.text_input("Name of the new config (lowercase, digits, _)", f"{base_name}_custom", key=f"{K}_name")
    cfg = {"name": name, "seed": int(base.get("seed", 42)), "inputs": inputs,
           "target": {"standardize_by": scale_by}, "splits": splits, "model": model, "train": train,
           "ensemble": {"n_members": members},
           "wandb": base.get("wandb") or {"project": "yieldsat", "entity": None, "mode": "disabled"}}
    if "paper_inputs" in base:
        cfg = {"name": name, "paper_inputs": base["paper_inputs"], **{k: v for k, v in cfg.items() if k != "name"}}
    header = (f"# Configured in the visualizer from {base_name} ({Path(configs[base_name]).relative_to(REPO_ROOT)}).\n"
              "# Train: python scripts/LSTM/04_train_v2.py --config configs/custom/" + name + ".yaml --no-wandb\n")
    text = header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=110)

    problems = []
    if not re.fullmatch(r"[a-z0-9_]+", name or ""):
        problems.append("The name may only contain lowercase letters, digits and _.")
    elif name in configs and "/custom/" not in configs[name]:
        problems.append(f"`{name}` is an existing config outside configs/custom; choose another name.")
    elif (data.RUNS / name).exists():
        problems.append(f"Runs named `{name}` exist already; the new config would skip or mix with them.")
    if not splits:
        problems.append("Choose at least one split.")
    if not any(f.partition(":")[0] in F.DYNAMIC_GROUPS for f in feats):
        problems.append("Choose at least one per-time-step feature.")
    for p in problems:
        st.error(p)

    # size of the training inputs per split (before training filters)
    n_dyn = sum(len(F.S2_BANDS) if f == "s2" else len(f.split(":", 1)[1].split(",")) if ":" in f
                else {"indices": len(F.INDICES), "observed": 1, "weather": len(F.WEATHER_DEFAULT)}.get(f, 0)
                for f in feats if f.partition(":")[0] in F.DYNAMIC_GROUPS)
    if splits:
        ft = data.fields_table().set_index("field")["n_pixels"]
        rows = []
        for s in splits:
            tab, _ = data.split_table(*s.split("/"))
            n = int(ft.reindex(tab.loc[tab["set"] == "train", "field"]).sum())
            rows.append({"split": s, "train pixels": f"{n:,}",
                         "train inputs (GB, float16)": round(n * n_steps * n_dyn * 2 / 1e9, 1)})
        size = pd.DataFrame(rows)
        st.markdown(f"**{len(feats)} feature specs, {n_dyn} per-step features × {n_steps} steps.** "
                    "Training inputs per split (before training-pixel filters; the machine has 31 GB RAM, "
                    "~13 GB was fine, ~29 GB peak was the limit):")
        st.dataframe(size, hide_index=True, width="stretch")

    st.code(text, language="yaml")
    c1, c2 = st.columns(2)
    if c1.button("Check inputs on a few pixels", disabled=bool(problems)):
        from src.loader import load
        with st.spinner("Loading 5 pixels per field of the first split's val set ..."):
            try:
                d = load(splits[0], "val", feats, grid=grid, align=align, season=tuple(season),
                         pixels_per_field=5, patch=model.get("patch_size"), dtype=np.float32, verbose=False)
                miss = np.isnan(d.dyn).mean(axis=(0, 1)) if d.dyn.size else []
                st.success(f"OK: {len(d):,} pixels, dynamic {d.dyn.shape}, static {d.static.shape}, days "
                           f"{d.days[0]}..{d.days[-1]}" + (f", patch {d.nbr.shape[1]} cells" if d.nbr is not None
                                                           else ""))
                st.dataframe(pd.DataFrame([np.round(miss, 2)], columns=d.dyn_names, index=["missing share"]),
                             width="stretch")
            except Exception as e:  # show the loader's message (unknown feature, bad season, ...)
                st.error(f"{type(e).__name__}: {e}")
    if c2.button("Save config", type="primary", disabled=bool(problems)):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{name}.yaml"
        existed = path.exists()
        path.write_text(text)
        base_configs.clear()
        st.success(f"{'Overwrote' if existed else 'Saved'} `{path.relative_to(REPO_ROOT)}`. Train it detached with:")
        st.code(f"""cd {REPO_ROOT}
setsid nohup bash -c "source /scratch2/oscipal/cropyield/bin/activate && \\
  python scripts/LSTM/04_train_v2.py --config configs/custom/{name}.yaml --no-wandb" \\
  >> /scratch2/oscipal/yieldsat_work/logs/{name}.log 2>&1 < /dev/null &""", language="bash")
        st.caption("Results then appear on the Runs and Predictions pages (under \"Other\" until the config is "
                   "added to MODEL_FAMILIES in viz/data.py).")


# ---------------------------------------------------------------- summary of the choices (shown above the tabs)

def feat_text(spec: str) -> str:
    g, _, arg = spec.partition(":")
    if g == "s2":
        return "S2 (all 12 bands)" if not arg else f"S2 ({arg.replace(',', ', ')})"
    if g == "indices":
        return "indices (" + (arg or ",".join(F.INDICES)).replace(",", ", ") + ")"
    if g == "weather":
        return "weather (" + ("default set" if not arg else "all" if arg == "all" else arg.replace(",", ", ")) + ")"
    if g == "terrain":
        return "terrain (" + ("all 6" if not arg else arg.replace(",", ", ")) + ")"
    if g == "soil":
        props, (a, b) = soil.parse(arg)
        return f"soil ({'all 8' if len(props) == len(soil.PROPS) else ', '.join(props)}, {a}–{b} cm)"
    return {"observed": "observed flag", "coords": "coords", "crop": "crop", "country": "country"}.get(g, spec)


def settings_rows(feats, grid, align, season, filters, model, train, members, scale_by, splits) -> dict[str, str]:
    dyn = [feat_text(f) for f in feats if f.partition(":")[0] in F.DYNAMIC_GROUPS]
    sta = [feat_text(f) for f in feats if f.partition(":")[0] in F.STATIC_GROUPS]
    w = WINDOW[grid]
    steps = len(range(-(-season[0] // w), -(-season[1] // w)))
    aug = train.get("augment") or {}
    aug_txt = ", ".join(x for x in [f"shift ±{aug['time_shift'] * w} d" if aug.get("time_shift") else "",
                                    f"step dropout {aug['step_dropout']}" if aug.get("step_dropout") else "",
                                    f"spectral noise {aug['spectral_noise']}" if aug.get("spectral_noise") else ""]
                        if x) or "none"
    mparams = ", ".join(f"{k} {v}" for k, v in model.items() if k != "type")
    flt = ", ".join(([f"edge buffer {filters['edge_buffer_m']} m"] if filters.get("edge_buffer_m") else [])
                    + ([f"≥ {filters['min_observed']} clear obs"] if filters.get("min_observed") else [])
                    + [f"{k}: {', '.join(map(str, filters[k]))}" for k in ("countries", "crops", "years")
                       if filters.get(k)]) or "all"
    pooled = {"europe", "south_america", "world"}
    split_txt = (f"{len(splits)}: " + ", ".join(splits)) if len(splits) <= 6 else f"{len(splits)} splits"
    return {
        "Time": f"{grid} grid · origin {align} · days {season[0]} to {season[1]} ({steps} steps)",
        "Per time step": ", ".join(dyn) or "–",
        "Per pixel": ", ".join(sta) or "–",
        "Training pixels": flt,
        "Model": f"{MODEL_LABEL[mtype]}: {mparams}",
        "Training": (f"{members} member{'s' * (members > 1)}, ≤ {train['max_epochs']} epochs (patience "
                     f"{train['patience']}), lr {train['lr']:g}, batch {train['batch_size']}, "
                     f"{train['pixels_per_field']} px/field/epoch, yield scaled per "
                     f"{' × '.join(scale_by) or 'nothing'}"),
        "Augmentation": aug_txt,
        "Splits": split_txt,
    }


m_base = {**({"type": mtype} if mtype != "lstm" else {}), **({"patch_size": m0.get("patch_size", 5)}
                                                           if mtype in ("patch", "convlstm") else {}),
          **{k: m0.get(k, d) for k, _, d, _ in MODEL_PARAMS[mtype]}}
t_base = {**{k: t0.get(k, d) for k, d in (("lr", 1e-3), ("weight_decay", 1e-4), ("batch_size", 1024),
                                           ("pixels_per_field", 500), ("max_epochs", 100), ("patience", 10))},
          "augment": {k: v for k, v in (t0.get("augment") or {}).items() if v}}
g0, a0 = inp0["grid"] or "10d", inp0["align"] or "calendar"
now = settings_rows(feats, grid, align, tuple(season), train_filters, model, train, members, scale_by, splits)
was = settings_rows(inp0["features"], g0, a0, tuple(inp0["season"]) if inp0["season"] else SEASON_DEFAULT[a0],
                    inp0.get("train_filters") or {}, m_base, t_base,
                    int((base.get("ensemble") or {}).get("n_members", 5)),
                    (base.get("target") or {}).get("standardize_by", ["crop", "country"]), s_dflt)
changed = [k for k in now if now[k] != was[k]]
with summary_box:
    label = MAIN.get(base_name, base_name)
    st.markdown(f"**Your run** — based on **{label}** · name `{name}` · "
                + (f"**{len(changed)} change{'s' * (len(changed) > 1)}** from the base (marked ✎)"
                   if changed else "no changes from the base yet"))
    st.markdown("\n".join(["| | picked |", "|---|---|"]
                          + [f"| {'✎ ' if k in changed else ''}**{k}** | {v} |" for k, v in now.items()]))
