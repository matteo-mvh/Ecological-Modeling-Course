# %% Compare averaged observations with MIKE output

import os
import numpy as np
import pandas as pd
import mikeio
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================
model_path = r"C:\Users\Mmm\OneDrive\Master Studies\4. Semester\Marine Ecological Modeling\Ecological-Modeling-Course\exercises\7\model output"
obs_path   = r"C:\Users\Mmm\OneDrive\Master Studies\4. Semester\Marine Ecological Modeling\Ecological-Modeling-Course\exercises\7\model data\csv_filtered"

files = {
    "bottom_ctd": "TS_VIB3727-00001_bottom-ctd.dfs0",
    "bottom_wq": "TS_VIB3727-00001_bottom-wq.dfs0",
    "surface_ctd": "TS_VIB3727-00001_surface-ctd.dfs0",
    "surface_wq": "TS_VIB3727-00001_surface-wq.dfs0",
    "obs_ctd": "ctd_VIB3727-00001_avg_combined.csv",
    "obs_wq": "wq_VIB3727-00001_avg_combined.csv",
}

# ============================================================
# SETTINGS
# ============================================================
USE_LAST_YEAR_ONLY = False
OBS_TIME_SLICE = True
OBS_TSTART = "2010-01-01"
OBS_TEND   = "2016-12-31"

PLOT_CTD = True
PLOT_WQ  = True

# ============================================================
# READ MIKE FILES
# ============================================================
ds_bottom_ctd  = mikeio.read(os.path.join(model_path, files["bottom_ctd"]))
ds_bottom_wq   = mikeio.read(os.path.join(model_path, files["bottom_wq"]))
ds_surface_ctd = mikeio.read(os.path.join(model_path, files["surface_ctd"]))
ds_surface_wq  = mikeio.read(os.path.join(model_path, files["surface_wq"]))

df_bottom_ctd  = ds_bottom_ctd.to_dataframe()
df_bottom_wq   = ds_bottom_wq.to_dataframe()
df_surface_ctd = ds_surface_ctd.to_dataframe()
df_surface_wq  = ds_surface_wq.to_dataframe()

# optional model slicing
if USE_LAST_YEAR_ONLY:
    t_end = df_bottom_ctd.index[-1]
    t_start = t_end - pd.Timedelta(days=365)

    df_bottom_ctd  = df_bottom_ctd.loc[t_start:]
    df_bottom_wq   = df_bottom_wq.loc[t_start:]
    df_surface_ctd = df_surface_ctd.loc[t_start:]
    df_surface_wq  = df_surface_wq.loc[t_start:]

    print(f"Model slicing enabled: {t_start} -> {t_end}")
else:
    print("Model slicing disabled")

# ============================================================
# READ AVERAGED OBSERVATIONS
# ============================================================
obs_ctd = pd.read_csv(os.path.join(obs_path, files["obs_ctd"]), sep=";", encoding="utf-8-sig")
obs_wq  = pd.read_csv(os.path.join(obs_path, files["obs_wq"]),  sep=";", encoding="utf-8-sig")

# ============================================================
# HELPER: detect time/value columns in averaged csv
# ============================================================
def get_obs_columns(df, dataset_type):
    if dataset_type.lower() == "ctd":
        time_candidates = ["Dato", "time"]
        value_candidates = ["AverageValue", "KorrigeretResultat", "Resultat"]
    elif dataset_type.lower() == "wq":
        time_candidates = ["time", "Dato"]
        value_candidates = ["AverageValue", "Resultat", "KorrigeretResultat"]
    else:
        raise ValueError("dataset_type must be 'ctd' or 'wq'")

    time_col = None
    value_col = None

    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break

    for c in value_candidates:
        if c in df.columns:
            value_col = c
            break

    if time_col is None:
        raise KeyError(f"Could not find time column in {dataset_type} observation file.")
    if value_col is None:
        raise KeyError(f"Could not find value column in {dataset_type} observation file.")

    return time_col, value_col

ctd_time_col, ctd_value_col = get_obs_columns(obs_ctd, "ctd")
wq_time_col,  wq_value_col  = get_obs_columns(obs_wq,  "wq")

# convert time/value columns
obs_ctd[ctd_time_col] = pd.to_datetime(obs_ctd[ctd_time_col], errors="coerce")
obs_wq[wq_time_col]   = pd.to_datetime(obs_wq[wq_time_col], errors="coerce")

obs_ctd[ctd_value_col] = pd.to_numeric(obs_ctd[ctd_value_col], errors="coerce")
obs_wq[wq_value_col]   = pd.to_numeric(obs_wq[wq_value_col], errors="coerce")

# Convert WQ obs from µg/L to mg/L or mmol-scale comparison, except oxygen
mask_not_oxygen = ~obs_wq["Parameter"].astype(str).str.strip().str.lower().eq("oxygen indhold")
obs_wq.loc[mask_not_oxygen, wq_value_col] = obs_wq.loc[mask_not_oxygen, wq_value_col] / 1000.0

obs_ctd["Parameter"] = obs_ctd["Parameter"].astype(str).str.strip()
obs_wq["Parameter"]  = obs_wq["Parameter"].astype(str).str.strip()
obs_ctd["Layer"]     = obs_ctd["Layer"].astype(str).str.strip()
obs_wq["Layer"]      = obs_wq["Layer"].astype(str).str.strip()

# optional obs slicing
if OBS_TIME_SLICE:
    obs_ctd = obs_ctd[
        (obs_ctd[ctd_time_col] >= OBS_TSTART) &
        (obs_ctd[ctd_time_col] <= OBS_TEND)
    ].copy()

    obs_wq = obs_wq[
        (obs_wq[wq_time_col] >= OBS_TSTART) &
        (obs_wq[wq_time_col] <= OBS_TEND)
    ].copy()

    print(f"Obs slicing enabled: {OBS_TSTART} -> {OBS_TEND}")
else:
    print("Obs slicing disabled")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_model_col(columns, candidates):
    for cand in candidates:
        for col in columns:
            if cand.lower() in col.lower():
                return col
    return None

def obs_subset(df, param_name, layer_name, time_col, value_col):
    out = df[
        (df["Parameter"].astype(str).str.strip() == param_name) &
        (df["Layer"].astype(str).str.strip().str.lower() == layer_name.lower())
    ].copy()
    out = out.dropna(subset=[time_col, value_col])
    return out

# ============================================================
# PARAMETER MAPPING
# ============================================================
ctd_map = {
    "Temperatur": {
        "obs_param": "Temperatur",
        "model_candidates": ["Temperature"]
    },
    "Salinitet": {
        "obs_param": "Salinitet",
        "model_candidates": ["Salinity"]
    },
    "Oxygen indhold": {
        "obs_param": "Oxygen indhold",
        "model_candidates": ["DO", "Dissolved oxygen"]
    },
}

wq_map = {
    "Klorofyl a": {
        "obs_param": "Klorofyl a",
        "model_candidates": ["CH", "Chlorophyll"]
    },
    "Ammoniak+ammonium-N": {
        "obs_param": "Ammoniak+ammonium-N",
        "model_candidates": ["NH4"]
    },
    "Nitrit+nitrat-N": {
        "obs_param": "Nitrit+nitrat-N",
        "model_candidates": ["NO3"]
    },
    "Ortho-phosphat-P": {
        "obs_param": "Ortho-phosphat-P",
        "model_candidates": ["IP", "phosphate"]
    },
    "Nitrogen,total N": {
        "obs_param": "Nitrogen,total N",
        "model_candidates": ["Tot-N"]
    },
    "Phosphor, total-P": {
        "obs_param": "Phosphor, total-P",
        "model_candidates": ["Tot-P"]
    },
}

print("\nAvailable CTD obs parameters:")
print(sorted(obs_ctd["Parameter"].dropna().unique()))

print("\nAvailable WQ obs parameters:")
print(sorted(obs_wq["Parameter"].dropna().unique()))

print("\nAvailable bottom CTD model columns:")
print(list(df_bottom_ctd.columns))

print("\nAvailable bottom WQ model columns:")
print(list(df_bottom_wq.columns))

# ============================================================
# PLOT CTD (Temp, Sal, Oxygen in 3 rows)
# ============================================================
if PLOT_CTD:
    available_ctd = []

    for label, cfg in ctd_map.items():

        # observations must exist in CTD obs file
        if cfg["obs_param"] not in obs_ctd["Parameter"].unique():
            continue

        # Temperature and Salinity from CTD dfs0
        if label in ["Temperatur", "Salinitet"]:
            surf_df = df_surface_ctd
            bot_df  = df_bottom_ctd
            model_col = find_model_col(surf_df.columns, cfg["model_candidates"])

        # Oxygen from WQ dfs0
        elif label == "Oxygen indhold":
            surf_df = df_surface_wq
            bot_df  = df_bottom_wq
            model_col = find_model_col(surf_df.columns, cfg["model_candidates"])

        else:
            continue

        if model_col is not None:
            available_ctd.append((label, cfg, model_col, surf_df, bot_df))

    if len(available_ctd) > 0:
        # force 3 rows, 1 column
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axs = np.array(axs).reshape(-1)

        for i, (label, cfg, model_col, surf_df, bot_df) in enumerate(available_ctd):
            ax = axs[i]

            # MIKE lines
            ax.plot(surf_df.index, surf_df[model_col], label="MIKE Surface")
            ax.plot(bot_df.index,  bot_df[model_col],  label="MIKE Bottom")

            # CTD observations
            obs_surf = obs_subset(obs_ctd, cfg["obs_param"], "Surface", ctd_time_col, ctd_value_col)
            obs_bot  = obs_subset(obs_ctd, cfg["obs_param"], "Bottom",  ctd_time_col, ctd_value_col)

            if len(obs_surf) > 0:
                ax.scatter(obs_surf[ctd_time_col], obs_surf[ctd_value_col], s=28, marker="o", label="Obs Surface")
            if len(obs_bot) > 0:
                ax.scatter(obs_bot[ctd_time_col], obs_bot[ctd_value_col], s=28, marker="s", label="Obs Bottom")

            ax.set_title(label)
            # units for CTD
            unit_map_ctd = {
                "Temperatur": "°C",
                "Salinitet": "‰",
                "Oxygen indhold": "mg/L",
            }
            
            ax.set_ylabel(f"{unit_map_ctd.get(label, '')}")
            ax.grid(True)
            ax.legend()

        axs[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

# ============================================================
# PLOT WQ
# ============================================================
if PLOT_WQ:
    available_wq = []
    for label, cfg in wq_map.items():
        if cfg["obs_param"] in obs_wq["Parameter"].unique():
            col = find_model_col(df_surface_wq.columns, cfg["model_candidates"])
            if col is not None:
                available_wq.append((label, cfg, col))

    if len(available_wq) > 0:
        nvars = len(available_wq)
        ncols = 2
        nrows = int(np.ceil(nvars / ncols))

        fig, axs = plt.subplots(nrows, ncols, figsize=(14, 3.8*nrows), sharex=True)
        axs = np.array(axs).reshape(-1)

        for i, (label, cfg, model_col) in enumerate(available_wq):
            ax = axs[i]

            # MIKE lines
            ax.plot(df_surface_wq.index, df_surface_wq[model_col], label="MIKE Surface")
            ax.plot(df_bottom_wq.index, df_bottom_wq[model_col], label="MIKE Bottom")

            # averaged observations
            obs_surf = obs_subset(obs_wq, cfg["obs_param"], "Surface", wq_time_col, wq_value_col)
            obs_bot  = obs_subset(obs_wq, cfg["obs_param"], "Bottom",  wq_time_col, wq_value_col)

            if len(obs_surf) > 0:
                ax.scatter(obs_surf[wq_time_col], obs_surf[wq_value_col], s=24, marker="o", label="Obs Surface")
            if len(obs_bot) > 0:
                ax.scatter(obs_bot[wq_time_col], obs_bot[wq_value_col], s=24, marker="s", label="Obs Bottom")

            ax.set_title(label)
            # all WQ now in mg/L after scaling
            ax.set_ylabel("mg/L")
            ax.grid(True)
            ax.legend(fontsize=8)

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        for ax in axs[max(0, len(axs)-ncols):]:
            ax.set_xlabel("Time")

        plt.tight_layout()
        plt.show()
        
        