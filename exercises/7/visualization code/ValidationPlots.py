# %% Compare selected averaged observations with MIKE output
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

# ============================================================
# DEFINE TIME WINDOW
# ============================================================
if USE_LAST_YEAR_ONLY:
    t_end = df_bottom_ctd.index[-1]
    t_start = t_end - pd.Timedelta(days=365)
    print(f"Last-year slicing enabled: {t_start} -> {t_end}")
else:
    t_start = pd.to_datetime(OBS_TSTART)
    t_end   = pd.to_datetime(OBS_TEND)
    print("Last-year slicing disabled")

# ============================================================
# APPLY MODEL SLICING
# ============================================================
if USE_LAST_YEAR_ONLY:
    df_bottom_ctd  = df_bottom_ctd.loc[t_start:t_end]
    df_bottom_wq   = df_bottom_wq.loc[t_start:t_end]
    df_surface_ctd = df_surface_ctd.loc[t_start:t_end]
    df_surface_wq  = df_surface_wq.loc[t_start:t_end]
    print(f"Model sliced to last year: {t_start} -> {t_end}")
else:
    print("Model full period retained")

# ============================================================
# READ AVERAGED OBSERVATIONS
# ============================================================
obs_ctd = pd.read_csv(os.path.join(obs_path, files["obs_ctd"]), sep=";", encoding="utf-8-sig")
obs_wq  = pd.read_csv(os.path.join(obs_path, files["obs_wq"]),  sep=";", encoding="utf-8-sig")

# ============================================================
# HELPERS
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

    time_col = next((c for c in time_candidates if c in df.columns), None)
    value_col = next((c for c in value_candidates if c in df.columns), None)

    if time_col is None:
        raise KeyError(f"Could not find time column in {dataset_type} observation file.")
    if value_col is None:
        raise KeyError(f"Could not find value column in {dataset_type} observation file.")

    return time_col, value_col

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
# OBS CLEANING
# ============================================================
ctd_time_col, ctd_value_col = get_obs_columns(obs_ctd, "ctd")
wq_time_col,  wq_value_col  = get_obs_columns(obs_wq,  "wq")

obs_ctd[ctd_time_col] = pd.to_datetime(obs_ctd[ctd_time_col], errors="coerce")
obs_wq[wq_time_col]   = pd.to_datetime(obs_wq[wq_time_col], errors="coerce")

obs_ctd[ctd_value_col] = pd.to_numeric(obs_ctd[ctd_value_col], errors="coerce")
obs_wq[wq_value_col]   = pd.to_numeric(obs_wq[wq_value_col], errors="coerce")

obs_ctd["Parameter"] = obs_ctd["Parameter"].astype(str).str.strip()
obs_wq["Parameter"]  = obs_wq["Parameter"].astype(str).str.strip()
obs_ctd["Layer"]     = obs_ctd["Layer"].astype(str).str.strip()
obs_wq["Layer"]      = obs_wq["Layer"].astype(str).str.strip()

# Convert WQ obs from µg/L to mg/L, except oxygen
mask_not_oxygen = ~obs_wq["Parameter"].str.lower().eq("oxygen indhold")
obs_wq.loc[mask_not_oxygen, wq_value_col] = obs_wq.loc[mask_not_oxygen, wq_value_col] / 1000.0

# ============================================================
# APPLY OBS SLICING
# ============================================================
if USE_LAST_YEAR_ONLY:
    obs_ctd = obs_ctd[
        (obs_ctd[ctd_time_col] >= t_start) &
        (obs_ctd[ctd_time_col] <= t_end)
    ].copy()

    obs_wq = obs_wq[
        (obs_wq[wq_time_col] >= t_start) &
        (obs_wq[wq_time_col] <= t_end)
    ].copy()

    print(f"Obs last-year slicing enabled: {t_start} -> {t_end}")

elif OBS_TIME_SLICE:
    obs_ctd = obs_ctd[
        (obs_ctd[ctd_time_col] >= pd.to_datetime(OBS_TSTART)) &
        (obs_ctd[ctd_time_col] <= pd.to_datetime(OBS_TEND))
    ].copy()

    obs_wq = obs_wq[
        (obs_wq[wq_time_col] >= pd.to_datetime(OBS_TSTART)) &
        (obs_wq[wq_time_col] <= pd.to_datetime(OBS_TEND))
    ].copy()

    print(f"Obs slicing enabled: {OBS_TSTART} -> {OBS_TEND}")

else:
    print("Obs slicing disabled")

# ============================================================
# SELECTED VARIABLES ONLY
# ============================================================
selected_vars = [
    {
        "label": "Salinity",
        "obs_df": obs_ctd,
        "obs_param": "Salinitet",
        "time_col": ctd_time_col,
        "value_col": ctd_value_col,
        "surf_df": df_surface_ctd,
        "bot_df": df_bottom_ctd,
        "model_candidates": ["Salinity"],
        "unit": "‰",
    },
    {
        "label": "Chlorophyll-a",
        "obs_df": obs_wq,
        "obs_param": "Klorofyl a",
        "time_col": wq_time_col,
        "value_col": wq_value_col,
        "surf_df": df_surface_wq,
        "bot_df": df_bottom_wq,
        "model_candidates": ["CH", "Chlorophyll"],
        "unit": "mg/L",
    },
    {
        "label": "Oxygen",
        "obs_df": obs_ctd,
        "obs_param": "Oxygen indhold",
        "time_col": ctd_time_col,
        "value_col": ctd_value_col,
        "surf_df": df_surface_wq,
        "bot_df": df_bottom_wq,
        "model_candidates": ["DO", "Dissolved oxygen"],
        "unit": "mg/L",
    },
]

# ============================================================
# PLOT
# ============================================================
fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axs = np.array(axs).reshape(-1)

for i, cfg in enumerate(selected_vars):
    ax = axs[i]

    model_col = find_model_col(cfg["surf_df"].columns, cfg["model_candidates"])
    if model_col is None:
        ax.set_title(f'{cfg["label"]} (model column not found)')
        ax.grid(True)
        continue

    obs_surf = obs_subset(cfg["obs_df"], cfg["obs_param"], "Surface", cfg["time_col"], cfg["value_col"])
    obs_bot  = obs_subset(cfg["obs_df"], cfg["obs_param"], "Bottom",  cfg["time_col"], cfg["value_col"])

    # model lines
    ax.plot(cfg["surf_df"].index, cfg["surf_df"][model_col], label="MIKE Surface")
    ax.plot(cfg["bot_df"].index,  cfg["bot_df"][model_col],  label="MIKE Bottom")

    # obs points
    if len(obs_surf) > 0:
        ax.scatter(
            obs_surf[cfg["time_col"]],
            obs_surf[cfg["value_col"]],
            s=28,
            marker="o",
            label="Obs Surface"
        )
    if len(obs_bot) > 0:
        ax.scatter(
            obs_bot[cfg["time_col"]],
            obs_bot[cfg["value_col"]],
            s=28,
            marker="s",
            label="Obs Bottom"
        )

    ax.set_title(cfg["label"])
    ax.set_ylabel(cfg["unit"])
    ax.grid(True)
    ax.legend(loc="lower right")

axs[-1].set_xlabel("Time")
plt.tight_layout()
plt.show()

# ============================================================
# SCATTER PLOTS: MODELED VS OBSERVED
# ============================================================

def match_obs_to_model(obs_df, obs_param, time_col, value_col,
                       surf_df, bot_df, model_col,
                       max_time_diff="12H"):
    """
    Match surface and bottom observations to nearest model timestamps.
    Returns one dataframe with columns:
    time, obs, model, Layer
    """
    out_frames = []

    for layer_name, model_df in [("Surface", surf_df), ("Bottom", bot_df)]:
        obs_sub = obs_subset(obs_df, obs_param, layer_name, time_col, value_col).copy()

        if len(obs_sub) == 0:
            continue

        obs_sub = obs_sub[[time_col, value_col]].copy()
        obs_sub = obs_sub.rename(columns={time_col: "time", value_col: "obs"})
        obs_sub = obs_sub.sort_values("time")

        model_sub = model_df[[model_col]].copy()
        model_sub = model_sub.reset_index()

        # make sure first column is called time
        if "time" not in model_sub.columns:
            model_sub = model_sub.rename(columns={model_sub.columns[0]: "time"})
        model_sub = model_sub.rename(columns={model_col: "model"})
        model_sub = model_sub.sort_values("time")

        matched = pd.merge_asof(
            obs_sub,
            model_sub,
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta(max_time_diff)
        )

        matched["Layer"] = layer_name
        out_frames.append(matched)

    if len(out_frames) == 0:
        return pd.DataFrame(columns=["time", "obs", "model", "Layer"])

    out = pd.concat(out_frames, ignore_index=True)
    out = out.dropna(subset=["obs", "model"])
    return out


scatter_vars = [
    {
        "label": "Salinity",
        "obs_df": obs_ctd,
        "obs_param": "Salinitet",
        "time_col": ctd_time_col,
        "value_col": ctd_value_col,
        "surf_df": df_surface_ctd,
        "bot_df": df_bottom_ctd,
        "model_candidates": ["Salinity"],
        "unit": "‰",
    },
    {
        "label": "Chlorophyll-a",
        "obs_df": obs_wq,
        "obs_param": "Klorofyl a",
        "time_col": wq_time_col,
        "value_col": wq_value_col,
        "surf_df": df_surface_wq,
        "bot_df": df_bottom_wq,
        "model_candidates": ["CH", "Chlorophyll"],
        "unit": "mg/L",
    },
    {
        "label": "Oxygen",
        "obs_df": obs_ctd,
        "obs_param": "Oxygen indhold",
        "time_col": ctd_time_col,
        "value_col": ctd_value_col,
        "surf_df": df_surface_wq,
        "bot_df": df_bottom_wq,
        "model_candidates": ["DO", "Dissolved oxygen"],
        "unit": "mg/L",
    },
]

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs = np.array(axs).reshape(-1)

for i, cfg in enumerate(scatter_vars):
    ax = axs[i]

    model_col = find_model_col(cfg["surf_df"].columns, cfg["model_candidates"])
    if model_col is None:
        ax.set_title(f'{cfg["label"]}\n(model column not found)')
        ax.grid(True)
        continue

    matched = match_obs_to_model(
        obs_df=cfg["obs_df"],
        obs_param=cfg["obs_param"],
        time_col=cfg["time_col"],
        value_col=cfg["value_col"],
        surf_df=cfg["surf_df"],
        bot_df=cfg["bot_df"],
        model_col=model_col,
        max_time_diff="12h"   # change if needed
    )

    if len(matched) == 0:
        ax.set_title(f'{cfg["label"]}\n(no matched data)')
        ax.grid(True)
        continue

    surf = matched[matched["Layer"] == "Surface"]
    bot  = matched[matched["Layer"] == "Bottom"]

    if len(surf) > 0:
        ax.scatter(surf["obs"], surf["model"], s=32, marker="o", label="Surface")
    if len(bot) > 0:
        ax.scatter(bot["obs"], bot["model"], s=32, marker="s", label="Bottom")

    # 1:1 line
    xy_min = min(matched["obs"].min(), matched["model"].min())
    xy_max = max(matched["obs"].max(), matched["model"].max())
    ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", linewidth=1)

    # Pearson correlation R
    R_all = matched["obs"].corr(matched["model"]) if len(matched) >= 2 else np.nan
    R_surf = surf["obs"].corr(surf["model"]) if len(surf) >= 2 else np.nan
    R_bot  = bot["obs"].corr(bot["model"]) if len(bot) >= 2 else np.nan
    
    # RMSE
    if len(matched) > 0:
        rmse_all = np.sqrt(np.mean((matched["model"] - matched["obs"])**2))
    else:
        rmse_all = np.nan
    
    if len(surf) > 0:
        rmse_surf = np.sqrt(np.mean((surf["model"] - surf["obs"])**2))
    else:
        rmse_surf = np.nan
    
    if len(bot) > 0:
        rmse_bot = np.sqrt(np.mean((bot["model"] - bot["obs"])**2))
    else:
        rmse_bot = np.nan
    
    ax.text(
        0.05, 0.95,
        f"R(all) = {R_all:.2f}\n"
        f"R(surf) = {R_surf:.2f}\n"
        f"R(bot) = {R_bot:.2f}\n",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )
    
    ax.set_title(cfg["label"])
    ax.set_xlabel(f'Observed [{cfg["unit"]}]')
    ax.set_ylabel(f'Modeled [{cfg["unit"]}]')
    ax.grid(True)
    ax.legend(loc="lower right")

plt.tight_layout()
plt.show()


# ============================================================
# BIAS PLOTS: MODELED - OBSERVED
# ============================================================

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs = np.array(axs).reshape(-1)

for i, cfg in enumerate(scatter_vars):
    ax = axs[i]

    model_col = find_model_col(cfg["surf_df"].columns, cfg["model_candidates"])
    if model_col is None:
        ax.set_title(f'{cfg["label"]}\n(model column not found)')
        ax.grid(True)
        continue

    matched = match_obs_to_model(
        obs_df=cfg["obs_df"],
        obs_param=cfg["obs_param"],
        time_col=cfg["time_col"],
        value_col=cfg["value_col"],
        surf_df=cfg["surf_df"],
        bot_df=cfg["bot_df"],
        model_col=model_col,
        max_time_diff="12h"
    )

    if len(matched) == 0:
        ax.set_title(f'{cfg["label"]}\n(no matched data)')
        ax.grid(True)
        continue

    matched = matched.copy()
    matched["bias"] = matched["model"] - matched["obs"]

    surf = matched[matched["Layer"] == "Surface"]
    bot  = matched[matched["Layer"] == "Bottom"]

    if len(surf) > 0:
        ax.scatter(surf["obs"], surf["bias"], s=32, marker="o", label="Surface")
    if len(bot) > 0:
        ax.scatter(bot["obs"], bot["bias"], s=32, marker="s", label="Bottom")

    # zero-bias line
    ax.axhline(0.0, linestyle="--", linewidth=1)

    # mean bias values
    bias_all = matched["bias"].mean() if len(matched) > 0 else np.nan
    bias_surf = surf["bias"].mean() if len(surf) > 0 else np.nan
    bias_bot  = bot["bias"].mean() if len(bot) > 0 else np.nan

    ax.set_title(cfg["label"])
    ax.set_xlabel(f'Observed [{cfg["unit"]}]')
    ax.set_ylabel(f'Bias = Model - Obs [{cfg["unit"]}]')
    ax.grid(True)
    ax.legend()

    ax.text(
        0.05, 0.95,
        f"Bias(all) = {bias_all:.2f}\nBias(surf) = {bias_surf:.2f}\nBias(bot) = {bias_bot:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

plt.tight_layout()
plt.show()