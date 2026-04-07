import os
import mikeio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================
path = r"C:\Users\Mmm\OneDrive\Master Studies\4. Semester\Marine Ecological Modeling\Ecological-Modeling-Course\exercises\7\model output"

files = {
    "bottom_ctd": "TS_VIB3727-00001_bottom-ctd.dfs0",
    "bottom_wq": "TS_VIB3727-00001_bottom-wq.dfs0",
    "surface_ctd": "TS_VIB3727-00001_surface-ctd.dfs0",
    "surface_wq": "TS_VIB3727-00001_surface-wq.dfs0",
}

# ============================================================
# SETTINGS
# ============================================================
USE_LAST_YEAR_ONLY = False   # <-- turn ON/OFF here

# ============================================================
# Read files
# ============================================================
ds_bottom_ctd = mikeio.read(os.path.join(path, files["bottom_ctd"]))
ds_bottom_wq = mikeio.read(os.path.join(path, files["bottom_wq"]))
ds_surface_ctd = mikeio.read(os.path.join(path, files["surface_ctd"]))
ds_surface_wq = mikeio.read(os.path.join(path, files["surface_wq"]))

df_bottom_ctd = ds_bottom_ctd.to_dataframe()
df_bottom_wq = ds_bottom_wq.to_dataframe()
df_surface_ctd = ds_surface_ctd.to_dataframe()
df_surface_wq = ds_surface_wq.to_dataframe()

# ============================================================
# Optional slicing
# ============================================================
if USE_LAST_YEAR_ONLY:
    t_end = df_bottom_ctd.index[-1]
    t_start = t_end - pd.Timedelta(days=365)

    df_bottom_ctd = df_bottom_ctd.loc[t_start:]
    df_bottom_wq  = df_bottom_wq.loc[t_start:]
    df_surface_ctd = df_surface_ctd.loc[t_start:]
    df_surface_wq  = df_surface_wq.loc[t_start:]

    print(f"Slicing enabled: {t_start} → {t_end}")
else:
    print("Slicing disabled: using full time range")


# ============================================================
# 1) Plot CTD variables: bottom vs surface
# ============================================================
fig_ctd, axs_ctd = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for col in df_bottom_ctd.columns:
    axs_ctd[0].plot(df_bottom_ctd.index, df_bottom_ctd[col], label=f"Bottom - {col}")
for col in df_surface_ctd.columns:
    axs_ctd[0].plot(df_surface_ctd.index, df_surface_ctd[col], linestyle="--", label=f"Surface - {col}")

axs_ctd[0].set_title("CTD variables: Bottom vs Surface")
axs_ctd[0].set_ylabel("Value")
axs_ctd[0].grid(True)
axs_ctd[0].legend()

# Plot temperature only
temp_cols_bottom = [c for c in df_bottom_ctd.columns if "Temperature" in c]
temp_cols_surface = [c for c in df_surface_ctd.columns if "Temperature" in c]

for col in temp_cols_bottom:
    axs_ctd[1].plot(df_bottom_ctd.index, df_bottom_ctd[col], label="Bottom Temperature")
for col in temp_cols_surface:
    axs_ctd[1].plot(df_surface_ctd.index, df_surface_ctd[col], linestyle="--", label="Surface Temperature")

sal_cols_bottom = [c for c in df_bottom_ctd.columns if "Salinity" in c]
sal_cols_surface = [c for c in df_surface_ctd.columns if "Salinity" in c]

for col in sal_cols_bottom:
    axs_ctd[1].plot(df_bottom_ctd.index, df_bottom_ctd[col], label="Bottom Salinity")
for col in sal_cols_surface:
    axs_ctd[1].plot(df_surface_ctd.index, df_surface_ctd[col], linestyle="--", label="Surface Salinity")

axs_ctd[1].set_title("Temperature and Salinity")
axs_ctd[1].set_ylabel("Value")
axs_ctd[1].set_xlabel("Time")
axs_ctd[1].grid(True)
axs_ctd[1].legend()

plt.tight_layout()
plt.show()

# ============================================================
# 2) Plot WQ variables: TWO COLUMNS
# ============================================================
wq_columns = df_bottom_wq.columns
nvars = len(wq_columns)

ncols = 2
nrows = int(np.ceil(nvars / ncols))

fig_wq, axs_wq = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharex=True)

# flatten axes for easy looping
axs_wq = axs_wq.flatten()

for i, col in enumerate(wq_columns):
    axs_wq[i].plot(df_bottom_wq.index, df_bottom_wq[col], label="Bottom")
    axs_wq[i].plot(df_surface_wq.index, df_surface_wq[col], linestyle="--", label="Surface")
    axs_wq[i].set_title(col)
    axs_wq[i].set_ylabel("Value")
    axs_wq[i].grid(True)
    axs_wq[i].legend()

# remove empty subplots (if odd number of variables)
for j in range(i + 1, len(axs_wq)):
    fig_wq.delaxes(axs_wq[j])

# only label bottom row
for ax in axs_wq[-ncols:]:
    ax.set_xlabel("Time")

plt.tight_layout()
plt.show()