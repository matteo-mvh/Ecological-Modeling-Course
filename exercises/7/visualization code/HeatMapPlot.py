# %% Heat map of observations over time and depth
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.tri as tri

# ============================================================
# SETTINGS
# ============================================================
data_path = r"C:\Users\Mmm\OneDrive\Master Studies\4. Semester\Marine Ecological Modeling\Ecological-Modeling-Course\exercises\7\model data\csv_filtered"

# Choose which CSV to load:
# Example 1: full station CSV
csv_file = os.path.join(data_path, "ctd_VIB3727-00001.csv")

# Example 2: parameter-specific CSV
# csv_file = os.path.join(data_path, "ctd_VIB3727-00001_Salinitet.csv")

station_name = "VIB3727-00001"
parameter = "Temperatur"   # ignored if csv already only contains this parameter

dataset_type = "ctd"      # "ctd" or "wq"

USE_TIME_SLICE = True
t_start = "2010-01-01"
t_end   = "2017-12-31"

MIN_POINTS = 3            # minimum required for triangulation
LEVELS = 30               # number of color levels

# ============================================================
# LOAD CSV
# ============================================================
df = pd.read_csv(csv_file, sep=";", encoding="utf-8-sig")

# ============================================================
# COLUMN SETUP
# ============================================================
if dataset_type.lower() == "ctd":
    time_col = "Dato"
    depth_col = "Dybde (m)"
    value_col = "KorrigeretResultat"
    param_col = "Parameter"
elif dataset_type.lower() == "wq":
    time_col = "time"
    depth_col = "GennemsnitsDybde_m"
    value_col = "Resultat"
    param_col = "Parameter"
else:
    raise ValueError("dataset_type must be 'ctd' or 'wq'")

# ============================================================
# CLEAN / CONVERT
# ============================================================
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

if param_col in df.columns:
    df[param_col] = df[param_col].astype(str).str.strip()

if "ObservationsStedNavn" in df.columns:
    df["ObservationsStedNavn"] = df["ObservationsStedNavn"].astype(str).str.strip()

# ============================================================
# OPTIONAL FILTERS
# ============================================================
if "ObservationsStedNavn" in df.columns:
    df = df[df["ObservationsStedNavn"] == station_name].copy()

if param_col in df.columns and parameter is not None:
    df = df[df[param_col] == parameter].copy()

if USE_TIME_SLICE:
    df = df[(df[time_col] >= t_start) & (df[time_col] <= t_end)].copy()

# remove bad rows
df = df.dropna(subset=[time_col, depth_col, value_col]).copy()

# ============================================================
# CHECK DATA
# ============================================================
print(f"Rows after filtering: {len(df)}")

if len(df) < MIN_POINTS:
    raise ValueError("Not enough points to make a heat map.")

print(df[[time_col, depth_col, value_col]].head())

# ============================================================
# PREPARE TRIANGULATION
# ============================================================
x = mdates.date2num(df[time_col])
y = df[depth_col].to_numpy()
z = df[value_col].to_numpy()

triang = tri.Triangulation(x, y)

# Mask overly flat / skinny triangles if needed
# This helps a bit for scattered field data
mask = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio=0.01)
triang.set_mask(mask)

# ============================================================
# PLOT HEAT MAP
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))

contour = ax.tricontourf(triang, z, levels=LEVELS)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label(value_col)

# optional: overlay measurement points
ax.scatter(df[time_col], df[depth_col], s=8, c="k", alpha=0.25)

ax.set_title(f"{parameter} at {station_name}" if parameter else f"Observations at {station_name}")
ax.set_xlabel("Time")
ax.set_ylabel("Depth (m)")
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

# nicer date formatting
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()