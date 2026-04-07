# %% Depth-averaging script for both CTD and WQ
import os
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
data_path = r"C:\Users\Mmm\OneDrive\Master Studies\4. Semester\Marine Ecological Modeling\Ecological-Modeling-Course\exercises\7\model data\csv_filtered"

# depth thresholds
SURFACE_MAX = 1.5
BOTTOM_MIN  = 3.5

files = {
    "ctd": os.path.join(data_path, "ctd_VIB3727-00001.csv"),
    "wq":  os.path.join(data_path, "wq_VIB3727-00001.csv"),
}

# ============================================================
# PROCESS FUNCTION
# ============================================================
def make_depth_averages(csv_file, dataset_type):
    df = pd.read_csv(csv_file, sep=";", encoding="utf-8-sig")

    # --------------------------------------------------------
    # Column setup
    # --------------------------------------------------------
    if dataset_type.lower() == "ctd":
        time_col = "Dato"
        depth_col = "Dybde (m)"
        value_col = "KorrigeretResultat"
    elif dataset_type.lower() == "wq":
        time_col = "time"
        depth_col = "GennemsnitsDybde_m"
        value_col = "Resultat"
    else:
        raise ValueError("dataset_type must be 'ctd' or 'wq'")

    param_col = "Parameter"

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[param_col] = df[param_col].astype(str).str.strip()

    df = df.dropna(subset=[time_col, depth_col, value_col])

    # --------------------------------------------------------
    # Surface average
    # --------------------------------------------------------
    df_surface = df[df[depth_col] <= SURFACE_MAX].copy()

    surface_avg = (
        df_surface
        .groupby([time_col, param_col])[value_col]
        .mean()
        .reset_index()
    )
    surface_avg["Layer"] = "Surface"

    # --------------------------------------------------------
    # Bottom average
    # --------------------------------------------------------
    df_bottom = df[df[depth_col] >= BOTTOM_MIN].copy()

    bottom_avg = (
        df_bottom
        .groupby([time_col, param_col])[value_col]
        .mean()
        .reset_index()
    )
    bottom_avg["Layer"] = "Bottom"

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------
    combined = pd.concat([surface_avg, bottom_avg], ignore_index=True)
    combined = combined.sort_values(by=[time_col, param_col, "Layer"])

    # optional: rename value column to something generic
    combined = combined.rename(columns={value_col: "AverageValue"})

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    station_name = os.path.splitext(os.path.basename(csv_file))[0]
    out_file = os.path.join(data_path, f"{station_name}_avg_combined.csv")

    combined.to_csv(out_file, sep=";", index=False, encoding="utf-8-sig")

    print(f"Saved combined file for {dataset_type.upper()}:")
    print(out_file)
    print(f"Rows: {len(combined)}\n")

    return combined

# ============================================================
# RUN BOTH
# ============================================================
for dataset_type, csv_file in files.items():
    if os.path.exists(csv_file):
        make_depth_averages(csv_file, dataset_type)
    else:
        print(f"File not found for {dataset_type.upper()}: {csv_file}")