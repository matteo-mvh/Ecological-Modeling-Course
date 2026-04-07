import pandas as pd
import matplotlib.pyplot as plt
import mikeio
import os

# ==========================================================
# FOLDERS
# ==========================================================
scenario_folder = r"C:\Users\Mmm\OneDrive\Master Studies\4. Semester\Marine Ecological Modeling\Ecological-Modeling-Course\exercises\9\Lovns_scenario_results-NOVANA_st\Lovns-scen2"
baseline_folder = r"C:\Users\Mmm\OneDrive\Master Studies\4. Semester\Marine Ecological Modeling\Ecological-Modeling-Course\exercises\9\Lovns_scenario_results-NOVANA_st\Lovns-baseline"

# ==========================================================
# FILES
# ==========================================================
scen_surface_path = os.path.join(scenario_folder, "TS_VIB3728-00001_surface-scen2.dfs0")
scen_bottom_path  = os.path.join(scenario_folder, "TS_VIB3728-00001_bottom-scen2.dfs0")
base_surface_path = os.path.join(baseline_folder, "TS_VIB3728-00001_surface-baseline.dfs0")
base_bottom_path  = os.path.join(baseline_folder, "TS_VIB3728-00001_bottom-baseline.dfs0")

# ==========================================================
# LOAD MIKE DFS0 FILES
# ==========================================================
dfs_scen_surface = mikeio.read(scen_surface_path)
dfs_scen_bottom  = mikeio.read(scen_bottom_path)
dfs_base_surface = mikeio.read(base_surface_path)
dfs_base_bottom  = mikeio.read(base_bottom_path)

# ==========================================================
# CONVERT TO DATAFRAME
# ==========================================================
df_scen_surface = dfs_scen_surface.to_dataframe().reset_index()
df_scen_bottom  = dfs_scen_bottom.to_dataframe().reset_index()
df_base_surface = dfs_base_surface.to_dataframe().reset_index()
df_base_bottom  = dfs_base_bottom.to_dataframe().reset_index()

# rename first column to datetime if needed
for df in [df_scen_surface, df_scen_bottom, df_base_surface, df_base_bottom]:
    if "time" in df.columns:
        df.rename(columns={"time": "datetime"}, inplace=True)
    elif df.columns[0] != "datetime":
        df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

# ==========================================================
# ROBUST COLUMN RENAME FROM MIKE LONG NAMES
# ==========================================================
def standardize_mike_columns(df):
    rename_map = {}
    for col in df.columns:
        col_low = str(col).lower()

        if col == "datetime":
            continue
        elif "chlorophyll-a" in col_low or ", ch," in col_low:
            rename_map[col] = "Chlorophyll_a_ug_L"
        elif "dissolved oxygen" in col_low or ", do," in col_low:
            rename_map[col] = "DO_mg_L"
        elif "temperature" in col_low or ", temp," in col_low:
            rename_map[col] = "Temperature_degC"
        elif "tot-n" in col_low or "total nitrogen" in col_low:
            rename_map[col] = "TotalN_mg_L"
        elif "tot-p" in col_low or "total phosphorus" in col_low:
            rename_map[col] = "TotalP_mg_L"
        elif "phosphate" in col_low or ", ip," in col_low:
            rename_map[col] = "Phosphate_P_mg_L"
        elif "nh4" in col_low:
            rename_map[col] = "NH4_N_mg_L"
        elif "no3" in col_low:
            rename_map[col] = "NO3_N_mg_L"
        elif "salinity" in col_low or ", psu," in col_low:
            rename_map[col] = "Salinity_psu"

    df = df.rename(columns=rename_map)
    return df

df_scen_surface = standardize_mike_columns(df_scen_surface)
df_scen_bottom  = standardize_mike_columns(df_scen_bottom)
df_base_surface = standardize_mike_columns(df_base_surface)
df_base_bottom  = standardize_mike_columns(df_base_bottom)

print("Scenario surface columns:")
print(df_scen_surface.columns.tolist())
print("\nScenario surface head:")
print(df_scen_surface.head())

# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================
required_cols = [
    "datetime",
    "Temperature_degC",
    "DO_mg_L",
    "TotalN_mg_L",
    "TotalP_mg_L",
    "Chlorophyll_a_ug_L",
]

for name, df in {
    "df_scen_surface": df_scen_surface,
    "df_scen_bottom": df_scen_bottom,
    "df_base_surface": df_base_surface,
    "df_base_bottom": df_base_bottom,
}.items():
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")

# ==========================================================
# TIMESTEP CHECK
# ==========================================================
dt = df_scen_surface["datetime"].diff().dropna()
timestep = dt.mode()[0]
steps_per_day = pd.Timedelta("1D") / timestep

print(f"\nTimestep: {timestep}")
print(f"Timesteps per day: {int(steps_per_day)}")

# ==========================================================
# MERGE FOR DIFFERENCE CALCULATION
# ==========================================================
surf = pd.merge(
    df_scen_surface,
    df_base_surface,
    on="datetime",
    suffixes=("_scen", "_base")
)

bot = pd.merge(
    df_scen_bottom,
    df_base_bottom,
    on="datetime",
    suffixes=("_scen", "_base")
)

# ==========================================================
# DIFFERENCE = BASELINE - SCENARIO
# ==========================================================
diff_surface = pd.DataFrame({"datetime": surf["datetime"]})
diff_bottom  = pd.DataFrame({"datetime": bot["datetime"]})

vars_to_diff = [
    "TotalP_mg_L",
    "TotalN_mg_L",
    "Chlorophyll_a_ug_L",
    "DO_mg_L",
    "Temperature_degC"
]

for var in vars_to_diff:
    diff_surface[var] = surf[f"{var}_base"] - surf[f"{var}_scen"]
    diff_bottom[var]  = bot[f"{var}_base"] - bot[f"{var}_scen"]

# ==========================================================
# COLORS
# ==========================================================
surface_color = "darkblue"
bottom_color = "orange"

# ==========================================================
# FIGURE 1B: BASELINE VALUES
# ==========================================================
fig1b, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axes[0].plot(df_base_surface["datetime"], df_base_surface["Temperature_degC"], color=surface_color, label="Surface")
axes[0].plot(df_base_bottom["datetime"], df_base_bottom["Temperature_degC"], color=bottom_color, label="Bottom")
axes[0].set_ylabel("Temp (°C)")
axes[0].set_title("Baseline: Temperature")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(df_base_surface["datetime"], df_base_surface["TotalP_mg_L"], color=surface_color, label="Surface Total P")
axes[1].plot(df_base_bottom["datetime"], df_base_bottom["TotalP_mg_L"], color=bottom_color, label="Bottom Total P")
axes[1].plot(df_base_surface["datetime"], df_base_surface["TotalN_mg_L"], color=surface_color, linestyle="--", label="Surface Total N")
axes[1].plot(df_base_bottom["datetime"], df_base_bottom["TotalN_mg_L"], color=bottom_color, linestyle="--", label="Bottom Total N")
axes[1].set_ylabel("mg/L")
axes[1].set_title("Baseline: Total Phosphorus and Total Nitrogen")
axes[1].grid(True, alpha=0.3)
axes[1].legend(ncol=2)

axes[2].plot(df_base_surface["datetime"], df_base_surface["Chlorophyll_a_ug_L"], color=surface_color, label="Surface")
axes[2].plot(df_base_bottom["datetime"], df_base_bottom["Chlorophyll_a_ug_L"], color=bottom_color, label="Bottom")
axes[2].set_ylabel("Chl-a (mg/L)")
axes[2].set_title("Baseline: Chlorophyll-a")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

axes[3].plot(df_base_surface["datetime"], df_base_surface["DO_mg_L"], color=surface_color, label="Surface")
axes[3].plot(df_base_bottom["datetime"], df_base_bottom["DO_mg_L"], color=bottom_color, label="Bottom")
axes[3].set_ylabel("DO (mg/L)")
axes[3].set_title("Baseline: Dissolved Oxygen")
axes[3].set_xlabel("Time")
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 1: SCENARIO VALUES
# ==========================================================
fig1, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axes[0].plot(df_scen_surface["datetime"], df_scen_surface["Temperature_degC"], color=surface_color, label="Surface")
axes[0].plot(df_scen_bottom["datetime"], df_scen_bottom["Temperature_degC"], color=bottom_color, label="Bottom")
axes[0].set_ylabel("Temp (°C)")
axes[0].set_title("Scenario: Temperature")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(df_scen_surface["datetime"], df_scen_surface["TotalP_mg_L"], color=surface_color, label="Surface Total P")
axes[1].plot(df_scen_bottom["datetime"], df_scen_bottom["TotalP_mg_L"], color=bottom_color, label="Bottom Total P")
axes[1].plot(df_scen_surface["datetime"], df_scen_surface["TotalN_mg_L"], color=surface_color, linestyle="--", label="Surface Total N")
axes[1].plot(df_scen_bottom["datetime"], df_scen_bottom["TotalN_mg_L"], color=bottom_color, linestyle="--", label="Bottom Total N")
axes[1].set_ylabel("mg/L")
axes[1].set_title("Scenario: Total Phosphorus and Total Nitrogen")
axes[1].grid(True, alpha=0.3)
axes[1].legend(ncol=2)

axes[2].plot(df_scen_surface["datetime"], df_scen_surface["Chlorophyll_a_ug_L"], color=surface_color, label="Surface")
axes[2].plot(df_scen_bottom["datetime"], df_scen_bottom["Chlorophyll_a_ug_L"], color=bottom_color, label="Bottom")
axes[2].set_ylabel("Chl-a (mg/L)")
axes[2].set_title("Scenario: Chlorophyll-a")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

axes[3].plot(df_scen_surface["datetime"], df_scen_surface["DO_mg_L"], color=surface_color, label="Surface")
axes[3].plot(df_scen_bottom["datetime"], df_scen_bottom["DO_mg_L"], color=bottom_color, label="Bottom")
axes[3].set_ylabel("DO (mg/L)")
axes[3].set_title("Scenario: Dissolved Oxygen")
axes[3].set_xlabel("Time")
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE: SURFACE TOTAL N (BASELINE vs SCENARIO)
# ==========================================================
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(df_scen_surface["datetime"], df_scen_surface["TotalN_mg_L"], color="orange", label="Scenario Surface")
ax.plot(df_base_surface["datetime"], df_base_surface["TotalN_mg_L"], color="darkblue", label="Baseline Surface")
ax.set_title("Surface Total Nitrogen: Baseline vs Scenario")
ax.set_ylabel("Total N (mg/L)")
ax.set_xlabel("Time")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 2A: BASELINE - SCENARIO (NUTRIENTS)
# ==========================================================
fig2, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(diff_surface["datetime"], diff_surface["TotalP_mg_L"], color=surface_color, label="Surface")
axes[0].plot(diff_bottom["datetime"], diff_bottom["TotalP_mg_L"], color=bottom_color, label="Bottom")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_ylabel("Δ Total P (mg/L)")
axes[0].set_title("Baseline - Scenario: Total Phosphorus")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(diff_surface["datetime"], diff_surface["TotalN_mg_L"], color=surface_color, label="Surface")
axes[1].plot(diff_bottom["datetime"], diff_bottom["TotalN_mg_L"], color=bottom_color, label="Bottom")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Δ Total N (mg/L)")
axes[1].set_title("Baseline - Scenario: Total Nitrogen")
axes[1].set_xlabel("Time")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 2B: BASELINE - SCENARIO (CHLOROPHYLL + OXYGEN)
# ==========================================================
fig3, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(diff_surface["datetime"], diff_surface["Chlorophyll_a_ug_L"], color=surface_color, label="Surface")
axes[0].plot(diff_bottom["datetime"], diff_bottom["Chlorophyll_a_ug_L"], color=bottom_color, label="Bottom")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_ylabel("Δ Chl-a (mg/L)")
axes[0].set_title("Baseline - Scenario: Chlorophyll-a")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(diff_surface["datetime"], diff_surface["DO_mg_L"], color=surface_color, label="Surface")
axes[1].plot(diff_bottom["datetime"], diff_bottom["DO_mg_L"], color=bottom_color, label="Bottom")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Δ DO (mg/L)")
axes[1].set_title("Baseline - Scenario: Dissolved Oxygen")
axes[1].set_xlabel("Time")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 3: BOTTOM OXYGEN + SUMMER/AUTUMN MEAN + HYPOXIA
# ==========================================================
summer_months = [6, 7, 8, 9, 10]
hypoxia_limit = 2.0

df_scen_bottom["month"] = df_scen_bottom["datetime"].dt.month
df_base_bottom["month"] = df_base_bottom["datetime"].dt.month

summer_scen = df_scen_bottom[df_scen_bottom["month"].isin(summer_months)].copy()
summer_base = df_base_bottom[df_base_bottom["month"].isin(summer_months)].copy()

mean_scen = summer_scen["DO_mg_L"].mean()
mean_base = summer_base["DO_mg_L"].mean()

t_start = summer_scen["datetime"].min()
t_end   = summer_scen["datetime"].max()

summer_scen["date"] = summer_scen["datetime"].dt.date
summer_base["date"] = summer_base["datetime"].dt.date

hypoxic_days_scen = summer_scen.loc[summer_scen["DO_mg_L"] < hypoxia_limit, "date"].nunique()
hypoxic_days_base = summer_base.loc[summer_base["DO_mg_L"] < hypoxia_limit, "date"].nunique()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_scen_bottom["datetime"], df_scen_bottom["DO_mg_L"], color="orange", label=f"Scenario ({hypoxic_days_scen} hypoxic days)")
ax.plot(df_base_bottom["datetime"], df_base_bottom["DO_mg_L"], color="darkblue", label=f"Baseline ({hypoxic_days_base} hypoxic days)")

ax.plot([t_start, t_end], [mean_scen, mean_scen], color="black", linestyle="--", linewidth=2, label="Scenario Summer/Autumn Mean")
ax.plot([t_start, t_end], [mean_base, mean_base], color="black", linestyle="-", linewidth=2, label="Baseline Summer/Autumn Mean")
ax.plot([t_start, t_end], [hypoxia_limit, hypoxia_limit], color="red", linestyle="--", linewidth=1.8, label="Hypoxia limit (2 mg/L)")
ax.axvspan(t_start, t_end, color="grey", alpha=0.1, label="Critical period")

ax.set_title("Bottom Oxygen with Summer/Autumn Mean and Hypoxia Threshold")
ax.set_ylabel("DO (mg/L)")
ax.set_xlabel("Time")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 4: TOTAL N : TOTAL P RATIO
# ==========================================================
eps = 1e-12

df_scen_surface["NP_ratio"] = df_scen_surface["TotalN_mg_L"] / (df_scen_surface["TotalP_mg_L"] + eps)
df_scen_bottom["NP_ratio"]  = df_scen_bottom["TotalN_mg_L"] / (df_scen_bottom["TotalP_mg_L"] + eps)
df_base_surface["NP_ratio"] = df_base_surface["TotalN_mg_L"] / (df_base_surface["TotalP_mg_L"] + eps)
df_base_bottom["NP_ratio"]  = df_base_bottom["TotalN_mg_L"] / (df_base_bottom["TotalP_mg_L"] + eps)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(df_scen_surface["datetime"], df_scen_surface["NP_ratio"], color="orange", label="Scenario Surface")
axes[0].plot(df_base_surface["datetime"], df_base_surface["NP_ratio"], color="darkblue", label="Baseline Surface")
axes[0].set_ylabel("N:P ratio")
axes[0].set_title("Surface Total N : Total P Ratio")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(df_scen_bottom["datetime"], df_scen_bottom["NP_ratio"], color="orange", label="Scenario Bottom")
axes[1].plot(df_base_bottom["datetime"], df_base_bottom["NP_ratio"], color="darkblue", label="Baseline Bottom")
axes[1].set_ylabel("N:P ratio")
axes[1].set_title("Bottom Total N : Total P Ratio")
axes[1].set_xlabel("Time")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE 5: EQUILIBRIUM OXYGEN, AOU, DELTA DO, DELTA AOU
# ==========================================================
import numpy as np

def oxygen_equilibrium_weiss_mgL(temp_degC, sal_psu):
    """
    Estimate oxygen equilibrium solubility in seawater [mg/L]
    from temperature [degC] and salinity [PSU].
    """
    T = np.asarray(temp_degC, dtype=float) + 273.15
    S = np.asarray(sal_psu, dtype=float)
    Ts = T / 100.0

    DO0 = 1.42905 * np.exp(
        -173.4292
        + 249.6339 * (100.0 / T)
        + 143.3483 * np.log(Ts)
        - 21.8492 * Ts
    )

    Fs = np.exp(
        -S * (
            0.033096
            - 0.014259 * Ts
            + 0.0017000 * Ts**2
        )
    )

    return DO0 * Fs

# ----------------------------------------------------------
# Calculate equilibrium oxygen for bottom water
# ----------------------------------------------------------
df_base_bottom["O2_eq_mg_L"] = oxygen_equilibrium_weiss_mgL(
    df_base_bottom["Temperature_degC"],
    df_base_bottom["Salinity_psu"]
)

df_scen_bottom["O2_eq_mg_L"] = oxygen_equilibrium_weiss_mgL(
    df_scen_bottom["Temperature_degC"],
    df_scen_bottom["Salinity_psu"]
)

# ----------------------------------------------------------
# AOU = O2_eq - bottom DO
# ----------------------------------------------------------
df_base_bottom["AOU_mg_L"] = df_base_bottom["O2_eq_mg_L"] - df_base_bottom["DO_mg_L"]
df_scen_bottom["AOU_mg_L"] = df_scen_bottom["O2_eq_mg_L"] - df_scen_bottom["DO_mg_L"]

# ----------------------------------------------------------
# Merge for delta calculations
# ----------------------------------------------------------
merge_o2 = pd.merge(
    df_base_bottom[["datetime", "DO_mg_L", "AOU_mg_L"]],
    df_scen_bottom[["datetime", "DO_mg_L", "AOU_mg_L"]],
    on="datetime",
    suffixes=("_base", "_scen")
)

merge_o2["delta_DO_mg_L"] = merge_o2["DO_mg_L_base"] - merge_o2["DO_mg_L_scen"]
merge_o2["delta_AOU_mg_L"] = merge_o2["AOU_mg_L_base"] - merge_o2["AOU_mg_L_scen"]

# ==========================================================
# FIGURE A: EQUILIBRIUM OXYGEN + AOU
# ==========================================================
figA, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# 1. Equilibrium oxygen
axes[0].plot(
    df_base_bottom["datetime"],
    df_base_bottom["O2_eq_mg_L"],
    color="black",
    label="Baseline equilibrium O$_2$"
)
axes[0].set_ylabel("O$_2$ eq (mg/L)")
axes[0].set_title("Estimated Bottom Equilibrium Oxygen")
axes[0].grid(True, alpha=0.3)

# 2. AOU
axes[1].plot(
    df_scen_bottom["datetime"],
    df_scen_bottom["AOU_mg_L"],
    color="orange",
    label="Scenario AOU"
)
axes[1].plot(
    df_base_bottom["datetime"],
    df_base_bottom["AOU_mg_L"],
    color="darkblue",
    label="Baseline AOU"
)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("AOU (mg/L)")
axes[1].set_title("Bottom Apparent Oxygen Utilization (AOU)")
axes[1].set_xlabel("Time")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()

# ==========================================================
# FIGURE B: DELTA DO + DELTA AOU
# ==========================================================
figB, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# 3. Delta DO
axes[0].plot(
    merge_o2["datetime"],
    merge_o2["delta_DO_mg_L"],
    color="black",
    label="Baseline - Scenario DO"
)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_ylabel("Δ DO (mg/L)")
axes[0].set_title("Delta Bottom Dissolved Oxygen (Baseline - Scenario)")
axes[0].grid(True, alpha=0.3)


# 4. Delta AOU
axes[1].plot(
    merge_o2["datetime"],
    merge_o2["delta_AOU_mg_L"],
    color="black",
    label="Baseline - Scenario AOU"
)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Δ AOU (mg/L)")
axes[1].set_title("Delta AOU (Baseline - Scenario)")
axes[1].set_xlabel("Time")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()