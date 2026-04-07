import os
import openpyxl
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================
USE_TIME_SLICE = True
t_start = "2010-01-01"
t_end   = "2017-12-31"

# ============================================================
# Paths
# ============================================================
data_path = r"C:\Users\Mmm\OneDrive\Master Studies\4. Semester\Marine Ecological Modeling\Ecological-Modeling-Course\exercises\7\model data"
out_path  = os.path.join(data_path, "csv_filtered")
os.makedirs(out_path, exist_ok=True)

file_ctd = os.path.join(data_path, "SkiveFjord-NOVANA-ctd-2000-2025.xlsx")
file_wq  = os.path.join(data_path, "SkiveFjord-NOVANA-wq-2000-2025.xlsx")

station_name = "VIB3727-00001"

# ============================================================
# Read Excel with openpyxl
# ============================================================
def xlsx_to_df(filename, sheet_name=None):
    wb = openpyxl.load_workbook(filename, data_only=True, read_only=True)
    ws = wb[wb.sheetnames[0] if sheet_name is None else sheet_name]
    rows = list(ws.values)
    header = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)

ctd_obs = xlsx_to_df(file_ctd)
wq_obs  = xlsx_to_df(file_wq)

# ============================================================
# Clean / convert columns
# ============================================================
ctd_obs["ObservationsStedNavn"] = ctd_obs["ObservationsStedNavn"].astype(str).str.strip()
wq_obs["ObservationsStedNavn"]  = wq_obs["ObservationsStedNavn"].astype(str).str.strip()

ctd_obs["Parameter"] = ctd_obs["Parameter"].astype(str).str.strip()
wq_obs["Parameter"]  = wq_obs["Parameter"].astype(str).str.strip()

# --- CTD ---
ctd_obs["Dato"] = pd.to_datetime(
    ctd_obs["Dato"].astype(str).str.strip(),
    format="%Y%m%d",
    errors="coerce"
)
ctd_obs["KorrigeretResultat"] = pd.to_numeric(ctd_obs["KorrigeretResultat"], errors="coerce")
ctd_obs["OriginalResultat"] = pd.to_numeric(ctd_obs["OriginalResultat"], errors="coerce")
ctd_obs["Dybde (m)"] = pd.to_numeric(ctd_obs["Dybde (m)"], errors="coerce")

# --- WQ ---
wq_obs["Startdato"] = pd.to_datetime(
    wq_obs["Startdato"].astype(str).str.strip(),
    format="%Y%m%d",
    errors="coerce"
)
wq_obs["Startklok"] = (
    pd.to_numeric(wq_obs["Startklok"], errors="coerce")
    .fillna(0)
    .astype(int)
    .astype(str)
    .str.zfill(4)
)

wq_obs["Resultat"] = pd.to_numeric(wq_obs["Resultat"], errors="coerce")
wq_obs["GennemsnitsDybde_m"] = pd.to_numeric(wq_obs["GennemsnitsDybde_m"], errors="coerce")

wq_obs["time"] = pd.to_datetime(
    wq_obs["Startdato"].dt.strftime("%Y-%m-%d") + " " + wq_obs["Startklok"],
    format="%Y-%m-%d %H%M",
    errors="coerce"
)

# ============================================================
# Filter station
# ============================================================
ctd_station = ctd_obs[ctd_obs["ObservationsStedNavn"] == station_name].copy()
wq_station  = wq_obs[wq_obs["ObservationsStedNavn"] == station_name].copy()

print("CTD rows after station filter:", len(ctd_station))
print("WQ rows after station filter:", len(wq_station))

print("\nCTD parameters:")
print(sorted(ctd_station["Parameter"].dropna().unique()))

print("\nWQ parameters:")
print(sorted(wq_station["Parameter"].dropna().unique()))

# ============================================================
# Optional time slicing
# ============================================================
if USE_TIME_SLICE:
    ctd_station = ctd_station[
        (ctd_station["Dato"] >= t_start) &
        (ctd_station["Dato"] <= t_end)
    ].copy()

    wq_station = wq_station[
        (wq_station["time"] >= t_start) &
        (wq_station["time"] <= t_end)
    ].copy()

    print(f"Slicing enabled: {t_start} → {t_end}")
else:
    print("No time slicing applied")

# ============================================================
# Format datetime columns before saving
# ============================================================
ctd_station["Dato"] = ctd_station["Dato"].dt.strftime("%Y-%m-%d %H:%M:%S")
wq_station["Startdato"] = wq_station["Startdato"].dt.strftime("%Y-%m-%d")
wq_station["time"] = wq_station["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# Save filtered full station CSVs
# ============================================================
ctd_station_file = os.path.join(out_path, f"ctd_{station_name}.csv")
wq_station_file  = os.path.join(out_path, f"wq_{station_name}.csv")

ctd_station.to_csv(ctd_station_file, sep=";", index=False, encoding="utf-8-sig")
wq_station.to_csv(wq_station_file, sep=";", index=False, encoding="utf-8-sig")

print("\nSaved:")
print(ctd_station_file)
print(wq_station_file)