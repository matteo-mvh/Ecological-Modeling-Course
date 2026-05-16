# ============================================================
# PLOT DIATOM DEPTH PROFILES FOR DIFFERENT CELL AMOUNTS
# ============================================================

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FILE HELPERS
# ============================================================
def get_base_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


# ============================================================
# SETTINGS
# ============================================================
base_dir = get_base_dir()

# Folder containing saved depth-profile txt files
diatom_depth_dir = base_dir / "results_files" / "DiatomOverDepth"
out_dir = diatom_depth_dir
out_dir.mkdir(parents=True, exist_ok=True)

file_pattern = "DiatomOverDepth_*.txt"


# ============================================================
# FIND FILES
# ============================================================
files = sorted(diatom_depth_dir.glob(file_pattern))

if len(files) == 0:
    raise FileNotFoundError(
        f"No files found in:\n{diatom_depth_dir}\n"
        f"Expected files named like: DiatomOverDepth_30.txt"
    )


# ============================================================
# READ FILES
# ============================================================
profiles = []

for file in files:
    match = re.search(r"DiatomOverDepth_(\d+)\.txt", file.name)

    if match is None:
        print(f"Skipping file with unexpected name: {file.name}")
        continue

    nz = int(match.group(1))

    data = np.loadtxt(file, comments="#")

    if data.ndim != 2 or data.shape[1] < 2:
        print(f"Skipping file with unexpected data format: {file.name}")
        continue

    depth = data[:, 0]
    pdn = data[:, 1]

    profiles.append({
        "nz": nz,
        "depth": depth,
        "pdn": pdn,
        "file": file,
    })

if len(profiles) == 0:
    raise RuntimeError("No valid diatom depth profiles could be loaded.")

profiles = sorted(profiles, key=lambda x: x["nz"])


# ============================================================
# PLOT: DIATOM CONCENTRATION OVER DEPTH
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))

for prof in profiles:
    ax.plot(
        prof["pdn"],
        prof["depth"],
        lw=2,
        marker="o",
        markersize=3,
        label=fr"$n_z={prof['nz']}$"
    )

ax.invert_yaxis()
ax.set_xlabel(r"$P_{D,N}$ [mmol N m$^{-3}$]")
ax.set_ylabel("Depth [m]")
ax.grid(True)
ax.legend(title="Cell amount")

plt.tight_layout()

out_file = out_dir / "DiatomOverDepth_resolution_comparison.png"
fig.savefig(out_file, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved figure to: {out_file}")