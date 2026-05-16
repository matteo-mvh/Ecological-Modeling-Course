# ============================================================
# STEADY-STATE / SEASONAL HEATMAP SWEEP
#
# If Seasonality = True:
#   - diagnostics are computed over the LAST 365 DAYS
#   - plots show water-column MEAN and MAX/MIN values side by side
#   - no surface/bottom plots are made
#   - additionally plots upper-10 m bloom amplitude and bloom timing
#
# If Seasonality = False:
#   - diagnostics are computed from the FINAL timestep only
#   - plots keep the old SURFACE/BOTTOM behaviour
# ============================================================

from pathlib import Path
from datetime import datetime
from dataclasses import fields
import json
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ModelCode import (
    Params,
    make_initial_conditions,
    rhs,
    build_parameter_signature,
    build_idx_from_saved_case,
)

# ============================================================
# USER SETTINGS
# ============================================================
heatmap_name = "seasonal_sweep"

N_range = np.linspace(0.0, 30.0, 21)
S_range = np.linspace(0.0, 5.0, 21)

verbose = True

pdn_quota_threshold = 1e-9
upper_bloom_depth_m = 10.0


# ============================================================
# OUTPUT FOLDER
# ============================================================
def get_base_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = get_base_dir()
RESULTS_DIR = BASE_DIR / "results_files"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEATMAP_DIR = RESULTS_DIR / f"{heatmap_name}_heatmap_{timestamp}"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

CACHE_ROOT = RESULTS_DIR / "cached_runs"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

print(f"Saving heatmap sweep to: {HEATMAP_DIR}")


# ============================================================
# SERIALIZATION HELPERS
# ============================================================
def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    return value


def params_to_dict(p: Params) -> dict:
    out = {}
    for f in fields(p):
        out[f.name] = to_serializable(getattr(p, f.name))
    return out


def make_hash(payload: dict) -> str:
    txt = json.dumps(payload, sort_keys=True)
    return hashlib.md5(txt.encode("utf-8")).hexdigest()[:16]


def save_parameter_cache_metadata(param_cache_dir: Path, param_signature: dict, param_hash: str) -> None:
    json_path = param_cache_dir / "parameter_signature.json"
    txt_path = param_cache_dir / "parameter_summary.txt"

    payload = {
        "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameter_hash": param_hash,
        "signature": param_signature,
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "PARAMETER CACHE SUMMARY",
        "=" * 60,
        f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Parameter hash: {param_hash}",
        "",
        json.dumps(param_signature, indent=2),
    ]

    txt_path.write_text("\n".join(lines), encoding="utf-8")


def make_case_cache_name(N0_mean: float, S0_mean: float) -> str:
    return f"case_N_{N0_mean:.8f}_S_{S0_mean:.8f}.npz"


def save_figure(fig, out_dir: Path, name: str, dpi: int = 300) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")


# ============================================================
# BASE PARAMETER SET
# ============================================================
p = Params()

parameter_signature = build_parameter_signature(p)
parameter_hash = make_hash(parameter_signature)

PARAM_CACHE_DIR = CACHE_ROOT / parameter_hash
RUN_CACHE_DIR = PARAM_CACHE_DIR / "runs"
SWEEP_EXPORT_DIR = PARAM_CACHE_DIR / "sweep_exports"

PARAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

save_parameter_cache_metadata(PARAM_CACHE_DIR, parameter_signature, parameter_hash)

print(f"Parameter hash: {parameter_hash}")
print(f"Parameter cache folder: {PARAM_CACHE_DIR}")


# ============================================================
# STORAGE ARRAYS
# Rows = S_range, Cols = N_range
# ============================================================
nS = len(S_range)
nN = len(N_range)


def empty_map():
    return np.full((nS, nN), np.nan)


# ------------------------------------------------------------
# Seasonal water-column maps
# These are used when p.Seasonality = True
# ------------------------------------------------------------
phyto_mean_map = empty_map()
phyto_max_map = empty_map()

PP_mean_map = empty_map()
PP_max_map = empty_map()

PDN_mean_map = empty_map()
PDN_max_map = empty_map()

PDS_mean_map = empty_map()
PDS_max_map = empty_map()

O_mean_map = empty_map()
O_min_map = empty_map()

Q_mean_map = empty_map()
Q_max_map = empty_map()

R_mean_map = empty_map()
R_max_map = empty_map()

# Upper-10 m bloom diagnostics
upper10_phyto_max_map = empty_map()
upper10_bloom_day_map = empty_map()


# ------------------------------------------------------------
# Non-seasonal surface/bottom maps
# These are used when p.Seasonality = False
# ------------------------------------------------------------
phyto_surface_mean_map = empty_map()
phyto_bottom_mean_map = empty_map()
phyto_surface_max_map = empty_map()
phyto_bottom_max_map = empty_map()

PP_surface_mean_map = empty_map()
PP_bottom_mean_map = empty_map()
PP_surface_max_map = empty_map()
PP_bottom_max_map = empty_map()

PDN_surface_mean_map = empty_map()
PDN_bottom_mean_map = empty_map()
PDN_surface_max_map = empty_map()
PDN_bottom_max_map = empty_map()

PDS_surface_mean_map = empty_map()
PDS_bottom_mean_map = empty_map()
PDS_surface_max_map = empty_map()
PDS_bottom_max_map = empty_map()

O_surface_mean_map = empty_map()
O_bottom_mean_map = empty_map()
O_surface_min_map = empty_map()
O_bottom_min_map = empty_map()

Q_surface_mean_map = empty_map()
Q_bottom_mean_map = empty_map()
Q_surface_max_map = empty_map()
Q_bottom_max_map = empty_map()

R_surface_mean_map = empty_map()
R_bottom_mean_map = empty_map()
R_surface_max_map = empty_map()
R_bottom_max_map = empty_map()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_window(sol_t: np.ndarray, p: Params):
    """
    If seasonality is ON:
        use the last 365 days.

    If seasonality is OFF:
        use final timestep only.
    """
    if p.Seasonality:
        mask = sol_t >= (p.t_max - 365.0)
        t_win = sol_t[mask] - (p.t_max - 365.0)
    else:
        mask = np.zeros_like(sol_t, dtype=bool)
        mask[-1] = True
        t_win = np.array([sol_t[-1]])

    return t_win, mask


def safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.nan
    return float(np.nanmean(x))


def safe_nanmax(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.nan
    return float(np.nanmax(x))


def safe_nanmin(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.nan
    return float(np.nanmin(x))


def safe_quota(PDS_ts, PDN_ts, threshold):
    Q = np.full_like(PDN_ts, np.nan, dtype=float)
    valid = PDN_ts > threshold
    Q[valid] = PDS_ts[valid] / PDN_ts[valid]
    return Q


def depth_mean_ts(arr_2d: np.ndarray) -> np.ndarray:
    """
    Converts depth x time array to one time series by averaging over depth.
    """
    return np.nanmean(arr_2d, axis=0)


def diagnose_case(sol, idx, p: Params, pdn_quota_threshold: float):
    """
    If seasonality is ON:
        diagnostics are based on the last 365 days.
        seasonal plots use full water-column depth-mean time series:
            mean = time mean of depth-mean value
            max  = time max of depth-mean value
            min  = time min of depth-mean value

    If seasonality is OFF:
        diagnostics are based only on the final timestep.
        non-seasonal plots still use surface and bottom values.
    """
    t_win, mask = extract_window(sol.t, p)

    PP = sol.y[idx["iPP"]][:, mask]
    PDN = sol.y[idx["iPDN"]][:, mask]
    PDS = sol.y[idx["iPDS"]][:, mask]
    O = sol.y[idx["iO"]][:, mask]

    phyto = PP + PDN

    Q = np.full_like(PDN, np.nan, dtype=float)
    valid_Q = PDN > pdn_quota_threshold
    Q[valid_Q] = PDS[valid_Q] / PDN[valid_Q]

    R = np.where(
        phyto > 1e-12,
        PDN / (phyto + 1e-12),
        np.nan,
    )

    # --------------------------------------------------------
    # Water-column mean time series
    # Used for seasonal mean/max/min maps
    # --------------------------------------------------------
    phyto_wc_ts = depth_mean_ts(phyto)
    PP_wc_ts = depth_mean_ts(PP)
    PDN_wc_ts = depth_mean_ts(PDN)
    PDS_wc_ts = depth_mean_ts(PDS)
    O_wc_ts = depth_mean_ts(O)
    Q_wc_ts = depth_mean_ts(Q)
    R_wc_ts = depth_mean_ts(R)

    # --------------------------------------------------------
    # Upper-10 m bloom diagnostics
    # Based on upper-10 m depth-averaged total phytoplankton:
    #     P_P + P_D,N
    # --------------------------------------------------------
    upper_mask = p.z <= upper_bloom_depth_m

    if not np.any(upper_mask):
        upper_mask = np.ones_like(p.z, dtype=bool)

    phyto_upper_ts = np.nanmean(phyto[upper_mask, :], axis=0)

    if np.any(np.isfinite(phyto_upper_ts)):
        idx_bloom = int(np.nanargmax(phyto_upper_ts))
        upper10_phyto_max = float(phyto_upper_ts[idx_bloom])

        if p.Seasonality:
            upper10_bloom_day = float(t_win[idx_bloom])
        else:
            upper10_bloom_day = np.nan
    else:
        upper10_phyto_max = np.nan
        upper10_bloom_day = np.nan

    # --------------------------------------------------------
    # Surface and bottom time series
    # Kept so non-seasonal behaviour stays unchanged
    # --------------------------------------------------------
    PP_surface = PP[0, :]
    PP_bottom = PP[-1, :]

    PDN_surface = PDN[0, :]
    PDN_bottom = PDN[-1, :]

    PDS_surface = PDS[0, :]
    PDS_bottom = PDS[-1, :]

    O_surface = O[0, :]
    O_bottom = O[-1, :]

    phyto_surface = PP_surface + PDN_surface
    phyto_bottom = PP_bottom + PDN_bottom

    Q_surface = safe_quota(PDS_surface, PDN_surface, pdn_quota_threshold)
    Q_bottom = safe_quota(PDS_bottom, PDN_bottom, pdn_quota_threshold)

    R_surface = PDN_surface / (PP_surface + PDN_surface + 1e-12)
    R_bottom = PDN_bottom / (PP_bottom + PDN_bottom + 1e-12)

    out = {
        # ----------------------------------------------------
        # Seasonal water-column values
        # ----------------------------------------------------
        "phyto_mean": safe_nanmean(phyto_wc_ts),
        "phyto_max": safe_nanmax(phyto_wc_ts),

        "PP_mean": safe_nanmean(PP_wc_ts),
        "PP_max": safe_nanmax(PP_wc_ts),

        "PDN_mean": safe_nanmean(PDN_wc_ts),
        "PDN_max": safe_nanmax(PDN_wc_ts),

        "PDS_mean": safe_nanmean(PDS_wc_ts),
        "PDS_max": safe_nanmax(PDS_wc_ts),

        "O_mean": safe_nanmean(O_wc_ts),
        "O_min": safe_nanmin(O_wc_ts),

        "Q_mean": safe_nanmean(Q_wc_ts),
        "Q_max": safe_nanmax(Q_wc_ts),

        "R_mean": safe_nanmean(R_wc_ts),
        "R_max": safe_nanmax(R_wc_ts),

        "upper10_phyto_max": upper10_phyto_max,
        "upper10_bloom_day": upper10_bloom_day,

        # ----------------------------------------------------
        # Non-seasonal surface/bottom values
        # ----------------------------------------------------
        "phyto_surface_mean": safe_nanmean(phyto_surface),
        "phyto_bottom_mean": safe_nanmean(phyto_bottom),
        "phyto_surface_max": safe_nanmax(phyto_surface),
        "phyto_bottom_max": safe_nanmax(phyto_bottom),

        "PP_surface_mean": safe_nanmean(PP_surface),
        "PP_bottom_mean": safe_nanmean(PP_bottom),
        "PP_surface_max": safe_nanmax(PP_surface),
        "PP_bottom_max": safe_nanmax(PP_bottom),

        "PDN_surface_mean": safe_nanmean(PDN_surface),
        "PDN_bottom_mean": safe_nanmean(PDN_bottom),
        "PDN_surface_max": safe_nanmax(PDN_surface),
        "PDN_bottom_max": safe_nanmax(PDN_bottom),

        "PDS_surface_mean": safe_nanmean(PDS_surface),
        "PDS_bottom_mean": safe_nanmean(PDS_bottom),
        "PDS_surface_max": safe_nanmax(PDS_surface),
        "PDS_bottom_max": safe_nanmax(PDS_bottom),

        "O_surface_mean": safe_nanmean(O_surface),
        "O_bottom_mean": safe_nanmean(O_bottom),
        "O_surface_min": safe_nanmin(O_surface),
        "O_bottom_min": safe_nanmin(O_bottom),

        "Q_surface_mean": safe_nanmean(Q_surface),
        "Q_bottom_mean": safe_nanmean(Q_bottom),
        "Q_surface_max": safe_nanmax(Q_surface),
        "Q_bottom_max": safe_nanmax(Q_bottom),

        "R_surface_mean": safe_nanmean(R_surface),
        "R_bottom_mean": safe_nanmean(R_bottom),
        "R_surface_max": safe_nanmax(R_surface),
        "R_bottom_max": safe_nanmax(R_bottom),
    }

    return out


def save_case_to_cache(
    run_cache_dir: Path,
    N0_mean: float,
    S0_mean: float,
    sol,
    idx: dict,
    diagnostics: dict,
    p: Params,
):
    case_path = run_cache_dir / make_case_cache_name(N0_mean, S0_mean)

    idx_serializable = {}
    for key, value in idx.items():
        if isinstance(value, slice):
            idx_serializable[key] = np.array(
                [value.start, value.stop, -1 if value.step is None else value.step],
                dtype=int
            )
        else:
            idx_serializable[key] = np.array([int(value)], dtype=int)

    np.savez_compressed(
        case_path,
        N_total_mean=float(N0_mean),
        S_total_mean=float(S0_mean),
        O_total_mean=float(p.O_total_mean),
        t=sol.t,
        y=sol.y,
        diagnostics_json=json.dumps(diagnostics),
        nz=np.array([p.nz], dtype=int),
        depth=np.array([p.depth], dtype=float),
        dz=np.array([p.dz], dtype=float),
        z=p.z,
        **idx_serializable,
    )


def load_case_from_cache(run_cache_dir: Path, N0_mean: float, S0_mean: float):
    case_path = run_cache_dir / make_case_cache_name(N0_mean, S0_mean)

    if not case_path.exists():
        return None

    data = np.load(case_path, allow_pickle=True)

    return {
        "path": case_path,
        "t": data["t"],
        "y": data["y"],
        "raw": data,
    }


# ============================================================
# MAIN SWEEP
# ============================================================
total_cases = nS * nN
case_counter = 0
n_loaded_from_cache = 0
n_solved_new = 0

sample_idx = None

for iS, S0_mean in enumerate(S_range):
    for iN, N0_mean in enumerate(N_range):
        case_counter += 1

        if verbose:
            print(f"Case {case_counter:>3}/{total_cases}: N0={N0_mean:.3f}, S0={S0_mean:.3f}")

        cached = load_case_from_cache(RUN_CACHE_DIR, float(N0_mean), float(S0_mean))

        if cached is not None:
            sol_cached = type("CachedSolution", (), {})()
            sol_cached.t = cached["t"]
            sol_cached.y = cached["y"]

            idx = build_idx_from_saved_case(cached["raw"])

            d = diagnose_case(sol_cached, idx, p, pdn_quota_threshold)

            n_loaded_from_cache += 1

            if verbose:
                print("  -> loaded solution from cache and recomputed diagnostics")

        else:
            y0, idx = make_initial_conditions(
                p,
                N_total_mean=float(N0_mean),
                S_total_mean=float(S0_mean),
                O_total_mean=p.O_total_mean,
            )

            sample_idx = idx

            sol = solve_ivp(
                lambda t, y: rhs(t, y, p, idx),
                p.t_span,
                y0,
                t_eval=p.t_eval,
                method="BDF",
                rtol=1e-7,
                atol=1e-10,
                max_step=1.0,
            )

            if not sol.success:
                print(f"  -> Solver failed for N0={N0_mean:.3f}, S0={S0_mean:.3f}")
                continue

            d = diagnose_case(sol, idx, p, pdn_quota_threshold)

            save_case_to_cache(
                RUN_CACHE_DIR,
                float(N0_mean),
                float(S0_mean),
                sol,
                idx,
                d,
                p,
            )

            n_solved_new += 1

            if verbose:
                print("  -> solved and saved to parameter-cache")

        # ----------------------------------------------------
        # Store seasonal water-column diagnostics
        # ----------------------------------------------------
        phyto_mean_map[iS, iN] = d["phyto_mean"]
        phyto_max_map[iS, iN] = d["phyto_max"]

        PP_mean_map[iS, iN] = d["PP_mean"]
        PP_max_map[iS, iN] = d["PP_max"]

        PDN_mean_map[iS, iN] = d["PDN_mean"]
        PDN_max_map[iS, iN] = d["PDN_max"]

        PDS_mean_map[iS, iN] = d["PDS_mean"]
        PDS_max_map[iS, iN] = d["PDS_max"]

        O_mean_map[iS, iN] = d["O_mean"]
        O_min_map[iS, iN] = d["O_min"]

        Q_mean_map[iS, iN] = d["Q_mean"]
        Q_max_map[iS, iN] = d["Q_max"]

        R_mean_map[iS, iN] = d["R_mean"]
        R_max_map[iS, iN] = d["R_max"]

        upper10_phyto_max_map[iS, iN] = d["upper10_phyto_max"]
        upper10_bloom_day_map[iS, iN] = d["upper10_bloom_day"]

        # ----------------------------------------------------
        # Store non-seasonal surface/bottom diagnostics
        # ----------------------------------------------------
        phyto_surface_mean_map[iS, iN] = d["phyto_surface_mean"]
        phyto_bottom_mean_map[iS, iN] = d["phyto_bottom_mean"]
        phyto_surface_max_map[iS, iN] = d["phyto_surface_max"]
        phyto_bottom_max_map[iS, iN] = d["phyto_bottom_max"]

        PP_surface_mean_map[iS, iN] = d["PP_surface_mean"]
        PP_bottom_mean_map[iS, iN] = d["PP_bottom_mean"]
        PP_surface_max_map[iS, iN] = d["PP_surface_max"]
        PP_bottom_max_map[iS, iN] = d["PP_bottom_max"]

        PDN_surface_mean_map[iS, iN] = d["PDN_surface_mean"]
        PDN_bottom_mean_map[iS, iN] = d["PDN_bottom_mean"]
        PDN_surface_max_map[iS, iN] = d["PDN_surface_max"]
        PDN_bottom_max_map[iS, iN] = d["PDN_bottom_max"]

        PDS_surface_mean_map[iS, iN] = d["PDS_surface_mean"]
        PDS_bottom_mean_map[iS, iN] = d["PDS_bottom_mean"]
        PDS_surface_max_map[iS, iN] = d["PDS_surface_max"]
        PDS_bottom_max_map[iS, iN] = d["PDS_bottom_max"]

        O_surface_mean_map[iS, iN] = d["O_surface_mean"]
        O_bottom_mean_map[iS, iN] = d["O_bottom_mean"]
        O_surface_min_map[iS, iN] = d["O_surface_min"]
        O_bottom_min_map[iS, iN] = d["O_bottom_min"]

        Q_surface_mean_map[iS, iN] = d["Q_surface_mean"]
        Q_bottom_mean_map[iS, iN] = d["Q_bottom_mean"]
        Q_surface_max_map[iS, iN] = d["Q_surface_max"]
        Q_bottom_max_map[iS, iN] = d["Q_bottom_max"]

        R_surface_mean_map[iS, iN] = d["R_surface_mean"]
        R_bottom_mean_map[iS, iN] = d["R_bottom_mean"]
        R_surface_max_map[iS, iN] = d["R_surface_max"]
        R_bottom_max_map[iS, iN] = d["R_bottom_max"]


print("")
print("Sweep finished.")
print(f"Loaded from cache: {n_loaded_from_cache}")
print(f"Solved new:        {n_solved_new}")


# ============================================================
# SAVE RAW ARRAYS
# ============================================================
sweep_export_path = SWEEP_EXPORT_DIR / f"heatmap_results_{timestamp}.npz"

save_payload = dict(
    N_range=N_range,
    S_range=S_range,

    # Seasonal water-column maps
    phyto_mean_map=phyto_mean_map,
    phyto_max_map=phyto_max_map,

    PP_mean_map=PP_mean_map,
    PP_max_map=PP_max_map,

    PDN_mean_map=PDN_mean_map,
    PDN_max_map=PDN_max_map,

    PDS_mean_map=PDS_mean_map,
    PDS_max_map=PDS_max_map,

    O_mean_map=O_mean_map,
    O_min_map=O_min_map,

    Q_mean_map=Q_mean_map,
    Q_max_map=Q_max_map,

    R_mean_map=R_mean_map,
    R_max_map=R_max_map,

    upper10_phyto_max_map=upper10_phyto_max_map,
    upper10_bloom_day_map=upper10_bloom_day_map,

    # Non-seasonal surface/bottom maps
    phyto_surface_mean_map=phyto_surface_mean_map,
    phyto_bottom_mean_map=phyto_bottom_mean_map,
    phyto_surface_max_map=phyto_surface_max_map,
    phyto_bottom_max_map=phyto_bottom_max_map,

    PP_surface_mean_map=PP_surface_mean_map,
    PP_bottom_mean_map=PP_bottom_mean_map,
    PP_surface_max_map=PP_surface_max_map,
    PP_bottom_max_map=PP_bottom_max_map,

    PDN_surface_mean_map=PDN_surface_mean_map,
    PDN_bottom_mean_map=PDN_bottom_mean_map,
    PDN_surface_max_map=PDN_surface_max_map,
    PDN_bottom_max_map=PDN_bottom_max_map,

    PDS_surface_mean_map=PDS_surface_mean_map,
    PDS_bottom_mean_map=PDS_bottom_mean_map,
    PDS_surface_max_map=PDS_surface_max_map,
    PDS_bottom_max_map=PDS_bottom_max_map,

    O_surface_mean_map=O_surface_mean_map,
    O_bottom_mean_map=O_bottom_mean_map,
    O_surface_min_map=O_surface_min_map,
    O_bottom_min_map=O_bottom_min_map,

    Q_surface_mean_map=Q_surface_mean_map,
    Q_bottom_mean_map=Q_bottom_mean_map,
    Q_surface_max_map=Q_surface_max_map,
    Q_bottom_max_map=Q_bottom_max_map,

    R_surface_mean_map=R_surface_mean_map,
    R_bottom_mean_map=R_bottom_mean_map,
    R_surface_max_map=R_surface_max_map,
    R_bottom_max_map=R_bottom_max_map,

    parameter_hash=parameter_hash,
    Seasonality=bool(p.Seasonality),
    upper_bloom_depth_m=float(upper_bloom_depth_m),
)

np.savez_compressed(
    HEATMAP_DIR / "heatmap_results.npz",
    **save_payload,
)

np.savez_compressed(
    sweep_export_path,
    **save_payload,
)


# ============================================================
# PLOTTING
# ============================================================
def make_edges(
    x: np.ndarray,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> np.ndarray:
    """
    Build pcolormesh edges from coordinate centers.

    Important fix:
    For non-seasonal runs, low-N / low-S values can look stretched if the
    first cell extends to a negative edge and is then clipped by set_xlim(0).
    So we clamp the first/last edges to the actual sampled range.
    """
    x = np.asarray(x, dtype=float)

    if len(x) == 1:
        half_width = 0.5
        edges = np.array([x[0] - half_width, x[0] + half_width], dtype=float)
    else:
        dx = np.diff(x)
        edges = np.empty(len(x) + 1, dtype=float)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * dx[0]
        edges[-1] = x[-1] + 0.5 * dx[-1]

    if clamp_min is not None:
        edges[0] = float(clamp_min)

    if clamp_max is not None:
        edges[-1] = float(clamp_max)

    return edges


# ------------------------------------------------------------
# IMPORTANT FIX:
# clamp the outer edges to the real sampled parameter range
# so the first/last cells do not visually stretch
# ------------------------------------------------------------
N_edges = make_edges(
    N_range,
    clamp_min=float(N_range[0]),
    clamp_max=float(N_range[-1]),
)

S_edges = make_edges(
    S_range,
    clamp_min=float(S_range[0]),
    clamp_max=float(S_range[-1]),
)


def get_shared_vmin_vmax(data_a, data_b):
    finite_values = np.concatenate([
        data_a[np.isfinite(data_a)].ravel(),
        data_b[np.isfinite(data_b)].ravel(),
    ])

    if finite_values.size > 0:
        return np.nanmin(finite_values), np.nanmax(finite_values)

    return None, None


def setup_heatmap_axes(ax):
    ax.set_xlabel(r"Initial total N [mmol N m$^{-3}$]")
    ax.set_ylabel(r"Initial total Si [mmol Si m$^{-3}$]")

    # Use the actual plotted domain
    ax.set_xlim(N_edges[0], N_edges[-1])
    ax.set_ylim(S_edges[0], S_edges[-1])


def plot_surface_bottom_pair(surface_data, bottom_data, cbar_label, filename):
    """
    Used for non-seasonal runs.
    """
    fig, axs = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharey=True,
        constrained_layout=True,
    )

    vmin, vmax = get_shared_vmin_vmax(surface_data, bottom_data)

    im1 = axs[0].pcolormesh(
        N_edges,
        S_edges,
        surface_data,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    im2 = axs[1].pcolormesh(
        N_edges,
        S_edges,
        bottom_data,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(
        im2,
        ax=axs,
        location="right",
        shrink=0.88,
        pad=0.02,
    )
    cbar.set_label(cbar_label)

    setup_heatmap_axes(axs[0])
    setup_heatmap_axes(axs[1])
    axs[1].set_ylabel("")

    axs[0].text(
        0.02, 0.98, "Surface",
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    axs[1].text(
        0.02, 0.98, "Bottom",
        transform=axs[1].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    save_figure(fig, HEATMAP_DIR, filename)
    plt.show()


def plot_mean_extreme_pair(mean_data, extreme_data, cbar_label, filename, extreme_label):
    """
    Used for seasonal runs.
    Shows mean and max/min side by side.
    """
    fig, axs = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharey=True,
        constrained_layout=True,
    )

    vmin, vmax = get_shared_vmin_vmax(mean_data, extreme_data)

    im1 = axs[0].pcolormesh(
        N_edges,
        S_edges,
        mean_data,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    im2 = axs[1].pcolormesh(
        N_edges,
        S_edges,
        extreme_data,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(
        im2,
        ax=axs,
        location="right",
        shrink=0.88,
        pad=0.02,
    )
    cbar.set_label(cbar_label)

    setup_heatmap_axes(axs[0])
    setup_heatmap_axes(axs[1])
    axs[1].set_ylabel("")

    axs[0].text(
        0.02, 0.98, "Mean",
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    axs[1].text(
        0.02, 0.98, extreme_label,
        transform=axs[1].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    save_figure(fig, HEATMAP_DIR, filename)
    plt.show()


def plot_metric(
    mean_data,
    extreme_data,
    surface_mean,
    bottom_mean,
    surface_extreme,
    bottom_extreme,
    cbar_label,
    filename,
    extreme_type,
):
    """
    If seasonality is ON:
        plot water-column mean and max/min.

    If seasonality is OFF:
        keep old surface/bottom final plotting behaviour.
    """
    if p.Seasonality:
        plot_mean_extreme_pair(
            mean_data,
            extreme_data,
            cbar_label,
            f"{filename}_mean_{extreme_type}",
            extreme_type.capitalize(),
        )

    else:
        plot_surface_bottom_pair(
            surface_mean,
            bottom_mean,
            cbar_label,
            f"{filename}_final_surface_bottom",
        )


def plot_upper10_bloom_pair():
    """
    Seasonal-only plot:
    left  = maximum upper-10 m phytoplankton biomass
    right = timing of that maximum during the last model year.

    The right panel uses the actual bloom day values for continuous shading,
    but the colorbar range is restricted to February-May only.
    """
    fig, axs = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharey=True,
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # Left: bloom amplitude
    # --------------------------------------------------------
    im1 = axs[0].pcolormesh(
        N_edges,
        S_edges,
        upper10_phyto_max_map,
        shading="auto",
    )

    cbar1 = fig.colorbar(
        im1,
        ax=axs[0],
        location="right",
        shrink=0.88,
        pad=0.02,
    )
    cbar1.set_label(r"Max upper-10 m $P_P + P_{D,N}$ [mmol N m$^{-3}$]")

    # --------------------------------------------------------
    # Right: bloom timing with continuous shading,
    # but only show February-May on the colorbar
    # --------------------------------------------------------
    feb_start = 31
    mar_start = 59
    apr_start = 90
    may_start = 120
    jun_start = 151

    bloom_day_plot = upper10_bloom_day_map

    im2 = axs[1].pcolormesh(
        N_edges,
        S_edges,
        bloom_day_plot,
        shading="auto",
        vmin=feb_start,
        vmax=jun_start,
    )

    month_tick_positions = [
        0.5 * (feb_start + mar_start),   # Feb
        0.5 * (mar_start + apr_start),   # Mar
        0.5 * (apr_start + may_start),   # Apr
        0.5 * (may_start + jun_start),   # May
    ]
    month_tick_labels = ["Feb", "Mar", "Apr", "May"]

    cbar2 = fig.colorbar(
        im2,
        ax=axs[1],
        location="right",
        shrink=0.88,
        pad=0.02,
        ticks=month_tick_positions,
    )
    cbar2.ax.set_yticklabels(month_tick_labels)
    cbar2.set_label("Bloom timing")

    setup_heatmap_axes(axs[0])
    setup_heatmap_axes(axs[1])
    axs[1].set_ylabel("")

    axs[0].text(
        0.02, 0.98, "Bloom amplitude",
        transform=axs[0].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    axs[1].text(
        0.02, 0.98, "Bloom timing",
        transform=axs[1].transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    save_figure(fig, HEATMAP_DIR, "08_upper10_bloom_amplitude_and_timing")
    plt.show()


# ============================================================
# MAKE PLOTS
# ============================================================

# 1) Total phytoplankton N: PP + PDN
plot_metric(
    phyto_mean_map,
    phyto_max_map,
    phyto_surface_mean_map,
    phyto_bottom_mean_map,
    phyto_surface_max_map,
    phyto_bottom_max_map,
    r"$P_P + P_{D,N}$ [mmol N m$^{-3}$]",
    "01_total_phyto_N",
    "max",
)

# 2) Non-diatom phytoplankton
plot_metric(
    PP_mean_map,
    PP_max_map,
    PP_surface_mean_map,
    PP_bottom_mean_map,
    PP_surface_max_map,
    PP_bottom_max_map,
    r"$P_P$ [mmol N m$^{-3}$]",
    "02_PP",
    "max",
)

# 3) Diatom nitrogen biomass
plot_metric(
    PDN_mean_map,
    PDN_max_map,
    PDN_surface_mean_map,
    PDN_bottom_mean_map,
    PDN_surface_max_map,
    PDN_bottom_max_map,
    r"$P_{D,N}$ [mmol N m$^{-3}$]",
    "03_PDN",
    "max",
)

# 4) Diatom silicate biomass
plot_metric(
    PDS_mean_map,
    PDS_max_map,
    PDS_surface_mean_map,
    PDS_bottom_mean_map,
    PDS_surface_max_map,
    PDS_bottom_max_map,
    r"$P_{D,S}$ [mmol Si m$^{-3}$]",
    "04_PDS",
    "max",
)

# 5) Oxygen
plot_metric(
    O_mean_map,
    O_min_map,
    O_surface_mean_map,
    O_bottom_mean_map,
    O_surface_min_map,
    O_bottom_min_map,
    r"$O_2$ [mmol m$^{-3}$]",
    "05_oxygen",
    "min",
)

# 6) Diatom Si:N quota
plot_metric(
    Q_mean_map,
    Q_max_map,
    Q_surface_mean_map,
    Q_bottom_mean_map,
    Q_surface_max_map,
    Q_bottom_max_map,
    r"$Q = P_{D,S}/P_{D,N}$ [mmol Si mmol N$^{-1}$]",
    "06_quota_Q",
    "max",
)

# 7) Diatom fraction
plot_metric(
    R_mean_map,
    R_max_map,
    R_surface_mean_map,
    R_bottom_mean_map,
    R_surface_max_map,
    R_bottom_max_map,
    r"$R = P_{D,N}/(P_P + P_{D,N})$ [-]",
    "07_diatom_fraction_R",
    "max",
)

# 8) Upper-10 m bloom amplitude and timing
if p.Seasonality:
    plot_upper10_bloom_pair()