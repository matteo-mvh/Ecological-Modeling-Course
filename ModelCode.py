# ============================================================
# START OF SCRIPT
# SNPZDO version with silicate cycle and diatoms
#
# Fixed to match the PDF notation:
#   N       = dissolved inorganic nitrogen
#   P_P     = non-diatom phytoplankton in N units
#   P_D,N   = diatom biomass in N units
#   P_D,S   = living diatom silica content
#   Z       = zooplankton
#   D_N     = nitrogen detritus
#   S       = dissolved silicate
#   D_S     = biogenic silica detritus
#   O       = dissolved oxygen
#
# ============================================================

# %% IMPORTS
from dataclasses import dataclass, field, fields
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import json
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


run_name = "PEAK_FIXED_SINGLE_SI_UPTAKE"


# ============================================================
# 1) PARAMETER DATACLASS
# ============================================================
@dataclass
class Params:
    # --------------------------------------------------------
    # 1H) INITIAL MEAN CONCENTRATIONS
    # --------------------------------------------------------
    N_total_mean: float = 11.0
    S_total_mean: float = 2.0
    O_total_mean: float = 240.0

    # --------------------------------------------------------
    # 1A) Biological rate parameters
    # --------------------------------------------------------
    mu_max_P: float = 1.2
    mu_max_D: float = 1.4
    gmax: float = 0.7

    k_L: float = 100.0
    k_N_P: float = 0.10
    k_N_D: float = 0.50
    k_S: float = 0.5
    k_O: float = 100.0

    # grazing half-saturation parameters
    k_Z_P: float = 2.0
    k_Z_D_base: float = 2.0
    k_Z_D_slope: float = 1.0

    m_P: float = 0.1
    m_D: float = 0.1
    m_Z: float = 0.1

    e_N: float = 0.3
    e_D: float = 0.3

    r_ae: float = 0.05
    r_an: float = 0.003

    # Stoichiometry / oxygen coupling
    y_P: float = 9.0
    y_N: float = 6.625

    # Silicate cycle
    r_S: float = 0.045

    # --------------------------------------------------------
    # 1A-extra) Droop-type silica quota parameters
    # --------------------------------------------------------
    Qmin: float = 0.4
    Qmax: float = 2.0
    Vmax_Si: float = 1.0

    # --------------------------------------------------------
    # 1B) Boundary / sediment parameters
    # --------------------------------------------------------
    O2_atm: float = 260.0
    k_N_bot: float = 10.0
    z_sed: float = 0.05

    # --------------------------------------------------------
    # 1C) Domain and grid
    # --------------------------------------------------------
    depth: float = 12
    nz: int = 10

    # --------------------------------------------------------
    # 1D) Sinking / advection velocities
    # order: N, P_P, P_D,N, P_D,S, Z, D_N, S, D_S, O
    # --------------------------------------------------------
    W: list[float] = field(default_factory=lambda: [
        0.0,  # N
        0.1,  # P_P
        0.3,  # P_D,N
        0.3,  # P_D,S
        0.0,  # Z
        3.0,  # D_N
        0.0,  # S
        3.0,  # D_S
        0.0,  # O
    ])

    # --------------------------------------------------------
    # 1E) Time setup
    # --------------------------------------------------------
    years: int = 3
    n_eval: int = 400

    # --------------------------------------------------------
    # 1F) Optional forcing switches
    # --------------------------------------------------------
    Lightswitch: bool = True
    Seasonality: bool = True
    bio_attenuation: bool = True
    use_quota_grazing: bool = True

    oxyg_switch: bool = True
    delta_O: float = 10.0

    # --------------------------------------------------------
    # 1G) Gas exchange scaling
    # --------------------------------------------------------
    kappa_ref: float = 10.0
    k_O_ref: float = 10.0

    # --------------------------------------------------------
    # GRID HELPERS
    # --------------------------------------------------------
    @property
    def dz(self) -> float:
        return self.depth / self.nz

    @property
    def z(self) -> np.ndarray:
        return np.linspace(self.dz / 2, self.depth - self.dz / 2, self.nz)

    @property
    def z_edges(self) -> np.ndarray:
        return np.linspace(0.0, self.depth, self.nz + 1)

    @property
    def t_max(self) -> float:
        return 365.0 * self.years

    @property
    def t_span(self) -> tuple[float, float]:
        return (0.0, self.t_max)

    @property
    def t_eval(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max, self.n_eval)


# ============================================================
# X) FILE / CACHE HELPERS
# ============================================================
def get_base_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def get_results_dir(folder_name: str = "results_files") -> Path:
    out_dir = get_base_dir() / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def build_parameter_signature(p: Params) -> dict:
    """
    Shared parameter signature with the sweep script.

    IMPORTANT:
    - excludes initial-condition means on purpose
    - case identity is handled separately by (N_total_mean, S_total_mean)
    """
    param_dict = params_to_dict(p)

    param_dict.pop("N_total_mean", None)
    param_dict.pop("S_total_mean", None)
    param_dict.pop("O_total_mean", None)

    return {
        "params": param_dict,
        "years_per_case": p.years,
        "n_eval_per_case": p.n_eval,
        "Seasonality": bool(p.Seasonality),
        "solver": {
            "method": "BDF",
            "rtol": 1e-7,
            "atol": 1e-10,
            "max_step": 1.0,
        },
    }


def make_hash(payload: dict) -> str:
    txt = json.dumps(payload, sort_keys=True)
    return hashlib.md5(txt.encode("utf-8")).hexdigest()[:16]


def save_parameter_cache_metadata(param_cache_dir: Path, param_signature: dict, parameter_hash: str) -> None:
    payload = {
        "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameter_hash": parameter_hash,
        "signature": param_signature,
    }

    (param_cache_dir / "parameter_signature.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8"
    )

    lines = [
        "PARAMETER CACHE SUMMARY",
        "=" * 60,
        f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Parameter hash: {parameter_hash}",
        "",
        json.dumps(param_signature, indent=2),
    ]

    (param_cache_dir / "parameter_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def save_params_to_txt(p: Params, run_name: str, run_dir: Path, parameter_hash: str) -> Path:
    out_path = run_dir / f"{run_name}_parameters.txt"

    lines = [
        "MODEL PARAMETER SUMMARY",
        "=" * 60,
        f"Run name: {run_name}",
        f"Parameter hash: {parameter_hash}",
        f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for f in fields(p):
        lines.append(f"{f.name}: {getattr(p, f.name)}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_params_to_json(p: Params, out_path: Path, parameter_hash: str) -> None:
    payload = {
        "parameter_hash": parameter_hash,
        "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": params_to_dict(p),
        "derived": {
            "dz": p.dz,
            "t_max": p.t_max,
            "t_span": list(p.t_span),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_figure(fig, run_dir: Path, name: str, dpi: int = 300) -> None:
    fig.savefig(run_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")


def make_case_cache_name(N0_mean: float, S0_mean: float) -> str:
    return f"case_N_{N0_mean:.8f}_S_{S0_mean:.8f}.npz"


def save_case_to_cache(
    run_cache_dir: Path,
    N0_mean: float,
    S0_mean: float,
    O0_mean: float,
    sol,
    idx: dict,
    y0: np.ndarray,
) -> Path:
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
        O_total_mean=float(O0_mean),
        t=sol.t,
        y=sol.y,
        y0=y0,
        success=np.array([sol.success], dtype=bool),
        status=np.array([sol.status], dtype=int),
        message=np.array([str(sol.message)], dtype=object),
        nfev=np.array([getattr(sol, "nfev", -1)], dtype=int),
        njev=np.array([getattr(sol, "njev", -1)], dtype=int),
        nlu=np.array([getattr(sol, "nlu", -1)], dtype=int),
        nz=np.array([idx["iN"].stop - idx["iN"].start], dtype=int),
        **idx_serializable,
    )
    return case_path


def load_case_from_cache(run_cache_dir: Path, N0_mean: float, S0_mean: float):
    case_path = run_cache_dir / make_case_cache_name(N0_mean, S0_mean)
    if not case_path.exists():
        return None

    data = np.load(case_path, allow_pickle=True)

    # --------------------------------------------------------
    # Compatibility:
    # Some cache files are produced by the sweep script and do
    # not contain y0, success, status, message, nfev, njev, nlu.
    # Therefore use safe fallbacks when keys are missing.
    # --------------------------------------------------------
    files = set(data.files)

    y0 = data["y0"] if "y0" in files else None

    success = bool(data["success"][0]) if "success" in files else True
    status = int(data["status"][0]) if "status" in files else 0
    message = str(data["message"][0]) if "message" in files else "Loaded from compatible cache without solver metadata."

    nfev = int(data["nfev"][0]) if "nfev" in files else -1
    njev = int(data["njev"][0]) if "njev" in files else -1
    nlu = int(data["nlu"][0]) if "nlu" in files else -1

    return SimpleNamespace(
        path=case_path,
        t=data["t"],
        y=data["y"],
        y0=y0,
        success=success,
        status=status,
        message=message,
        nfev=nfev,
        njev=njev,
        nlu=nlu,
        raw=data,
    )

def build_idx_from_saved_case(data) -> dict[str, slice | int]:
    def parse_entry(arr):
        arr = np.asarray(arr)
        if arr.size == 3:
            start, stop, step = [int(v) for v in arr]
            return slice(start, stop, None if step == -1 else step)
        return int(arr[0])

    keys = ["iN", "iPP", "iPDN", "iPDS", "iZ", "iDN", "iS", "iDS", "iO", "iNsed", "iDsed"]

    # backwards compatibility if an old cache used iPN
    out = {}
    for k in keys:
        cache_key = "iPN" if k == "iPP" and "iPP" not in data.files else k
        out[k] = parse_entry(data[cache_key])

    return out


# ============================================================
# 2) INITIAL CONDITIONS
# ============================================================
def make_initial_conditions(
    p: Params,
    N_total_mean: float = 11.31,
    S_total_mean: float = 2.10,
    O_total_mean: float = 240.0,
    frac_PP: float = 0.1,
    frac_PDN: float = 0.001,
    frac_Z: float = 0.01,
    frac_DN: float = 0.01,
    Q_init: float = 0.8,
    frac_DS: float = 0.2,
) -> tuple[np.ndarray, dict[str, slice | int]]:

    z = p.z
    nz = p.nz

    def scale_to_mean(shape: np.ndarray, target_mean: float) -> np.ndarray:
        if target_mean <= 0:
            return np.zeros_like(shape, dtype=float)

        shape = np.asarray(shape, dtype=float)
        shape_mean = np.mean(shape)

        if shape_mean <= 0:
            raise ValueError("Shape mean must be positive.")

        return shape * (target_mean / shape_mean)

    frac_N_diss = 1.0 - (frac_PP + frac_PDN + frac_Z + frac_DN)

    if frac_N_diss < 0:
        raise ValueError(
            f"N fractions exceed 1.0. Remaining dissolved fraction is {frac_N_diss:.6f}"
        )

    N_mean   = max(N_total_mean * frac_N_diss, 0.0)
    PP_mean  = max(N_total_mean * frac_PP, 0.0)
    PDN_mean = max(N_total_mean * frac_PDN, 0.0)
    Z_mean   = max(N_total_mean * frac_Z, 0.0)
    DN_mean  = max(N_total_mean * frac_DN, 0.0)

    PDS_mean = max(Q_init * PDN_mean, 0.0)

    S_remaining = S_total_mean - PDS_mean

    if S_remaining <= 0:
        S_mean = 0.0
        DS_mean = 0.0
    else:
        S_mean = S_remaining * (1.0 - frac_DS)
        DS_mean = S_remaining * frac_DS

    O_mean = max(O_total_mean, 0.0)

    N_shape   = 1.0 + 0.8 * (z / p.depth)
    PP_shape  = 1.0 - 0.8 * (z / p.depth)
    PDN_shape = PP_shape.copy()
    PDS_shape = PDN_shape.copy()
    Z_shape   = PP_shape.copy()
    DN_shape  = PP_shape.copy()
    S_shape   = N_shape.copy()
    DS_shape  = PDN_shape.copy()
    O_shape   = 1.0 - 0.15 * (z / p.depth)

    N0   = scale_to_mean(N_shape,   N_mean)
    PP0  = scale_to_mean(PP_shape,  PP_mean)
    PDN0 = scale_to_mean(PDN_shape, PDN_mean)
    PDS0 = scale_to_mean(PDS_shape, PDS_mean)
    Z0   = scale_to_mean(Z_shape,   Z_mean)
    DN0  = scale_to_mean(DN_shape,  DN_mean)
    S0   = scale_to_mean(S_shape,   S_mean)
    DS0  = scale_to_mean(DS_shape,  DS_mean)
    O0   = scale_to_mean(O_shape,   O_mean)

    Nsed0 = max(N0[-1] * p.z_sed, 0.0)
    Dsed0 = max(DN0[-1] * p.z_sed, 0.0)

    y0 = np.concatenate([
        N0, PP0, PDN0, PDS0, Z0, DN0, S0, DS0, O0,
        [Nsed0], [Dsed0]
    ])

    idx = {
        "iN": slice(0, nz),
        "iPP": slice(nz, 2 * nz),
        "iPDN": slice(2 * nz, 3 * nz),
        "iPDS": slice(3 * nz, 4 * nz),
        "iZ": slice(4 * nz, 5 * nz),
        "iDN": slice(5 * nz, 6 * nz),
        "iS": slice(6 * nz, 7 * nz),
        "iDS": slice(7 * nz, 8 * nz),
        "iO": slice(8 * nz, 9 * nz),
        "iNsed": 9 * nz,
        "iDsed": 9 * nz + 1,
    }

    return y0, idx


# ============================================================
# 3) HELPER FUNCTIONS
# ============================================================
def day_of_year(t: float) -> float:
    return float(t % 365.0)


def season_from_doy(doy: float) -> str:
    if doy < 90:
        return "Winter"
    if doy < 172:
        return "Spring"
    if doy < 265:
        return "Summer"
    return "Autumn"


def season_ticks(ax) -> None:
    season_positions = [0, 80, 172, 264, 355]
    season_labels = ["Winter", "Spring", "Summer", "Autumn", "Winter"]
    ax.set_xticks(season_positions)
    ax.set_xticklabels(season_labels)


def quota_SiN(PDN: np.ndarray, PDS: np.ndarray) -> np.ndarray:
    return PDS / (PDN + 1e-12)


def kZ_diatom_from_quota(Q: np.ndarray, k_base: float, k_slope: float, Qmin: float) -> np.ndarray:
    Q_excess = np.maximum(Q - Qmin, 0.0)
    return k_base + k_slope * Q_excess


# ============================================================
# 4) LIMITATION FUNCTIONS
# ============================================================
def droop_lim(Q: np.ndarray, Qmin: float) -> np.ndarray:
    return np.maximum(0.0, 1.0 - Qmin / (Q + 1e-12))


def storage_drive(Q: np.ndarray, Qmin: float, Qmax: float) -> np.ndarray:
    x = (Q - Qmin) / (Qmax - Qmin + 1e-12)
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - x**3


def get_limits(
    p: Params,
    Lz: np.ndarray,
    N: np.ndarray,
    S: np.ndarray,
    PP: np.ndarray,
    PDN: np.ndarray,
    Q: np.ndarray,
    O: np.ndarray,
) -> dict[str, np.ndarray]:

    PPp = np.maximum(PP, 0.0)
    PDNp = np.maximum(PDN, 0.0)

    L_lim = Lz / (Lz + p.k_L + 1e-12)
    N_lim_P = N / (N + p.k_N_P + 1e-12)
    N_lim_D = N / (N + p.k_N_D + 1e-12)
    S_lim = S / (S + p.k_S + 1e-12)

    Q_lim = droop_lim(Q, p.Qmin)
    S_store = storage_drive(Q, p.Qmin, p.Qmax)

    n = 1.5

    G_P = PPp**n / (PPp**n + p.k_Z_P**n + 1e-12)

    if p.use_quota_grazing:
        k_Z_D_eff = kZ_diatom_from_quota(Q, p.k_Z_D_base, p.k_Z_D_slope, p.Qmin)
    else:
        k_Z_D_eff = p.k_Z_D_base * np.ones_like(Q)

    G_D = PDNp**n / (PDNp**n + k_Z_D_eff**n + 1e-12)

    prey_total = PPp + PDNp
    theta_P = PPp / (prey_total + 1e-12)
    theta_D = PDNp / (prey_total + 1e-12)

    G_tot = theta_P * G_P + theta_D * G_D

    if p.oxyg_switch:
        O_lim = 0.5 * (1.0 + np.tanh((O - p.k_O) / (p.delta_O + 1e-12)))
    else:
        O_lim = O / (O + p.k_O + 1e-12)

    return {
        "L_lim": L_lim,
        "N_lim_P": N_lim_P,
        "N_lim_D": N_lim_D,
        "S_lim": S_lim,
        "O_lim": O_lim,
        "Q_lim": Q_lim,
        "S_store": S_store,
        "k_Z_D_eff": k_Z_D_eff,
        "G_P": G_P,
        "G_D": G_D,
        "G_tot": G_tot,
        "prey_total": prey_total,
        "theta_P": theta_P,
        "theta_D": theta_D,
    }


# ============================================================
# 5) SEASONAL FORCING: MIXING + LIGHT
# ============================================================
def getLIGHTandKAPPAS(
    p: Params,
    t: float,
    PP: np.ndarray | None = None,
    PDN: np.ndarray | None = None,
    PDS: np.ndarray | None = None,
    DN: np.ndarray | None = None,
    DS: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:

    z = p.z
    dz = p.dz
    nz = p.nz

    k_water = 0.30
    k_bio = 0.04

    if p.bio_attenuation and (PP is not None) and (PDN is not None) and (PDS is not None) and (DN is not None) and (DS is not None):
        bio_particles = (
            np.maximum(PP, 0.0)
            + np.maximum(PDN, 0.0)
            + np.maximum(PDS, 0.0)
            + np.maximum(DN, 0.0)
            + np.maximum(DS, 0.0)
        )
        bio_integral = np.cumsum(bio_particles) * dz
    else:
        bio_integral = 0.0

    if p.Seasonality:
        doy = day_of_year(t)

        zMix = 0.1 * p.depth
        zMixWinter = 0.8 * p.depth
        tMaxSpring = 90
        zetaMaxSteep = 2.0

        z_mix = (
            0.5
            * (1 - np.sin(2 * np.pi * (doy - tMaxSpring) / 365.0)) ** zetaMaxSteep
            * (zMixWinter - zMix)
            + zMix
        )

        kappa_top_summer = 7.0
        kappa_top_winter = 15.0
        kappa_bottom_summer = 2.0
        kappa_bottom_winter = 15.0

        season_shape = 0.5 * (1 - np.sin(2 * np.pi * (doy - tMaxSpring) / 365.0))

        kappa_top = kappa_top_summer + (kappa_top_winter - kappa_top_summer) * (season_shape**zetaMaxSteep)
        kappa_bottom = kappa_bottom_summer + (kappa_bottom_winter - kappa_bottom_summer) * (season_shape**zetaMaxSteep)

        zeta_mix = 5.0
        kappa_center = (
            0.5 * (1 - np.tanh((z - z_mix) / zeta_mix)) * (kappa_top - kappa_bottom)
            + kappa_bottom
        )

        L0_min = 30.0
        L0_max = 500.0

        if p.Lightswitch:
            spring_center = 120.0
            autumn_center = 230.0
            spring_width = 40.0
            autumn_width = 40.0

            spring_switch = 0.5 * (1 + np.tanh((doy - spring_center) / spring_width))
            autumn_switch = 0.5 * (1 - np.tanh((doy - autumn_center) / autumn_width))
            seasonal_shape = spring_switch * autumn_switch

            L0 = L0_min + (L0_max - L0_min) * seasonal_shape
        else:
            phase_shift = 80.0
            seasonal_shape = ((1 + np.sin(2 * np.pi * (doy - phase_shift) / 365.0)) / 2.0) ** 3
            L0 = L0_min + (L0_max - L0_min) * seasonal_shape

    else:
        kappa_surface = 10.0
        kappa_bottom = 5.0
        z_transition = 50.0
        zeta_mix = 10.0

        kappa_center = (
            0.5 * (1 - np.tanh((z - z_transition) / zeta_mix)) * (kappa_surface - kappa_bottom)
            + kappa_bottom
        )

        L0 = 1000.0

    kappa_interface = np.zeros(nz + 1)
    kappa_interface[1:nz] = 0.5 * (kappa_center[1:] + kappa_center[:-1])
    kappa_interface[0] = kappa_center[0]
    kappa_interface[nz] = kappa_center[-1]

    if p.bio_attenuation and (PP is not None) and (PDN is not None) and (PDS is not None) and (DN is not None) and (DS is not None):
        Lz = L0 * np.exp(-k_water * z - k_bio * bio_integral)
    else:
        Lz = L0 * np.exp(-k_water * z)

    return kappa_interface, Lz, L0


# ============================================================
# 6) BOUNDARY FLUXES AND VERTICAL TRANSPORT
# ============================================================
def kappa_scaled(p: Params, kappa_surface: float) -> float:
    return p.k_O_ref * (kappa_surface / p.kappa_ref)


def surface_flux(p: Params, tracer_name: str, C_surface: float, kappa_surface: float) -> float:
    if tracer_name == "O":
        k_O_surf = kappa_scaled(p, kappa_surface)
        return (p.O2_atm - C_surface) * k_O_surf
    return 0.0


def bottom_flux(p: Params, tracer_name: str, C_bottom: float, sed: float | None = None) -> float:
    if tracer_name == "N":
        Nsed = 0.0 if sed is None else sed
        return -p.k_N_bot * (Nsed / p.z_sed - C_bottom)
    elif tracer_name == "DN":
        return p.W[5] * C_bottom
    return 0.0


def vertical_transport(
    p: Params,
    C: np.ndarray,
    kappa_interface: np.ndarray,
    w: float = 0.0,
    tracer_name: str = "",
    sed: float | None = None,
) -> np.ndarray:

    nz = p.nz
    dz = p.dz

    J = np.zeros(nz + 1)

    if w >= 0:
        Ja = w * C[:-1]
    else:
        Ja = w * C[1:]

    Jd = -kappa_interface[1:nz] * (C[1:] - C[:-1]) / dz
    J[1:nz] = Ja + Jd

    J[0] = surface_flux(p, tracer_name, C[0], kappa_interface[0])
    J[nz] = bottom_flux(p, tracer_name, C[-1], sed=sed)

    return -(J[1:] - J[:-1]) / dz


# ============================================================
# 7) MODEL RIGHT-HAND SIDE
# ============================================================
def rhs(t: float, y: np.ndarray, p: Params, idx: dict[str, slice | int]) -> np.ndarray:
    N = y[idx["iN"]]
    PP = y[idx["iPP"]]
    PDN = y[idx["iPDN"]]
    PDS = y[idx["iPDS"]]
    Z = y[idx["iZ"]]
    DN = y[idx["iDN"]]
    S = y[idx["iS"]]
    DS = y[idx["iDS"]]
    O = y[idx["iO"]]
    Nsed = y[idx["iNsed"]]
    Dsed = y[idx["iDsed"]]

    Np = np.maximum(N, 0.0)
    PPp = np.maximum(PP, 0.0)
    PDNp = np.maximum(PDN, 0.0)
    PDSp = np.maximum(PDS, 0.0)
    Zp = np.maximum(Z, 0.0)
    DNp = np.maximum(DN, 0.0)
    Sp = np.maximum(S, 0.0)
    DSp = np.maximum(DS, 0.0)
    Op = np.maximum(O, 0.0)

    kappa_interface, Lz, _ = getLIGHTandKAPPAS(
        p, t, PP=PPp, PDN=PDNp, PDS=PDSp, DN=DNp, DS=DSp
    )

    Q = quota_SiN(PDNp, PDSp)

    lims = get_limits(
        p=p,
        Lz=Lz,
        N=Np,
        S=Sp,
        PP=PPp,
        PDN=PDNp,
        Q=Q,
        O=Op,
    )

    L_lim = lims["L_lim"]
    N_lim_P = lims["N_lim_P"]
    N_lim_D = lims["N_lim_D"]
    S_lim = lims["S_lim"]
    O_lim = lims["O_lim"]
    Q_lim = lims["Q_lim"]
    S_store = lims["S_store"]
    G_P = lims["G_P"]
    G_D = lims["G_D"]
    theta_P = lims["theta_P"]
    theta_D = lims["theta_D"]

    # --------------------------------------------------------
    # Production
    # F_prod,P = mu_P P_P
    # mu_P = mu_max,P min(L_lim, N_lim,P)
    # F_prod,D = mu_D P_D,N
    # mu_D = mu_max,D min(L_lim, N_lim,D, Q_lim)
    # --------------------------------------------------------
    mu_P = p.mu_max_P * np.minimum(L_lim, N_lim_P)
    mu_D = p.mu_max_D * np.minimum.reduce([L_lim, N_lim_D, Q_lim])

    Fprod_P = mu_P * PPp
    Fprod_D = mu_D * PDNp

    # --------------------------------------------------------
    # Single silica uptake flux
    # F_uptake,S = Vmax,S min(L_lim, S_lim, S_store) P_D,N
    # --------------------------------------------------------
    Fuptake_S = p.Vmax_Si * np.minimum.reduce([L_lim, S_lim, S_store]) * PDNp

    # --------------------------------------------------------
    # Grazing
    # --------------------------------------------------------
    Fgraz_P = p.gmax * theta_P * np.minimum(G_P, O_lim) * Zp
    Fgraz_D = p.gmax * theta_D * np.minimum(G_D, O_lim) * Zp
    Fgraz_tot = Fgraz_P + Fgraz_D

    Fgraz_S_living = Q * Fgraz_D
    Frecy_S = p.e_N * Fgraz_S_living
    Fdet_S = (1.0 - p.e_N) * Fgraz_S_living

    # --------------------------------------------------------
    # Mortality
    # --------------------------------------------------------
    Fmort_P = p.m_P * PPp
    Fmort_D = p.m_D * PDNp
    Fmort_Z = p.m_Z * Zp**2

    Fmort_S = Q * Fmort_D

    # --------------------------------------------------------
    # Dissolution and remineralization
    # --------------------------------------------------------
    Fdiss_S = p.r_S * DSp

    Fremin_ae_wc = p.r_ae * O_lim * DNp
    Fremin_an_wc = p.r_an * DNp
    Fremin_wc = Fremin_ae_wc + Fremin_an_wc

    ox_bot = O_lim[-1]
    Fben_ae = p.r_ae * ox_bot * Dsed
    Fben_an = p.r_an * Dsed
    Fben = Fben_ae + Fben_an

    J_DN_bot = bottom_flux(p, "DN", DN[-1])
    J_N_bot = bottom_flux(p, "N", N[-1], sed=Nsed)

    # --------------------------------------------------------
    # Reaction terms
    # --------------------------------------------------------
    dPP_reac = Fprod_P - Fgraz_P - Fmort_P
    dPDN_reac = Fprod_D - Fgraz_D - Fmort_D
    dPDS_reac = Fuptake_S - Fgraz_S_living - Fmort_S

    dZ_reac = (1.0 - p.e_N - p.e_D) * Fgraz_tot - Fmort_Z

    dS_reac = -Fuptake_S + Frecy_S + Fdiss_S
    dDS_reac = Fmort_S + Fdet_S - Fdiss_S

    dN_reac = -Fprod_P - Fprod_D + p.e_N * Fgraz_tot + Fremin_wc

    dDN_reac = (
        Fmort_P
        + Fmort_D
        + p.e_D * Fgraz_tot
        + Fmort_Z
        - Fremin_wc
    )

    dO_reac = (
        p.y_P * (Fprod_P + Fprod_D)
        - p.y_N * p.e_N * Fgraz_tot
        - p.y_N * Fremin_ae_wc
    )

    # --------------------------------------------------------
    # Transport + reactions
    # --------------------------------------------------------
    dN = vertical_transport(p, N, kappa_interface, w=p.W[0], tracer_name="N", sed=Nsed) + dN_reac
    dPP = vertical_transport(p, PP, kappa_interface, w=p.W[1], tracer_name="PP") + dPP_reac
    dPDN = vertical_transport(p, PDN, kappa_interface, w=p.W[2], tracer_name="PDN") + dPDN_reac
    dPDS = vertical_transport(p, PDS, kappa_interface, w=p.W[3], tracer_name="PDS") + dPDS_reac
    dZ = vertical_transport(p, Z, kappa_interface, w=p.W[4], tracer_name="Z") + dZ_reac
    dDN = vertical_transport(p, DN, kappa_interface, w=p.W[5], tracer_name="DN") + dDN_reac
    dS = vertical_transport(p, S, kappa_interface, w=p.W[6], tracer_name="S") + dS_reac
    dDS = vertical_transport(p, DS, kappa_interface, w=p.W[7], tracer_name="DS") + dDS_reac
    dO = vertical_transport(p, O, kappa_interface, w=p.W[8], tracer_name="O") + dO_reac

    dO[-1] -= p.y_N * Fben_ae / p.dz

    dDsed = J_DN_bot - Fben
    dNsed = J_N_bot + Fben

    return np.concatenate([
        dN, dPP, dPDN, dPDS, dZ, dDN, dS, dDS, dO,
        [dNsed], [dDsed]
    ])


# ============================================================
# DIAGNOSTIC LIMITATIONS THROUGH TIME
# ============================================================
def compute_limitations_all_times(sol, p: Params, idx: dict):
    nt = len(sol.t)
    nz = p.nz

    out = {
        "L_lim": np.zeros((nz, nt)),
        "N_lim_P": np.zeros((nz, nt)),
        "N_lim_D": np.zeros((nz, nt)),
        "S_lim": np.zeros((nz, nt)),
        "Q_lim": np.zeros((nz, nt)),
        "S_store": np.zeros((nz, nt)),
        "G_P": np.zeros((nz, nt)),
        "G_D": np.zeros((nz, nt)),
        "G_tot": np.zeros((nz, nt)),
        "k_Z_D_eff": np.zeros((nz, nt)),
        "phyto_growth_lim": np.zeros((nz, nt)),
        "diatom_growth_lim": np.zeros((nz, nt)),
        "diatom_si_uptake_lim": np.zeros((nz, nt)),
    }

    for it in range(nt):
        N_t = np.maximum(sol.y[idx["iN"], it], 0.0)
        PP_t = np.maximum(sol.y[idx["iPP"], it], 0.0)
        PDN_t = np.maximum(sol.y[idx["iPDN"], it], 0.0)
        PDS_t = np.maximum(sol.y[idx["iPDS"], it], 0.0)
        DN_t = np.maximum(sol.y[idx["iDN"], it], 0.0)
        DS_t = np.maximum(sol.y[idx["iDS"], it], 0.0)
        O_t = np.maximum(sol.y[idx["iO"], it], 0.0)
        S_t = np.maximum(sol.y[idx["iS"], it], 0.0)

        _, Lz_t, _ = getLIGHTandKAPPAS(
            p,
            sol.t[it],
            PP=PP_t,
            PDN=PDN_t,
            PDS=PDS_t,
            DN=DN_t,
            DS=DS_t,
        )

        Q_t = quota_SiN(PDN_t, PDS_t)

        lims_t = get_limits(
            p=p,
            Lz=Lz_t,
            N=N_t,
            S=S_t,
            PP=PP_t,
            PDN=PDN_t,
            Q=Q_t,
            O=O_t,
        )

        for key in ["L_lim", "N_lim_P", "N_lim_D", "S_lim", "Q_lim", "S_store", "G_P", "G_D", "G_tot", "k_Z_D_eff"]:
            out[key][:, it] = lims_t[key]

        out["phyto_growth_lim"][:, it] = np.minimum(lims_t["L_lim"], lims_t["N_lim_P"])
        out["diatom_growth_lim"][:, it] = np.minimum.reduce([
            lims_t["L_lim"],
            lims_t["N_lim_D"],
            lims_t["Q_lim"],
        ])
        out["diatom_si_uptake_lim"][:, it] = np.minimum.reduce([
            lims_t["L_lim"],
            lims_t["S_lim"],
            lims_t["S_store"],
        ])

    return out


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":

    p = Params()

    parameter_signature = build_parameter_signature(p)
    PARAMETER_HASH = make_hash(parameter_signature)

    RESULTS_DIR = get_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = RESULTS_DIR / f"{run_name}_run_{timestamp}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    CACHE_DIR = RESULTS_DIR / "cached_runs"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    PARAM_CACHE_DIR = CACHE_DIR / PARAMETER_HASH
    RUN_CACHE_DIR = PARAM_CACHE_DIR / "runs"
    SWEEP_EXPORT_DIR = PARAM_CACHE_DIR / "sweep_exports"

    PARAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    save_parameter_cache_metadata(PARAM_CACHE_DIR, parameter_signature, PARAMETER_HASH)

    print(f"Run folder: {RUN_DIR}")
    print(f"Parameter hash: {PARAMETER_HASH}")
    print(f"Parameter cache folder: {PARAM_CACHE_DIR}")
    print(f"Run cache folder: {RUN_CACHE_DIR}")

    param_txt_file = save_params_to_txt(p, run_name, RUN_DIR, PARAMETER_HASH)
    save_params_to_json(p, RUN_DIR / f"{run_name}_parameters.json", PARAMETER_HASH)
    print(f"Saved parameter file to: {param_txt_file}")

    y0, idx = make_initial_conditions(
        p,
        N_total_mean=p.N_total_mean,
        S_total_mean=p.S_total_mean,
        O_total_mean=p.O_total_mean,
    )

    cached_case = load_case_from_cache(RUN_CACHE_DIR, p.N_total_mean, p.S_total_mean)

    # ============================================================
    # 8) RUN SIMULATION OR LOAD CACHED RESULT
    # ============================================================
    if cached_case is not None:
        print("Cached case found for this parameter set and initial means.")
        print("Loading cached solution instead of rerunning solve_ivp...")
        sol = cached_case
        idx = build_idx_from_saved_case(cached_case.raw)
    else:
        print("No cached case found for this parameter set and initial means.")
        print("Running solve_ivp...")

        sol_ivp = solve_ivp(
            lambda t, y: rhs(t, y, p, idx),
            p.t_span,
            y0,
            t_eval=p.t_eval,
            method="BDF",
            rtol=1e-7,
            atol=1e-10,
            max_step=1.0,
        )

        if not sol_ivp.success:
            raise RuntimeError(f"solve_ivp failed: {sol_ivp.message}")

        case_path = save_case_to_cache(
            RUN_CACHE_DIR,
            p.N_total_mean,
            p.S_total_mean,
            p.O_total_mean,
            sol_ivp,
            idx,
            y0,
        )
        print(f"Saved case to cache: {case_path}")

        sol = SimpleNamespace(
            t=sol_ivp.t,
            y=sol_ivp.y,
            success=sol_ivp.success,
            status=sol_ivp.status,
            message=sol_ivp.message,
            nfev=getattr(sol_ivp, "nfev", -1),
            njev=getattr(sol_ivp, "njev", -1),
            nlu=getattr(sol_ivp, "nlu", -1),
        )

    # ============================================================
    # 10) PLOTTING SETTINGS
    # ============================================================
    plot_top = False
    top_m = 50.0

    if plot_top:
        z_mask = p.z <= top_m
    else:
        z_mask = np.ones_like(p.z, dtype=bool)

    z_plot = p.z[z_mask]

    z_kappa_full = p.z_edges
    if plot_top:
        z_kappa_mask = z_kappa_full <= top_m
    else:
        z_kappa_mask = np.ones_like(z_kappa_full, dtype=bool)

    z_kappa_plot = z_kappa_full[z_kappa_mask]

    if plot_top:
        z_edges_plot = p.z_edges[:np.sum(z_mask) + 1]
    else:
        z_edges_plot = p.z_edges

    # ============================================================
    # 11) HEATMAPS FOR FINAL MODEL YEAR
    # ============================================================
    only_last365 = True

    if only_last365:
        t_mask = sol.t >= (p.t_max - 365.0)
        t_plot = sol.t[t_mask] - (p.t_max - 365.0)
    else:
        t_mask = slice(None)
        t_plot = sol.t

    dt = t_plot[1] - t_plot[0]
    t_edges = np.linspace(t_plot[0] - dt / 2, t_plot[-1] + dt / 2, len(t_plot) + 1)

    N_all = sol.y[idx["iN"]][:, t_mask][z_mask, :]
    PP_all = sol.y[idx["iPP"]][:, t_mask][z_mask, :]
    PDN_all = sol.y[idx["iPDN"]][:, t_mask][z_mask, :]
    DN_all = sol.y[idx["iDN"]][:, t_mask][z_mask, :]
    S_all = sol.y[idx["iS"]][:, t_mask][z_mask, :]
    DS_all = sol.y[idx["iDS"]][:, t_mask][z_mask, :]

    fig_hm, axs_hm = plt.subplots(2, 3, figsize=(13, 7), sharex=True)

    im0 = axs_hm[0, 0].pcolormesh(t_edges, z_edges_plot, N_all, shading="auto")
    axs_hm[0, 0].set_title(r"$N$ [mmol N m$^{-3}$]")
    axs_hm[0, 0].set_ylabel("Depth [m]")
    fig_hm.colorbar(im0, ax=axs_hm[0, 0])

    im1 = axs_hm[0, 1].pcolormesh(t_edges, z_edges_plot, PP_all, shading="auto")
    axs_hm[0, 1].set_title(r"$P_P$ [mmol N m$^{-3}$]")
    fig_hm.colorbar(im1, ax=axs_hm[0, 1])

    im2 = axs_hm[0, 2].pcolormesh(t_edges, z_edges_plot, PDN_all, shading="auto")
    axs_hm[0, 2].set_title(r"$P_{D,N}$ [mmol N m$^{-3}$]")
    fig_hm.colorbar(im2, ax=axs_hm[0, 2])

    im3 = axs_hm[1, 0].pcolormesh(t_edges, z_edges_plot, DN_all, shading="auto")
    axs_hm[1, 0].set_title(r"$D_N$ [mmol N m$^{-3}$]")
    axs_hm[1, 0].set_ylabel("Depth [m]")
    fig_hm.colorbar(im3, ax=axs_hm[1, 0])

    im4 = axs_hm[1, 1].pcolormesh(t_edges, z_edges_plot, S_all, shading="auto")
    axs_hm[1, 1].set_title(r"$S$ [mmol Si m$^{-3}$]")
    fig_hm.colorbar(im4, ax=axs_hm[1, 1])

    im5 = axs_hm[1, 2].pcolormesh(t_edges, z_edges_plot, DS_all, shading="auto")
    axs_hm[1, 2].set_title(r"$D_S$ [mmol Si m$^{-3}$]")
    fig_hm.colorbar(im5, ax=axs_hm[1, 2])

    for ax in axs_hm.ravel():
        ax.invert_yaxis()

    if only_last365:
        for ax in axs_hm[1, :]:
            season_ticks(ax)
            ax.set_xlabel("Season")
    else:
        for ax in axs_hm[1, :]:
            ax.set_xlabel("Time [days]")

    plt.tight_layout()
    save_figure(fig_hm, RUN_DIR, "11_heatmaps_final_year")
    plt.show()

    # ============================================================
    # 13) SEASONAL P_P, P_D,N, N, S PROFILES
    # ============================================================
    season_days_pzd = {
        "Winter": 0,
        "Spring": 90,
        "Summer": 172,
        "Autumn": 260,
    }

    max_val = 0.0
    for season, day in season_days_pzd.items():
        t_target = p.t_max - 365 + day
        idx_t = np.argmin(np.abs(sol.t - t_target))

        PP_prof = sol.y[idx["iPP"], idx_t][z_mask]
        PDN_prof = sol.y[idx["iPDN"], idx_t][z_mask]
        N_prof = sol.y[idx["iN"], idx_t][z_mask]
        S_prof = sol.y[idx["iS"], idx_t][z_mask]

        max_val = max(max_val, PP_prof.max(), PDN_prof.max(), N_prof.max(), S_prof.max())

    fig_seas, axs_seas = plt.subplots(1, 4, figsize=(13, 6), sharey=True)

    for i, (season, day) in enumerate(season_days_pzd.items()):
        t_target = p.t_max - 365 + day
        idx_t = np.argmin(np.abs(sol.t - t_target))

        PP_prof = sol.y[idx["iPP"], idx_t][z_mask]
        PDN_prof = sol.y[idx["iPDN"], idx_t][z_mask]
        N_prof = sol.y[idx["iN"], idx_t][z_mask]
        S_prof = sol.y[idx["iS"], idx_t][z_mask]

        axs_seas[i].plot(PP_prof, z_plot, lw=2, label=r"$P_P$")
        axs_seas[i].plot(PDN_prof, z_plot, lw=2, label=r"$P_{D,N}$")
        axs_seas[i].plot(N_prof, z_plot, lw=2, label=r"$N$")
        axs_seas[i].plot(S_prof, z_plot, lw=2, label=r"$S$")

        axs_seas[i].set_title(season)
        axs_seas[i].set_xlim(0, max_val * 1.05)
        axs_seas[i].grid(True)

    axs_seas[0].invert_yaxis()
    axs_seas[-1].legend()

    for ax in axs_seas:
        ax.set_ylim(p.depth, 0)
        ax.set_xlim(1e-1, 1e2)
        ax.set_xscale("log")

    fig_seas.supxlabel(r"Concentration [mmol m$^{-3}$]")
    fig_seas.supylabel("Depth [m]")

    plt.tight_layout()
    save_figure(fig_seas, RUN_DIR, "13_seasonal_profiles")
    plt.show()

    # ============================================================
    # 14) SEASONAL FORCING AND OXYGEN PROFILES
    # ============================================================
    season_days = {"Winter": 0, "Spring": 90, "Summer": 172, "Autumn": 265}

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0, 0:2])
    for season, day in season_days.items():
        kappa_prof_full, _, _ = getLIGHTandKAPPAS(p, day)
        kappa_prof = kappa_prof_full[z_kappa_mask]
        ax1.plot(kappa_prof, z_kappa_plot, label=season)
    ax1.set_xlabel(r"$\kappa$ [m$^2$ d$^{-1}$]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title(r"Seasonal $\kappa$ profiles")
    ax1.invert_yaxis()
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2:4])
    for season, day in season_days.items():
        t_target = p.t_max - 365.0 + day
        idx_t = np.argmin(np.abs(sol.t - t_target))
        _, L_profile_full, _ = getLIGHTandKAPPAS(
            p,
            sol.t[idx_t],
            PP=sol.y[idx["iPP"], idx_t],
            PDN=sol.y[idx["iPDN"], idx_t],
            PDS=sol.y[idx["iPDS"], idx_t],
            DN=sol.y[idx["iDN"], idx_t],
            DS=sol.y[idx["iDS"], idx_t],
        )
        L_profile = L_profile_full[z_mask]
        ax2.plot(L_profile, z_plot, label=season)
    ax2.set_xlabel(r"Light [$\mu$mol m$^{-2}$ s$^{-1}$]")
    ax2.set_ylabel("Depth [m]")
    ax2.set_title("Seasonal light profiles")
    ax2.invert_yaxis()
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 4:6])
    o2_max = 0.0
    for season, day in season_days.items():
        t_target = p.t_max - 365.0 + day
        idx_t = np.argmin(np.abs(sol.t - t_target))
        O_profile = sol.y[idx["iO"], idx_t][z_mask]
        ax3.plot(O_profile, z_plot, label=season)
        o2_max = max(o2_max, np.max(O_profile))
    ax3.axvline(p.k_O, color="black", linestyle="--", linewidth=1, label=r"critical O$_2$")
    ax3.set_xlabel(r"$O$ [mmol O$_2$ m$^{-3}$]")
    ax3.set_ylabel("Depth [m]")
    ax3.set_title(r"Seasonal $O_2$ profiles")
    ax3.set_xlim(0, o2_max * 1.05)
    ax3.invert_yaxis()
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0:3])
    days = np.arange(0, 365)
    L0_year = np.array([getLIGHTandKAPPAS(p, d)[2] for d in days])
    ax4.plot(days, L0_year)
    season_ticks(ax4)
    ax4.set_xlabel("Season")
    ax4.set_ylabel(r"Surface light $L_0$ [$\mu$mol m$^{-2}$ s$^{-1}$]")
    ax4.set_title("Seasonal surface light")
    ax4.grid(True)

    ax5 = fig.add_subplot(gs[1, 3:6])
    kappa_surface_year = np.array([getLIGHTandKAPPAS(p, d)[0][0] for d in days])
    k_O_surf_year = p.k_O_ref * (kappa_surface_year / p.kappa_ref)
    ax5.plot(days, k_O_surf_year, color="black")
    season_ticks(ax5)
    ax5.set_xlabel("Season")
    ax5.set_ylabel(r"$k_{O,\mathrm{surf}}$ [m d$^{-1}$]")
    ax5.set_title(r"Surface gas exchange $k_{O,\mathrm{surf}}$")
    ax5.grid(True)

    plt.tight_layout()
    save_figure(fig, RUN_DIR, "14_seasonal_forcing_oxygen")
    plt.show()

    # ============================================================
    # 15) DEPTH-INTEGRATED P_P & P_D,N LAST YEAR
    # ============================================================
    PP_all_time = sol.y[idx["iPP"], :]
    PDN_all_time = sol.y[idx["iPDN"], :]

    PP_int = np.sum(PP_all_time, axis=0) * p.dz
    PDN_int = np.sum(PDN_all_time, axis=0) * p.dz
    Ptot_int = PP_int + PDN_int

    t_mask = sol.t >= (p.t_max - 365.0)
    t_plot = sol.t[t_mask] - (p.t_max - 365.0)

    fig_15 = plt.figure(figsize=(12, 5))
    plt.plot(t_plot, PP_int[t_mask], lw=2, label=r"$P_P$")
    plt.plot(t_plot, PDN_int[t_mask], lw=2, label=r"$P_{D,N}$")
    plt.plot(t_plot, Ptot_int[t_mask], lw=2.5, ls="--", color="black", label=r"$P_P + P_{D,N}$")

    season_ticks(plt.gca())
    plt.xlabel("Season")
    plt.ylabel(r"Depth-integrated biomass [mmol N m$^{-2}$]")
    plt.title("Depth-integrated phytoplankton biomass, last year")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure(fig_15, RUN_DIR, "15_depth_integrated_last_year")
    plt.show()
    
    # ============================================================
    # 15B) FULL TIME SERIES OF DEPTH-INTEGRATED P_P AND P_D,N
    # ============================================================
    PP_all_time = sol.y[idx["iPP"], :]
    PDN_all_time = sol.y[idx["iPDN"], :]
    
    PP_int_full = np.sum(PP_all_time, axis=0) * p.dz
    PDN_int_full = np.sum(PDN_all_time, axis=0) * p.dz
    Ptot_int_full = PP_int_full + PDN_int_full
    
    fig_15b = plt.figure(figsize=(12, 5))
    plt.plot(sol.t, PP_int_full, lw=2, label=r"$P_P$")
    plt.plot(sol.t, PDN_int_full, lw=2, label=r"$P_{D,N}$")
    plt.plot(sol.t, Ptot_int_full, lw=2.5, ls="--", color="black", label=r"$P_P + P_{D,N}$")
    
    plt.xlabel("Time [days]")
    plt.ylabel(r"Depth-integrated biomass [mmol N m$^{-2}$]")
    plt.title("Full time series of depth-integrated phytoplankton biomass")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure(fig_15b, RUN_DIR, "15B_full_time_series_phytoplankton_diatom")
    plt.show()
    
    # ============================================================
    # 15C) SAVE DIATOM DEPTH PROFILE AT FIXED SUMMER DAY
    # ============================================================
    # Fixed summer day in final-year day coordinates
    # Summer is defined as day 172 to 265 in season_from_doy().
    fixed_summer_day = 200.0
   
    last_year_start = p.t_max - 365.0
    target_time = last_year_start + fixed_summer_day
   
    # Use the closest available model output time
    fixed_idx = np.argmin(np.abs(sol.t - target_time))
   
    actual_time = sol.t[fixed_idx]
    actual_day_final_year = actual_time - last_year_start
   
    PDN_profile_fixed = sol.y[idx["iPDN"], fixed_idx]
   
    diatom_depth_out = np.column_stack([
        p.z,
        PDN_profile_fixed
    ])
   
    # Save outside the specific run folder, but in the same parent directory as RUN_DIR
    diatom_depth_dir = RUN_DIR.parent / "DiatomOverDepth"
    diatom_depth_dir.mkdir(parents=True, exist_ok=True)
   
    diatom_depth_file = diatom_depth_dir / f"DiatomOverDepth_{p.nz}.txt"
   
    header = (
        f"Diatom depth profile at fixed summer day\n"
        f"Run name: {run_name}\n"
        f"Parameter hash: {PARAMETER_HASH}\n"
        f"nz: {p.nz}\n"
        f"Depth: {p.depth} m\n"
        f"Requested fixed summer day: {fixed_summer_day:.6f}\n"
        f"Closest model time: {actual_time:.6f} days\n"
        f"Actual day in final year: {actual_day_final_year:.6f}\n"
        f"Depth-integrated P_D,N at selected day: {PDN_int_full[fixed_idx]:.10e} mmol N m^-2\n"
        f"Columns: depth_m, P_DN_mmol_N_m-3"
    )
   
    np.savetxt(
        diatom_depth_file,
        diatom_depth_out,
        header=header,
        comments="# ",
        fmt="%.10e",
        delimiter="\t"
    )
   
    print(f"Saved diatom depth profile at fixed summer day to: {diatom_depth_file}")
     
    # ============================================================
    # 16) YEARLY MEAN DIFFERENCE DIAGNOSTIC
    # ============================================================
    N_int_full = np.sum(sol.y[idx["iN"], :], axis=0) * p.dz
    PP_int_full = np.sum(sol.y[idx["iPP"], :], axis=0) * p.dz
    PDN_int_full = np.sum(sol.y[idx["iPDN"], :], axis=0) * p.dz
    PDS_int_full = np.sum(sol.y[idx["iPDS"], :], axis=0) * p.dz
    Z_int_full = np.sum(sol.y[idx["iZ"], :], axis=0) * p.dz
    DN_int_full = np.sum(sol.y[idx["iDN"], :], axis=0) * p.dz

    Ptot_int_full = PP_int_full + PDN_int_full
    TotalN_int_full = N_int_full + PP_int_full + PDN_int_full + Z_int_full + DN_int_full

    year_numbers = np.arange(1, p.years)

    N_year_mean = np.zeros(p.years - 1)
    PP_year_mean = np.zeros(p.years - 1)
    PDN_year_mean = np.zeros(p.years - 1)
    PDS_year_mean = np.zeros(p.years - 1)
    Z_year_mean = np.zeros(p.years - 1)
    DN_year_mean = np.zeros(p.years - 1)
    Ptot_year_mean = np.zeros(p.years - 1)
    TotalN_year_mean = np.zeros(p.years - 1)

    for yr in range(p.years - 1):
        t0 = yr * 365.0
        t1 = (yr + 1) * 365.0
        mask = (sol.t >= t0) & (sol.t < t1)

        N_year_mean[yr] = np.mean(N_int_full[mask])
        PP_year_mean[yr] = np.mean(PP_int_full[mask])
        PDN_year_mean[yr] = np.mean(PDN_int_full[mask])
        PDS_year_mean[yr] = np.mean(PDS_int_full[mask])
        Z_year_mean[yr] = np.mean(Z_int_full[mask])
        DN_year_mean[yr] = np.mean(DN_int_full[mask])
        Ptot_year_mean[yr] = np.mean(Ptot_int_full[mask])
        TotalN_year_mean[yr] = np.mean(TotalN_int_full[mask])

    dN = np.diff(N_year_mean)
    dPP = np.diff(PP_year_mean)
    dPDN = np.diff(PDN_year_mean)
    dPDS = np.diff(PDS_year_mean)
    dZ = np.diff(Z_year_mean)
    dDN = np.diff(DN_year_mean)
    dTotal = np.diff(TotalN_year_mean)

    years_diff = year_numbers[1:]
    labels = [f"{i}->{i+1}" for i in range(1, p.years - 1)]

    fig_16 = plt.figure(figsize=(12, 6))
    plt.plot(years_diff, dN, marker="o", lw=2, label=r"$\Delta N$")
    plt.plot(years_diff, dPP, marker="o", lw=2, label=r"$\Delta P_P$")
    plt.plot(years_diff, dPDN, marker="o", lw=2, label=r"$\Delta P_{D,N}$")
    plt.plot(years_diff, dPDS, marker="o", lw=2, label=r"$\Delta P_{D,S}$")
    plt.plot(years_diff, dZ, marker="o", lw=2, label=r"$\Delta Z$")
    plt.plot(years_diff, dDN, marker="o", lw=2, label=r"$\Delta D_N$")
    plt.plot(years_diff, dTotal, marker="o", lw=2, ls="--", color="black", label=r"$\Delta$ Total N")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Year interval")
    plt.ylabel("Year-to-year change")
    if len(years_diff) > 0:
        plt.xticks(years_diff, labels)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure(fig_16, RUN_DIR, "16_yearly_mean_difference")
    plt.show()

    # ============================================================
    # 17) THEORETICAL LIMITATION CURVES
    # ============================================================
    L_range = np.linspace(0, 1000, 200)
    N_range = np.linspace(0, 30, 200)
    S_range = np.linspace(0, 30, 200)
    P_range = np.linspace(0, 10, 200)
    O_range = np.linspace(0, 400, 200)
    Q_range = np.linspace(max(1e-4, 0.1 * p.Qmin), 2.0 * p.Qmax, 200)

    L_lim_theory = L_range / (L_range + p.k_L + 1e-12)
    N_lim_P_theory = N_range / (N_range + p.k_N_P + 1e-12)
    N_lim_D_theory = N_range / (N_range + p.k_N_D + 1e-12)
    S_lim_theory = S_range / (S_range + p.k_S + 1e-12)

    n = 1.5
    G_P_theory = P_range**n / (P_range**n + p.k_Z_P**n + 1e-12)

    Q_low = p.Qmin
    Q_high = p.Qmax
    kZ_D_low = kZ_diatom_from_quota(np.array([Q_low]), p.k_Z_D_base, p.k_Z_D_slope, p.Qmin)[0]
    kZ_D_high = kZ_diatom_from_quota(np.array([Q_high]), p.k_Z_D_base, p.k_Z_D_slope, p.Qmin)[0]

    G_D_low_theory = P_range**n / (P_range**n + kZ_D_low**n + 1e-12)
    G_D_high_theory = P_range**n / (P_range**n + kZ_D_high**n + 1e-12)

    if p.oxyg_switch:
        O_lim_theory = 0.5 * (1.0 + np.tanh((O_range - p.k_O) / (p.delta_O + 1e-12)))
    else:
        O_lim_theory = O_range / (O_range + p.k_O + 1e-12)

    Q_lim_theory = droop_lim(Q_range, p.Qmin)
    S_store_theory = storage_drive(Q_range, p.Qmin, p.Qmax)

    fig_lim, axs_lim = plt.subplots(3, 2, figsize=(12, 10))
    axs_lim = axs_lim.ravel()

    axs_lim[0].plot(L_range, L_lim_theory, lw=2)
    axs_lim[0].axvline(p.k_L, color="black", ls="--", lw=1.5, label=fr"$k_L={p.k_L}$")
    axs_lim[0].axhline(0.5, color="gray", ls=":", lw=1)
    axs_lim[0].plot(p.k_L, 0.5, "o", color="black")
    axs_lim[0].set_xlabel(r"Light $L$ [$\mu$mol photons m$^{-2}$ s$^{-1}$]")
    axs_lim[0].set_ylabel("Limitation [-]")
    axs_lim[0].set_title("Light limitation")
    axs_lim[0].set_ylim(0, 1.05)
    axs_lim[0].grid(True)
    axs_lim[0].legend()

    axs_lim[1].plot(N_range, N_lim_P_theory, lw=2, label=fr"$P_P$ ($k_{{N,P}}={p.k_N_P}$)")
    axs_lim[1].plot(N_range, N_lim_D_theory, lw=2, ls="--", label=fr"$P_{{D,N}}$ ($k_{{N,D}}={p.k_N_D}$)")
    axs_lim[1].axvline(p.k_N_P, color="black", ls="--", lw=1.2)
    axs_lim[1].axvline(p.k_N_D, color="gray", ls=":", lw=1.2)
    axs_lim[1].axhline(0.5, color="gray", ls=":", lw=1)
    axs_lim[1].set_xlabel(r"Nitrogen $N$ [mmol N m$^{-3}$]")
    axs_lim[1].set_ylabel("Limitation [-]")
    axs_lim[1].set_title("Nitrogen limitation")
    axs_lim[1].set_ylim(0, 1.05)
    axs_lim[1].grid(True)
    axs_lim[1].legend()

    axs_lim[2].plot(S_range, S_lim_theory, lw=2)
    axs_lim[2].axvline(p.k_S, color="black", ls="--", lw=1.5, label=fr"$k_S={p.k_S}$")
    axs_lim[2].axhline(0.5, color="gray", ls=":", lw=1)
    axs_lim[2].plot(p.k_S, 0.5, "o", color="black")
    axs_lim[2].set_xlabel(r"Silicate $S$ [mmol Si m$^{-3}$]")
    axs_lim[2].set_ylabel("Limitation [-]")
    axs_lim[2].set_title("Silicate limitation")
    axs_lim[2].set_ylim(0, 1.05)
    axs_lim[2].grid(True)
    axs_lim[2].legend()

    axs_lim[3].plot(P_range, G_P_theory, lw=2, label=r"$G_P$")
    axs_lim[3].plot(P_range, G_D_low_theory, lw=2, ls="--", label=fr"$G_D$, Q={Q_low:.2f}")
    axs_lim[3].plot(P_range, G_D_high_theory, lw=2, ls=":", label=fr"$G_D$, Q={Q_high:.2f}")
    axs_lim[3].set_xlabel(r"Prey biomass [mmol N m$^{-3}$]")
    axs_lim[3].set_ylabel("Limitation [-]")
    axs_lim[3].set_title("Grazing limitation")
    axs_lim[3].set_ylim(0, 1.05)
    axs_lim[3].grid(True)
    axs_lim[3].legend()

    axs_lim[4].plot(Q_range, Q_lim_theory, lw=2, color="black", label=r"$Q_{\mathrm{lim}}$")
    axs_lim[4].plot(Q_range, S_store_theory, lw=2, ls="--", color="gray", label=r"$S_{\mathrm{store}}$")
    axs_lim[4].axvline(p.Qmin, color="black", ls="--", lw=1.5, label=fr"$Q_{{min}}={p.Qmin}$")
    axs_lim[4].axvline(p.Qmax, color="gray", ls=":", lw=1.5, label=fr"$Q_{{max}}={p.Qmax}$")
    axs_lim[4].set_xlabel(r"Quota $Q = P_{D,S}/P_{D,N}$ [mmol Si mmol N$^{-1}$]")
    axs_lim[4].set_ylabel("Limitation [-]")
    axs_lim[4].set_title("Quota limitation and storage capacity")
    axs_lim[4].set_ylim(0, 1.05)
    axs_lim[4].grid(True)
    axs_lim[4].legend()

    axs_lim[5].plot(O_range, O_lim_theory, lw=2, color="black")
    axs_lim[5].axvline(p.k_O, color="black", ls="--", lw=1.5, label=fr"$k_O={p.k_O}$")
    axs_lim[5].axhline(0.5, color="gray", ls=":", lw=1)
    axs_lim[5].plot(p.k_O, 0.5, "o", color="black")
    axs_lim[5].set_xlabel(r"Oxygen $O$ [mmol O$_2$ m$^{-3}$]")
    axs_lim[5].set_ylabel("Limitation [-]")
    axs_lim[5].set_title("Oxygen limitation")
    axs_lim[5].set_ylim(0, 1.05)
    axs_lim[5].grid(True)
    axs_lim[5].legend()

    plt.tight_layout()
    save_figure(fig_lim, RUN_DIR, "17_theoretical_limitation_curves")
    plt.show()

    # ============================================================
    # 18) QUOTA AND DIATOM BIOMASS FRACTION HEATMAPS
    # ============================================================
    only_last365 = p.Seasonality

    PP_all_time = sol.y[idx["iPP"], :]
    PDN_all_time = sol.y[idx["iPDN"], :]
    PDS_all_time = sol.y[idx["iPDS"], :]

    Q_all_time = np.where(
        PDN_all_time > 1e-8,
        PDS_all_time / (PDN_all_time + 1e-12),
        np.nan
    )

    R_all_time = np.where(
        (PP_all_time + PDN_all_time) > 1e-8,
        PDN_all_time / (PP_all_time + PDN_all_time + 1e-12),
        np.nan
    )

    if only_last365:
        t_mask = sol.t >= (p.t_max - 365.0)
        t_plot = sol.t[t_mask] - (p.t_max - 365.0)
        Q_plot = Q_all_time[:, t_mask]
        R_plot = R_all_time[:, t_mask]
    else:
        t_plot = sol.t
        Q_plot = Q_all_time
        R_plot = R_all_time

    dt = t_plot[1] - t_plot[0]
    t_edges = np.linspace(t_plot[0] - dt / 2, t_plot[-1] + dt / 2, len(t_plot) + 1)

    fig_18, axs_18 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    im_q = axs_18[0].pcolormesh(
        t_edges, p.z_edges, Q_plot, shading="auto",
        vmin=np.nanmin(Q_plot), vmax=np.nanmax(Q_plot)
    )
    axs_18[0].set_ylabel("Depth [m]")
    axs_18[0].set_title(r"Living diatom quota $Q=P_{D,S}/P_{D,N}$")
    fig_18.colorbar(im_q, ax=axs_18[0], label=r"$Q$ [mmol Si mmol N$^{-1}$]")

    im_r = axs_18[1].pcolormesh(
        t_edges, p.z_edges, R_plot, shading="auto",
        vmin=np.nanmin(R_plot), vmax=np.nanmax(R_plot)
    )
    axs_18[1].set_title("Diatom biomass fraction")
    fig_18.colorbar(im_r, ax=axs_18[1], label=r"$R = \frac{P_{D,N}}{P_P + P_{D,N}}$ [-]")

    for ax in axs_18:
        ax.set_ylim(p.z_edges[-1], p.z_edges[0])

    if only_last365:
        for ax in axs_18:
            season_ticks(ax)
            ax.set_xlabel("Season")
    else:
        for ax in axs_18:
            ax.set_xlabel("Time [days]")

    plt.tight_layout()
    save_figure(fig_18, RUN_DIR, "18_quota_and_biomass_fraction_heatmaps")
    plt.show()

    # ============================================================
    # 19A) DEPTH-AVERAGED Q AND R OVER TIME
    # ============================================================
    Q_depthavg = np.nanmean(Q_all_time, axis=0)
    R_depthavg = np.nanmean(R_all_time, axis=0)

    if only_last365:
        t_mask = sol.t >= (p.t_max - 365.0)
        t_plot = sol.t[t_mask] - (p.t_max - 365.0)
        Q_plot = Q_depthavg[t_mask]
        R_plot = R_depthavg[t_mask]
    else:
        t_plot = sol.t
        Q_plot = Q_depthavg
        R_plot = R_depthavg

    fig_19, axs_19 = plt.subplots(1, 2, figsize=(13, 4), sharex=False)

    axs_19[0].plot(t_plot, Q_plot, lw=2)
    axs_19[0].axhline(p.Qmin, color="black", ls="--", lw=1.5, label=r"$Q_{\min}$")
    axs_19[0].axhline(p.Qmax, color="gray", ls=":", lw=1.5, label=r"$Q_{\max}$")
    axs_19[0].set_ylabel(r"Depth-averaged quota [mmol Si mmol N$^{-1}$]")
    axs_19[0].set_title(r"Depth-averaged diatom quota $Q$")
    axs_19[0].grid(True)
    axs_19[0].legend()

    axs_19[1].plot(t_plot, R_plot, lw=2)
    axs_19[1].set_ylabel("Depth-averaged fraction [-]")
    axs_19[1].set_title(r"Depth-averaged diatom fraction $R$")
    axs_19[1].set_ylim(0, 1.05)
    axs_19[1].grid(True)

    if only_last365:
        for ax in axs_19:
            season_ticks(ax)
            ax.set_xlabel("Season")
    else:
        for ax in axs_19:
            ax.set_xlabel("Time [days]")

    plt.tight_layout()
    save_figure(fig_19, RUN_DIR, "19_depth_averaged_Q_and_R")
    plt.show()

    # ============================================================
    # 20) TOTAL CONSERVATION + SYSTEM-MEAN CONCENTRATIONS
    # ============================================================
    N_wc_int = np.sum(sol.y[idx["iN"], :], axis=0) * p.dz
    PP_int = np.sum(sol.y[idx["iPP"], :], axis=0) * p.dz
    PDN_int = np.sum(sol.y[idx["iPDN"], :], axis=0) * p.dz
    Z_int = np.sum(sol.y[idx["iZ"], :], axis=0) * p.dz
    DN_int = np.sum(sol.y[idx["iDN"], :], axis=0) * p.dz

    S_int = np.sum(sol.y[idx["iS"], :], axis=0) * p.dz
    PDS_int = np.sum(sol.y[idx["iPDS"], :], axis=0) * p.dz
    DS_int = np.sum(sol.y[idx["iDS"], :], axis=0) * p.dz

    Nsed_int = sol.y[idx["iNsed"], :]
    Dsed_int = sol.y[idx["iDsed"], :]

    TotalN_int = N_wc_int + PP_int + PDN_int + Z_int + DN_int + Nsed_int + Dsed_int
    TotalSi_int = S_int + PDS_int + DS_int

    TotalN0 = TotalN_int[0]
    TotalSi0 = TotalSi_int[0]

    dTotalN_rel = (TotalN_int - TotalN0) / (TotalN0 + 1e-12)
    dTotalSi_rel = (TotalSi_int - TotalSi0) / (TotalSi0 + 1e-12)

    N_mean_system = TotalN_int / p.depth
    Si_mean_system = TotalSi_int / p.depth

    fig_20, axs_20 = plt.subplots(1, 2, figsize=(14, 4))

    axs_20[0].plot(sol.t, 100 * dTotalN_rel, lw=2, label="Total N drift")
    axs_20[0].plot(sol.t, 100 * dTotalSi_rel, lw=2, label="Total Si drift")
    axs_20[0].axhline(0, color="black", ls="--", lw=1)
    axs_20[0].set_xlabel("Time [days]")
    axs_20[0].set_ylabel("Relative drift [%]")
    axs_20[0].set_title("Relative conservation drift")
    axs_20[0].grid(True)
    axs_20[0].legend()

    axs_20[1].plot(sol.t, N_mean_system, lw=2, label="Mean total N")
    axs_20[1].plot(sol.t, Si_mean_system, lw=2, label="Mean total Si")
    axs_20[1].set_xlabel("Time [days]")
    axs_20[1].set_ylabel(r"Equivalent mean concentration [mmol m$^{-3}$]")
    axs_20[1].set_title("System-mean concentrations")
    axs_20[1].grid(True)
    axs_20[1].legend()

    plt.tight_layout()
    save_figure(fig_20, RUN_DIR, "20_conservation_and_system_means")
    plt.show()

    # ============================================================
    # 19B) TIME SERIES OF LIMITATIONS
    # ============================================================
    lim_all = compute_limitations_all_times(sol, p, idx)

    only_last365 = p.Seasonality
    if only_last365:
        t_mask = sol.t >= (p.t_max - 365.0)
        t_plot = sol.t[t_mask] - (p.t_max - 365.0)
    else:
        t_mask = slice(None)
        t_plot = sol.t

    def depthavg(key):
        return np.nanmean(lim_all[key][:, t_mask], axis=0)

    fig_lim_ts, axs_lim_ts = plt.subplots(1, 2, figsize=(14, 4), sharex=False)

    axs_lim_ts[0].plot(t_plot, depthavg("L_lim"), lw=2, label=r"$L_{\mathrm{lim}}$")
    axs_lim_ts[0].plot(t_plot, depthavg("N_lim_P"), lw=2, label=r"$N_{\mathrm{lim},P}$")
    axs_lim_ts[0].plot(t_plot, depthavg("N_lim_D"), lw=2, ls="--", label=r"$N_{\mathrm{lim},D}$")
    axs_lim_ts[0].plot(t_plot, depthavg("S_lim"), lw=2, label=r"$S_{\mathrm{lim}}$")
    axs_lim_ts[0].plot(t_plot, depthavg("Q_lim"), lw=2, label=r"$Q_{\mathrm{lim}}$")
    axs_lim_ts[0].plot(t_plot, depthavg("S_store"), lw=2, label=r"$S_{\mathrm{store}}$")
    axs_lim_ts[0].plot(t_plot, depthavg("G_P"), lw=2, ls=":", label=r"$G_P$")
    axs_lim_ts[0].plot(t_plot, depthavg("G_D"), lw=2, ls="-.", label=r"$G_D$")
    axs_lim_ts[0].set_ylabel("Depth-averaged limitation [-]")
    axs_lim_ts[0].set_title("Individual limitation factors")
    axs_lim_ts[0].set_ylim(0, 1.05)
    axs_lim_ts[0].grid(True)
    axs_lim_ts[0].legend()

    axs_lim_ts[1].plot(t_plot, depthavg("phyto_growth_lim"), lw=2, label=r"$P_P$ growth: $\min(L,N_P)$")
    axs_lim_ts[1].plot(t_plot, depthavg("diatom_growth_lim"), lw=2, label=r"$P_{D,N}$ growth: $\min(L,N_D,Q)$")
    axs_lim_ts[1].plot(t_plot, depthavg("diatom_si_uptake_lim"), lw=2, ls="--", label=r"Si uptake: $\min(L,S,S_{\mathrm{store}})$")
    axs_lim_ts[1].plot(t_plot, depthavg("G_tot"), lw=2, ls=":", label="Total grazing limitation")
    axs_lim_ts[1].set_ylabel("Depth-averaged effective limitation [-]")
    axs_lim_ts[1].set_title("Effective limitations from model equations")
    axs_lim_ts[1].set_ylim(0, 1.05)
    axs_lim_ts[1].grid(True)
    axs_lim_ts[1].legend()

    if only_last365:
        for ax in axs_lim_ts:
            season_ticks(ax)
            ax.set_xlabel("Season")
    else:
        for ax in axs_lim_ts:
            ax.set_xlabel("Time [days]")

    plt.tight_layout()
    save_figure(fig_lim_ts, RUN_DIR, "19B_limitation_time_series")
    plt.show()

    # ============================================================
    # 19C) HEATMAPS OF LIMITATIONS OVER TIME AND DEPTH
    # ============================================================
    if only_last365:
        t_mask = sol.t >= (p.t_max - 365.0)
        t_plot = sol.t[t_mask] - (p.t_max - 365.0)
    else:
        t_mask = slice(None)
        t_plot = sol.t

    dt = t_plot[1] - t_plot[0]
    t_edges = np.linspace(t_plot[0] - dt / 2, t_plot[-1] + dt / 2, len(t_plot) + 1)

    L_plot = lim_all["L_lim"][:, t_mask]
    Np_plot = lim_all["N_lim_P"][:, t_mask]
    Nd_plot = lim_all["N_lim_D"][:, t_mask]
    S_plot_lim = lim_all["S_lim"][:, t_mask]
    phyto_plot = lim_all["phyto_growth_lim"][:, t_mask]
    diatom_plot = lim_all["diatom_growth_lim"][:, t_mask]

    fig_19C, axs_19C = plt.subplots(
        2, 3, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True
    )

    im0 = axs_19C[0, 0].pcolormesh(t_edges, p.z_edges, L_plot, shading="auto", vmin=0.0, vmax=1.0)
    axs_19C[0, 0].set_title("Light limitation")
    axs_19C[0, 0].set_ylabel("Depth [m]")
    fig_19C.colorbar(im0, ax=axs_19C[0, 0], label="Limitation [-]")

    im1 = axs_19C[0, 1].pcolormesh(t_edges, p.z_edges, Np_plot, shading="auto", vmin=0.0, vmax=1.0)
    axs_19C[0, 1].set_title(r"Nitrogen limitation $P_P$")
    fig_19C.colorbar(im1, ax=axs_19C[0, 1], label="Limitation [-]")

    im2 = axs_19C[1, 1].pcolormesh(t_edges, p.z_edges, Nd_plot, shading="auto", vmin=0.0, vmax=1.0)
    axs_19C[1, 1].set_title(r"Nitrogen limitation $P_{D,N}$")
    fig_19C.colorbar(im2, ax=axs_19C[1, 1], label="Limitation [-]")

    im3 = axs_19C[1, 0].pcolormesh(t_edges, p.z_edges, S_plot_lim, shading="auto", vmin=0.0, vmax=1.0)
    axs_19C[1, 0].set_title("Silicate limitation")
    axs_19C[1, 0].set_ylabel("Depth [m]")
    fig_19C.colorbar(im3, ax=axs_19C[1, 0], label="Limitation [-]")

    im4 = axs_19C[0, 2].pcolormesh(t_edges, p.z_edges, phyto_plot, shading="auto", vmin=0.0, vmax=1.0)
    axs_19C[0, 2].set_title(r"$P_P$ growth limitation" + "\n" + r"$\min(L_{\mathrm{lim}},N_{\mathrm{lim},P})$")
    fig_19C.colorbar(im4, ax=axs_19C[0, 2], label="Growth limitation [-]")

    im5 = axs_19C[1, 2].pcolormesh(t_edges, p.z_edges, diatom_plot, shading="auto", vmin=0.0, vmax=1.0)
    axs_19C[1, 2].set_title(r"Diatom growth limitation" + "\n" + r"$\min(L_{\mathrm{lim}},N_{\mathrm{lim},D},Q_{\mathrm{lim}})$")
    fig_19C.colorbar(im5, ax=axs_19C[1, 2], label="Growth limitation [-]")

    for ax in axs_19C.ravel():
        ax.set_ylim(p.z_edges[-1], p.z_edges[0])

    if only_last365:
        for ax in axs_19C[1, :]:
            season_ticks(ax)
            ax.set_xlabel("Season")
    else:
        for ax in axs_19C[1, :]:
            ax.set_xlabel("Time [days]")

    save_figure(fig_19C, RUN_DIR, "19C_limitation_heatmaps")
    plt.show()

    # ============================================================
    # 21) COMPARE COPERNICUS FULL-WATER-COLUMN DEPTH-AVERAGED
    #     NITRATE AND PHYTO FOR 2024 + OWN MODEL RESULTS
    # ============================================================
    try:
        import xarray as xr
        import pandas as pd

        cop_dir = get_base_dir() / "copernicus_downloads_2024"

        no3_file = cop_dir / "no3_nut_no3_2024_fullwatercolumn_depthavg.nc"
        phyc_file = cop_dir / "phyc_pft_phyc_2024_fullwatercolumn_depthavg.nc"

        print("\nOpening Copernicus files:")
        print(no3_file)
        print(phyc_file)

        if no3_file.exists() and phyc_file.exists():
            ds_no3 = xr.open_dataset(no3_file)
            ds_phyc = xr.open_dataset(phyc_file)

            print("\nNO3 dataset:")
            print(ds_no3)
            print("\nPHYC dataset:")
            print(ds_phyc)

            def get_main_var(ds, preferred):
                data_vars = list(ds.data_vars)
                for cand in preferred:
                    if cand in ds.data_vars:
                        return cand
                if len(data_vars) == 1:
                    return data_vars[0]
                return data_vars[0]

            no3_var = get_main_var(ds_no3, ["no3"])
            phyc_var = get_main_var(ds_phyc, ["phyc"])

            print(f"\nUsing NO3 variable:  {no3_var}")
            print(f"Using PHYC variable: {phyc_var}")

            da_no3 = ds_no3[no3_var]
            da_phyc = ds_phyc[phyc_var]

            def horizontal_mean(da):
                spatial_dims = [d for d in da.dims if d.lower() not in ["time"]]
                if len(spatial_dims) > 0:
                    return da.mean(dim=spatial_dims, skipna=True)
                return da

            no3_wc_avg = horizontal_mean(da_no3).sortby("time")
            phyc_wc_avg = horizontal_mean(da_phyc).sortby("time")

            model_mask = sol.t >= (p.t_max - 365.0)
            t_model = sol.t[model_mask] - (p.t_max - 365.0)

            N_model_wc_avg = np.mean(sol.y[idx["iN"]][:, model_mask], axis=0)

            P_model_N_wc_avg = np.mean(
                sol.y[idx["iPP"]][:, model_mask] + sol.y[idx["iPDN"]][:, model_mask],
                axis=0
            )

            P_model_C_wc_avg = P_model_N_wc_avg * p.y_N

            # match the comparison year in the Copernicus files
            model_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(t_model, unit="D")

            no3_units = da_no3.attrs.get("units", "")
            phyc_units = da_phyc.attrs.get("units", "")

            fig_cmp, axs_cmp = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

            axs_cmp[0].plot(
                no3_wc_avg["time"].values,
                no3_wc_avg.values,
                lw=2,
                label="Copernicus NO3"
            )
            axs_cmp[0].plot(
                model_dates,
                N_model_wc_avg,
                lw=2,
                ls="--",
                label=r"Model $N$"
            )
            axs_cmp[0].set_ylabel(f"{no3_var} [{no3_units}]" if no3_units else no3_var)
            axs_cmp[0].set_title("2024 full-water-column depth-averaged nitrate")
            axs_cmp[0].grid(True)
            axs_cmp[0].legend()

            axs_cmp[1].plot(
                phyc_wc_avg["time"].values,
                phyc_wc_avg.values,
                lw=2,
                label="Copernicus PHYC"
            )
            axs_cmp[1].plot(
                model_dates,
                P_model_C_wc_avg,
                lw=2,
                ls="--",
                label=r"Model phyto $(P_P+P_{D,N})$ converted to C"
            )
            axs_cmp[1].set_ylabel(f"{phyc_var} [{phyc_units}]" if phyc_units else phyc_var)
            axs_cmp[1].set_title("2024 full-water-column depth-averaged phytoplankton")
            axs_cmp[1].set_xlabel("Time")
            axs_cmp[1].grid(True)
            axs_cmp[1].legend()

            plt.tight_layout()
            save_figure(fig_cmp, RUN_DIR, "21_copernicus_vs_model_wcavg_2024")
            plt.show()

            ds_no3.close()
            ds_phyc.close()
        else:
            print("\nCopernicus comparison skipped because one or both files were not found.")
            print(f"Missing? {no3_file}: {not no3_file.exists()}")
            print(f"Missing? {phyc_file}: {not phyc_file.exists()}")

    except Exception as exc:
        print("\nCopernicus comparison skipped due to an error:")
        print(exc)
