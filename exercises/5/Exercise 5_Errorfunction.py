import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def run_model(nz):
    # ============================================================
    # 1) PARAMETERS & UNITS
    # ============================================================
    P_growth_max = 2.0
    Z_consum_max = 1.0

    k_L = 20.0
    k_N = 0.3
    k_Z = 1.5
    k_O = 100.0

    m_P = 0.07
    m_Z = 0.1

    e_N = 0.3
    e_D = 0.3

    r = 0.5

    y_P = 9.0
    y_N = 6.625

    O2_atm  = 260.0
    k_O_surf = 1.0
    O2_n     = 0.0
    k_O_bot  = 1.0

    depth = 200.0
    z = np.linspace(0.0, depth, nz)
    dz = z[1] - z[0]

    W = [0.0, 5.0, 0.0, 7.0, 0.0]

    # ============================================================
    # 2) INITIAL CONDITIONS
    # ============================================================
    N0 = np.linspace(0, 100, nz)
    P0 = np.linspace(0.5, 0, nz)
    Z0 = 0.1 * np.ones(nz)
    D0 = np.linspace(0, 100, nz)
    O0 = np.linspace(250, 0, nz)

    y0 = np.concatenate([N0, P0, Z0, D0, O0])

    iN = slice(0, nz)
    iP = slice(nz, 2*nz)
    iZ = slice(2*nz, 3*nz)
    iD = slice(3*nz, 4*nz)
    iO = slice(4*nz, 5*nz)

    years = 5
    t_max = 365.0 * years
    t_span = (0.0, t_max)
    t_eval = np.linspace(0.0, t_max, 1000)

    def day_of_year(t):
        return float(t % 365.0)

    def get_limits(Lz, N, P, O, delta_O=10.0, oxyg_switch=True):
        light_lim = Lz / (Lz + k_L + 1e-12)
        nut_lim   = N  / (N  + k_N + 1e-12)
        graze_lim = P  / (P  + k_Z + 1e-12)

        if oxyg_switch:
            oxyg_lim = 0.5 * (1.0 + np.tanh((O - k_O) / (delta_O + 1e-12)))
        else:
            oxyg_lim = O / (O + k_O + 1e-12)

        return light_lim, nut_lim, graze_lim, oxyg_lim

    def getLIGHTandKAPPAS(t, P=None, D=None, Lightswitch=True, Seasonality=True, bio_attenuation=True):
        k_water = 0.1
        k_bio   = 0.20

        if bio_attenuation and (P is not None) and (D is not None):
            P_pos = np.maximum(P, 0.0)
            D_pos = np.maximum(D, 0.0)
            bio_integral = np.cumsum(P_pos + D_pos) * dz
        else:
            bio_integral = 0.0

        if Seasonality:
            doy = day_of_year(t)

            zMix       = 0.05 * depth
            zMixWinter = 0.8  * depth
            tMaxSpring = 90
            zetaMaxSteep = 2.0
            z_mix = 0.5 * (1 - np.sin(2*np.pi*(doy - tMaxSpring)/365.0))**zetaMaxSteep * (zMixWinter - zMix) + zMix

            kappa_top_summer = 5.0
            kappa_top_winter = 15.0
            kappa_bottom_summer = 0.5
            kappa_bottom_winter = 15.0

            season_shape = 0.5 * (1 - np.sin(2*np.pi*(doy - tMaxSpring)/365.0))
            kappa_top = kappa_top_summer + (kappa_top_winter - kappa_top_summer) * (season_shape**zetaMaxSteep)
            kappa_bottom = kappa_bottom_summer + (kappa_bottom_winter - kappa_bottom_summer) * (season_shape**zetaMaxSteep)

            zeta_mix = 10.0
            kappa_center = 0.5*(1 - np.tanh((z - z_mix)/zeta_mix))*(kappa_top - kappa_bottom) + kappa_bottom

            L0_min = 50.0
            L0_max = 1000.0

            if Lightswitch:
                spring_center = 90.0
                autumn_center = 260.0
                spring_width = 20.0
                autumn_width = 30.0

                spring_switch = 0.5 * (1 + np.tanh((doy - spring_center) / spring_width))
                autumn_switch = 0.5 * (1 - np.tanh((doy - autumn_center) / autumn_width))
                seasonal_shape = spring_switch * autumn_switch
                L0 = L0_min + (L0_max - L0_min) * seasonal_shape
            else:
                phase_shift = 80.0
                seasonal_shape = (1 + np.sin(2*np.pi*(doy - phase_shift)/365.0))**2
                L0 = L0_min + (L0_max - L0_min) * seasonal_shape / 4.0

        else:
            kappa_surface = 10.0
            kappa_bottom  = 1.0
            z_transition  = 50.0
            zeta_mix      = 10.0
            kappa_center = 0.5*(1 - np.tanh((z - z_transition)/zeta_mix))*(kappa_surface - kappa_bottom) + kappa_bottom
            L0 = 1400.0

        kappa_interface = np.zeros(nz + 1)
        kappa_interface[1:nz] = 0.5*(kappa_center[1:] + kappa_center[:-1])
        kappa_interface[0] = kappa_center[0]
        kappa_interface[nz] = kappa_center[-1]

        if bio_attenuation and (P is not None) and (D is not None):
            Lz = L0 * np.exp(-k_water*z - k_bio*bio_integral)
        else:
            Lz = L0 * np.exp(-k_water*z)

        return kappa_interface, Lz, L0

    def surface_flux(tracer_name, C_surface):
        if tracer_name == "O":
            return (O2_atm - C_surface) * k_O_surf
        return 0.0

    def bottom_flux(tracer_name, C_bottom):
        if tracer_name == "O":
            O_use = 0.0
            return (O_use + O2_n) * k_O_bot
        return 0.0

    def vertical_transport(C, kappa_interface, w=0.0, tracer_name=""):
        J = np.zeros(nz + 1)

        if w >= 0:
            Ja = w * C[:-1]
        else:
            Ja = w * C[1:]

        Jd = -kappa_interface[1:nz] * (C[1:] - C[:-1]) / dz
        J[1:nz] = Ja + Jd

        J[0]  = surface_flux(tracer_name, C[0])
        J[nz] = bottom_flux(tracer_name, C[-1])

        return -(J[1:] - J[:-1]) / dz

    def rhs(t, y):
        N = y[iN]
        P = y[iP]
        Z = y[iZ]
        D = y[iD]
        O = y[iO]

        kappa_interface, Lz, _ = getLIGHTandKAPPAS(t, P=P, D=D)
        light_lim, nut_lim, graze_lim, oxyg_lim = get_limits(Lz, N, P, O)

        P_growth = P_growth_max * np.minimum(light_lim, nut_lim)
        Z_consum = Z_consum_max * np.minimum(graze_lim, oxyg_lim)

        eps = 0.0
        r_factor = r * (eps + (1 - eps) * oxyg_lim)

        N_uptake  = P_growth * P
        P_mort    = m_P * P
        Z_grazing = Z_consum * Z
        Z_mort    = m_Z * Z**2
        remin     = r_factor * D

        dN_reac = -N_uptake + e_N*Z_grazing + remin
        dP_reac =  N_uptake - Z_grazing - P_mort
        dZ_reac = (1 - e_N - e_D)*Z_grazing - Z_mort
        dD_reac =  P_mort + e_D*Z_grazing + Z_mort - remin
        dO_reac = -y_N*e_N*Z_grazing - y_N*remin + y_P*N_uptake

        dN = vertical_transport(N, kappa_interface, w=W[0], tracer_name="N") + dN_reac
        dP = vertical_transport(P, kappa_interface, w=W[1], tracer_name="P") + dP_reac
        dZ = vertical_transport(Z, kappa_interface, w=W[2], tracer_name="Z") + dZ_reac
        dD = vertical_transport(D, kappa_interface, w=W[3], tracer_name="D") + dD_reac
        dO = vertical_transport(O, kappa_interface, w=W[4], tracer_name="O") + dO_reac

        return np.concatenate([dN, dP, dZ, dD, dO])

    sol = solve_ivp(
        rhs, t_span, y0, t_eval=t_eval,
        method="BDF", rtol=1e-7, atol=1e-10, max_step=1.0
    )

    return {
        "nz": nz,
        "z": z,
        "dz": dz,
        "t": sol.t,
        "N": sol.y[iN, :],
        "P": sol.y[iP, :],
        "Z": sol.y[iZ, :],
        "D": sol.y[iD, :],
        "O": sol.y[iO, :],
    }

def interp_to_ref_grid(z_src, field_src, z_ref):
    """
    Interpolate a field from source grid to reference grid.

    Parameters
    ----------
    z_src : (nz_src,) array
    field_src : (nz_src,) or (nz_src, nt) array
    z_ref : (nz_ref,) array

    Returns
    -------
    field_ref : interpolated field on z_ref
    """
    f = interp1d(z_src, field_src, axis=0, bounds_error=False, fill_value="extrapolate")
    return f(z_ref)


def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


def relative_l2_error(a, b, eps=1e-14):
    """
    Relative L2 error of a against reference b.
    """
    num = np.sqrt(np.mean((a - b)**2))
    den = np.sqrt(np.mean(b**2)) + eps
    return num / den


def final_profile_error(run, ref_run, tracer="P"):
    """
    Compare final profile of one tracer to reference run.
    """
    z_ref = ref_run["z"]
    prof = interp_to_ref_grid(run["z"], run[tracer][:, -1], z_ref)
    prof_ref = ref_run[tracer][:, -1]

    return {
        "rmse": rmse(prof, prof_ref),
        "rel_l2": relative_l2_error(prof, prof_ref),
    }


def last_year_timeseries_error(run, ref_run, tracer="P"):
    """
    Compare last-year depth-integrated inventory time series.
    Assumes same t-grid for all runs.
    """
    t = run["t"]
    t_ref = ref_run["t"]

    if not np.allclose(t, t_ref):
        raise ValueError("Time grids do not match. Use same t_eval for all runs.")

    mask = t >= (t[-1] - 365.0)

    field = interp_to_ref_grid(run["z"], run[tracer][:, mask], ref_run["z"])
    field_ref = ref_run[tracer][:, mask]

    inv = np.trapz(field, ref_run["z"], axis=0)
    inv_ref = np.trapz(field_ref, ref_run["z"], axis=0)

    return {
        "rmse": rmse(inv, inv_ref),
        "rel_l2": relative_l2_error(inv, inv_ref),
    }


def bottom_oxygen_error(run, ref_run):
    """
    Compare last-year bottom oxygen time series.
    """
    t = run["t"]
    t_ref = ref_run["t"]

    if not np.allclose(t, t_ref):
        raise ValueError("Time grids do not match. Use same t_eval for all runs.")

    mask = t >= (t[-1] - 365.0)

    O_bottom = run["O"][-1, mask]
    O_bottom_ref = ref_run["O"][-1, mask]

    return {
        "rmse": rmse(O_bottom, O_bottom_ref),
        "rel_l2": relative_l2_error(O_bottom, O_bottom_ref),
    }

nz_list = [10,20,30,40,50,60,70,80,90,100]

runs = {}
for nz_val in nz_list:
    print(f"Running nz = {nz_val}")
    runs[nz_val] = run_model(nz_val)

ref_nz = max(nz_list)
ref_run = runs[ref_nz]

tracers = ["N", "P", "Z", "D", "O"]

error_summary = {}

for nz_val in nz_list:
    if nz_val == ref_nz:
        continue

    error_summary[nz_val] = {}

    for tracer in tracers:
        error_summary[nz_val][f"{tracer}_final_profile"] = final_profile_error(
            runs[nz_val], ref_run, tracer=tracer
        )
        error_summary[nz_val][f"{tracer}_lastyear_inventory"] = last_year_timeseries_error(
            runs[nz_val], ref_run, tracer=tracer
        )

    error_summary[nz_val]["O_bottom"] = bottom_oxygen_error(runs[nz_val], ref_run)
    
import matplotlib.pyplot as plt

nz_plot = []
p_err = []
o_err = []
obot_err = []

for nz_val in nz_list:
    if nz_val == ref_nz:
        continue
    nz_plot.append(nz_val)
    p_err.append(error_summary[nz_val]["P_final_profile"]["rel_l2"])
    o_err.append(error_summary[nz_val]["O_final_profile"]["rel_l2"])
    obot_err.append(error_summary[nz_val]["O_bottom"]["rel_l2"])

# %%


plt.figure(figsize=(8, 5))
plt.plot(nz_plot, p_err, marker="o", label="P final profile")
plt.plot(nz_plot, o_err, marker="o", label="O final profile")
plt.plot(nz_plot, obot_err, marker="o", label="Bottom O2 last year")
plt.xlabel("Number of vertical layers nz")
plt.ylabel("Relative L2 error")
plt.grid(True)
plt.legend()
plt.show()