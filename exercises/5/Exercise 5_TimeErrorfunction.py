import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def run_model_timeconv(nz=50, max_step=1.0, rtol=1e-7, atol=1e-10, nt_eval=1000):
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
    t_eval = np.linspace(0.0, t_max, nt_eval)

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
        method="BDF", rtol=rtol, atol=atol, max_step=max_step
    )

    return {
        "nz": nz,
        "max_step": max_step,
        "rtol": rtol,
        "atol": atol,
        "z": z,
        "t": sol.t,
        "N": sol.y[iN, :],
        "P": sol.y[iP, :],
        "Z": sol.y[iZ, :],
        "D": sol.y[iD, :],
        "O": sol.y[iO, :],
    }

# ============================================================
# SPIN-UP CONVERGENCE TEST
# ============================================================
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def yearly_difference(run, tracer="P"):
    
    t = run["t"]
    z = run["z"]
    field = run[tracer]

    # Only keep times where t-365 exists
    mask = t >= 365

    field_now = field[:, mask]
    field_prev = np.zeros_like(field_now)

    # interpolate previous year
    for k in range(field.shape[0]):
        interp = interp1d(t, field[k], kind="linear", fill_value="extrapolate")
        field_prev[k] = interp(t[mask] - 365)

    # Surface
    surf_rmse = rmse(field_now[0], field_prev[0])

    # Depth integrated
    inv_now = np.trapezoid(field_now, z, axis=0)
    inv_prev = np.trapezoid(field_prev, z, axis=0)

    inv_rmse = rmse(inv_now, inv_prev)

    # Full profile
    prof_rmse = rmse(field_now, field_prev)

    return surf_rmse, inv_rmse, prof_rmse


# ============================================================
# RUN MODEL WITH INCREASING SPIN-UP TIME
# ============================================================

years_list = [2,3,4,5,6,8,10,12,15,20]

spinup_runs = {}
dt = 0.5  # days
for yrs in years_list:
    print(f"Running {yrs} years")

    spinup_runs[yrs] = run_model_timeconv(
        nz=50,
        max_step=1.0,
        rtol=1e-7,
        atol=1e-10,
        t_eval = np.arange(0, yrs*365, dt)
    )


# ============================================================
# COMPUTE CONVERGENCE METRICS
# ============================================================

surf_err = []
inv_err = []
prof_err = []

for yrs in years_list:

    s,i,p = yearly_difference(spinup_runs[yrs], tracer="P")

    surf_err.append(s)
    inv_err.append(i)
    prof_err.append(p)


# ============================================================
# PLOT CONVERGENCE
# ============================================================

plt.figure(figsize=(10,5))

plt.yscale("log")

plt.plot(years_list, surf_err, marker="o", label="Surface P")
plt.plot(years_list, prof_err, marker="o", label="Full profile P")

plt.xlabel("Simulation length [years]")
plt.ylabel("RMSE between consecutive years")

plt.title("Model Spin-up Convergence")

plt.grid(True)
plt.legend()

plt.show()