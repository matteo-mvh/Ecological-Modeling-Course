# START OF SCRIPT

# %% IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# %% PARAMETER INPUT

# ============================================================
# 1) PARAMETERS & UNITS
# ============================================================
# Vertical 1D NPZD-O model
# State variables (concentrations, cell centers):
#   N, P, Z, D     : [mmol N m^-3]
#   O              : [mmol O2 m^-3]
#   N_sed, D_sed   : [mmol N m^-2]
#
# Vertical diffusivity:
#   kappa      : [m^2 d^-1]
#
# Sinking/advection velocities:
#   w          : [m d^-1]  (positive downward)
#
# Boundary fluxes:
#   J          : [mmol m^-2 d^-1]
#   Convention: J > 0 is DOWNWARD (from surface into ocean; from bottom into sediments)

# ---- Biology parameters ----
P_growth_max = 1.5   # [d^-1]
Z_consum_max = 0.7   # [d^-1]

k_L = 80.0           # [µmol photons m^-2 s^-1]
k_N = 0.4            # [mmol N m^-3]
k_Z = 4.0            # [mmol N m^-3]
k_O = 100.0          # critical oxygen for switch [mmol O2 m^-3]

m_P = 0.25           # [d^-1]
m_Z = 0.2            # [d^-1]

e_N = 0.3            # [-]
e_D = 0.3            # [-]

r_ae = 0.2           # aerobic detritus remin rate [d^-1]
r_an = 0.01          # anaerobic detritus remin rate [d^-1]

# Stoichiometry: O2 production/consumption per N transformed
y_P = 9.0
y_N = 6.625

# ---- Air-sea and bottom O2 exchange/sink ----
O2_atm  = 260.0      # "equilibrium/atmospheric" O2 [mmol O2 m^-3]

# ---- bottom N/D exchange ----
k_N_bot = 10.0         # [m d^-1]
z_sed = 0.05      # [m]

# ---- Domain/grid ----
depth = 20.0        # [m]
nz = 50
z = np.linspace(0.0, depth, nz)   # [m], 0 at surface, positive downward
dz = z[1] - z[0]                  # [m]

# ---- Transport settings (positive downward) ----
#      N    P    Z    D    O
W = [0.0, 0.0, 0.0, 5.0, 0.0]     # [m d^-1]

# ============================================================
# 2) INITIAL CONDITIONS
# ============================================================
N0 = 20.0 + 8.0 * (z / depth)  #  normal
N0 = 100.0 + 8.0 * (z / depth)               
P0 = 1.0 * np.exp(-z / 3.0) + 0.01
Z0 = 0.2 * np.exp(-z / 5.0) + 0.02
D0 = 0.2 + 1.5 * (z / depth)                 # low at surface, moderate at bottom
O0 = 260.0 - 40.0 * (z / depth)              # 260 -> 220
Nsed0 = N0[-1] * z_sed                # [mmol N m^-2]
Dsed0 = D0[-1] * z_sed                # [mmol N m^-2]

y0 = np.concatenate([N0, P0, Z0, D0, O0, [Nsed0], [Dsed0]])

# Convenient slices
iN     = slice(0, nz)
iP     = slice(nz, 2*nz)
iZ     = slice(2*nz, 3*nz)
iD     = slice(3*nz, 4*nz)
iO     = slice(4*nz, 5*nz)
iNsed  = 5*nz
iDsed  = 5*nz + 1


# ============================================================
# 3) TIME SETUP
# ============================================================
years = 3
t_max = 365.0 * years
t_span = (0.0, t_max)
t_eval = np.linspace(0.0, t_max, 1000)

# %% FUNCTION SETUP

# ------------------------------------------------------------
# 3A) FIXED SUMMER FORCING
# ------------------------------------------------------------
years_ss = 3
t_max_ss = 365.0 * years_ss
t_span_ss = (0.0, t_max_ss)
t_eval_ss = np.linspace(0.0, t_max_ss, 1200)

def getLIGHTandKAPPAS_summer(t, P=None, D=None):
    """
    Fixed summer conditions:
      - no seasonality
      - relatively weak stratified mixing
      - high light
    """
    kappa_top = 6.0
    kappa_bottom = 3.0
    z_mix = 0.1 * depth
    zeta_mix = 5.0

    kappa_center = (
        0.5 * (1 - np.tanh((z - z_mix) / zeta_mix)) * (kappa_top - kappa_bottom)
        + kappa_bottom
    )

    # interface kappa
    kappa_interface = np.zeros(nz + 1)
    kappa_interface[1:nz] = 0.5 * (kappa_center[1:] + kappa_center[:-1])
    kappa_interface[0] = kappa_center[0]
    kappa_interface[nz] = kappa_center[-1]

    # fixed summer light
    L0 = 700.0
    k_water = 0.35
    Lz = L0 * np.exp(-k_water * z)

    return kappa_interface, Lz, L0

# ============================================================
# 4) SMALL UTILITIES
# ============================================================
def day_of_year(t):
    return float(t % 365.0)

def season_from_doy(doy):
    # simple 4-season bins matching your season_days choices
    # Winter: [0..90), Spring: [90..172), Summer: [172..265), Autumn: [265..365)
    if doy < 90:
        return "Winter"
    if doy < 172:
        return "Spring"
    if doy < 265:
        return "Summer"
    return "Autumn"

def season_ticks(ax):
    season_positions = [0, 80, 172, 264, 355]
    season_labels = ["Winter", "Spring", "Summer", "Autumn", "Winter"]
    ax.set_xticks(season_positions)
    ax.set_xticklabels(season_labels)

# ============================================================
# 5) LIMITATION FUNCTIONS
# ============================================================
def get_limits(Lz, N, P, O, delta_O=10.0, oxyg_switch=True):
    """
    Returns limitation factors in [0,1]:
      light_lim(Lz), nut_lim(N), graze_lim(P), oxyg_lim(O)
    """
    light_lim = Lz / (Lz + k_L + 1e-12)
    nut_lim   = N  / (N  + k_N + 1e-12)
    n = 1.5
    graze_lim = P**n  / (P**n  + k_Z**n + 1e-12)

    if oxyg_switch:
        # smooth step around k_O with width delta_O
        oxyg_lim = 0.5 * (1.0 + np.tanh((O - k_O) / (delta_O + 1e-12)))
    else:
        oxyg_lim = O / (O + k_O + 1e-12)

    return light_lim, nut_lim, graze_lim, oxyg_lim

# ============================================================
# 6) SEASONAL FORCING: KAPPA + LIGHT
# ============================================================
def getLIGHTandKAPPAS(t, P=None, D=None,Lightswitch = False, Seasonality=False, bio_attenuation=True):
    """
    Returns:
      kappa_interface : [m^2 d^-1] at interfaces, size nz+1
      Lz              : [µmol photons m^-2 s^-1] at cell centers
      L0              : surface light [µmol photons m^-2 s^-1]
    """
    # light attenuation
    k_water = 0.25   # [m^-1]
    k_bio   = 0.05   # [m^2 mmol^-1]

    # self-shading integral (dimensionless in exponent)
    if bio_attenuation and (P is not None) and (D is not None):
        P_pos = np.maximum(P, 0.0)
        D_pos = np.maximum(D, 0.0)
        bio_integral = np.cumsum(P_pos + D_pos) * dz
    else:
        bio_integral = 0.0

    if Seasonality:
        doy = day_of_year(t)

        # --- Seasonal mixing depth z_mix(t) [m] ---
        zMix       = 0.1 * depth
        zMixWinter = 0.8 * depth
        tMaxSpring = 90
        zetaMaxSteep = 2.0
        z_mix = 0.5 * (1 - np.sin(2*np.pi*(doy - tMaxSpring)/365.0))**zetaMaxSteep * (zMixWinter - zMix) + zMix

        # --- Seasonal surface diffusivity kappa_top(t) [m^2 d^-1] ---
        kappa_top_summer = 7
        kappa_top_winter = 15
        kappa_bottom_summer = 2
        kappa_bottom_winter = 15
        
        season_shape = 0.5 * (1 - np.sin(2*np.pi*(doy - tMaxSpring)/365.0)) 
        kappa_top = kappa_top_summer + (kappa_top_winter - kappa_top_summer) * (season_shape**zetaMaxSteep)
        kappa_bottom = kappa_bottom_summer + (kappa_bottom_winter - kappa_bottom_summer) * (season_shape ** zetaMaxSteep)
        zeta_mix = 5.0
        kappa_center = 0.5*(1 - np.tanh((z - z_mix)/zeta_mix))*(kappa_top - kappa_bottom) + kappa_bottom

        # --- Seasonal surface light L0(t) ---
        L0_min = 50.0
        L0_max = 900.0
        if Lightswitch:
            # --- Seasonal surface light L0(t) using double tanh ---
            #Timing (days of year)
            spring_center  = 120.0   # center of spring transition
            autumn_center  = 210.0   # center of autumn transition
            # Controls sharpness (smaller = sharper)
            spring_width = 30.0
            autumn_width = 30.0
            # Spring switch: 0 → 1
            spring_switch = 0.5 * (1 + np.tanh((doy - spring_center) / spring_width))
            # Autumn switch: 1 → 0
            autumn_switch = 0.5 * (1 - np.tanh((doy - autumn_center) / autumn_width))
            seasonal_shape = spring_switch * autumn_switch
            L0 = L0_min + (L0_max - L0_min) * seasonal_shape
        else:
            phase_shift = 80.0
            seasonal_shape = (1 + np.sin(2*np.pi*(doy - phase_shift)/365.0))**3
            L0 = L0_min + (L0_max - L0_min) * seasonal_shape / 4.0

    else:
        kappa_surface = 10.0
        kappa_bottom  = 5.0
        z_transition  = 50.0
        zeta_mix      = 10.0
        kappa_center = 0.5*(1 - np.tanh((z - z_transition)/zeta_mix))*(kappa_surface - kappa_bottom) + kappa_bottom
        L0 = 1000.0

    # interface κ (nz+1)
    kappa_interface = np.zeros(nz + 1)
    kappa_interface[1:nz] = 0.5*(kappa_center[1:] + kappa_center[:-1])
    kappa_interface[0] = kappa_center[0]
    kappa_interface[nz] = kappa_center[-1]

    # light profile
    if bio_attenuation and (P is not None) and (D is not None):
        Lz = L0 * np.exp(-k_water*z - k_bio*bio_integral)
    else:
        Lz = L0 * np.exp(-k_water*z)

    return kappa_interface, Lz, L0

# ============================================================
# 7) BOUNDARY FLUXES + VERTICAL TRANSPORT
# ============================================================

def kappa_scaled(kappa_surface, kappa_ref = 10.0, k_O_ref = 10.0):
        return  k_O_ref * (kappa_surface / kappa_ref)
    
def surface_flux(tracer_name, C_surface, kappa_surface):
    """Return downward flux J[0] [mmol m^-2 d^-1] (positive into ocean)."""

    if tracer_name == "O":

        k_O_surf = kappa_scaled(kappa_surface)

        return (O2_atm - C_surface) * k_O_surf

    return 0.0

def bottom_flux(tracer_name, C_bottom, sed=None):
    """Return downward flux J[nz] [mmol m^-2 d^-1] (positive removes tracer into sediments)."""

    if tracer_name == "N":
        # downward-positive convention
        # if sediment concentration > bottom-water concentration,
        # this becomes negative = upward flux into water column
        Nsed = 0.0 if sed is None else sed
        return -k_N_bot * (Nsed / z_sed - C_bottom)

    elif tracer_name == "D":
        # burial/export of detritus into sediment
        return W[3] * C_bottom

    return 0.0

def vertical_transport(C, kappa_interface, w=0.0, tracer_name="", sed=None):
    """
    dC/dt from 1D vertical advection + diffusion + boundary fluxes.
    w [m d^-1] positive downward.

    Flux form:
      dC/dt = -(J_{k+1/2} - J_{k-1/2}) / dz
    with J in [mmol m^-2 d^-1].
    """
    J = np.zeros(nz + 1)

    # interior interfaces 1..nz-1
    if w >= 0:
        Ja = w * C[:-1]     # upwind from above
    else:
        Ja = w * C[1:]      # upwind from below

    Jd = -kappa_interface[1:nz] * (C[1:] - C[:-1]) / dz
    J[1:nz] = Ja + Jd

    # boundary fluxes
    J[0] = surface_flux(tracer_name, C[0], kappa_interface[0])
    J[nz] = bottom_flux(tracer_name, C[-1],sed=sed)

    return -(J[1:] - J[:-1]) / dz

# ============================================================
# 8) ODE RHS
# ============================================================
def rhs(t, y):
    N    = y[iN]
    P    = y[iP]
    Z    = y[iZ]
    D    = y[iD]
    O    = y[iO]
    Nsed = y[iNsed]
    Dsed = y[iDsed]

    # forcing
    kappa_interface, Lz, _ = getLIGHTandKAPPAS(t, P=P, D=D)

    # limitations
    light_lim, nut_lim, graze_lim, oxyg_lim = get_limits(Lz, N, P, O)

    # rates
    P_growth = P_growth_max * np.minimum(light_lim, nut_lim)
    Z_consum = Z_consum_max * np.minimum(graze_lim, oxyg_lim)

    # processes
    N_uptake  = P_growth * P
    P_mort    = m_P * P
    Z_grazing = Z_consum * Z
    Z_mort    = m_Z * Z**2

    # water-column remineralization
    remin_ae_wc = r_ae * oxyg_lim  * D
    remin_an_wc = r_an * D
    remin_wc    = remin_ae_wc + remin_an_wc

    # sediment remineralization
    ox_bot      = oxyg_lim[-1]
    remin_ae_bot = r_ae * ox_bot  * Dsed
    remin_an_bot = r_an * Dsed
    remin_bot    = remin_ae_bot + remin_an_bot

    # boundary fluxes
    J_D_bot = bottom_flux("D", D[-1])
    J_N_bot = bottom_flux("N", N[-1], sed=Nsed)

    # reactions
    dN_reac = -N_uptake + e_N * Z_grazing + remin_wc
    dP_reac =  N_uptake - Z_grazing - P_mort
    dZ_reac = (1 - e_N - e_D) * Z_grazing - Z_mort
    dD_reac =  P_mort + e_D * Z_grazing + Z_mort - remin_wc
    dO_reac = -y_N * e_N * Z_grazing - y_N * remin_ae_wc + y_P * N_uptake

    # transport + reactions
    dN = vertical_transport(N, kappa_interface, w=W[0], tracer_name="N", sed=Nsed) + dN_reac
    dP = vertical_transport(P, kappa_interface, w=W[1], tracer_name="P") + dP_reac
    dZ = vertical_transport(Z, kappa_interface, w=W[2], tracer_name="Z") + dZ_reac
    dD = vertical_transport(D, kappa_interface, w=W[3], tracer_name="D") + dD_reac
    dO = vertical_transport(O, kappa_interface, w=W[4], tracer_name="O") + dO_reac

    # benthic aerobic O2 demand acts only on bottom water cell
    dO[-1] -= y_N * remin_ae_bot / dz

    # sediment boxes
    dDsed = J_D_bot - remin_bot
    dNsed = J_N_bot + remin_bot

    return np.concatenate([dN, dP, dZ, dD, dO, [dNsed], [dDsed]])


# ------------------------------------------------------------
# 2) CASE-SPECIFIC LIMITATION FUNCTION
# ------------------------------------------------------------
def get_limits_case(Lz, N, P, O, k_N_case, delta_O=10.0, oxyg_switch=True):
    """
    Same as get_limits(), but with case-specific nutrient half-saturation.
    """
    light_lim = Lz / (Lz + k_L + 1e-12)
    nut_lim   = N  / (N  + k_N_case + 1e-12)

    n = 1.5
    graze_lim = P**n / (P**n + k_Z**n + 1e-12)

    if oxyg_switch:
        oxyg_lim = 0.5 * (1.0 + np.tanh((O - k_O) / (delta_O + 1e-12)))
    else:
        oxyg_lim = O / (O + k_O + 1e-12)

    return light_lim, nut_lim, graze_lim, oxyg_lim

# ------------------------------------------------------------
# INITIAL CONDITION BUILDER
# ------------------------------------------------------------
def make_initial_state(N_surface):
    """
    Builds a consistent NPZD-O initial state.
    N_surface controls the nutrient level at the surface.
    """

    N0 = N_surface + 8.0 * (z / depth)
    P0 = 1.0 * np.exp(-z / 3.0) + 0.01
    Z0 = 0.2 * np.exp(-z / 5.0) + 0.02
    D0 = 0.2 + 1.5 * (z / depth)
    O0 = 260.0 - 40.0 * (z / depth)

    Nsed0 = N0[-1] * z_sed
    Dsed0 = D0[-1] * z_sed

    y0 = np.concatenate([N0, P0, Z0, D0, O0, [Nsed0], [Dsed0]])

    return y0
# ------------------------------------------------------------
# 3) CASE-SPECIFIC SUMMER RHS
# ------------------------------------------------------------
def make_rhs_summer_kN(k_N_case):
    def rhs_summer_kN(t, y):
        N    = y[iN]
        P    = y[iP]
        Z    = y[iZ]
        D    = y[iD]
        O    = y[iO]
        Nsed = y[iNsed]
        Dsed = y[iDsed]

        # fixed summer forcing
        kappa_interface, Lz, _ = getLIGHTandKAPPAS_summer(t, P=P, D=D)

        # limitations with case-specific k_N
        light_lim, nut_lim, graze_lim, oxyg_lim = get_limits_case(
            Lz, N, P, O, k_N_case=k_N_case
        )

        # rates
        P_growth = P_growth_max * np.minimum(light_lim, nut_lim)
        Z_consum = Z_consum_max * np.minimum(graze_lim, oxyg_lim)

        # processes
        N_uptake  = P_growth * P
        P_mort    = m_P * P
        Z_grazing = Z_consum * Z
        Z_mort    = m_Z * Z**2

        # water-column remineralization
        remin_ae_wc = r_ae * oxyg_lim * D
        remin_an_wc = r_an * D
        remin_wc    = remin_ae_wc + remin_an_wc

        # sediment remineralization
        ox_bot       = oxyg_lim[-1]
        remin_ae_bot = r_ae * ox_bot * Dsed
        remin_an_bot = r_an * Dsed
        remin_bot    = remin_ae_bot + remin_an_bot

        # boundary fluxes
        J_D_bot = bottom_flux("D", D[-1])
        J_N_bot = bottom_flux("N", N[-1], sed=Nsed)

        # reactions
        dN_reac = -N_uptake + e_N * Z_grazing + remin_wc
        dP_reac =  N_uptake - Z_grazing - P_mort
        dZ_reac = (1 - e_N - e_D) * Z_grazing - Z_mort
        dD_reac =  P_mort + e_D * Z_grazing + Z_mort - remin_wc
        dO_reac = -y_N * e_N * Z_grazing - y_N * remin_ae_wc + y_P * N_uptake

        # transport + reactions
        dN = vertical_transport(N, kappa_interface, w=W[0], tracer_name="N", sed=Nsed) + dN_reac
        dP = vertical_transport(P, kappa_interface, w=W[1], tracer_name="P") + dP_reac
        dZ = vertical_transport(Z, kappa_interface, w=W[2], tracer_name="Z") + dZ_reac
        dD = vertical_transport(D, kappa_interface, w=W[3], tracer_name="D") + dD_reac
        dO = vertical_transport(O, kappa_interface, w=W[4], tracer_name="O") + dO_reac

        # benthic aerobic O2 demand on bottom water cell
        dO[-1] -= y_N * remin_ae_bot / dz

        # sediment boxes
        dDsed = J_D_bot - remin_bot
        dNsed = J_N_bot + remin_bot

        return np.concatenate([dN, dP, dZ, dD, dO, [dNsed], [dDsed]])

    return rhs_summer_kN


# ------------------------------------------------------------
# 4) RUN TWO k_N CASES
# ------------------------------------------------------------
# Keep nutrient loading fixed
N_surface_fixed = 5
y0_case = make_initial_state(N_surface_fixed)

# Choose two half-saturation constants
k_N_low = 0.4
k_N_high = 0.4 * 1.25

rhs_low_kN = make_rhs_summer_kN(k_N_low)
rhs_high_kN = make_rhs_summer_kN(k_N_high)

sol_low_kN = solve_ivp(
    rhs_low_kN, t_span_ss, y0_case, t_eval=t_eval_ss,
    method="BDF", rtol=1e-7, atol=1e-10, max_step=1.0
)

sol_high_kN = solve_ivp(
    rhs_high_kN, t_span_ss, y0_case, t_eval=t_eval_ss,
    method="BDF", rtol=1e-7, atol=1e-10, max_step=1.0
)


# ------------------------------------------------------------
# 5) EXTRACT STEADY STATES
# ------------------------------------------------------------
N_lowk  = sol_low_kN.y[iN, -1]
P_lowk  = sol_low_kN.y[iP, -1]
D_lowk  = sol_low_kN.y[iD, -1]
O_lowk  = sol_low_kN.y[iO, -1]

N_highk = sol_high_kN.y[iN, -1]
P_highk = sol_high_kN.y[iP, -1]
D_highk = sol_high_kN.y[iD, -1]
O_highk = sol_high_kN.y[iO, -1]


# ------------------------------------------------------------
# 6) FOUR PANEL STEADY-STATE COMPARISON (1x4 layout)
# ------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(14,6), sharey=True)

# Nutrients
axs[0].plot(N_lowk, z, color="darkblue", lw=2)
axs[0].plot(N_highk, z, color="darkblue", lw=2, ls="--")
axs[0].set_title("Nutrients")
axs[0].set_xlabel("N [mmol N m$^{-3}$]")
axs[0].set_ylabel("Depth [m]")
axs[0].grid(True)

# Oxygen
axs[1].plot(O_lowk, z, color="black", lw=2)
axs[1].plot(O_highk, z, color="black", lw=2, ls="--")
axs[1].set_title("Oxygen")
axs[1].set_xlabel("O$_2$ [mmol O$_2$ m$^{-3}$]")
axs[1].grid(True)

# Phytoplankton
axs[2].plot(P_lowk, z, color="green", lw=2)
axs[2].plot(P_highk, z, color="green", lw=2, ls="--")
axs[2].set_title("Phytoplankton")
axs[2].set_xlabel("P [mmol N m$^{-3}$]")
axs[2].grid(True)

# Detritus
axs[3].plot(D_lowk, z, color="red", lw=2)
axs[3].plot(D_highk, z, color="red", lw=2, ls="--")
axs[3].set_title("Detritus")
axs[3].set_xlabel("D [mmol N m$^{-3}$]")
axs[3].grid(True)

# Formatting
for ax in axs:
    ax.invert_yaxis()
    ax.set_ylim(depth, 0)

from matplotlib.lines import Line2D
legend_lines = [
    Line2D([0], [0], color="black", lw=2, linestyle="-"),
    Line2D([0], [0], color="black", lw=2, linestyle="--"),
]

fig.legend(
    legend_lines,
    [fr"Low $k_N$ = {k_N_low}", fr"High $k_N$ = {k_N_high}"],
    bbox_to_anchor=(0.5, 0.97),
    loc="upper center",
    ncol=2,
    frameon=True
)

plt.tight_layout(rect=[0,0,1,0.92])
plt.show()