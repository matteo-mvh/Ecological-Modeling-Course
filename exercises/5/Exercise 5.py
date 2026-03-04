import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# 1) PARAMETERS & UNITS
# ============================================================
# Vertical 1D NPZD-O model
# State variables (concentrations, cell centers):
#   N, P, Z, D : [mmol N m^-3]
#   O          : [mmol O2 m^-3]
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
P_growth_max = 2.0   # [d^-1]
Z_consum_max = 1.0   # [d^-1]

k_L = 20.0           # [µmol photons m^-2 s^-1]
k_N = 0.3            # [mmol N m^-3]
k_Z = 1.5            # [mmol N m^-3]
k_O = 100.0          # critical oxygen for switch [mmol O2 m^-3]

m_P = 0.07           # [d^-1]
m_Z = 0.1            # [d^-1]

e_N = 0.3            # [-]
e_D = 0.3            # [-]

r = 0.5              # detritus remin rate [d^-1]

# Stoichiometry: O2 production/consumption per N transformed
y_P = 9.0
y_N = 6.625

# ---- Air-sea and bottom O2 exchange/sink ----
O2_atm  = 260.0      # "equilibrium/atmospheric" O2 [mmol O2 m^-3]
k_O_surf = 1.0       # piston velocity [m d^-1]
O2_n     = 0.0       # bottom demand/background [mmol O2 m^-3]
k_O_bot  = 1.0       # bottom exchange/consumption velocity [m d^-1]

# ---- Domain/grid ----
depth = 200.0        # [m]
nz = 50
z = np.linspace(0.0, depth, nz)   # [m], 0 at surface, positive downward
dz = z[1] - z[0]                  # [m]

# ---- Transport settings (positive downward) ----
#        N   P   Z    D   O
W = [0.0, 5.0, 0.0, 10.0, 0.0]     # [m d^-1]

# ============================================================
# 2) INITIAL CONDITIONS
# ============================================================
N0 = np.linspace(0, 20, nz)          # [mmol N m^-3]
P0 = np.linspace(1, 0, nz)           # [mmol N m^-3]
Z0 = 0.2 * np.ones(nz)               # [mmol N m^-3]
D0 = 1.5 * np.ones(nz)               # [mmol N m^-3]
O0 = np.linspace(300, 0, nz)         # [mmol O2 m^-3]

y0 = np.concatenate([N0, P0, Z0, D0, O0])

# Convenient slices
iN = slice(0, nz)
iP = slice(nz, 2*nz)
iZ = slice(2*nz, 3*nz)
iD = slice(3*nz, 4*nz)
iO = slice(4*nz, 5*nz)

# ============================================================
# 3) TIME SETUP
# ============================================================
years = 5
t_max = 365.0 * years
t_span = (0.0, t_max)
t_eval = np.linspace(0.0, t_max, 1000)

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
    graze_lim = P  / (P  + k_Z + 1e-12)

    if oxyg_switch:
        # smooth step around k_O with width delta_O
        oxyg_lim = 0.5 * (1.0 + np.tanh((O - k_O) / (delta_O + 1e-12)))
    else:
        oxyg_lim = O / (O + k_O + 1e-12)

    return light_lim, nut_lim, graze_lim, oxyg_lim

# ============================================================
# 6) SEASONAL FORCING: KAPPA + LIGHT
# ============================================================
def getLIGHTandKAPPAS(t, P=None, D=None,Lightswitch = True, Seasonality=True, bio_attenuation=True):
    """
    Returns:
      kappa_interface : [m^2 d^-1] at interfaces, size nz+1
      Lz              : [µmol photons m^-2 s^-1] at cell centers
      L0              : surface light [µmol photons m^-2 s^-1]
    """
    # light attenuation
    k_water = 0.1   # [m^-1]
    k_bio   = 0.20  # [m^2 mmol^-1] (since P,D are mmol m^-3 and integral uses dz)

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
        zMix       = 0.05 * depth
        zMixWinter = 0.8  * depth
        tMaxSpring = 90.0
        zetaMaxSteep = 2.0
        z_mix = 0.5 * (1 - np.sin(2*np.pi*(doy - tMaxSpring)/365.0))**zetaMaxSteep * (zMixWinter - zMix) + zMix

        # --- Seasonal surface diffusivity kappa_top(t) [m^2 d^-1] ---
        kappa_top_summer = 5.0
        kappa_top_winter = 15
        kappa_bottom_summer = 0.5
        kappa_bottom_winter = 15
        
        season_shape = 0.5 * (1 - np.sin(2*np.pi*(doy - tMaxSpring)/365.0)) 
        kappa_top = kappa_top_summer + (kappa_top_winter - kappa_top_summer) * (season_shape**zetaMaxSteep)
        kappa_bottom = kappa_bottom_summer + (kappa_bottom_winter - kappa_bottom_summer) * (season_shape ** zetaMaxSteep)
        zeta_mix = 10.0
        kappa_center = 0.5*(1 - np.tanh((z - z_mix)/zeta_mix))*(kappa_top - kappa_bottom) + kappa_bottom

        # --- Seasonal surface light L0(t) ---
        L0_min = 50.0
        L0_max = 1000.0
        if Lightswitch:
            # --- Seasonal surface light L0(t) using double tanh ---
            #Timing (days of year)
            spring_center  = 90.0   # center of spring transition
            autumn_center  = 260.0   # center of autumn transition
            # Controls sharpness (smaller = sharper)
            spring_width = 20.0
            autumn_width = 30.0
            # Spring switch: 0 → 1
            spring_switch = 0.5 * (1 + np.tanh((doy - spring_center) / spring_width))
            # Autumn switch: 1 → 0
            autumn_switch = 0.5 * (1 - np.tanh((doy - autumn_center) / autumn_width))
            seasonal_shape = spring_switch * autumn_switch
            L0 = L0_min + (L0_max - L0_min) * seasonal_shape
        else:
            phase_shift = 80.0
            seasonal_shape = (1 + np.sin(2*np.pi*(doy - phase_shift)/365.0))**2
            L0 = L0_min + (L0_max - L0_min) * seasonal_shape / 4.0

    else:
        kappa_surface = 5.0
        kappa_bottom  = 1.0
        z_transition  = 50.0
        zeta_mix      = 10.0
        kappa_center = 0.5*(1 - np.tanh((z - z_transition)/zeta_mix))*(kappa_surface - kappa_bottom) + kappa_bottom
        L0 = 1400.0

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
def surface_flux(tracer_name, C_surface):
    """Return downward flux J[0] [mmol m^-2 d^-1] (positive into ocean)."""
    if tracer_name == "O":
        return (O2_atm - C_surface) * k_O_surf
    return 0.0

def bottom_flux(tracer_name, C_bottom):
    """Return downward flux J[nz] [mmol m^-2 d^-1] (positive removes tracer into sediments)."""
    if tracer_name == "O":
        O_use = max(C_bottom, 0.0)
        return (O_use + O2_n) * k_O_bot
    return 0.0

def vertical_transport(C, kappa_interface, w=0.0, tracer_name=""):
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
    J[0]  = surface_flux(tracer_name, C[0])
    J[nz] = bottom_flux(tracer_name, C[-1])

    return -(J[1:] - J[:-1]) / dz

# ============================================================
# 8) ODE RHS
# ============================================================
def rhs(t, y):
    N = y[iN]
    P = y[iP]
    Z = y[iZ]
    D = y[iD]
    O = y[iO]

    # forcing
    kappa_interface, Lz, _ = getLIGHTandKAPPAS(t, P=P, D=D)

    # limitations
    light_lim, nut_lim, graze_lim, oxyg_lim = get_limits(Lz, N, P, O)

    # rates
    P_growth = P_growth_max * np.minimum(light_lim, nut_lim)
    Z_consum = Z_consum_max * np.minimum(graze_lim, oxyg_lim)

    eps = 0.1  # [-] minimum aerobic fraction
    r_factor = r * (eps + (1 - eps) * oxyg_lim)

    # processes
    N_uptake  = P_growth * P
    P_mort    = m_P * P
    Z_grazing = Z_consum * Z
    Z_mort    = m_Z * Z**2
    remin     = r_factor * D

    # reactions
    dN_reac = -N_uptake + e_N*Z_grazing + remin
    dP_reac =  N_uptake - Z_grazing - P_mort
    dZ_reac = (1 - e_N - e_D)*Z_grazing - Z_mort
    dD_reac =  P_mort + e_D*Z_grazing + Z_mort - remin
    dO_reac = -y_N*e_N*Z_grazing - y_N*remin + y_P*N_uptake

    # transport + reactions
    dN = vertical_transport(N, kappa_interface, w=W[0], tracer_name="N") + dN_reac
    dP = vertical_transport(P, kappa_interface, w=W[1], tracer_name="P") + dP_reac
    dZ = vertical_transport(Z, kappa_interface, w=W[2], tracer_name="Z") + dZ_reac
    dD = vertical_transport(D, kappa_interface, w=W[3], tracer_name="D") + dD_reac
    dO = vertical_transport(O, kappa_interface, w=W[4], tracer_name="O") + dO_reac

    # (optional) simple non-negativity safeguard for biology-only vars
    # NOTE: we do NOT clip before transport; solver can handle small negatives.
    N = np.maximum(N, 0.0)
    P = np.maximum(P, 0.0)
    Z = np.maximum(Z, 0.0)
    D = np.maximum(D, 0.0)
    O = np.maximum(O, 0.0)

    return np.concatenate([dN, dP, dZ, dD, dO])

# ============================================================
# 9) RUN SIMULATION
# ============================================================
sol = solve_ivp(
    rhs, t_span, y0, t_eval=t_eval,
    method="BDF", rtol=1e-7, atol=1e-10, max_step=1.0
)

# ============================================================
# 10) PLOTTING HELPERS
# ============================================================
def set_time_axis_seasons_if_last365(ax, t_plot, only_last365):
    """
    If plotting last 365 days, relabel x-axis as seasons (0..365).
    Otherwise keep days.
    """
    if only_last365:
        # Convert to "day-of-year within last year": 0..365
        ax.set_xlim(0, 365)
        season_ticks(ax)
        ax.set_xlabel("Season")
    else:
        ax.set_xlabel("Time [days]")

# ============================================================
# 11) HEATMAPS (LAST 365 DAYS)
# ============================================================
only_last365 = True

if only_last365:
    t_mask = sol.t >= (t_max - 365.0)
    t_plot = sol.t[t_mask] - (t_max - 365.0)   # 0..365 for last year
else:
    t_mask = slice(None)
    t_plot = sol.t

N_all = sol.y[iN, t_mask]
P_all = sol.y[iP, t_mask]
Z_all = sol.y[iZ, t_mask]
D_all = sol.y[iD, t_mask]
O_all = sol.y[iO, t_mask]
S_all = np.zeros_like(N_all)  # same shape [nz, nt]

# --- sediment placeholder (example: bottom flux time series, repeated over depth) ---
# If you don't have sediment state yet, plot bottom O2 flux as a diagnostic placeholder.
# J_bot_O2 is [mmol m^-2 d^-1]; we "paint" it over depth just for a placeholder panel.
if only_last365:
    # times in absolute model time for flux calculation
    t_abs = sol.t[t_mask]
else:
    t_abs = sol.t

O_bottom_ts = sol.y[iO, t_mask][-1, :]  # bottom cell O2 over time
J_bot_O2_ts = np.array([bottom_flux("O", ob) for ob in O_bottom_ts])  # [mmol m^-2 d^-1]

fig, axs = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)

im0 = axs[0, 0].contourf(t_plot, -z, N_all, levels=50)
axs[0, 0].set_title("N [mmol N m$^{-3}$]")
axs[0, 0].set_ylabel("Depth [m]")
fig.colorbar(im0, ax=axs[0, 0])

im1 = axs[0, 1].contourf(t_plot, -z, P_all, levels=50)
axs[0, 1].set_title("P [mmol N m$^{-3}$]")
fig.colorbar(im1, ax=axs[0, 1])

im2 = axs[1, 0].contourf(t_plot, -z, Z_all, levels=50)
axs[1, 0].set_title("Z [mmol N m$^{-3}$]")
fig.colorbar(im2, ax=axs[1, 0])

im3 = axs[1, 1].contourf(t_plot, -z, D_all, levels=50)
axs[1, 1].set_title("D [mmol N m$^{-3}$]")
axs[1, 1].set_ylabel("Depth [m]")
fig.colorbar(im3, ax=axs[1, 1])

im4 = axs[0, 2].contourf(t_plot, -z, O_all, levels=50)
axs[0, 2].set_title("O$_2$ [mmol O$_2$ m$^{-3}$]")
fig.colorbar(im4, ax=axs[0, 2])

im5 = axs[1, 2].contourf(t_plot, -z, S_all, levels=50)
axs[1, 2].set_title("Sediment [placeholder]")
fig.colorbar(im5, ax=axs[1, 2])

# --- axis formatting ---
for ax in axs.flat:
    ax.invert_yaxis()
    ax.grid(False)

# Use seasonal tick labels ONLY when last365, but DO NOT write xlabel "Season"
if only_last365:
    season_positions = [0, 80, 172, 264, 355]
    season_labels = ["Winter", "Spring", "Summer", "Autumn", "Winter"]
    for ax in axs[1, :]:  # label only the bottom row to avoid clutter
        ax.set_xticks(season_positions)
        ax.set_xticklabels(season_labels)

    # remove xlabels entirely (so "Season" doesn't show up)
    for ax in axs.flat:
        ax.set_xlabel("")
else:
    for ax in axs[1, :]:
        ax.set_xlabel("Time [days]")

plt.tight_layout()
plt.show()

# ============================================================
# 12) PROFILE + LIMITATIONS (WITH SEASON LABEL)
# ============================================================
time_index = -172
t_selected = sol.t[time_index]
doy_sel = day_of_year(t_selected)
season_sel = season_from_doy(doy_sel)
year_sel = int(t_selected // 365) + 1

N_sel = sol.y[iN, time_index]
P_sel = sol.y[iP, time_index]
Z_sel = sol.y[iZ, time_index]
D_sel = sol.y[iD, time_index]
O_sel = sol.y[iO, time_index]

T_sel = N_sel + P_sel + Z_sel + D_sel

kappa_interface, Lz, _ = getLIGHTandKAPPAS(t_selected, P=P_sel, D=D_sel)
light_lim, nut_lim, graze_lim, oxyg_lim = get_limits(Lz, N_sel, P_sel, O_sel)

fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# Left: profiles
axs[0].plot(np.maximum(N_sel, 1e-8), -z, label="N", lw=2, color = 'darkblue')
axs[0].plot(np.maximum(P_sel, 1e-8), -z, label="P", lw=2, color = 'green')
axs[0].plot(np.maximum(Z_sel, 1e-8), -z, label="Z", lw=2, color = 'orange')
axs[0].plot(np.maximum(D_sel, 1e-8), -z, label="D", lw=2, color = 'red')
axs[0].plot(np.maximum(T_sel, 1e-8), -z, label="Total N", lw=2, ls="--", color = 'black')

axs[0].set_xscale("log")
axs[0].set_xlabel("Concentration (log)")
axs[0].set_ylabel("Depth [m]")
axs[0].invert_yaxis()
axs[0].grid(True, which="both")
axs[0].legend()
axs[0].set_title(f"Profiles (Year {year_sel}, DOY {doy_sel:.0f} ~ {season_sel})")

# Right: limitations
axs[1].plot(light_lim, -z, label="Light", lw=2, color = 'orange')
axs[1].plot(nut_lim,   -z, label="Nutrient", lw=2, color = 'darkblue')
axs[1].plot(graze_lim, -z, label="Grazing", lw=2, ls="--", color = 'red')
axs[1].plot(oxyg_lim,  -z, label="Oxygen", lw=2, ls="--", color = 'black')

axs[1].set_xlabel("Limitation (0–1)")
axs[1].invert_yaxis()
axs[1].grid(True)
axs[1].legend()
axs[1].set_title(f"Limitations (Year {year_sel}, DOY {doy_sel:.0f} ~ {season_sel})")

plt.tight_layout()
plt.show()

# ============================================================
# 13) SEASONAL PROFILES (KAPPA, LIGHT, O2)
# ============================================================
season_days = {"Winter": 0, "Spring": 90, "Summer": 172, "Autumn": 265}

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])

# κ profiles
ax1 = fig.add_subplot(gs[0, 0])
for season, day in season_days.items():
    kappa_prof, _, _ = getLIGHTandKAPPAS(day)
    z_kappa = np.r_[z, z[-1] + dz]
    ax1.plot(kappa_prof, z_kappa, label=season)
ax1.set_xlabel("κ [m$^2$ d$^{-1}$]")
ax1.set_ylabel("Depth [m]")
ax1.set_title("Seasonal κ profiles")
ax1.invert_yaxis()
ax1.grid(True)
ax1.legend()

# Light profiles
ax2 = fig.add_subplot(gs[0, 1])
for season, day in season_days.items():
    _, L_profile, _ = getLIGHTandKAPPAS(day)
    ax2.plot(L_profile, z, label=season)
ax2.set_xlabel("Light [µmol m$^{-2}$ s$^{-1}$]")
ax2.set_ylabel("Depth [m]")
ax2.set_title("Seasonal light profiles")
ax2.invert_yaxis()
ax2.grid(True)
ax2.legend()

# O2 profiles from LAST YEAR
ax3 = fig.add_subplot(gs[0, 2])
for season, day in season_days.items():
    t_target = t_max - 365.0 + day
    idx = np.argmin(np.abs(sol.t - t_target))
    O_profile = sol.y[iO, idx]
    ax3.plot(O_profile, z, label=season)
ax3.set_xlabel("O$_2$ [mmol m$^{-3}$]")
ax3.set_ylabel("Depth [m]")
ax3.set_title("Seasonal O$_2$ profiles (last year)")
ax3.invert_yaxis()
ax3.grid(True)
ax3.legend()

# L0 over year (season ticks)
ax4 = fig.add_subplot(gs[1, :])
days = np.arange(0, 365)
L0_year = np.array([getLIGHTandKAPPAS(d)[2] for d in days])
ax4.plot(days, L0_year, color = 'orange')
season_ticks(ax4)
ax4.set_xlabel("Season")
ax4.set_ylabel("Surface light L0 [µmol m$^{-2}$ s$^{-1}$]")
ax4.set_title("Seasonal surface light over year")
ax4.grid(True)

plt.tight_layout()

plt.show()
