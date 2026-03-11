# ============================================================
# START OF SCRIPT

# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# %% PARAMETER INPUT

# ============================================================
# 1) MODEL PARAMETERS AND UNITS
# ============================================================
# Vertical 1D NPZD-O model
#
# State variables at cell centers:
#   N      : dissolved inorganic nitrogen              [mmol N m^-3]
#   P      : phytoplankton biomass                     [mmol N m^-3]
#   Z      : zooplankton biomass                       [mmol N m^-3]
#   D      : detritus                                  [mmol N m^-3]
#   O      : dissolved oxygen                          [mmol O2 m^-3]
#   N_sed  : sediment nitrogen inventory               [mmol N m^-2]
#   D_sed  : sediment detritus inventory               [mmol N m^-2]
#
# Transport:
#   kappa  : vertical diffusivity                      [m^2 d^-1]
#   w      : sinking/advection velocity                [m d^-1]
#
# Flux convention:
#   J > 0 means downward flux                          [mmol m^-2 d^-1]

# ------------------------------------------------------------
# 1A) Biological rate parameters
# ------------------------------------------------------------
P_growth_max = 1.5     # phytoplankton max growth rate              [d^-1]
Z_consum_max = 0.7     # zooplankton max consumption rate           [d^-1]

k_L = 80.0             # light half-saturation                      [µmol photons m^-2 s^-1]
k_N = 0.4              # nutrient half-saturation                   [mmol N m^-3]
k_Z = 4.0              # grazing half-saturation wrt prey           [mmol N m^-3]
k_O = 100.0            # critical oxygen threshold                  [mmol O2 m^-3]

m_P = 0.25             # phytoplankton mortality                    [d^-1]
m_Z = 0.2              # zooplankton quadratic mortality            [d^-1]

e_N = 0.3              # fraction of grazing excreted to DIN        [-]
e_D = 0.3              # fraction of grazing routed to detritus     [-]

r_ae = 0.07            # aerobic detritus remineralization rate     [d^-1]
r_an = 0.005           # anaerobic detritus remineralization rate   [d^-1]

# Stoichiometry
y_P = 9.0              # oxygen produced per N uptake               [mmol O2 / mmol N]
y_N = 6.625            # oxygen consumed per aerobic N recycling    [mmol O2 / mmol N]

# ------------------------------------------------------------
# 1B) Boundary / sediment parameters
# ------------------------------------------------------------
O2_atm = 260.0         # atmospheric / equilibrium oxygen           [mmol O2 m^-3]

k_N_bot = 10.0         # sediment-water N exchange velocity         [m d^-1]
z_sed = 0.05           # sediment box thickness                     [m]

# ------------------------------------------------------------
# 1C) Domain and grid
# ------------------------------------------------------------
depth = 20.0
nz = 50

dz = depth / nz
z = np.linspace(dz/2, depth - dz/2, nz)
z_edges = np.linspace(0.0, depth, nz + 1)   # cell interfaces / edges

# ------------------------------------------------------------
# 1D) Sinking / advection velocities
# ------------------------------------------------------------
# Order corresponds to: N, P, Z, D, O
W = [0.0, 0.0, 0.0, 5.0, 0.0]     # downward velocity               [m d^-1]


# ============================================================
# 2) INITIAL CONDITIONS
# ============================================================
# Initial vertical profiles for all tracers.
# Note: the second N0 line overwrites the first one, preserved
# exactly from your original script.

N0 = 20.0 + 8.0 * (z / depth)     # initial nutrient profile        [mmol N m^-3]
N0 = 100.0 + 8.0 * (z / depth)    # overwritten nutrient profile    [mmol N m^-3]

P0 = 1.0 * np.exp(-z / 3.0) + 0.01   # initial phytoplankton        [mmol N m^-3]
Z0 = 0.2 * np.exp(-z / 5.0) + 0.02   # initial zooplankton          [mmol N m^-3]
D0 = 0.2 + 1.5 * (z / depth)         # initial detritus             [mmol N m^-3]
O0 = 260.0 - 40.0 * (z / depth)      # initial oxygen               [mmol O2 m^-3]

Nsed0 = N0[-1] * z_sed            # initial sediment nitrogen        [mmol N m^-2]
Dsed0 = D0[-1] * z_sed            # initial sediment detritus        [mmol N m^-2]

# Full model state vector
y0 = np.concatenate([N0, P0, Z0, D0, O0, [Nsed0], [Dsed0]])

# Convenient index slices into the state vector
iN    = slice(0, nz)
iP    = slice(nz, 2 * nz)
iZ    = slice(2 * nz, 3 * nz)
iD    = slice(3 * nz, 4 * nz)
iO    = slice(4 * nz, 5 * nz)
iNsed = 5 * nz
iDsed = 5 * nz + 1


# ============================================================
# 3) TIME SETUP
# ============================================================
years = 3                                    # simulation length       [years]
t_max = 365.0 * years                        # total simulation time   [days]
t_span = (0.0, t_max)                        # integration interval    [days]
t_eval = np.linspace(0.0, t_max, 1000)       # output times            [days]

# %% FUNCTION SETUP

# ============================================================
# 4) SMALL UTILITY FUNCTIONS
# ============================================================
def day_of_year(t):
    """Return day-of-year from model time t [days]."""
    return float(t % 365.0)


def season_from_doy(doy):
    """Map day-of-year to a simple 4-season label."""
    if doy < 90:
        return "Winter"
    if doy < 172:
        return "Spring"
    if doy < 265:
        return "Summer"
    return "Autumn"


def season_ticks(ax):
    """Apply seasonal x-axis ticks to a plot."""
    season_positions = [0, 80, 172, 264, 355]
    season_labels = ["Winter", "Spring", "Summer", "Autumn", "Winter"]
    ax.set_xticks(season_positions)
    ax.set_xticklabels(season_labels)


# ============================================================
# 5) LIMITATION FUNCTIONS
# ============================================================
def get_limits(Lz, N, P, O, delta_O=10.0, oxyg_switch=True):
    """
    Compute biological limitation factors in the range [0, 1].

    Parameters
    ----------
    Lz : light profile                                  [µmol photons m^-2 s^-1]
    N  : dissolved inorganic nitrogen                   [mmol N m^-3]
    P  : phytoplankton biomass                          [mmol N m^-3]
    O  : dissolved oxygen                               [mmol O2 m^-3]

    Returns
    -------
    light_lim : light limitation                        [-]
    nut_lim   : nutrient limitation                     [-]
    graze_lim : prey-dependent grazing limitation       [-]
    oxyg_lim  : oxygen limitation                       [-]
    """
    light_lim = Lz / (Lz + k_L + 1e-12)
    nut_lim = N / (N + k_N + 1e-12)

    n = 1.5
    graze_lim = P**n / (P**n + k_Z**n + 1e-12)

    if oxyg_switch:
        oxyg_lim = 0.5 * (1.0 + np.tanh((O - k_O) / (delta_O + 1e-12)))
    else:
        oxyg_lim = O / (O + k_O + 1e-12)

    return light_lim, nut_lim, graze_lim, oxyg_lim


# ============================================================
# 6) SEASONAL FORCING: MIXING + LIGHT
# ============================================================
def getLIGHTandKAPPAS(t, P=None, D=None, Lightswitch=False, Seasonality=True, bio_attenuation=True):
    """
    Return vertical diffusivity and light profile at time t.

    Returns
    -------
    kappa_interface : diffusivity at cell interfaces    [m^2 d^-1]
    Lz              : light at cell centers             [µmol photons m^-2 s^-1]
    L0              : surface light                     [µmol photons m^-2 s^-1]
    """
    # Background optical attenuation
    k_water = 0.25   # pure-water attenuation coefficient         [m^-1]
    k_bio = 0.05     # biological self-shading coefficient        [m^2 mmol^-1]

    # Depth-integrated biological shading term
    if bio_attenuation and (P is not None) and (D is not None):
        P_pos = np.maximum(P, 0.0)
        D_pos = np.maximum(D, 0.0)
        bio_integral = np.cumsum(P_pos + D_pos) * dz
    else:
        bio_integral = 0.0

    if Seasonality:
        doy = day_of_year(t)

        # --------------------------------------------------------
        # Seasonal mixing depth
        # --------------------------------------------------------
        zMix = 0.1 * depth
        zMixWinter = 0.8 * depth
        tMaxSpring = 90
        zetaMaxSteep = 2.0

        z_mix = (
            0.5
            * (1 - np.sin(2 * np.pi * (doy - tMaxSpring) / 365.0)) ** zetaMaxSteep
            * (zMixWinter - zMix)
            + zMix
        )

        # --------------------------------------------------------
        # Seasonal diffusivity
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Seasonal surface light
        # --------------------------------------------------------
        L0_min = 50.0
        L0_max = 900.0

        if Lightswitch:
            spring_center = 120.0
            autumn_center = 210.0
            spring_width = 30.0
            autumn_width = 30.0

            spring_switch = 0.5 * (1 + np.tanh((doy - spring_center) / spring_width))
            autumn_switch = 0.5 * (1 - np.tanh((doy - autumn_center) / autumn_width))
            seasonal_shape = spring_switch * autumn_switch

            L0 = L0_min + (L0_max - L0_min) * seasonal_shape
        else:
            phase_shift = 80.0
            seasonal_shape = (1 + np.sin(2 * np.pi * (doy - phase_shift) / 365.0)) ** 3
            L0 = L0_min + (L0_max - L0_min) * seasonal_shape / 4.0

    else:
        # --------------------------------------------------------
        # Non-seasonal fallback forcing
        # --------------------------------------------------------
        kappa_surface = 10.0
        kappa_bottom = 5.0
        z_transition = 50.0
        zeta_mix = 10.0

        kappa_center = (
            0.5 * (1 - np.tanh((z - z_transition) / zeta_mix)) * (kappa_surface - kappa_bottom)
            + kappa_bottom
        )

        L0 = 1000.0

    # Diffusivity at interfaces
    kappa_interface = np.zeros(nz + 1)
    kappa_interface[1:nz] = 0.5 * (kappa_center[1:] + kappa_center[:-1])
    kappa_interface[0] = kappa_center[0]
    kappa_interface[nz] = kappa_center[-1]

    # Light profile
    if bio_attenuation and (P is not None) and (D is not None):
        Lz = L0 * np.exp(-k_water * z - k_bio * bio_integral)
    else:
        Lz = L0 * np.exp(-k_water * z)

    return kappa_interface, Lz, L0


# ============================================================
# 7) BOUNDARY FLUXES AND VERTICAL TRANSPORT
# ============================================================
def kappa_scaled(kappa_surface, kappa_ref=10.0, k_O_ref=10.0):
    """
    Scale the air-sea gas transfer velocity with surface diffusivity.

    Parameters
    ----------
    kappa_surface : surface diffusivity                 [m^2 d^-1]

    Returns
    -------
    k_O_surf : surface oxygen exchange velocity         [m d^-1]
    """
    return k_O_ref * (kappa_surface / kappa_ref)


def surface_flux(tracer_name, C_surface, kappa_surface):
    """
    Surface boundary flux J[0].

    Positive is downward into the ocean.                [mmol m^-2 d^-1]
    """
    if tracer_name == "O":
        k_O_surf = kappa_scaled(kappa_surface)
        return (O2_atm - C_surface) * k_O_surf

    return 0.0


def bottom_flux(tracer_name, C_bottom, sed=None):
    """
    Bottom boundary flux J[nz].

    Positive is downward into sediments.                [mmol m^-2 d^-1]
    """
    if tracer_name == "N":
        Nsed = 0.0 if sed is None else sed
        return -k_N_bot * (Nsed / z_sed - C_bottom)

    elif tracer_name == "D":
        return W[3] * C_bottom

    return 0.0


def vertical_transport(C, kappa_interface, w=0.0, tracer_name="", sed=None):
    """
    Vertical 1D advection-diffusion operator in flux form.

    Parameters
    ----------
    C               : tracer concentration              [mmol m^-3 or mmol O2 m^-3]
    kappa_interface : interface diffusivity             [m^2 d^-1]
    w               : downward advection/sinking        [m d^-1]

    Returns
    -------
    dCdt_transport  : transport tendency                [mmol m^-3 d^-1]
    """
    J = np.zeros(nz + 1)

    # Interior advection
    if w >= 0:
        Ja = w * C[:-1]
    else:
        Ja = w * C[1:]

    # Interior diffusion
    Jd = -kappa_interface[1:nz] * (C[1:] - C[:-1]) / dz
    J[1:nz] = Ja + Jd

    # Boundary fluxes
    J[0] = surface_flux(tracer_name, C[0], kappa_interface[0])
    J[nz] = bottom_flux(tracer_name, C[-1], sed=sed)

    return -(J[1:] - J[:-1]) / dz

# ============================================================
# 8) MODEL RIGHT-HAND SIDE
# ============================================================
def rhs(t, y):
    """
    Full NPZD-O model tendency function.

    Returns
    -------
    dydt : time tendency of the full state vector       [state units per day]
    """
    N = y[iN]
    P = y[iP]
    Z = y[iZ]
    D = y[iD]
    O = y[iO]
    Nsed = y[iNsed]
    Dsed = y[iDsed]

    # Forcing
    kappa_interface, Lz, _ = getLIGHTandKAPPAS(t, P=P, D=D)

    # Limitation functions
    light_lim, nut_lim, graze_lim, oxyg_lim = get_limits(Lz, N, P, O)

    # Biological rates
    P_growth = P_growth_max * np.minimum(light_lim, nut_lim)
    Z_consum = Z_consum_max * np.minimum(graze_lim, oxyg_lim)

    # Process rates
    N_uptake = P_growth * P
    P_mort = m_P * P
    Z_grazing = Z_consum * Z
    Z_mort = m_Z * Z**2

    # Water-column remineralization
    remin_ae_wc = r_ae * oxyg_lim * D
    remin_an_wc = r_an * D
    remin_wc = remin_ae_wc + remin_an_wc

    # Sediment remineralization
    ox_bot = oxyg_lim[-1]
    remin_ae_bot = r_ae * ox_bot * Dsed
    remin_an_bot = r_an * Dsed
    remin_bot = remin_ae_bot + remin_an_bot

    # Boundary fluxes
    J_D_bot = bottom_flux("D", D[-1])
    J_N_bot = bottom_flux("N", N[-1], sed=Nsed)

    # Reaction tendencies
    dN_reac = -N_uptake + e_N * Z_grazing + remin_wc
    dP_reac = N_uptake - Z_grazing - P_mort
    dZ_reac = (1 - e_N - e_D) * Z_grazing - Z_mort
    dD_reac = P_mort + e_D * Z_grazing + Z_mort - remin_wc
    dO_reac = -y_N * e_N * Z_grazing - y_N * remin_ae_wc + y_P * N_uptake

    # Transport + reaction
    dN = vertical_transport(N, kappa_interface, w=W[0], tracer_name="N", sed=Nsed) + dN_reac
    dP = vertical_transport(P, kappa_interface, w=W[1], tracer_name="P") + dP_reac
    dZ = vertical_transport(Z, kappa_interface, w=W[2], tracer_name="Z") + dZ_reac
    dD = vertical_transport(D, kappa_interface, w=W[3], tracer_name="D") + dD_reac
    dO = vertical_transport(O, kappa_interface, w=W[4], tracer_name="O") + dO_reac

    # Bottom-water oxygen demand from benthic aerobic remineralization
    dO[-1] -= y_N * remin_ae_bot / dz

    # Sediment box tendencies
    dDsed = J_D_bot - remin_bot
    dNsed = J_N_bot + remin_bot

    return np.concatenate([dN, dP, dZ, dD, dO, [dNsed], [dDsed]])

# %% RUN SIMULATION

# ============================================================
# 9) RUN SIMULATION
# ============================================================
sol = solve_ivp(
    rhs,
    t_span,
    y0,
    t_eval=t_eval,
    method="BDF",
    rtol=1e-7,
    atol=1e-10,
    max_step=1.0
)

# %% PLOTTING CELL

# ============================================================
# 10) PLOTTING HELPER
# ============================================================
def set_time_axis_seasons_if_last365(ax, t_plot, only_last365):
    """Apply seasonal x-axis if plotting the final model year."""
    if only_last365:
        ax.set_xlim(0, 365)
        season_ticks(ax)
        ax.set_xlabel("Season")
    else:
        ax.set_xlabel("Time [days]")


# ============================================================
# 11) PLOTTING SETTINGS
# ============================================================
plot_top = False
top_m = 50.0

if plot_top:
    z_mask = z <= top_m
else:
    z_mask = slice(None)

z_plot = z[z_mask]
z_kappa_full = z_edges

if plot_top:
    z_kappa_mask = z_kappa_full <= top_m
else:
    z_kappa_mask = slice(None)
z_kappa_plot = z_kappa_full[z_kappa_mask]

# ============================================================
# 12) HEATMAPS FOR FINAL MODEL YEAR
# ============================================================
only_last365 = True

if only_last365:
    t_mask = sol.t >= (t_max - 365.0)
    t_plot = sol.t[t_mask] - (t_max - 365.0)
else:
    t_mask = slice(None)
    t_plot = sol.t

dt = t_plot[1] - t_plot[0]
t_edges = np.linspace(t_plot[0] - dt/2, t_plot[-1] + dt/2, len(t_plot) + 1)

N_all = sol.y[iN][:, t_mask][z_mask, :]
P_all = sol.y[iP][:, t_mask][z_mask, :]
Z_all = sol.y[iZ][:, t_mask][z_mask, :]
D_all = sol.y[iD][:, t_mask][z_mask, :]
O_all = sol.y[iO][:, t_mask][z_mask, :]

Nsed_all = sol.y[iNsed, t_mask]
Dsed_all = sol.y[iDsed, t_mask]
N_bottom = sol.y[iN][:, t_mask][-1, :]

fig_hm, axs_hm = plt.subplots(2, 3, figsize=(12, 7), sharex=True)

im0 = axs_hm[0, 0].pcolormesh(t_edges, z_edges, N_all, shading="auto")
axs_hm[0, 0].set_title("N [mmol N m$^{-3}$]")
axs_hm[0, 0].set_ylabel("Depth [m]")
fig_hm.colorbar(im0, ax=axs_hm[0, 0])

im1 = axs_hm[0, 1].pcolormesh(t_edges, z_edges, P_all, shading="auto")
axs_hm[0, 1].set_title("P [mmol N m$^{-3}$]")
fig_hm.colorbar(im1, ax=axs_hm[0, 1])

im2 = axs_hm[1, 0].pcolormesh(t_edges, z_edges, Z_all, shading="auto")
axs_hm[1, 0].set_title("Z [mmol N m$^{-3}$]")
axs_hm[1, 0].set_ylabel("Depth [m]")
fig_hm.colorbar(im2, ax=axs_hm[1, 0])

im3 = axs_hm[1, 1].pcolormesh(t_edges, z_edges, D_all, shading="auto")
axs_hm[1, 1].set_title("D [mmol N m$^{-3}$]")
axs_hm[1, 1].set_ylabel("Depth [m]")
fig_hm.colorbar(im3, ax=axs_hm[1, 1])

im4 = axs_hm[0, 2].pcolormesh(t_edges, z_edges, O_all, shading="auto")
axs_hm[0, 2].set_title("O$_2$ [mmol O$_2$ m$^{-3}$]")
fig_hm.colorbar(im4, ax=axs_hm[0, 2])

axs_hm[1, 2].plot(t_plot, Nsed_all / z_sed, label="Nsed", lw=2, color="darkblue")
axs_hm[1, 2].plot(t_plot, Dsed_all / z_sed, label="Dsed", lw=2, color="red")
axs_hm[1, 2].plot(t_plot, N_bottom, ls="--", color="black", label="Bottom N")
axs_hm[1, 2].set_title(r"Sediment concentrations [mmol N m$^{-3}$]")
axs_hm[1, 2].set_yscale("log")
axs_hm[1, 2].set_ylabel(r"Concentration [mmol N m$^{-3}$]")
axs_hm[1, 2].grid(True, which="both")
axs_hm[1, 2].legend()

for ax in [axs_hm[0, 0], axs_hm[0, 1], axs_hm[0, 2], axs_hm[1, 0], axs_hm[1, 1]]:
    ax.invert_yaxis()

if only_last365:
    season_positions = [0, 80, 172, 264, 355]
    season_labels = ["Winter", "Spring", "Summer", "Autumn", "Winter"]
    for ax in axs_hm[1, :]:
        ax.set_xticks(season_positions)
        ax.set_xticklabels(season_labels)
else:
    for ax in axs_hm[1, :]:
        ax.set_xlabel("Time [days]")

plt.tight_layout()
plt.show()


# ============================================================
# 13) PROFILE + LIMITATIONS FOR A SELECTED DAY
# ============================================================
selected_day = 172   # selected day within the final year        [days]

t_target = t_max - 365.0 + selected_day
time_index = np.argmin(np.abs(sol.t - t_target))

t_selected = sol.t[time_index]
doy_sel = day_of_year(t_selected)
season_sel = season_from_doy(doy_sel)
year_sel = int(t_selected // 365) + 1

N_sel = sol.y[iN, time_index][z_mask]
P_sel = sol.y[iP, time_index][z_mask]
Z_sel = sol.y[iZ, time_index][z_mask]
D_sel = sol.y[iD, time_index][z_mask]
O_sel = sol.y[iO, time_index][z_mask]

T_sel = N_sel + P_sel + Z_sel + D_sel

kappa_interface_full, Lz_full, _ = getLIGHTandKAPPAS(
    t_selected,
    P=sol.y[iP, time_index],
    D=sol.y[iD, time_index]
)
Lz = Lz_full[z_mask]

light_lim, nut_lim, graze_lim, oxyg_lim = get_limits(Lz, N_sel, P_sel, O_sel)

fig_prof, axs_prof = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

axs_prof[0].plot(np.maximum(N_sel, 1e-8), z_plot, label="N", lw=2, color="darkblue")
axs_prof[0].plot(np.maximum(P_sel, 1e-8), z_plot, label="P", lw=2, color="green")
axs_prof[0].plot(np.maximum(Z_sel, 1e-8), z_plot, label="Z", lw=2, color="orange")
axs_prof[0].plot(np.maximum(D_sel, 1e-8), z_plot, label="D", lw=2, color="red")
axs_prof[0].plot(np.maximum(T_sel, 1e-8), z_plot, label="Total N", lw=2, ls="--", color="black")

axs_prof[0].set_xscale("log")
axs_prof[0].set_xlabel("Concentration [mmol N m$^{-3}$]")
axs_prof[0].set_ylabel("Depth [m]")
axs_prof[0].grid(True, which="both")
axs_prof[0].set_ylim(z_plot.max(), z_plot.min())
axs_prof[0].legend()
axs_prof[0].set_title(f"Profiles (Year {year_sel}, DOY {doy_sel:.0f} ~ {season_sel})")

axs_prof[1].plot(light_lim, z_plot, label="Light", lw=2, color="orange")
axs_prof[1].plot(nut_lim, z_plot, label="Nutrient", lw=2, color="darkblue")
axs_prof[1].plot(graze_lim, z_plot, label="Grazing", lw=2, ls="--", color="red")
axs_prof[1].plot(oxyg_lim, z_plot, label="Oxygen", lw=2, ls="--", color="black")

axs_prof[1].set_xlim(0, 1.05)
axs_prof[1].set_xlabel("Limitation [-]")
axs_prof[1].grid(True)
axs_prof[1].legend()
axs_prof[1].set_ylim(z_plot.max(), z_plot.min())
axs_prof[1].set_title("Growth limitations")

plt.tight_layout()
plt.show()


# ============================================================
# 14) SEASONAL P, Z, D PROFILES
# ============================================================
season_days_pzd = {
    "Winter": 0,
    "Spring": 90,
    "Summer": 172,
    "Autumn": 260
}

max_val = 0.0
for season, day in season_days_pzd.items():
    t_target = t_max - 365 + day
    idx = np.argmin(np.abs(sol.t - t_target))

    P_prof = sol.y[iP, idx][z_mask]
    Z_prof = sol.y[iZ, idx][z_mask]
    D_prof = sol.y[iD, idx][z_mask]

    max_val = max(max_val, P_prof.max(), Z_prof.max(), D_prof.max())

fig_seas, axs_seas = plt.subplots(1, 4, figsize=(12, 6), sharey=True)

for i, (season, day) in enumerate(season_days_pzd.items()):
    t_target = t_max - 365 + day
    idx = np.argmin(np.abs(sol.t - t_target))

    P_prof = sol.y[iP, idx][z_mask]
    Z_prof = sol.y[iZ, idx][z_mask]
    D_prof = sol.y[iD, idx][z_mask]

    axs_seas[i].plot(P_prof, z_plot, lw=2, color="green", label="P")
    axs_seas[i].plot(Z_prof, z_plot, lw=2, color="orange", label="Z")
    axs_seas[i].plot(D_prof, z_plot, lw=2, color="red", label="D")

    axs_seas[i].set_title(season)
    axs_seas[i].set_xlim(0, max_val)
    axs_seas[i].grid(True)

axs_seas[0].invert_yaxis()
axs_seas[-1].legend()

for ax in axs_seas:
    ax.set_ylim(20, 0)

fig_seas.supxlabel("Concentration [mmol N m$^{-3}$]")
fig_seas.supylabel("Depth [m]")

plt.tight_layout()
plt.show()


# ============================================================
# 15) SEASONAL FORCING AND OXYGEN PROFILES
# ============================================================
season_days = {"Winter": 0, "Spring": 90, "Summer": 172, "Autumn": 265}

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 6, height_ratios=[3, 1])

# ------------------------------------------------------------
# 15A) Diffusivity profiles
# ------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0:2])
for season, day in season_days.items():
    kappa_prof_full, _, _ = getLIGHTandKAPPAS(day)
    kappa_prof = kappa_prof_full[z_kappa_mask]
    ax1.plot(kappa_prof, z_kappa_plot, label=season)

ax1.set_xlabel("κ [m$^2$ d$^{-1}$]")
ax1.set_ylabel("Depth [m]")
ax1.set_title("Seasonal κ profiles")
ax1.invert_yaxis()
ax1.grid(True)
ax1.legend()

# ------------------------------------------------------------
# 15B) Light profiles
# ------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 2:4])
for season, day in season_days.items():
    _, L_profile_full, _ = getLIGHTandKAPPAS(day)
    L_profile = L_profile_full[z_mask]
    ax2.plot(L_profile, z_plot, label=season)

ax2.set_xlabel("Light [µmol m$^{-2}$ s$^{-1}$]")
ax2.set_ylabel("Depth [m]")
ax2.set_title("Seasonal light profiles")
ax2.invert_yaxis()
ax2.grid(True)
ax2.legend()

# ------------------------------------------------------------
# 15C) Oxygen profiles
# ------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 4:6])
o2_max = 0.0

for season, day in season_days.items():
    t_target = t_max - 365.0 + day
    idx = np.argmin(np.abs(sol.t - t_target))
    O_profile_full = sol.y[iO, idx]
    O_profile = O_profile_full[z_mask]
    ax3.plot(O_profile, z_plot, label=season)
    o2_max = max(o2_max, np.max(O_profile))

ax3.axvline(k_O, color="black", linestyle="--", linewidth=1, label=r"critical O$_2$ value")
ax3.set_xlabel("O$_2$ [mmol m$^{-3}$]")
ax3.set_ylabel("Depth [m]")
ax3.set_title("Seasonal O$_2$ profiles")
ax3.set_xlim(0, o2_max * 1.05)
ax3.invert_yaxis()
ax3.grid(True)
ax3.legend()

# ------------------------------------------------------------
# 15D) Surface light over the year
# ------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 0:3])
days = np.arange(0, 365)
L0_year = np.array([getLIGHTandKAPPAS(d)[2] for d in days])

ax4.plot(days, L0_year, color="orange")
season_ticks(ax4)
ax4.set_xlabel("Season")
ax4.set_ylabel("Surface light L0 [µmol m$^{-2}$ s$^{-1}$]")
ax4.set_title("Seasonal surface light")
ax4.grid(True)

# ------------------------------------------------------------
# 15E) Surface oxygen exchange over the year
# ------------------------------------------------------------
ax5 = fig.add_subplot(gs[1, 3:6])
kappa_surface_year = np.array([getLIGHTandKAPPAS(d)[0][0] for d in days])

kappa_ref = 10.0
k_O_ref = 20.0
k_O_surf_year = k_O_ref * np.sqrt(kappa_surface_year / kappa_ref)

ax5.plot(days, k_O_surf_year, color="black")
season_ticks(ax5)
ax5.set_xlabel("Season")
ax5.set_ylabel(r"$k_{O,\mathrm{surf}}$ [m d$^{-1}$]")
ax5.set_title(r"Surface gas exchange $k_{O,\mathrm{surf}}$")
ax5.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 16) DEPTH-INTEGRATED P, Z, D OVER FULL SIMULATION
# ============================================================
P_all_time = sol.y[iP, :]
Z_all_time = sol.y[iZ, :]
D_all_time = sol.y[iD, :]

PZD_total_int = np.sum(P_all_time + Z_all_time + D_all_time, axis=0) * dz
P_int = np.sum(P_all_time, axis=0) * dz
Z_int = np.sum(Z_all_time, axis=0) * dz
D_int = np.sum(D_all_time, axis=0) * dz

plt.figure(figsize=(12, 5))
plt.plot(sol.t, P_int, color="green", lw=1.5, alpha=0.8, label="P")
plt.plot(sol.t, Z_int, color="orange", lw=1.5, alpha=0.8, label="Z")
plt.plot(sol.t, D_int, color="red", lw=1.5, alpha=0.8, label="D")

for yr in range(1, years):
    plt.axvline(yr * 365, color="black", linestyle="--", linewidth=1.5)

plt.xlabel("Time [days]")
plt.ylabel(r"Depth-integrated biomass [mmol N m$^{-2}$]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 17) YEARLY MEAN DIFFERENCE DIAGNOSTIC
# ============================================================
N_int_full = np.sum(sol.y[iN, :], axis=0) * dz
P_int_full = np.sum(sol.y[iP, :], axis=0) * dz
Z_int_full = np.sum(sol.y[iZ, :], axis=0) * dz
D_int_full = np.sum(sol.y[iD, :], axis=0) * dz
PZD_int_full = P_int_full + Z_int_full + D_int_full
TotalN_int_full = N_int_full + P_int_full + Z_int_full + D_int_full

year_numbers = np.arange(1, years)

N_year_mean = np.zeros(years - 1)
P_year_mean = np.zeros(years - 1)
Z_year_mean = np.zeros(years - 1)
D_year_mean = np.zeros(years - 1)
PZD_year_mean = np.zeros(years - 1)
TotalN_year_mean = np.zeros(years - 1)

for yr in range(years - 1):
    t0 = yr * 365.0
    t1 = (yr + 1) * 365.0
    mask = (sol.t >= t0) & (sol.t < t1)

    N_year_mean[yr] = np.mean(N_int_full[mask])
    P_year_mean[yr] = np.mean(P_int_full[mask])
    Z_year_mean[yr] = np.mean(Z_int_full[mask])
    D_year_mean[yr] = np.mean(D_int_full[mask])
    PZD_year_mean[yr] = np.mean(PZD_int_full[mask])
    TotalN_year_mean[yr] = np.mean(TotalN_int_full[mask])

dN = np.diff(N_year_mean)
dP = np.diff(P_year_mean)
dZ = np.diff(Z_year_mean)
dD = np.diff(D_year_mean)
dTotal = np.diff(TotalN_year_mean)

years_diff = year_numbers[1:]
labels = [f"{i}->{i+1}" for i in range(1, years - 1)]

plt.figure(figsize=(12, 6))
plt.plot(years_diff, dN, marker="o", lw=2, color="darkblue", label="ΔN")
plt.plot(years_diff, dP, marker="o", lw=2, color="green", label="ΔP")
plt.plot(years_diff, dZ, marker="o", lw=2, color="orange", label="ΔZ")
plt.plot(years_diff, dD, marker="o", lw=2, color="red", label="ΔD")
plt.plot(years_diff, dTotal, marker="o", lw=2, ls="--", color="black", label="ΔTotal N")

plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Year interval")
plt.ylabel(r"Year-to-year change [mmol N m$^{-2}$]")
plt.xticks(years_diff, labels)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 18) THEORETICAL LIMITATION CURVES
# ============================================================
L_range = np.linspace(0, 1000, 70)   # light range                [µmol photons m^-2 s^-1]
N_range = np.linspace(0, 30, 70)     # nutrient range             [mmol N m^-3]
P_range = np.linspace(0, 10, 70)     # prey range                 [mmol N m^-3]
O_range = np.linspace(0, 400, 70)    # oxygen range               [mmol O2 m^-3]

light_lim_theory, nut_lim_theory, graze_lim_theory, oxyg_lim_theory = get_limits(
    L_range, N_range, P_range, O_range
)

fig_lim, axs_lim = plt.subplots(2, 2, figsize=(12, 8))

axs_lim[0, 0].plot(L_range, light_lim_theory, lw=2, color="orange")
axs_lim[0, 0].axvline(k_L, color="black", ls="--", lw=1.5, label=fr"$k_L={k_L}$")
axs_lim[0, 0].axhline(0.5, color="gray", ls=":", lw=1)
axs_lim[0, 0].plot(k_L, 0.5, "o", color="black")
axs_lim[0, 0].set_xlabel(r"Light $L$ [$\mu$mol photons m$^{-2}$ s$^{-1}$]")
axs_lim[0, 0].set_ylabel("Light limitation [-]")
axs_lim[0, 0].set_title("Light limitation")
axs_lim[0, 0].set_ylim(0, 1.05)
axs_lim[0, 0].grid(True)
axs_lim[0, 0].legend()

axs_lim[0, 1].plot(N_range, nut_lim_theory, lw=2, color="darkblue")
axs_lim[0, 1].axvline(k_N, color="black", ls="--", lw=1.5, label=fr"$k_N={k_N}$")
axs_lim[0, 1].axhline(0.5, color="gray", ls=":", lw=1)
axs_lim[0, 1].plot(k_N, 0.5, "o", color="black")
axs_lim[0, 1].set_xlabel(r"Nutrient $N$ [mmol N m$^{-3}$]")
axs_lim[0, 1].set_ylabel("Nutrient limitation [-]")
axs_lim[0, 1].set_title("Nutrient limitation")
axs_lim[0, 1].set_ylim(0, 1.05)
axs_lim[0, 1].grid(True)
axs_lim[0, 1].legend()

axs_lim[1, 0].plot(P_range, graze_lim_theory, lw=2, color="red")
axs_lim[1, 0].axvline(k_Z, color="black", ls="--", lw=1.5, label=fr"$k_Z={k_Z}$")
axs_lim[1, 0].axhline(0.5, color="gray", ls=":", lw=1)
axs_lim[1, 0].plot(k_Z, 0.5, "o", color="black")
axs_lim[1, 0].set_xlabel(r"Prey biomass $P$ [mmol N m$^{-3}$]")
axs_lim[1, 0].set_ylabel("Grazing limitation [-]")
axs_lim[1, 0].set_title("Grazing limitation")
axs_lim[1, 0].set_ylim(0, 1.05)
axs_lim[1, 0].grid(True)
axs_lim[1, 0].legend()

axs_lim[1, 1].plot(O_range, oxyg_lim_theory, lw=2, color="black")
axs_lim[1, 1].axvline(k_O, color="black", ls="--", lw=1.5, label=fr"$k_O={k_O}$")
axs_lim[1, 1].axhline(0.5, color="gray", ls=":", lw=1)
axs_lim[1, 1].plot(k_O, 0.5, "o", color="black")
axs_lim[1, 1].set_xlabel(r"Oxygen $O$ [mmol O$_2$ m$^{-3}$]")
axs_lim[1, 1].set_ylabel("Oxygen limitation [-]")
axs_lim[1, 1].set_title("Oxygen limitation")
axs_lim[1, 1].set_ylim(0, 1.05)
axs_lim[1, 1].grid(True)
axs_lim[1, 1].legend()

plt.tight_layout()
plt.show()


# END OF SCRIPT