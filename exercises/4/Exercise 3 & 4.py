import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------------------------------------
# 1) REACTION / MODEL PARAMETERS
# ------------------------------------------------------------

# STATE VARIABLES (all in mmol N m^-3)
# N  : Dissolved inorganic nitrogen [mmol N m^-3]
# P  : Phytoplankton nitrogen biomass [mmol N m^-3]
# Z  : Zooplankton nitrogen biomass [mmol N m^-3]
# D  : Detrital nitrogen [mmol N m^-3]

P_growth_max = 1.0   # maximum phytoplankton growth rate [d^-1]
Z_consum_max = 1.7   # maximum zooplankton grazing rate [d^-1]

k_L = 20.0           # half-saturation constant for light [µmol photons m^-2 s^-1]
k_N = 0.3            # half-saturation constant for nutrients [mmol N m^-3]
k_Z = 1.5            # grazing half-saturation constant [mmol N m^-3]

m_P = 0.07           # phytoplankton mortality [d^-1]
m_Z = 0.1            # zooplankton mortality [d^-1]

e_N = 0.3            # fraction of grazing excreted to DIN [-]
e_D = 0.3            # fraction of grazing to detritus [-]

r = 0.05             # remineralization rate of detritus [d^-1]

depth = 200          # total water column depth [m]
# ------------------------------------------------------------
# 2) VERTICAL GRID
# ------------------------------------------------------------
nz = 30
z = np.linspace(0, depth, nz)   # depth [m]
dz = z[1] - z[0]                # layer thickness [m]

# ------------------------------------------------------------
# 3) MIXING / ADVECTION SETTINGS
# ------------------------------------------------------------
W = [0 , 1 , 0 , 2]  # sinking velocity [m d^-1]

# ------------------------------------------------------------
# 4) INITIAL CONDITIONS
# ------------------------------------------------------------
N0 = np.linspace(0, 20, nz)
P0 = np.linspace(1, 0, nz)
Z0 = 0.2 * np.ones(nz)
D0 = 1.5 * np.ones(nz)

y0 = np.concatenate([N0, P0, Z0, D0])

# ------------------------------------------------------------
# 5) TIME SETUP
# ------------------------------------------------------------
t_max = 365 * 10
t_span = (0, t_max)
t_eval = np.linspace(0, t_max, 1000)

# ------------------------------------------------------------
# 6) SEASONALITY FUNCTION (returns kappa_interface, Lz)
# ------------------------------------------------------------
def getLIGHTandKAPPAS(t, P=None, D=None, Seasonality=True, bio_attenuation=True):
    """
    Returns:
        kappa_interface : vertical diffusivity at interfaces (size nz+1)
        Lz              : light profile at cell centers
    """
    
    # LIGHT ATTENUATION PARAMETERS
    k_water = 0.1   # background attenuation [m^-1]
    k_bio   = 0.20   # biological attenuation [m^2 mmol^-1]

    # Biological integral for self-shading
    if bio_attenuation and (P is not None) and (D is not None):
        bio_integral = np.cumsum(P + D) * dz
    else:
        bio_integral = 0.0

    # SEASONALITY
    if Seasonality:
        day_of_year = t % 365

        # Seasonal mixing depth z_mix(t)
        zMix = 0.05 * depth        # shallow summer mixing depth
        zMixWinter = 0.8 * depth   # deep winter mixing depth
        tMaxSpring = 90            # day of strongest shoaling
        zetaMaxSteep = 2.0         # sharpness of seasonal transition
        z_mix = (0.5* (1 - np.sin(2 * np.pi * (day_of_year - tMaxSpring) / 365))** zetaMaxSteep* (zMixWinter - zMix)+ zMix)
        
        # Diffusivity profile κ(z,t)
        kappa_top = 5.0        # surface diffusivity [m^2 d^-1]
        kappa_bottom = 0.5     # deep diffusivity [m^2 d^-1]
        zeta_mix = 10.0        # pycolocine thickness [m]
        kappa_center = (0.5* (1 - np.tanh((z - z_mix) / zeta_mix))* (kappa_top - kappa_bottom)+ kappa_bottom)

        # Seasonal surface light
        L0_min = 100      # winter minimum
        L0_max = 1400     # summer maximum
        phase_shift = 80  # day of peak light
        seasonal_shape = (1 + np.sin(2 * np.pi * (day_of_year - phase_shift) / 365))**2
        L0 = L0_min + (L0_max - L0_min) * seasonal_shape / 4

    else:
        # No seasonality (constant fields)
        kappa_surface = 5.0
        kappa_bottom  = 1.0
        z_transition  = 50.0   # fixed mixing depth
        zeta_mix      = 10.0   # transition thickness
        kappa_center = (0.5 * (1 - np.tanh((z - z_transition) / zeta_mix))* (kappa_surface - kappa_bottom)+ kappa_bottom)
        L0 = 1400.0

    # ============================================================
    # INTERFACE DIFFUSIVITY
    # ============================================================

    kappa_interface = np.zeros(nz + 1)
    kappa_interface[1:nz] = 0.5 * (kappa_center[1:] + kappa_center[:-1])
    kappa_interface[0] = kappa_center[0]
    kappa_interface[nz] = kappa_center[-1]

    # ============================================================
    # LIGHT PROFILE
    # ============================================================

    if bio_attenuation and (P is not None) and (D is not None):
        Lz = L0 * np.exp(-k_water * z - k_bio * bio_integral)
    else:
        Lz = L0 * np.exp(-k_water * z)

    return kappa_interface, Lz , L0

# ------------------------------------------------------------
# 7) ADVECTION / DIFFUSION (vertical transport)
# ------------------------------------------------------------
def vertical_transport(C, kappa_interface, w=0.0):
    J = np.zeros(nz + 1)

    # Interior fluxes
    Ja = w * C[:-1]
    Jd = -kappa_interface[1:nz] * (C[1:] - C[:-1]) / dz
    J[1:nz] = Ja + Jd

    # No-flux boundaries
    J[0] = 0.0
    J[nz] = 0.0

    return -(J[1:] - J[:-1]) / dz

# ------------------------------------------------------------
# 8) ODE SYSTEM (rhs)
# ------------------------------------------------------------
def rhs(t, y):
    N = y[0:nz]
    P = y[nz:2*nz]
    Z = y[2*nz:3*nz]
    D = y[3*nz:4*nz]
    
    # Get seasonal kappa_interface and light profile
    kappa_interface, Lz, _ = getLIGHTandKAPPAS(t, P=P, D=D)
         
    light_lim = Lz / (Lz + k_L)
    
    nut_lim   = N / (N + k_N + 1e-12)
    graze_lim = P / (P + k_Z + 1e-12)
    
    P_growth = P_growth_max * np.minimum(light_lim, nut_lim)
    Z_consum = Z_consum_max * graze_lim

    N_uptake  = P_growth * P
    P_mort    = m_P * P
    Z_grazing = Z_consum * Z
    Z_mort    = m_Z * Z
    remin     = r * D

    # --- Reaction terms ---
    dN_reac = -N_uptake + e_N * Z_grazing          + remin
    dP_reac =  N_uptake -       Z_grazing - P_mort
    dZ_reac = (1 - e_N - e_D) * Z_grazing - Z_mort
    dD_reac =  P_mort +  e_D  * Z_grazing + Z_mort - remin

    # --- Transport ---
    dN = vertical_transport(N, kappa_interface, w=W[0]) + dN_reac
    dP = vertical_transport(P, kappa_interface, w=W[1]) + dP_reac
    dZ = vertical_transport(Z, kappa_interface, w=W[2]) + dZ_reac
    dD = vertical_transport(D, kappa_interface, w=W[3]) + dD_reac
    
    # Prevent negative concentrations
    N = np.maximum(N, 0)
    P = np.maximum(P, 0)
    Z = np.maximum(Z, 0)
    D = np.maximum(D, 0)

    return np.concatenate([dN, dP, dZ, dD])

# %% SIMULATION CELL

# ------------------------------------------------------------
# 9) SOLVE ODEs
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 10) MASS CONSERVATION DIAGNOSTICS
# ------------------------------------------------------------
massCheck = False
if massCheck:
    def total_mass_derivative(t, y):
        """Returns d/dt (N + P + Z + D)"""
        return np.sum(rhs(t, y))
    
    dMdt = np.array([
        total_mass_derivative(t, sol.y[:, i])
        for i, t in enumerate(sol.t)
    ])
    
    total_mass = []
    for i in range(len(sol.t)):
        y = sol.y[:, i]
        N = y[0:nz]
        P = y[nz:2*nz]
        Z = y[2*nz:3*nz]
        D = y[3*nz:4*nz]
        total_mass.append(np.sum((N + P + Z + D) * dz))
    total_mass = np.array(total_mass)

# %% PLOT CELL 

# ------------------------------------------------------------
# 11) HEATMAP PLOTS (time-depth) — last 365 days option
# ------------------------------------------------------------
only_last365 = True
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

if only_last365:
    t_mask = sol.t >= (t_max - 365)
    t_plot = sol.t[t_mask]
    
    N_all = sol.y[0:nz, t_mask]
    P_all = sol.y[nz:2*nz, t_mask]
    Z_all = sol.y[2*nz:3*nz, t_mask]
    D_all = sol.y[3*nz:4*nz, t_mask]
else:
    t_plot = sol.t
    N_all = sol.y[0:nz, :]
    P_all = sol.y[nz:2*nz, :]
    Z_all = sol.y[2*nz:3*nz, :]
    D_all = sol.y[3*nz:4*nz, :]

# Nutrients
im0 = axs[0, 0].contourf(t_plot, -z, N_all, levels=50)
axs[0, 0].set_title("Nutrients in [mmol N/m^3] ")
axs[0, 0].set_ylabel("Depth [m]")
fig.colorbar(im0, ax=axs[0, 0])

# Phytoplankton
im1 = axs[0, 1].contourf(t_plot, -z, P_all, levels=50)
axs[0, 1].set_title("Phytoplankton in [mmol N/m^3]")
fig.colorbar(im1, ax=axs[0, 1])

# Zooplankton
im2 = axs[1, 0].contourf(t_plot, -z, Z_all, levels=50)
axs[1, 0].set_title("Zooplankton in [mmol N/m^3]")
axs[1, 0].set_xlabel("Time [days]")
axs[1, 0].set_ylabel("Depth [m]")
fig.colorbar(im2, ax=axs[1, 0])

# Detritus
im3 = axs[1, 1].contourf(t_plot, -z, D_all, levels=50)
axs[1, 1].set_title("Detritus in [mmol N/m^3]")
axs[1, 1].set_xlabel("Time [days]")
fig.colorbar(im3, ax=axs[1, 1])

# Invert depth axis (surface at top)
for ax in axs.flat:
    ax.invert_yaxis()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12) MASS CONSERVATION PLOT
# ------------------------------------------------------------
if massCheck:
    plt.figure(figsize=(12,10))
    plt.plot(sol.t, total_mass - total_mass[0], color='darkred', lw=2)
    plt.xlabel("Time [days]")
    plt.ylabel("Deviation from Initial Total Mass")
    plt.title("Mass Conservation Error")
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------
# 13) PROFILE + LIMITATION PLOT (same time index)
# ------------------------------------------------------------

time_index = -172   # same as before
t_selected = sol.t[time_index]

N_sel = sol.y[0:nz, time_index]
P_sel = sol.y[nz:2*nz, time_index]
Z_sel = sol.y[2*nz:3*nz, time_index]
D_sel = sol.y[3*nz:4*nz, time_index]

T_sel = N_sel + P_sel + Z_sel + D_sel

# Get seasonal light for this time
kappa_interface, Lz, _  = getLIGHTandKAPPAS(t_selected)

# ---- Limitations ----
light_lim = Lz / (Lz + k_L)
nut_lim = N_sel / (N_sel + k_N + 1e-12)

# ---- Plot ----
fig, axs = plt.subplots(1, 2, figsize=(12,5), sharey=True)

# ---- Left panel: Concentration profiles ----
N_safe = np.maximum(N_sel, 1e-8)
P_safe = np.maximum(P_sel, 1e-8)
Z_safe = np.maximum(Z_sel, 1e-8)
D_safe = np.maximum(D_sel, 1e-8)
T_safe = np.maximum(T_sel, 1e-8)

axs[0].plot(N_safe, -z, label="N", linewidth=2, color='darkblue')
axs[0].plot(P_safe, -z, label="P", linewidth=2, color='green')
axs[0].plot(Z_safe, -z, label="Z", linewidth=2, color='orange')
axs[0].plot(D_safe, -z, label="D", linewidth=2, color='red')
axs[0].plot(T_safe, -z, label="Total Nitrogen",
            linewidth=2, linestyle="--", color="black")

axs[0].set_xscale("log")   # <-- log scale here
axs[0].set_xlabel("Concentration (log scale)")
axs[0].set_ylabel("Depth [m]")
axs[0].invert_yaxis()
axs[0].legend()
axs[0].grid(True, which="both")
axs[0].set_title("State Variables")

# ---- Right panel: Limitation profiles ----
axs[1].plot(light_lim, -z, label="Light limitation", linewidth=2)
axs[1].plot(nut_lim,   -z, label="Nutrient limitation", linewidth=2)

axs[1].set_xlabel("Limitation (0–1)")
axs[1].invert_yaxis()
axs[1].legend()
axs[1].grid(True)
axs[1].set_title("Growth Limitations")

plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# 14) SEASONAL PROFILES (kappa_interface and Lz) — four seasons
# ------------------------------------------------------------
season_days = {
    "Winter": 0,
    "Spring": 90,
    "Summer": 172,
    "Autumn": 265
}


# Create figure with gridspec
fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(2, 2, height_ratios=[3,1])  # top row 3x height, bottom row 1x

# ---- Top left: Turbulence profiles ----
ax1 = fig.add_subplot(gs[0,0])
for season, day in season_days.items():
    kappa_prof, L_profile, L0 = getLIGHTandKAPPAS(day)
    
    z_kappa = np.zeros(nz+1)
    z_kappa[:-1] = z
    z_kappa[-1] = z[-1] + dz
    
    ax1.plot(kappa_prof, z_kappa, label=season)
ax1.set_xlabel("kappa [m²/d]")
ax1.set_ylabel("Depth [m]")
ax1.set_title("Seasonal Turbulence Profiles")
ax1.invert_yaxis()
ax1.grid(True)
ax1.legend()

# ---- Top right: Light profiles ----
ax2 = fig.add_subplot(gs[0,1])
for season, day in season_days.items():
    kappa_prof, L_profile, L0 = getLIGHTandKAPPAS(day)
    ax2.plot(L_profile, z, label=season)
ax2.set_xlabel("Light [µmol photons m⁻² s⁻¹]")
ax2.set_ylabel("Depth [m]")
ax2.set_title("Seasonal Light Profiles")
ax2.invert_yaxis()
ax2.grid(True)
ax2.legend()

# ---- Bottom: Surface light L0 over 365 days ----
ax3 = fig.add_subplot(gs[1,:])  # spans both columns
days = np.arange(0, 365)
L0_year = np.array([getLIGHTandKAPPAS(d)[2] for d in days])
ax3.plot(days, L0_year, color='orange')
season_positions = [0, 80, 172, 264, 355]
season_labels = ["Winter", "Spring", "Summer", "Autumn", "Winter"]
ax3.set_xticks(season_positions)
ax3.set_xticklabels(season_labels)
ax3.set_xlabel("Season")
ax3.set_ylabel("Surface Light [µmol m⁻² s⁻¹]")
ax3.set_title("Seasonal Surface Light Over Year")
ax3.grid(True)

plt.tight_layout()
plt.show()