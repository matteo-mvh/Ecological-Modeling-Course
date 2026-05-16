import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SIMPLE STEADY-FORCING DIATOM VS NON-DIATOM MODEL
# Based on the report parameterization, but intentionally reduced:
# - keeps only N, S, PN, PD
# - skips recycling, oxygen, detritus, sediment, explicit Z dynamics
# - uses constant nutrient loading and a flushing/loss term
# ============================================================

# -----------------------------
# Parameters from the report
# -----------------------------
params = {
    "mu_max_N": 1.5,   # d^-1
    "mu_max_D": 1.2,   # d^-1
    "k_L": 80.0,       # umol photons m^-2 s^-1
    "k_N": 0.5,        # mmol N m^-3
    "k_S": 2.0,        # mmol Si m^-3
    "m_N": 0.09,       # d^-1
    "m_D": 0.05,       # d^-1
    "g_max": 0.7,      # d^-1
    "k_Z": 4.0,        # mmol N m^-3
    "q_N": 1.0,        # -
    "q_D": 0.7,        # -
    "k_phi": 1.8,      # mmol N m^-3
    "R_SiN": 1.0,      # mmol Si mmol N^-1
}

# --------------------------------------------------------
# Extra forcing / closure parameters NOT from the report
# --------------------------------------------------------
# These are just to make the reduced model reach equilibrium.
forcing = {
    "L": 120.0,        # constant light [umol photons m^-2 s^-1]
    "loss_rate": 0.10, # d^-1, flushing/background export
    "Z_const": 0.6,    # fixed grazer biomass proxy [mmol N m^-3]
}

# ============================================================
# Helper functions
# ============================================================

def monod(x, k):
    return x / (x + k + 1e-12)

def light_limitation(L, k_L):
    return L / (L + k_L + 1e-12)

def diatom_protection(PD, k_phi):
    # from the report:
    # phi_D = k_phi / (PD + k_phi)
    return k_phi / (PD + k_phi + 1e-12)

def compute_rates(N, S, PN, PD, p, f):
    """
    Compute growth and grazing terms using:
    - report-style growth limitation
    - report-style diatom protection
    - report-style preference-weighted grazing partitioning
    - grazing limitation based on effective accessible prey
    """
    Llim = light_limitation(f["L"], p["k_L"])
    Nlim = monod(N, p["k_N"])
    Slim = monod(S, p["k_S"])

    # Growth rates from report structure
    mu_N = p["mu_max_N"] * min(Llim, Nlim)
    mu_D = p["mu_max_D"] * min(Llim, Nlim, Slim)

    # Diatom protection from the report:
    # phi_D(PD) = k_phi / (PD + k_phi)
    phi_D = diatom_protection(PD, p["k_phi"])

    # Effective accessible prey biomass seen by grazers
    prey_N_eff = p["q_N"] * PN
    prey_D_eff = p["q_D"] * phi_D * PD
    prey_eff = prey_N_eff + prey_D_eff

    # Grazing limitation term:
    # inferred reduced-form closure consistent with report structure
    # and k_Z half-saturation
    Glim = prey_eff / (prey_eff + p["k_Z"] + 1e-12)

    # Partitioning of grazing exactly following the report logic
    theta_N = prey_N_eff / (prey_eff + 1e-12)
    theta_D = prey_D_eff / (prey_eff + 1e-12)

    # Total grazing in reduced model:
    # report has gmax * min(Glim, Olim) * Z
    # here oxygen is omitted, so use only Glim with fixed grazer biomass
    Fgraz_tot = p["g_max"] * Glim * f["Z_const"]

    Fgraz_N = theta_N * Fgraz_tot
    Fgraz_D = theta_D * Fgraz_tot

    return {
        "mu_N": mu_N,
        "mu_D": mu_D,
        "Fgraz_N": Fgraz_N,
        "Fgraz_D": Fgraz_D,
        "Llim": Llim,
        "Nlim": Nlim,
        "Slim": Slim,
        "phi_D": phi_D,
        "Glim": Glim,
        "theta_N": theta_N,
        "theta_D": theta_D,
        "prey_eff": prey_eff,
    }

def rhs(state, N_load, Si_to_N_load_ratio, p, f):
    """
    State = [N, S, PN, PD]
    Loads are in mmol m^-3 d^-1.
    """
    N, S, PN, PD = state

    # Prevent negatives in rate calculations
    N = max(N, 0.0)
    S = max(S, 0.0)
    PN = max(PN, 0.0)
    PD = max(PD, 0.0)

    rates = compute_rates(N, S, PN, PD, p, f)

    mu_N = rates["mu_N"]
    mu_D = rates["mu_D"]
    Fgraz_N = rates["Fgraz_N"]
    Fgraz_D = rates["Fgraz_D"]

    loss = f["loss_rate"]
    S_load = Si_to_N_load_ratio * N_load

    # Nutrients: external loading - flushing - uptake
    dNdt = N_load - loss * N - mu_N * PN - mu_D * PD
    dSdt = S_load - loss * S - p["R_SiN"] * mu_D * PD

    # Biomass: growth - mortality - grazing - flushing
    dPNdt = mu_N * PN - p["m_N"] * PN - Fgraz_N - loss * PN
    dPDdt = mu_D * PD - p["m_D"] * PD - Fgraz_D - loss * PD

    return np.array([dNdt, dSdt, dPNdt, dPDdt], dtype=float)

def integrate_to_equilibrium(
    N_load,
    Si_to_N_load_ratio=1.0,
    p=params,
    f=forcing,
    tmax=400.0,
    dt=0.02,
    state0=None,
):
    """
    Simple forward Euler integration to a quasi-steady state.
    """
    if state0 is None:
        # [N, S, PN, PD]
        state = np.array([0.5, 1.0, 0.1, 0.1], dtype=float)
    else:
        state = np.array(state0, dtype=float)

    nt = int(tmax / dt)
    history = np.zeros((nt + 1, 5))
    history[0, :] = [0.0, *state]

    for i in range(1, nt + 1):
        dstate = rhs(state, N_load, Si_to_N_load_ratio, p, f)
        state = state + dt * dstate

        # enforce non-negative concentrations
        state = np.maximum(state, 0.0)
        history[i, :] = [i * dt, *state]

    return history

def summarize_equilibrium(history, p=params, f=forcing):
    """
    Return last-state summary including biomass ratio PD/PN.
    """
    t, N, S, PN, PD = history[-1]
    ratio = PD / (PN + 1e-12)

    rates = compute_rates(N, S, PN, PD, p, f)

    return {
        "t_end": t,
        "N": N,
        "S": S,
        "PN": PN,
        "PD": PD,
        "PD_to_PN": ratio,
        "mu_N": rates["mu_N"],
        "mu_D": rates["mu_D"],
        "Llim": rates["Llim"],
        "Nlim": rates["Nlim"],
        "Slim": rates["Slim"],
    }

# ============================================================
# Run one example
# ============================================================

hist = integrate_to_equilibrium(
    N_load=0.2,                 # mmol N m^-3 d^-1
    Si_to_N_load_ratio=0.0,     # dimensionless loading ratio
)

summary = summarize_equilibrium(hist)

print("Single run summary:")
for k, v in summary.items():
    print(f"{k:>10s}: {v:.4f}" if isinstance(v, float) else f"{k:>10s}: {v}")

# Plot time series for one run
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(hist[:, 0], hist[:, 1], label="N")
ax.plot(hist[:, 0], hist[:, 2], label="S")
ax.plot(hist[:, 0], hist[:, 3], label="PN")
ax.plot(hist[:, 0], hist[:, 4], label="PD")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Concentration / biomass [mmol m$^{-3}$]")
ax.set_title("Simple steady-forcing SNPZDO-Si reduction")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# HEATMAP: x = nitrogen loading, y = silicate loading
# ============================================================

N_load_vals = np.linspace(0.05, 2.0, 50)
S_load_vals = np.linspace(0.00, 1.6, 50)   # mmol Si m^-3 d^-1

ratio_map = np.full((len(S_load_vals), len(N_load_vals)), np.nan)

for i, S_load in enumerate(S_load_vals):
    for j, N_load in enumerate(N_load_vals):

        # convert to ratio used by model
        SiN_ratio = S_load / (N_load + 1e-12)

        hist = integrate_to_equilibrium(
            N_load=N_load,
            Si_to_N_load_ratio=SiN_ratio,
            tmax=100.0,
            dt=0.02
        )

        out = summarize_equilibrium(hist)
        PN = out["PN"]
        PD = out["PD"]
        total = PN + PD
        
        if total < 1e-8:
            ratio_map[i, j] = np.nan
        else:
            ratio_map[i, j] = PD / total

fig, ax = plt.subplots(figsize=(9, 6))

# Create meshgrid for contour
X, Y = np.meshgrid(N_load_vals, S_load_vals)

# Filled contour plot
cf = ax.contourf(
    X, Y, ratio_map,
    levels=20,
    cmap="viridis",
    vmin=0.0,
    vmax=1.0
)

# Add contour lines (optional but very nice)
cs = ax.contour(
    X, Y, ratio_map,
    levels=[0.2, 0.4, 0.5, 0.6, 0.8],
    colors="black",
    linewidths=0.8
)

ax.clabel(cs, fmt="%.1f", fontsize=8)

# Highlight the key boundary: PD = PN
cs_eq = ax.contour(
    X, Y, ratio_map,
    levels=[0.5],
    colors="white",
    linewidths=2
)

# --- Mean silicate line ---
mean_S = 0.5  # adjust if you want

ax.axhline(
    y=mean_S,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Typical Si loading (~0.5)"
)

ax.legend()

# Colorbar
cbar = plt.colorbar(cf, ax=ax)
cbar.set_label("Diatom fraction PD / (PD + PN) [-]")

# Labels
ax.set_xlabel("Steady nitrogen loading [mmol N m$^{-3}$ d$^{-1}$]")
ax.set_ylabel("Steady silicate loading [mmol Si m$^{-3}$ d$^{-1}$]")

plt.tight_layout()
plt.show()