import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


time_varying = False
# -----------------------------
# PARAMETERS
# -----------------------------
N_deep = 10.0        # mmol N m^-3
P_growth_max = 2.0   # d^-1
Z_consum_max = 1.0   # d^-1

k_L = 20.0           # light half-sat
k_N = 0.3            # mmol N m^-3
k_Z = 1.0            # mmol N m^-3

m_P = 0.07           # d^-1
m_Z = 0.2            # d^-1

e_N = 0.3            # -
e_D = 0.3            # -

r   = 0.1            # d^-1
w_D = 2.0            # m d^-1

kappa = 0.3          # m d^-1
M    = 60.0             # m

if time_varying:
    L_max = 20.0  # mmol photons/m^2/day
    def light(t):
        # Seasonal sinusoidal forcing: varies 0 -> L_max
        return L_max * (0.5 + 0.5 * np.sin(2 * np.pi * t / 365 - np.pi/2))
else:
    L = 50

# -----------------------------
# TIME SETUP
# -----------------------------
t_max = 365 * 5
t_span = (0, t_max)               # days
t_eval = np.linspace(0, t_max, 1000)

# -----------------------------
# INITIAL CONDITIONS
# -----------------------------
N0 = 0.1
P0 = 1.0
Z0 = 0.2
D0 = 1.5

y0 = [N0, P0, Z0, D0]

# -----------------------------
# ODE SYSTEM
# -----------------------------
def rhs(t, y):
    N, P, Z, D = y
    
    L_current = light(t) if time_varying else L
    
    P_growth = P_growth_max * min((L_current / (L_current + k_L)),
                                  (N / (N + k_N)))
    Z_consum = Z_consum_max * (P / (P + k_Z))

    N_uptake  = P_growth * P
    P_mort    = m_P * P
    Z_grazing = Z_consum * Z
    Z_mort    = m_Z * Z
    remin     = r * D

    dN = -N_uptake + e_N * Z_grazing + remin + (N_deep - N) * (kappa / M)
    dP =  N_uptake - Z_grazing - P_mort - (kappa / M) * P
    dZ = (1 - e_N - e_D) * Z_grazing - Z_mort
    dD = P_mort + e_D * Z_grazing + Z_mort - remin - ((kappa / M) + (w_D / M)) * D

    return [dN, dP, dZ, dD]

# -----------------------------
# SOLVE ODEs
# -----------------------------
sol = solve_ivp(
    rhs,
    t_span,
    y0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-6,
    atol=1e-9
)

# -------------------------------------------------
# TOTAL MASS DERIVATIVE FUNCTION
# -------------------------------------------------
def total_mass_derivative(t, y):
    """
    Returns d/dt (N + P + Z + D)
    """
    return np.sum(rhs(t, y))
# -------------------------------------------------
# EVALUATE MASS CONSERVATION
# -------------------------------------------------
dMdt = np.array([
    total_mass_derivative(t, sol.y[:, i])
    for i, t in enumerate(sol.t)
])
total_mass = sol.y.sum(axis=0)

# -------------------------------------------------
# PLOTS
# -------------------------------------------------
# Total mass tendency
plt.figure(figsize=(8, 3))
plt.plot(sol.t, dMdt, color="black")
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Time [days]")
plt.ylabel(r"$d(N+P+Z+D)/dt$")
plt.title("Mass conservation check (κ = w = 0)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Total mass
plt.figure(figsize=(8, 3))
plt.plot(sol.t, total_mass, color="black")
plt.xlabel("Time [days]")
plt.ylabel("mmol N")
plt.title("Total Nitrogen in the water coulmn")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# SIMPLE PLOT
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex=True)

axs[0, 0].plot(sol.t, sol.y[0], color="black")
axs[0, 0].set_title("Nutrients (N)")
axs[0, 0].set_ylabel("mmol N m$^{-3}$")

axs[0, 1].plot(sol.t, sol.y[1], color="black")
axs[0, 1].set_title("Phytoplankton (P)")

axs[1, 0].plot(sol.t, sol.y[2], color="black")
axs[1, 0].set_title("Zooplankton (Z)")
axs[1, 0].set_xlabel("Time [days]")
axs[1, 0].set_ylabel("mmol N m$^{-3}$")

axs[1, 1].plot(sol.t, sol.y[3], color="black")
axs[1, 1].set_title("Detritus (D)")
axs[1, 1].set_xlabel("Time [days]")

for ax in axs.flat:
    ax.grid(True)

plt.tight_layout()
plt.show()
