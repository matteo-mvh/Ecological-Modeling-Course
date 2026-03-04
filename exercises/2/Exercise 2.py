import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# DOMAIN
# -----------------------------
M = 200
nz = 50
z = np.linspace(0, M, nz)
dz = z[1] - z[0]

# -----------------------------
# Mixing profile
# -----------------------------
kappa_top = 5
kappa_deep = 1
z_mix = 20.0
zeta_mix = 3.0

kappa_center = (
    kappa_deep
    + 0.5 * (kappa_top - kappa_deep)
    * (1 - np.tanh((z - z_mix) / zeta_mix))
)

kappa_interface = np.zeros(nz + 1)
kappa_interface[1:nz] = 0.5 * (kappa_center[1:] + kappa_center[:-1])
kappa_interface[0] = kappa_center[0]
kappa_interface[nz] = kappa_center[-1]

# -----------------------------
# TRANSPORT FUNCTION
# -----------------------------
def vertical_transport(C, w=0.1):

    J = np.zeros(nz + 1)

    # Interior fluxes
    Ja = w * C[:-1]
    Jd = -kappa_interface[1:nz] * (C[1:] - C[:-1]) / dz
    J[1:nz] = Ja + Jd

    # No-flux boundaries
    J[0] = 0.0
    J[nz] = 0.0

    return -(J[1:] - J[:-1]) / dz


# -----------------------------
# ODE SYSTEM
# -----------------------------

def rhs(t, C):
    return vertical_transport(C)

# -----------------------------
# INITIAL CONDITION
# Gaussian tracer blob
# -----------------------------
C0 = np.exp(-(z - 40)**2 / (2 * 5**2))

# -----------------------------
# TIME
# -----------------------------
max_t = 2000
t_span = (0,max_t )
t_eval = np.linspace(0, max_t, 400)

# -----------------------------
# SOLVE
# -----------------------------
sol = solve_ivp(
    rhs,
    t_span,
    C0,
    t_eval=t_eval,
    method="BDF",
    rtol=1e-6,
    atol=1e-9
)

# -----------------------------
# PLOT HEATMAP
# -----------------------------
plt.figure(figsize=(7,6))
plt.contourf(sol.t, z, sol.y, levels=50)
plt.xlabel("Time [days]")
plt.ylabel("Depth [m]")
plt.title("Passive Tracer Transport")
plt.gca().invert_yaxis()
plt.colorbar(label="Concentration")
plt.tight_layout()
plt.show()

# -----------------------------
# MASS CHECK
# -----------------------------
mass = np.sum(sol.y * dz, axis=0)

plt.figure(figsize=(6,4))
plt.plot(sol.t, mass - mass[0])
plt.xlabel("Time [days]")
plt.ylabel("Mass deviation")
plt.title("Mass Conservation Check")
plt.grid(True)
plt.show()

# -----------------------------
# SELECT FOUR EQUALLY SPACED TIMES
# -----------------------------
times_to_plot = np.linspace(t_span[0], t_span[1], 5)

# Find closest indices in solution
indices = [np.argmin(np.abs(sol.t - t_sel)) for t_sel in times_to_plot]

# -----------------------------
# PLOT PROFILES
# -----------------------------
plt.figure(figsize=(6,8))

for idx in indices:
    plt.plot(sol.y[:, idx], z, label=f"t = {sol.t[idx]:.0f} d")

plt.xlabel("Concentration")
plt.ylabel("Depth [m]")
plt.title("Vertical Tracer Profiles at Selected Times")
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True)

plt.tight_layout()
plt.show()


