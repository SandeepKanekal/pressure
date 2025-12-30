import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Solve the 1D viscous Burgers equation:
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
with periodic boundary conditions.

Files: 
- flow-simulations/videos/burgers1d.mp4
"""

L = 2 * np.pi
N = 256
dx = L / N
nu = 0.01
tmax = 50
CFL = 0.4

x = np.linspace(0, L, N, endpoint=False)
u = np.sin(x)
t = 0.0
umax = max(np.max(np.abs(u)), 1e-6)
dt = CFL * min(dx / umax, dx**2 / nu)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
line_u, = axes[0].plot(x, u, lw=2)
axes[0].set_title("Velocity Field u(x,t)")
axes[0].set_xlim(0, L)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x,t)")

line_E, = axes[1].plot([], [], lw=2)
axes[1].set_title("Kinetic Energy E(t)")
axes[1].set_xlim(0, tmax)
axes[1].set_ylim(0, 1.2)
axes[1].set_xlabel("t")
axes[1].set_ylabel("E(t)")

line_eps, = axes[2].plot([], [], lw=2)
axes[2].set_title(r"Energy Dissipation Rate $\epsilon(t)$")
axes[2].set_xlim(0, tmax)
axes[2].set_ylim(0, 0.5)
axes[2].set_xlabel("t")
axes[2].set_ylabel(r"$\epsilon(t)$")

t_hist = []
E_hist = []
eps_hist = []

def update(frame):
    global u, t, dt
    
    for _ in range(5):
        if t >= tmax:
            break

        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        uxx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2

        umax = max(np.max(np.abs(u)), 1e-6)
        dt = CFL * min(dx / umax, dx**2 / nu)

        u = u - dt * u * ux + nu * dt * uxx
        t += dt

    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

    t_hist.append(t)
    E_hist.append(0.5 * np.sum(u**2) * dx)
    eps_hist.append(nu * np.sum(ux**2) * dx)

    line_u.set_ydata(u)
    line_E.set_data(t_hist, E_hist)
    line_eps.set_data(t_hist, eps_hist)
    
    if len(t_hist) > 1:
        axes[1].set_ylim(0, max(E_hist) * 1.1)
        axes[2].set_ylim(0, max(eps_hist) * 1.1)

    plt.suptitle(f"t = {t:.2f}")

    return line_u, line_E, line_eps

anim = FuncAnimation(
    fig,
    update,
    frames=480,
    interval=20,
    blit=False
)

anim.save('flow-simulations/videos/burgers1d.mp4', fps=60, dpi=300)
