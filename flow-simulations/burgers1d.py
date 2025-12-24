import numpy as np
import matplotlib.pyplot as plt

"""
Solve the 1D viscous Burgers equation:
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
with periodic boundary conditions.
"""

L = 2 * np.pi
N = 256
dx = L / N
nu = 0.01
tmax = 2.0
CFL = 0.4

x = np.linspace(0, L, N, endpoint=False)
u = np.sin(x)
t = 0.0

fig_u, ax_u = plt.subplots(figsize=(12, 4), constrained_layout=True)
line_u, = ax_u.plot(x, u, lw=2)
ax_u.set_xlim(0, L)
ax_u.set_ylim(-1.5, 1.5)
ax_u.set_xlabel("x")
ax_u.set_ylabel("u(x,t)")

fig_E, ax_E = plt.subplots(figsize=(12, 8), constrained_layout=True)
line_E, = ax_E.plot([], [], lw=2)
ax_E.set_xlim(0, tmax)
ax_E.set_ylim(0, 1.2)
ax_E.set_xlabel("t")
ax_E.set_ylabel("E(t)")

fig_eps, ax_eps = plt.subplots(figsize=(12, 8), constrained_layout=True)
line_eps, = ax_eps.plot([], [], lw=2)
ax_eps.set_xlim(0, tmax)
ax_eps.set_ylim(0, 0.5)
ax_eps.set_xlabel("t")
ax_eps.set_ylabel("ε(t)")

t_hist = []
E_hist = []
eps_hist = []

while t < tmax:
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    uxx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2

    dt = CFL * min(dx / np.max(np.abs(u)), dx**2 / nu)

    u = u - dt * u * ux + nu * dt * uxx
    t += dt

    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

    t_hist.append(t)
    E_hist.append(0.5 * np.sum(u**2) * dx)
    eps_hist.append(nu * np.sum(ux**2) * dx)

    line_u.set_ydata(u)
    line_E.set_data(t_hist, E_hist)
    line_eps.set_data(t_hist, eps_hist)

    ax_u.set_title(f"t = {t:.2f}")

    fig_u.canvas.draw_idle()
    fig_E.canvas.draw_idle()
    fig_eps.canvas.draw_idle()

    plt.pause(0.01)

dEdt = np.gradient(E_hist, t_hist)
plt.plot(t_hist, -dEdt, label='-dE/dt', color='red')
plt.legend()

plt.show()
