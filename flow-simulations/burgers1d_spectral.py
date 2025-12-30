import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Solve the 1D viscous Burgers equation
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
on a periodic domain using a Fourier-Galerkin method.
The solution is expanded in Fourier modes and the nonlinear term is computed pseudospectrally using FFTs.

In Fourier space, the Burgers equation becomes:
    ∂û_k/∂t = -i k/2 û²_k - ν k² û_k

Files: 
- flow-simulations/videos/burgers1d_spectral.mp4
"""

L = 2 * np.pi
N = 256
dx = L / N
nu1 = 0.02
nu2 = 0.01
tmax = 15
CFL = 0.2

x = np.linspace(0, L, N, endpoint=False)
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

u0 = np.sin(x)
u_hat = np.fft.fft(u0)
v_hat = np.fft.fft(u0)

# RHS in Fourier space
def rhs(u_hat, nu):
    u = np.fft.ifft(u_hat).real
    u2_hat = np.fft.fft(u**2)
    u2_hat[np.abs(k) > (2 * (N // 2)) // 3] = 0
    return -0.5j * k * u2_hat - nu * k**2 * u_hat

# RK4 time stepping
def step(u_hat, dt, nu):
    k1 = rhs(u_hat, nu)
    k2 = rhs(u_hat + 0.5 * dt * k1, nu)
    k3 = rhs(u_hat + 0.5 * dt * k2, nu)
    k4 = rhs(u_hat + dt * k3, nu)
    return u_hat + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def energy(u_hat):
    return 0.5 * np.sum(np.abs(u_hat)**2) * (L / N)

def dissipation(u_hat, nu):
    return 2 * nu * np.sum(k**2 * np.abs(u_hat)**2) * (L / N)

dt = CFL * dx / np.max(np.abs(u0))
t = 0.0
steps_per_frame = 5

E_u, E_v = [], []
eps_u, eps_v = [], []
time = []

fig, ax = plt.subplots(1, 3, figsize=(10, 4), tight_layout=True)

# Velocity plot
line_u, = ax[0].plot(x, np.fft.ifft(u_hat).real, lw=2, label=r"$u(x, t)$ and $\nu=0.02$")
line_v, = ax[0].plot(x, np.fft.ifft(v_hat).real, lw=2, color="orange", label=r"$v(x, t)$ and $\nu=0.01$")
ax[0].set_xlim(0, L)
ax[0].set_ylim(-1.5, 1.5)
ax[0].set_title("Velocity Fields")
ax[0].set_xlabel("x")
ax[0].set_ylabel("u, v")
ax[0].legend()

# Energy plot
line_Eu, = ax[1].plot([], [], lw=2, label=r"$E_u(t)$")
line_Ev, = ax[1].plot([], [], lw=2, color="orange", label=r"$E_v(t)$")
ax[1].set_xlim(0, tmax)
ax[1].set_title("Kinetic Energies")
ax[1].set_xlabel("t")
ax[1].set_ylabel("E_u, E_v")
ax[1].legend()

# Dissipation and energy dissipation rate plot
line_epsu, = ax[2].plot([], [], lw=2, label=r"$\varepsilon_u(t)$")
line_epsv, = ax[2].plot([], [], lw=2, color="orange", label=r"$\varepsilon_v(t)$")
line_dEdu, = ax[2].plot([], [], "--", color="black", label=r"$-dE_u/dt$")
line_dEdv, = ax[2].plot([], [], "--", color="gray", label=r"$-dE_v/dt$")
ax[2].set_title("Dissipation Rates")
ax[2].set_xlabel("t")
ax[2].set_ylabel(r"$\varepsilon_u, \varepsilon_v, -dE_u/dt, -dE_v/dt$")
ax[2].set_xlim(0, tmax)
ax[2].legend()


def update(frame):
    global u_hat, v_hat, t

    for _ in range(steps_per_frame):
        if t >= tmax:
            break

        u_hat = step(u_hat, dt, nu1)
        v_hat = step(v_hat, dt, nu2)
        t += dt

        E_u.append(energy(u_hat))
        E_v.append(energy(v_hat))
        eps_u.append(dissipation(u_hat, nu1))
        eps_v.append(dissipation(v_hat, nu2))
        time.append(t)

    u = np.fft.ifft(u_hat).real
    v = np.fft.ifft(v_hat).real
    line_u.set_ydata(u)
    line_v.set_ydata(v)

    line_Eu.set_data(time, E_u)
    line_Ev.set_data(time, E_v)
    ax[1].set_ylim(0, max(E_u[0], E_v[0]) * 1.1)

    line_epsu.set_data(time, eps_u)
    line_epsv.set_data(time, eps_v)

    if len(time) > 1:
        dEdu = -np.gradient(E_u, time)
        dEdv = -np.gradient(E_v, time)
        line_dEdu.set_data(time, dEdu)
        line_dEdv.set_data(time, dEdv)

    ax[2].set_ylim(
        0,
        max(
            max(eps_u),
            max(eps_v),
            max(dEdu) if len(time) > 1 else 0,
            max(dEdv) if len(time) > 1 else 0,
        ) * 1.1
    )

    plt.suptitle(f"t = {t:.2f}")

    return (
        line_u, line_v,
        line_Eu, line_Ev,
        line_epsu, line_epsv,
        line_dEdu, line_dEdv
    )

nframes = int(np.ceil(tmax / (steps_per_frame * dt)))
anim = FuncAnimation(fig, update, frames=nframes, interval=30, blit=False)
anim.save('flow-simulations/videos/burgers1d_spectral.mp4', fps=60, dpi=300)
