import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Solve the equation
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² + f(x, t)
on a periodic domain using a Fourier-Galerkin method.
The solution is expanded in Fourier modes and the nonlinear term is computed pseudospectrally using FFTs.

In Fourier space, the Burgers equation becomes:
    ∂û_k/∂t = -i k/2 û²_k - ν k² û_k + ˆf_k

f injects energy at large scales to maintain a statistically steady turbulent state.

\hat{f}_k is defined as:
    \sigma \zeta_k (t) for |k| ≤ k_f
    0 otherwise
where \zeta_k (t) is a complex Gaussian white noise process.
"""

L = 2 * np.pi
N = 256
dx = L / N
nu1 = 1e-3
nu2 = 0.01
tmax = 50.0
CFL = 0.4

x = np.linspace(0, L, N, endpoint=False)
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

u0 = np.sin(x)
u_hat = np.fft.fft(u0)
v_hat = np.fft.fft(u0)

def forcing():
    f_hat = np.zeros(N, dtype=complex)
    kf = 4
    sigma = 0.5
    for i, ki in enumerate(k):
        if abs(ki) <= kf:
            f_hat[i] = sigma * (np.random.randn() + 1j * np.random.randn())
    return f_hat

# RHS in Fourier space
def rhs(u_hat, nu):
    u = np.fft.ifft(u_hat).real
    u2_hat = np.fft.fft(u**2)
    u2_hat[np.abs(k) > (2 * (N // 2)) // 3] = 0
    return -0.5j * k * u2_hat - nu * k**2 * u_hat + forcing()

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
Espec_u, Espec_v = np.zeros(N), np.zeros(N)

fig, ax = plt.subplots(2, 3, figsize=(10, 4), tight_layout=True)

# Velocity plot
line_u, = ax[0, 0].plot(x, np.fft.ifft(u_hat).real, lw=2, label=r"$u(x, t)$ and $\nu={}$".format(nu1))
line_v, = ax[0, 0].plot(x, np.fft.ifft(v_hat).real, lw=2, color="orange", label=r"$v(x, t)$ and $\nu={}$".format(nu2))
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("u, v")
ax[0, 0].set_xlim(0, L)
ax[0, 0].set_ylim(-1.5, 1.5)
ax[0, 0].legend()

# Energy plot
line_Eu, = ax[0, 1].plot([], [], lw=2, label=r"$E_u(t)$")
line_Ev, = ax[0, 1].plot([], [], lw=2, color="orange", label=r"$E_v(t)$")
ax[0, 1].set_xlabel("Time")
ax[0, 1].set_ylabel("Energy")
ax[0, 1].set_xlim(0, tmax)
ax[0, 1].legend()

# Energy dissipation rate plot
line_dEdu, = ax[0, 2].plot([], [], lw=2, label=r"$-dE_u/dt$")
line_dEdv, = ax[0, 2].plot([], [], lw=2, color="orange", label=r"$-dE_v/dt$")
ax[0, 2].set_xlabel("Time")
ax[0, 2].set_ylabel("Energy Dissipation Rate")
ax[0, 2].set_xlim(0, tmax)
ax[0, 2].legend()

# Dissipation and energy dissipation rate plot
line_epsu, = ax[1, 0].plot([], [], lw=2, label=r"$\varepsilon_u(t)$")
line_epsv, = ax[1, 0].plot([], [], lw=2, color="orange", label=r"$\varepsilon_v(t)$")
ax[1, 0].set_xlabel("Time")
ax[1, 0].set_ylabel("Dissipation Rate")
ax[1, 0].set_xlim(0, tmax)
ax[1, 0].legend()

# Energy spectrum plot
line_spectrum_u, = ax[1, 1].plot([], [], lw=2, label=r"$E_u(k)$")
line_spectrum_v, = ax[1, 1].plot([], [], lw=2, color="orange", label=r"$E_v(k)$")
ax[1, 1].set_xlabel(r"$k^{-2}$")
ax[1, 1].set_ylabel("Energy Spectrum")
ax[1, 1].legend()

# Average dissipation rate plot
line_epsu_avg, = ax[1, 2].plot([], [], lw=2, label=r"$\langle \varepsilon_u \rangle$")
line_epsv_avg, = ax[1, 2].plot([], [], lw=2, color="orange", label=r"$\langle \varepsilon_v \rangle$")
ax[1, 2].set_xlabel("Time")
ax[1, 2].set_ylabel("Average Dissipation Rate")
ax[1, 2].set_xlim(0, tmax)
ax[1, 2].legend()

def update(frame):
    global u_hat, v_hat, t, Espec_u, Espec_v, samples

    for _ in range(steps_per_frame):
        if t >= tmax:
            break

        u_hat = step(u_hat, dt, nu1)
        v_hat = step(v_hat, dt, nu2)
        t += dt
        
        Espec_u += 0.5 * np.abs(u_hat)**2
        Espec_v += 0.5 * np.abs(v_hat)**2

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
    ax[0, 1].set_ylim(0, max(E_u[0], E_v[0]) * 1.1)

    line_epsu.set_data(time, eps_u)
    line_epsv.set_data(time, eps_v)

    if len(time) > 1:
        dEdu = -np.gradient(E_u, time)
        dEdv = -np.gradient(E_v, time)
        line_dEdu.set_data(time, dEdu)
        line_dEdv.set_data(time, dEdv)

    if len(time) > 1:
        ax[0, 2].set_ylim(0, max(max(dEdu), max(dEdv)) * 1.1)
    ax[1, 0].set_ylim(0, max(max(eps_u), max(eps_v)) * 1.1)
        
    Ek_u = Espec_u / N
    Ek_v = Espec_v / N
    Ek_u = Ek_u[k > 0]
    Ek_v = Ek_v[k > 0]
    k_positive = k[k > 0]
    line_spectrum_u.set_data(k_positive**-2, Ek_u)
    line_spectrum_v.set_data(k_positive**-2, Ek_v)
    ax[1, 1].set_ylim(0, max(max(Ek_u), max(Ek_v)) * 1.1)
    ax[1, 1].set_xlim(0, max(k_positive**-2) * 1.1)
    
    eps_u_avg = np.cumsum(eps_u) / np.arange(1, len(eps_u) + 1)
    eps_v_avg = np.cumsum(eps_v) / np.arange(1, len(eps_v) + 1)
    line_epsu_avg.set_data(time, eps_u_avg)
    line_epsv_avg.set_data(time, eps_v_avg)
    ax[1, 2].set_ylim(0, max(max(eps_u_avg), max(eps_v_avg)) * 1.1)
    
    plt.suptitle(f"Time: {t:.2f}")

    return (
        line_u, line_v,
        line_Eu, line_Ev,
        line_epsu, line_epsv,
        line_dEdu, line_dEdv,
        line_spectrum_u, line_spectrum_v,
        line_epsu_avg, line_epsv_avg
    )

nframes = int(np.ceil(tmax / (steps_per_frame * dt)))
anim = FuncAnimation(fig, update, frames=nframes, interval=30, blit=False)
plt.show()
