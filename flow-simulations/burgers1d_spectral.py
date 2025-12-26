import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Solve the 1D viscous Burgers equation
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
on a periodic domain using a Fourier–Galerkin method.
The solution is expanded in Fourier modes and the nonlinear term is computed pseudospectrally using FFTs.

In Fourier space, the Burgers equation becomes:
    ∂û_k/∂t = -i k/2 \hat{u^2}_k - ν k² û_k
"""

L = 2 * np.pi
N = 256
dx = L / N
nu1 = 0.1
nu2 = 0.01
tmax = 15
CFL = 0.01

x = np.linspace(0, L, N, endpoint=False)
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

u0 = np.sin(x)
u_hat = np.fft.fft(u0)
v_hat = np.fft.fft(u0)

# RHS in Fourier space
def rhs(u_hat, nu):
    u = np.fft.ifft(u_hat).real
    u2_hat = np.fft.fft(u**2)
    u2_hat[abs(k) > (2* N//2)//3] = 0  # dealiasing
    nonlinear = -0.5j * k * u2_hat
    diffusion = -nu * k**2 * u_hat
    return nonlinear + diffusion

# RK4 step
def step(u_hat, dt, nu):
    k1 = rhs(u_hat, nu)
    k2 = rhs(u_hat + 0.5 * dt * k1, nu)
    k3 = rhs(u_hat + 0.5 * dt * k2, nu)
    k4 = rhs(u_hat + dt * k3, nu)
    return u_hat + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# Time step
dt = CFL * dx / np.max(np.abs(u0))
t = 0.0
steps_per_frame = 5

# Figure
fig, ax = plt.subplots()
line_u, = ax.plot(x, np.fft.ifft(u_hat).real, lw=2, label=r"$\nu = {}$".format(nu1))
line_v, = ax.plot(x, np.fft.ifft(v_hat).real, lw=2, color='orange', label=r"$\nu = {}$".format(nu2))
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Time: 0.00")
ax.legend()

def update(frame):
    global u_hat, v_hat, t

    if t >= tmax:
        anim.event_source.stop()
        return line_u, line_v

    for _ in range(steps_per_frame):
        if t >= tmax:
            break
        u_hat = step(u_hat, dt, nu1)
        v_hat = step(v_hat, dt, nu2)
        t += dt
    u = np.fft.ifft(u_hat).real
    v = np.fft.ifft(v_hat).real
    line_u.set_ydata(u)
    line_v.set_ydata(v)
    
    ax.set_title(f"Time: {t:.2f}")

    return line_u, line_v

# Number of frames needed to reach tmax
nframes = int(np.ceil(tmax / (steps_per_frame * dt)))

anim = FuncAnimation(
    fig,
    update,
    frames=nframes,
    interval=30,
    blit=False
)

plt.show()
