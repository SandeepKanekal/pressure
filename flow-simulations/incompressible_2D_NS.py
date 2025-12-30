import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


"""
Solve for the vorticity field in a 2D incompressible Navier-Stokes flow using a spectral method.

The vorticity equation is given by:
∂ω/∂t + u·∇ω = ν∇²ω + f

with the condition
∇·u = 0

Files: 
- flow-simulations/incompressible_2D_NS.py
- flow-simulations/videos/incompressilbe_2D_NS_velocity.mp4
- flow-simulations/videos/incompressible_2D_NS_energy_spectrum.mp4
- flow-simulations/images/incompressible_2D_NS_energy_enstrophy.png
"""


L = 2.0 * np.pi
N = 256
dx = L / N
dy = L / N
nu = 0.01
tmax = 15
CFL = 0.1

x, y = np.meshgrid(np.linspace(0, L, N, endpoint=False), np.linspace(0, L, N, endpoint=False))
w0 = np.sin(x) * np.cos(y)
t = 0.0
w_hat = np.fft.fft2(w0)

k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
kx, ky = np.meshgrid(k, k)
k2 = kx**2 + ky**2
k2[0, 0] = 1.0  # Avoid division by zero

ux = np.fft.ifft2(1j * ky * (-w_hat / k2)).real
uy = np.fft.ifft2(-1j * kx * (-w_hat / k2)).real
umax = np.max(np.maximum(np.abs(ux), np.abs(uy)))
umax = max(umax, 1e-6)

dt_adv = CFL * dx / umax
dt_diff = CFL * dx**2 / nu
dt = min(dt_adv, dt_diff)

wmax = np.max(np.abs(w0))

def forcing():
    f_hat = np.zeros((N, N), dtype=complex)
    kf = 4
    sigma = 0.5
    for i in range(N):
        for j in range(N):
            k_mag = np.sqrt(kx[i, j]**2 + ky[i, j]**2)
            if k_mag <= kf:
                f_hat[i, j] = sigma * (np.random.randn() + 1j * np.random.randn())
    return f_hat
        

def rhs(w_hat):
    psi_hat = -w_hat / k2
    ux_hat = 1j * ky * psi_hat
    uy_hat = -1j * kx * psi_hat
    ux = np.fft.ifft2(ux_hat).real
    uy = np.fft.ifft2(uy_hat).real
    w = np.fft.ifft2(w_hat).real
    dwdx = np.fft.ifft2(1j * kx * w_hat).real
    dwdy = np.fft.ifft2(1j * ky * w_hat).real
    nonlinear_term = ux * dwdx + uy * dwdy
    nonlinear_term_hat = np.fft.fft2(nonlinear_term)
    nonlinear_term_hat[np.abs(kx) > (2 * (N // 2)) // 3] = 0
    nonlinear_term_hat[np.abs(ky) > (2 * (N // 2)) // 3] = 0
    return -nonlinear_term_hat - nu * k2 * w_hat + forcing()


def step(w_hat, dt):
    f1 = rhs(w_hat)
    f2 = rhs(w_hat + 0.5 * dt * f1)
    f3 = rhs(w_hat + 0.5 * dt * f2)
    f4 = rhs(w_hat + dt * f3)
    return w_hat + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)


def vorticity_simulation():
    global w_hat, t, dt
    fig, axes = plt.subplots(figsize=(10, 10))
    im = axes.imshow(w0, origin='lower', extent=(0, L, 0, L), vmin=-wmax, vmax=wmax, cmap='RdBu')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    title = axes.set_title(f'Time = {t:.2f}')
    cbar = fig.colorbar(im, ax=axes, label=r'$\omega(x, y, t)$')

    def update(frame):
        global w_hat, t, dt
        for _ in range(5):
            if t >= tmax:
                break
            w_hat = step(w_hat, dt)
            t += dt
        w = np.fft.ifft2(w_hat).real
        im.set_data(w)
        title.set_text(f'Time = {t:.2f}')
        return im,

    anim = FuncAnimation(fig, update, frames = int(tmax/(dt*5)), interval=50, blit=False)
    anim.save("flow-simulations/videos/incompressible_2D_NS.mp4", fps=60, dpi=300)


def energy_enstrophy_simulation():
    global w_hat, t, dt
    KE = []
    EN = []
    time = []

    while t<tmax:
        w_hat = step(w_hat, dt)
        t += dt
        print('T = ', t)
        KE.append(0.5*np.sum(np.abs(w_hat)**2 / k2) * dx * dy)
        EN.append(0.5*np.sum(np.abs(w_hat)**2) * dx * dy)
        time.append(t)

    plt.plot(time, KE, label='Kinetic Energy')
    plt.plot(time, EN, label='Enstrophy')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Kinetic Energy and Enstrophy over Time')
    plt.savefig('flow-simulations/images/incompressible_2D_NS_energy_enstrophy.png', dpi=300)


def velocity_simulation():
    global w_hat, t, dt
    ux_data, uy_data = [], []

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    im_ux = axes[0].imshow(np.fft.ifft2(1j * ky * (-w_hat / k2)).real, origin='lower', extent=(0, L, 0, L), cmap='viridis')
    axes[0].set_title('Velocity Field u_x')
    im_uy = axes[1].imshow(np.fft.ifft2(-1j * kx * (-w_hat / k2)).real, origin='lower', extent=(0, L, 0, L), cmap='viridis')
    axes[1].set_title('Velocity Field u_y')
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
    cbar_ux = fig.colorbar(im_ux, ax=axes[0], label=r'$u_x(x, y, t)$')
    cbar_uy = fig.colorbar(im_uy, ax=axes[1], label=r'$u_y(x, y, t)$')

    def update(frame):
        global w_hat, t, dt
        for _ in range(5):
            if t >= tmax:
                break
            what = step(w_hat, dt)
            t += dt
        psi_hat = -w_hat / k2
        ux_hat = 1j * ky * psi_hat
        uy_hat = -1j * kx * psi_hat
        ux = np.fft.ifft2(ux_hat).real
        uy = np.fft.ifft2(uy_hat).real
        ux_data.append(ux)
        uy_data.append(uy)
        im_ux.set_data(ux)
        im_uy.set_data(uy)
        return im_ux, im_uy

    anim = FuncAnimation(fig, update, frames = int(tmax/(dt*5)), interval=50, blit=False)
    anim.save("flow-simulations/videos/incompressible_2D_NS_velocity.mp4", fps=60, dpi=300)


def energy_spectrum_simulation():
    global w_hat, t, dt

    k_mag = np.sqrt(kx**2 + ky**2)
    k_bins = np.arange(1, N//2)
    E_spec = np.zeros(len(k_bins))
    samples = 0

    fig, axes = plt.subplots(figsize=(8, 6))
    line_Espec, = axes.loglog([], [], lw=2)

    axes.set_xlabel('Wavenumber k')
    axes.set_ylabel('Kinetic Energy Spectrum')
    axes.set_xlim(1, N//2)

    def update(frame):
        global w_hat, t, dt
        nonlocal E_spec, samples

        for _ in range(5):
            if t >= tmax:
                break
            w_hat = step(w_hat, dt)
            t += dt

        E_hat = 0.5 * np.abs(w_hat)**2 / k2
        E_shell = np.zeros_like(E_spec)
        for i in range(len(k_bins) - 1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            E_shell[i] = np.sum(E_hat[mask])

        E_spec += E_shell
        samples += 1
        
        if samples > 1:
            axes.set_ylim(1e-6, np.max(E_spec / samples) * 1.2)

        line_Espec.set_data(k_bins, E_spec / samples)
        axes.set_title(f'Time = {t:.2f}')

        return line_Espec,

    anim = FuncAnimation(fig, update, frames=int(tmax / (dt * 5)), interval=50, blit=False)
    anim.save("flow-simulations/videos/incompressible_2D_NS_energy_spectrum.mp4", fps=60, dpi=300)


if __name__ == "__main__":
    vorticity_simulation()
    energy_enstrophy_simulation()
    velocity_simulation()
    energy_spectrum_simulation()
