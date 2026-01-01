import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Solve for the vorticity field in a 2D incompressible Navier-Stokes flow using a spectral method.

The vorticity equation is given by:
∂ω/∂t + u·∇ω = ν∇²ω  - αω + f

with the condition
∇·u = 0

Files: 
- flow-simulations/videos/turbulence_with_linearfriction_vorticity.mp4
- flow-simulations/videos/turbulence_with_linearfriction_velocity.mp4
- flow-simulations/images/turbulence_with_linearfriction_energy_enstrophy.png
- flow-simulations/images/turbulence_with_linearfriction_energy_spectrum.png
"""

L = 2 * np.pi
N = 256
dx = L / N
nu = 1e-3
alpha = 0.1
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
k_mag = np.sqrt(k2)

ux = np.fft.ifft2(1j * ky * (-w_hat / k2)).real
uy = np.fft.ifft2(-1j * kx * (-w_hat / k2)).real
umax = np.max(np.maximum(np.abs(ux), np.abs(uy)))
umax = max(umax, 1e-6)

dt_adv = CFL * dx / umax
dt_diff = CFL * dx**2 / nu
dt = min(dt_adv, dt_diff)

wmax = np.max(np.abs(w0))

def reset_state():
    return np.fft.fft2(w0), 0.0

def compute_dt(w_hat):
    psi_hat = -w_hat / k2
    ux = np.fft.ifft2(1j * ky * psi_hat).real
    uy = np.fft.ifft2(-1j * kx * psi_hat).real
    umax = max(np.max(np.abs(ux)), np.max(np.abs(uy)), 1e-6)
    return min(CFL * dx / umax, CFL * dx**2 / nu)   

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

def rhs(w_hat, f_hat):
    psi_hat = -w_hat / k2
    ux = np.fft.ifft2(1j * ky * psi_hat).real
    uy = np.fft.ifft2(-1j * kx * psi_hat).real
    dwdx = np.fft.ifft2(1j * kx * w_hat).real
    dwdy = np.fft.ifft2(1j * ky * w_hat).real
    advective_term = ux * dwdx + uy * dwdy
    advective_term_hat = np.fft.fft2(advective_term)
    advective_term_hat[np.abs(kx) > (2 * (N // 2)) // 3] = 0
    advective_term_hat[np.abs(ky) > (2 * (N // 2)) // 3] = 0
    return -advective_term_hat - nu * k2 * w_hat - alpha * w_hat + f_hat

def step(w_hat, f_hat, dt):
    k1 = rhs(w_hat, f_hat)
    k2 = rhs(w_hat + 0.5 * dt * k1, f_hat)
    k3 = rhs(w_hat + 0.5 * dt * k2, f_hat)
    k4 = rhs(w_hat + dt * k3, f_hat)
    return w_hat + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
def vorticity_simulation():
    global w_hat, t, dt
    w_hat, t = reset_state()
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
            w_hat = step(w_hat, forcing(), dt)
            dt = compute_dt(w_hat)
            t += dt
        w = np.fft.ifft2(w_hat).real
        im.set_data(w)
        title.set_text(f'Time = {t:.2f}')
        return im,

    anim = FuncAnimation(fig, update, frames = 360, interval=50, blit=False)
    anim.save("flow-simulations/videos/turbulence_with_linearfriction_vorticity.mp4", fps=60, dpi=300)
    plt.close('all')


def energy_enstrophy_simulation():
    global w_hat, t, dt
    w_hat, t = reset_state()
    KE = []
    EN = []
    time = []

    while t<tmax:
        w_hat = step(w_hat, forcing(), dt)
        dt = compute_dt(w_hat)
        t += dt
        KE.append(0.5*np.sum(np.abs(w_hat)**2 / k2) * dx**2)
        EN.append(0.5*np.sum(np.abs(w_hat)**2) * dx**2)
        time.append(t)

    plt.plot(time, KE, label='Kinetic Energy')
    plt.plot(time, EN, label='Enstrophy')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Kinetic Energy and Enstrophy over Time')
    plt.savefig('flow-simulations/images/turbulence_with_linearfriction_energy_enstrophy.png', dpi=300)
    plt.close('all')

def velocity_simulation():
    global w_hat, t, dt
    w_hat, t = reset_state()
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
            w_hat = step(w_hat, forcing(), dt)
            dt = compute_dt(w_hat)
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
        plt.suptitle(f'Time = {t:.2f}')
        return im_ux, im_uy

    anim = FuncAnimation(fig, update, frames = 360, interval=50, blit=False)
    anim.save("flow-simulations/videos/turbulence_with_linearfriction_velocity.mp4", fps=60, dpi=300)
    plt.close('all')


def energy_spectrum_simulation():
    global w_hat, t, dt
    w_hat, t = reset_state()
    t_steady = 5.0

    k_mag = np.sqrt(kx**2 + ky**2)
    k_bins = np.arange(1, N//2)
    E_spec = np.zeros(len(k_bins))
    samples = 0

    while t < tmax:
        w_hat = step(w_hat, forcing(), dt)
        dt = compute_dt(w_hat)
        t += dt

        if t < t_steady:
            continue

        E_hat = 0.5 * np.abs(w_hat)**2 / k2
        E_shell = np.zeros_like(E_spec)

        for i in range(len(k_bins) - 1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
            E_shell[i] = np.sum(E_hat[mask])

        E_spec += E_shell
        samples += 1

    E_avg = E_spec / samples

    plt.figure(figsize=(8, 6))
    plt.loglog(k_bins, E_avg, lw=2)
    plt.xlabel(r"Wavenumber $k$")
    plt.ylabel(r"Kinetic Energy Spectrum $E(k)$")
    plt.title("Steady-State Energy Spectrum")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.savefig("flow-simulations/images/turbulence_with_linearfriction_energy_spectrum.png", dpi=300)
    plt.close('all')
    

if __name__ == "__main__":
    # vorticity_simulation()
    # energy_enstrophy_simulation()
    velocity_simulation()   
    # energy_spectrum_simulation()
