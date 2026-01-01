import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Solve the equation
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² + f(x, t)
on a periodic domain using a Fourier-Galerkin method.
The solution is expanded in Fourier modes and the nonlinear term is computed pseudospectrally using FFTs.

In Fourier space, the Burgers equation becomes:
    ∂û_k/∂t = -i k/2 û²_k - ν k² û_k + f̂_k

f injects energy at large scales to maintain a statistically steady turbulent state.

f̂_k is defined as:
    σ ζ_k(t) for |k| ≤ k_f
    0 otherwise
where ζ_k(t) is a complex Gaussian white noise process.

Files:
- flow-simulations/videos/stable_turbulence.mp4
"""

def main():
    L = 2 * np.pi
    N = 256
    dx = L / N
    nu1 = 1e-3
    nu2 = 0.01
    tmax = 50.0
    CFL = 0.2

    x = np.linspace(0, L, N, endpoint=False)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    u0 = np.sin(x)
    u_hat = np.fft.fft(u0)
    v_hat = np.fft.fft(u0)
    
    def compute_dt(u_hat, nu):
        u = np.fft.ifft(u_hat).real
        umax = max(np.max(np.abs(u)), 1e-6)
        dt_adv = CFL * dx / umax
        dt_diff = CFL * dx**2 / nu
        return min(dt_adv, dt_diff)

    def forcing():
        f_hat = np.zeros(N, dtype=complex)
        kf = 4
        sigma = 0.5
        for i, ki in enumerate(k):
            if abs(ki) <= kf:
                f_hat[i] = sigma * (np.random.randn() + 1j * np.random.randn())
        return f_hat

    def rhs(u_hat, f_hat, nu):
        u = np.fft.ifft(u_hat).real
        u2_hat = np.fft.fft(u**2)
        u2_hat[np.abs(k) > (2 * (N // 2)) // 3] = 0
        return -0.5j * k * u2_hat - nu * k**2 * u_hat + f_hat
    
    def step(u_hat, f_hat, dt, nu):
        k1 = rhs(u_hat, f_hat, nu)
        k2 = rhs(u_hat + 0.5 * dt * k1, f_hat, nu)
        k3 = rhs(u_hat + 0.5 * dt * k2, f_hat, nu)
        k4 = rhs(u_hat + dt * k3, f_hat, nu)
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
    Espec_u = np.zeros(N)
    Espec_v = np.zeros(N)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8), tight_layout=True)

    line_u, = ax[0, 0].plot(x, np.fft.ifft(u_hat).real, label=r"$u(x, t)$ and $\nu={}$".format(nu1))
    line_v, = ax[0, 0].plot(x, np.fft.ifft(v_hat).real, label=r"$v(x, t)$ and $\nu={}$".format(nu2))
    ax[0, 0].set_xlim(0, L)
    ax[0, 0].set_ylim(-1.5, 1.5)
    ax[0, 0].set_title("Velocity Fields")
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("u, v")

    line_Eu, = ax[0, 1].plot([], [], label='E_u(t)')
    line_Ev, = ax[0, 1].plot([], [], label='E_v(t)')
    ax[0, 1].set_xlim(0, tmax)
    ax[0, 1].set_title("Kinetic Energies")
    ax[0, 1].set_xlabel("t")
    ax[0, 1].set_ylabel("E_u, E_v")

    line_dEdu, = ax[0, 2].plot([], [], label=r"$-dE_u/dt$")
    line_dEdv, = ax[0, 2].plot([], [], label=r"$-dE_v/dt$")
    ax[0, 2].set_xlim(0, tmax)
    ax[0, 2].set_title("Dissipation Rates")
    ax[0, 2].set_xlabel("t")
    ax[0, 2].set_ylabel(r"$-dE_u/dt, -dE_v/dt$")

    line_epsu, = ax[1, 0].plot([], [], label=r"$\varepsilon_u(t)$")
    line_epsv, = ax[1, 0].plot([], [], label=r"$\varepsilon_v(t)$")
    ax[1, 0].set_xlim(0, tmax)
    ax[1, 0].set_title("Instantaneous Dissipation Rates")
    ax[1, 0].set_xlabel("t")
    ax[1, 0].set_ylabel(r"$\varepsilon_u(t), \varepsilon_v(t)$")

    line_spectrum_u, = ax[1, 1].loglog([], [], label="E_u(k)")
    line_spectrum_v, = ax[1, 1].loglog([], [], label="E_v(k)")
    ax[1, 1].set_title("Energy Spectra")
    ax[1, 1].set_xlabel("k")
    ax[1, 1].set_ylabel("E(k)")

    line_epsu_avg, = ax[1, 2].plot([], [], label=r"$\langle \varepsilon_u \rangle$")
    line_epsv_avg, = ax[1, 2].plot([], [], label=r"$\langle \varepsilon_v \rangle$")
    ax[1, 2].set_xlim(0, tmax)
    ax[1, 2].set_title("Average Dissipation Rates")
    ax[1, 2].set_xlabel("t")
    ax[1, 2].set_ylabel(r"$\langle \varepsilon_u \rangle, \langle \varepsilon_v \rangle$")
    
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 2].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[1, 2].legend()

    def update(frame):
        nonlocal u_hat, v_hat, t, dt, line_u, line_v, line_Eu, line_Ev, line_dEdu, line_dEdv, line_epsu, line_epsv, line_spectrum_u, line_spectrum_v, line_epsu_avg, line_epsv_avg, Espec_u, Espec_v, ax

        for _ in range(steps_per_frame):
            if t >= tmax:
                break

            u_hat = step(u_hat, forcing(), dt, nu1)
            v_hat = step(v_hat, forcing(), dt, nu2)
            dt_u = compute_dt(u_hat, nu1)
            dt_v = compute_dt(v_hat, nu2)
            dt = min(dt_u, dt_v)
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
        
        line_epsu.set_data(time, eps_u)
        line_epsv.set_data(time, eps_v)

        Ek_u = Espec_u / len(time)
        Ek_v = Espec_v / len(time)

        kp = k[k > 0]
        Ekmax = max(np.max(Ek_u[k > 0]), np.max(Ek_v[k > 0]))
        ax[1, 1].set_xlim(kp[0], kp[-1])
        ax[1, 1].set_ylim(1e-8 * Ekmax, 1.1 * Ekmax)
        
        line_spectrum_u.set_data(kp, Ek_u[k > 0])
        line_spectrum_v.set_data(kp, Ek_v[k > 0])

        eps_u_avg = np.cumsum(eps_u) / np.arange(1, len(eps_u) + 1)
        eps_v_avg = np.cumsum(eps_v) / np.arange(1, len(eps_v) + 1)
        line_epsu_avg.set_data(time, eps_u_avg)
        line_epsv_avg.set_data(time, eps_v_avg)

        if len(time) > 1:
            dEdu = -np.gradient(E_u, time)
            dEdv = -np.gradient(E_v, time)
            line_dEdu.set_data(time, dEdu)
            line_dEdv.set_data(time, dEdv)
            
            ax[0, 1].set_ylim(0, max(E_u[0], E_v[0]) * 1.1)
            ax[0, 2].set_ylim(0, max(max(dEdu), max(dEdv)) * 1.1)
            ax[1, 0].set_ylim(0, max(max(eps_u), max(eps_v)) * 1.1)
            ax[1, 2].set_ylim(0, max(np.max(eps_u_avg), np.max(eps_v_avg)) * 1.1)
        
        plt.suptitle(f"t = {t:.2f}")

        return (
            line_u, line_v,
            line_Eu, line_Ev,
            line_dEdu, line_dEdv,
            line_epsu, line_epsv,
            line_spectrum_u, line_spectrum_v,
            line_epsu_avg, line_epsv_avg
        )

    anim = FuncAnimation(fig, update, frames=960, interval=30, blit=False)
    anim.save("flow-simulations/videos/stable_turbulence.mp4", fps=60, dpi=300)


if __name__ == "__main__":
    main()
