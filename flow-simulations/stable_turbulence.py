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

    def rhs(u_hat, nu):
        u = np.fft.ifft(u_hat).real
        u2_hat = np.fft.fft(u**2)
        u2_hat[np.abs(k) > (2 * (N // 2)) // 3] = 0
        return -0.5j * k * u2_hat - nu * k**2 * u_hat + forcing()

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
    Espec_u = np.zeros(N)
    Espec_v = np.zeros(N)

    fig, ax = plt.subplots(2, 3, figsize=(10, 4), tight_layout=True)

    line_u, = ax[0, 0].plot(x, np.fft.ifft(u_hat).real)
    line_v, = ax[0, 0].plot(x, np.fft.ifft(v_hat).real)

    line_Eu, = ax[0, 1].plot([], [])
    line_Ev, = ax[0, 1].plot([], [])

    line_dEdu, = ax[0, 2].plot([], [])
    line_dEdv, = ax[0, 2].plot([], [])

    line_epsu, = ax[1, 0].plot([], [])
    line_epsv, = ax[1, 0].plot([], [])

    line_spectrum_u, = ax[1, 1].plot([], [])
    line_spectrum_v, = ax[1, 1].plot([], [])

    line_epsu_avg, = ax[1, 2].plot([], [])
    line_epsv_avg, = ax[1, 2].plot([], [])

    def update(frame):
        nonlocal u_hat, v_hat, t, Espec_u, Espec_v

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

        line_u.set_ydata(np.fft.ifft(u_hat).real)
        line_v.set_ydata(np.fft.ifft(v_hat).real)

        line_Eu.set_data(time, E_u)
        line_Ev.set_data(time, E_v)

        if len(time) > 1:
            dEdu = -np.gradient(E_u, time)
            dEdv = -np.gradient(E_v, time)
            line_dEdu.set_data(time, dEdu)
            line_dEdv.set_data(time, dEdv)

        Ek_u = Espec_u / len(time)
        Ek_v = Espec_v / len(time)

        kp = k[k > 0]
        line_spectrum_u.set_data(kp, Ek_u[k > 0])
        line_spectrum_v.set_data(kp, Ek_v[k > 0])

        eps_u_avg = np.cumsum(eps_u) / np.arange(1, len(eps_u) + 1)
        eps_v_avg = np.cumsum(eps_v) / np.arange(1, len(eps_v) + 1)
        line_epsu_avg.set_data(time, eps_u_avg)
        line_epsv_avg.set_data(time, eps_v_avg)

        plt.suptitle(f"t = {t:.2f}")

        return (
            line_u, line_v,
            line_Eu, line_Ev,
            line_dEdu, line_dEdv,
            line_epsu, line_epsv,
            line_spectrum_u, line_spectrum_v,
            line_epsu_avg, line_epsv_avg
        )

    nframes = int(np.ceil(tmax / (steps_per_frame * dt)))
    anim = FuncAnimation(fig, update, frames=nframes, interval=30, blit=False)
    anim.save("flow-simulations/videos/stable_turbulence.mp4", fps=60, dpi=300)


if __name__ == "__main__":
    main()
