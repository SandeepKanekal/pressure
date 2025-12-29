import numpy as np
import matplotlib.pyplot as plt

"""
Aim of this module is to compute the structure function of a 1D field. We solve the following equation for the velocity field:
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² + f(x, t)
on a periodic domain using a Fourier-Galerkin method.
The solution is expanded in Fourier modes and the nonlinear term is computed pseudospectrally using FFTs.

In Fourier space, the Burgers equation becomes:
    ∂û_k/∂t = -i k/2 û²_k - ν k² û_k + ˆf_k

After obtaining the velocity field, we compute strucuture functions of various orders. Defined as:
    S_p (r) = <|u(x + r) - u(x)|^p>
where <.> denotes spatial averaging over x.

Files:
- flow-simulations/images/structure_functions_1d.png
"""

L = 2 * np.pi
N = 256
dx = L / N
nu = 0.01
tmax = 15
CFL = 0.2

x = np.linspace(0, L, N, endpoint=False)
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
u0 = np.sin(x)
u_hat = np.fft.fft(u0)

def forcing():
    f_hat = np.zeros(N, dtype=complex)
    kf = 4.0
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

# Time integration
t = 0.0
while t < tmax:
    u = np.fft.ifft(u_hat).real
    umax = max(np.max(np.abs(u)), 1e-6)
    
    dt_advec = CFL * dx / umax
    dt_diff = CFL * dx**2 / nu
    dt = min(dt_advec, dt_diff)
    
    u_hat = step(u_hat, dt, nu)
    t += dt

u = np.fft.ifft(u_hat).real
r = np.arange(1, N//2) * dx
S1 = np.zeros_like(r)
S2 = np.zeros_like(r)
S3 = np.zeros_like(r)

for i, ri in enumerate(r):
    diffs = np.abs(u - np.roll(u, -i))  # Boundary conditions are periodic
    S1[i] = np.mean(diffs)
    S2[i] = np.mean(diffs**2)
    S3[i] = np.mean(diffs**3)

plt.loglog(r, S1, label='S1')
plt.loglog(r, S2, label='S2', color='orange')
plt.loglog(r, S3, label='S3', color='green')
plt.xlabel('r')
plt.ylabel('Structure Functions')
plt.legend()

r_min = 0.1
r_max = 1.0
mask = (r > r_min) & (r < r_max)
for S, name in zip([S1, S2, S3], ['S1', 'S2', 'S3']):
    coeffs = np.polyfit(np.log(r[mask]), np.log(S[mask]), 1)
    print(f"Scaling exponent for {name}: {coeffs[0]:.3f}")

plt.savefig('structure_functions_1d.png', dpi=300)
