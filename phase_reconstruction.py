import numpy as np

# Based on Holmes 2004
# 
# phi = m * Delta_theta * k0 * Delta_x
# N is the number of data points on the half-plane
# Hence, k0 Delta_x Delta_theta = pi / N



def solve_phase(s, phi, N=None):
    delta_s = np.diff(s)
    phi = (phi[:-1] + phi[1:]) / 2
    if N is None:
        N = (len(phi) - 1) // 2
    X = np.empty((len(phi), 2 * N))
    j = np.arange(N) + 1
    delta_phi = np.diff(phi)[0]
    prefactor = (1 - 2 * np.sin(delta_phi / 2))**j - 1
    X[:, :N] = np.cos(j[None, :] * phi[:, None]) * prefactor
    X[:, N:] = np.sin(j[None, :] * phi[:, None]) * prefactor
    b = np.linalg.inv(X.T @ X) @ (X.T @ delta_s)
    return X, b


