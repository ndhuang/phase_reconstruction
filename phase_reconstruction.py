import numpy as np

# Based on Holmes 2004
# https://ui.adsabs.harvard.edu/abs/2004JOSAA..21..697H/abstract


def solve_phase(s, phi, N=None):
    '''
    This implements the OLS fit prescribed in equations 11 and 14.

    Parameters
    ----------
    s : array-like
        The natural log of the Fourier magnitudes
    phi : array-like
        Formally, this the the phase of the argument of the Fourier transform
        (in the phasor formulation).  In practice, this (almost?) always maps to
        the wavenumber.  Must be regularly spaced and monotonically increasing.
    N : int
        Number of expansion terms.  If set to `None`, `N` is set to the maximum
        number of terms that can be constrained.


    Returns
    -------
    a, b : array-like
        The best fit a_j and b_j terms, which can be used to determine the
        Fourier phases when passed to `evaluate_phase`.
    '''
    delta_s = np.diff(s)
    phi = (phi[:-1] + phi[1:]) / 2
    if N is None:
        N = (len(phi) - 1) // 2
    X = np.empty((len(phi), 2 * N))
    j = np.arange(N) + 1
    delta_phi = np.diff(phi)[0]
    prefactor = (1 - 2 * np.sin(delta_phi / 2)) ** j - 1
    X[:, :N] = np.cos(j[None, :] * phi[:, None]) * prefactor
    X[:, N:] = np.sin(j[None, :] * phi[:, None]) * prefactor
    ab = np.linalg.inv(X.T @ X) @ (X.T @ delta_s)
    a = ab[:N]
    b = ab[N:]
    return a, b


def evaluate_phase(phi, a, b, rho=1):
    '''
    Evaluate the phase, given the coeffecients a_j and b_j.  This implements the
    sum in equation 13.

    Parameters
    ----------
    phi : array-like
        Formally, this the the phase of the argument of the Fourier transform
        (in the phasor formulation).  In practice, this (almost?) always maps to
        the wavenumber.
    a, b : array-like
        The coeffecients a_j and b_j (1 <= j <= N).  These can be determined
        using `solve_phase`.
    rho : float
        The magnitude of the argument of the Fourier transform.  Under (almost?)
        all circumstances, this is 1.

    Returns
    -------
    The Fourier phases at locations given by phi.
    '''
    return np.sum(
        [
            rho**j
            * (a[j] * np.cos((j + 1) * phi) + b[j] * np.sin((j + 1) * phi))
            for j in range(len(a))
        ],
        axis=0,
    )


def reconstruct_phase(s, phi):
    '''
    Reconstruct Fourier phases from the Fourier magnitudes.

    Parameters
    ----------
    s : array-like
        The natural log of the Fourier magnitudes
    phi : array-like
        Formally, this the the phase of the argument of the Fourier transform
        (in the phasor formulation).  In practice, this (almost?) always maps to
        the wavenumber.  Must be regularly spaced and monotonically increasing.

    Returns
    -------
    The Fourier phases at locations given by phi.
    '''
    return evaluate_phase(phi, *solve_phase(s, phi))
