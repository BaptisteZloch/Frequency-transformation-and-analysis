import numpy as np


def explicit_heat_smooth(prices: np.array, t_end: float = 3.0) -> np.array:
    """
    Smoothen out a time series using a simple explicit finite difference method.
    The scheme uses a first-order method in time, and a second-order centred
    difference approximation in space. The scheme is only numerically stable
    if the time-step 0<=k<=1.

    The prices are fixed at the end-points, so the interior is smoothed.

    Parameters
    ----------
    prices : np.array
        The price to smoothen
    t_end : float
        The time at which to terminate the smoothing (i.e. t = 2)


    Returns
    -------
    P : np.array
        The smoothened time-series

    """

    k = 0.1  # Time spacing

    # Set up the initial condition
    P = prices

    t = 0
    while t < t_end:
        # Solve the finite difference scheme for the next time-step
        P = k * (P[2:] + P[:-2]) + P[1:-1] * (1 - 2 * k)

        # Add the fixed boundary conditions since the above solves the interior
        # points only
        P = np.hstack(
            (
                np.array([prices[0]]),
                P,
                np.array([prices[-1]]),
            )
        )
        t += k

    return P


def heat_analytical_smooth(prices: np.array, t: float = 3.0, m: int = 200) -> np.array:
    """
    Find the analytical solution to the heat equation

    See: https://tutorial.math.lamar.edu/classes/de/heateqnnonzero.aspx

    Parameters
    ----------
    prices : np.array
        The price to smoothen.
    t : float
        The time at which to terminate the smootheing (i.e. t = 2)
    m : int
        The amount of terms in the solution's Fourier series

    Returns
    -------
    np.array
        The analytical solution to the heat equation

    """

    p0 = prices[0]
    pn = prices[-1]

    n = prices.shape[0]
    x = np.arange(0, n, dtype=np.float32)
    M = np.arange(1, m, dtype=np.float32)

    L = n - 1
    u_e = p0 + (pn - p0) * x / L

    mx = M.reshape(-1, 1) @ x.reshape(1, -1)
    sin_m_pi_x = np.sin(mx * np.pi / L)

    # Calculate the B_m terms using numerical quadrature (trapezium rule)
    bm = (
        2
        * np.sum(
            (sin_m_pi_x * (prices - u_e)).T,
            axis=0,
        )
        / n
    )

    return u_e + np.sum(
        (bm * np.exp(-t * (M * np.pi / L) ** 2)).reshape(-1, 1) * sin_m_pi_x,
        axis=0,
    )


def butter_lowpass_filter(data):
    fs = 1 / 300  # sample rate, Hz day :1/24/3600
    cutoff = 0.1  # desired cutoff frequency of the filter, Hz, slightly higher than actual 1.2 Hz
    b, a = butter(2, cutoff, btype="lowpass")  # low pass filter
    y = filtfilt(b, a, data)
    return y
