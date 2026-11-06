from functools import cache
from operator import index

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

# Module-level constants for magic numbers
IMPVOL_MIN = 1e-10
IMPVOL_MAX = 5.0


# Internal quadrature helpers and caches
def _validate_quadrature_order(n: int) -> int:
    if isinstance(n, (bool, np.bool_)):
        raise ValueError("n must be a positive integer.")

    try:
        n = index(n)
    except TypeError as exc:
        raise TypeError("n must be an integer.") from exc

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    return n


@cache
def _cached_leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


# Quadrature rules
def gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on the interval [a, b].

    Parameters
    ----------
    a : float
        Lower bound of the integration interval.
    b : float
        Upper bound of the integration interval.
    n : int
        Number of quadrature points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 1-D arrays:
        - Quadrature points on [a, b].
        - Quadrature weights on [a, b].
    """
    n = _validate_quadrature_order(n)
    knots, weights = _cached_leggauss(n)
    knots_a_b = 0.5 * (b - a) * knots + 0.5 * (b + a)
    weights_a_b = 0.5 * (b - a) * weights
    return knots_a_b, weights_a_b


# Linear algebra helpers for simulation routines
# Black pricing and implied-volatility routines
def black_price(K, T, F, vol, opttype: float | np.ndarray = 1.0):
    """
    Calculate the Black option price.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    opttype : float or np.ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        The Black price of the option.
    """
    T = float(T)
    K, F, vol, opttype = np.broadcast_arrays(
        np.asarray(K, dtype=float),
        np.asarray(F, dtype=float),
        np.asarray(vol, dtype=float),
        np.asarray(opttype, dtype=float),
    )

    if not np.all(np.abs(opttype) == 1.0):
        raise ValueError("opttype must be either 1 or -1.")

    valid = (K > 0.0) & (F > 0.0) & np.isfinite(K) & np.isfinite(F)
    intrinsic = np.maximum(opttype * (F - K), 0.0)
    price = np.where(valid, intrinsic, np.nan)

    positive_var = valid & (T > 0.0) & (vol > 0.0) & np.isfinite(vol)
    if np.any(positive_var):
        s = vol[positive_var] * T**0.5
        d1 = np.log(F[positive_var] / K[positive_var]) / s + 0.5 * s
        d2 = d1 - s
        price[positive_var] = opttype[positive_var] * (
            F[positive_var] * norm.cdf(opttype[positive_var] * d1)
            - K[positive_var] * norm.cdf(opttype[positive_var] * d2)
        )

    return price


def black_delta(K, T, F, vol, opttype=1):
    """
    Calculate the Black delta of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.
    opttype : int, optional
        Option type: 1 for call options, -1 for put options. Default is 1.

    Returns
    -------
    float
        The Black delta of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return opttype * norm.cdf(opttype * d1)


def black_gamma(K, T, F, vol):
    """
    Calculate the Black gamma of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The Black gamma of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return norm.pdf(d1) / (F * s)


def black_speed(K, T, F, vol):
    """
    Calculate the Black speed of an option.

    Parameters
    ----------
    K : float
        Strike price of the option.
    T : float
        Time to maturity of the option.
    F : float
        Forward price of the underlying asset.
    vol : float
        Volatility of the underlying asset.

    Returns
    -------
    float
        The Black speed of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return -(d1 / s + 1.0) * norm.pdf(d1) / (F**2 * s)


def black_impvol_brent(K, T, F, value, opttype: int | np.ndarray = 1, xtol=1e-13):
    """
    Black implied volatility by Brent's method, one strike at a time.

    Uses the same bracket [1e-8, 50] and volatility tolerance as the reference
    smiles, so approximations and references are inverted identically.

    Parameters
    ----------
    K : ndarray or float
        Strike(s).
    T : float
        Time to maturity.
    F : float
        Forward price.
    value : ndarray or float
        Option price(s), with the same shape as `K`.
    opttype : int or ndarray, optional
        1 for calls, -1 for puts. Default is 1.
    xtol : float, optional
        Absolute volatility tolerance. Default is 1e-13.

    Returns
    -------
    ndarray
        Implied volatilities; NaN where the price violates the Black bounds.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    value = np.atleast_1d(np.asarray(value, dtype=float))
    if K.shape != value.shape:
        raise ValueError("K and value must have the same shape.")
    try:
        opttype = np.broadcast_to(np.asarray(opttype, dtype=float), K.shape)
    except ValueError as exc:
        raise ValueError("opttype must be scalar or have the same shape as K.") from exc
    if not np.all(np.abs(opttype) == 1):
        raise ValueError("opttype must be either 1 or -1.")

    F = float(F)
    T = float(T)
    impvol = np.full(K.shape, np.nan)
    if T <= 0 or F <= 0:
        return impvol
    sqrt_t = np.sqrt(T)

    for index_, (strike, price, sign) in enumerate(
        zip(K.flat, value.flat, opttype.flat)
    ):
        intrinsic = max(sign * (F - strike), 0.0)
        upper = F if sign == 1 else strike
        if not (strike > 0 and np.isfinite(price) and intrinsic < price < upper):
            continue

        def discrepancy(vol, strike=strike, price=price, sign=sign):
            total = vol * sqrt_t
            d1 = np.log(F / strike) / total + total / 2
            return (
                sign * (F * ndtr(sign * d1) - strike * ndtr(sign * (d1 - total)))
                - price
            )

        impvol.flat[index_] = brentq(discrepancy, 1e-8, 50.0, xtol=xtol)
    return impvol


# Initial forward variance curve for the Heston model with different shapes
# Statistical approximations for sums of lognormals
def sum_lognorm_single_lognorm_approx(lbd, mu_1, mu_2, sig_1, sig_2):
    """
    Approximate
    X = lbd * exp(mu_1 + sig_1 * Z) + (1 - lbd) * exp(mu_2 + sig_2 * Z), Z = N(0, 1)
    by a lognormal Y = exp(mu_y + sig_y * Z) by matching the first two moments of X and
    Y.
    """

    # First moment of X
    m1_x = lbd * np.exp(mu_1 + 0.5 * sig_1**2) + (1 - lbd) * np.exp(
        mu_2 + 0.5 * sig_2**2
    )
    # Second moment of X; both components are driven by the same Z, so the
    # cross moment is E[exp((mu_1 + mu_2) + (sig_1 + sig_2) Z)].
    m2_x = (
        lbd**2 * np.exp(2 * mu_1 + 2 * sig_1**2)
        + (1 - lbd) ** 2 * np.exp(2 * mu_2 + 2 * sig_2**2)
        + 2 * lbd * (1 - lbd) * np.exp(mu_1 + mu_2 + 0.5 * (sig_1 + sig_2) ** 2)
    )

    # Parameters of the approximating lognormal
    sig_y = np.log(m2_x / m1_x**2) ** 0.5
    mu_y = np.log(m1_x) - 0.5 * sig_y**2

    # Moments of Y for verification
    m1_y = np.exp(mu_y + 0.5 * sig_y**2)
    m2_y = np.exp(2 * mu_y + 2 * sig_y**2)

    assert np.isclose(m1_x, m1_y), "First moments do not match!"
    assert np.isclose(m2_x, m2_y), "Second moments do not match!"

    return {
        "mu_y": mu_y,
        "sig_y": sig_y,
    }


# Non-uniform grids and relative error calculations
# Plotting helpers
