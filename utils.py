import warnings
from functools import cache
from operator import index

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import root_scalar
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
def _cached_hermgauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    knots, weights = np.polynomial.hermite.hermgauss(n)
    return knots * np.sqrt(2.0), weights / np.sqrt(np.pi)


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


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Hermite quadrature points and weights.

    Integration is with respect to the Gaussian density. It corresponds to the
    probabilist's Hermite polynomials.

    Parameters
    ----------
    n: int
        Number of quadrature points.

    Returns
    -------
    knots: array-like
        Gauss-Hermite knots.
    weight: array-like
        Gauss-Hermite weights.
    """
    n = _validate_quadrature_order(n)
    knots, weights = _cached_hermgauss(n)
    return knots.copy(), weights.copy()


# Linear algebra helpers for simulation routines
def cholesky_from_svd(a: np.ndarray) -> np.ndarray:
    """
    Compute a square-root factor of a positive semi-definite matrix.

    This function is a fallback for numerically positive semi-definite matrices.

    Parameters
    ----------
    a : np.ndarray
        The input matrix.

    Returns
    -------
    np.ndarray
        Matrix factor `b` such that `b @ b.T` approximates the input matrix.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square matrix.")

    a = 0.5 * (a + a.T)
    eigvals, eigvecs = np.linalg.eigh(a)
    tol = np.finfo(float).eps * a.shape[0] * max(1.0, np.max(np.abs(eigvals)))

    if np.any(eigvals < -tol):
        raise np.linalg.LinAlgError("Matrix is not positive semi-definite.")

    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs * np.sqrt(eigvals)


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


def black_vega(K, T, F, vol):
    """
    Calculate the Black vega of an option.

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
        The Black vega of the option.
    """
    s = vol * T**0.5
    d1 = np.log(F / K) / s + 0.5 * s
    return F * norm.pdf(d1) * np.sqrt(T)


def black_impvol(
    K, T, F, value, opttype: int | np.ndarray = 1, TOL=1e-10, MAX_ITER=1000
):
    """
    Calculate the Black implied volatility using a bisection method.

    Parameters
    ----------
    K : ndarray or float
        Strike price(s) of the option(s).
    T : float
        Time to maturity of the option(s).
    F : float
        Forward price of the underlying asset.
    value : ndarray or float
        Observed market price(s) of the option(s).
    opttype : int or ndarray, optional
        Option type: 1 for call options, -1 for put options. Default is 1.
    TOL : float, optional
        Tolerance for convergence of the implied volatility. Default is 1e-10.
    MAX_ITER : int, optional
        Maximum number of iterations for the bisection method. Default is 1000.

    Returns
    -------
    ndarray or float
        Implied volatility(ies) corresponding to the input option prices. If the
        input arrays are multidimensional, the output will have the same shape.
        Returns NaN if the implied volatility does not converge or if invalid
        inputs are provided.

    Raises
    ------
    ValueError
        If `K` and `value` do not have the same shape.
        If `opttype` is not 1 or -1.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    value = np.atleast_1d(np.asarray(value, dtype=float))
    try:
        opttype = np.broadcast_to(np.asarray(opttype, dtype=float), K.shape)
    except ValueError as exc:
        raise ValueError("opttype must be scalar or have the same shape as K.") from exc

    if K.shape != value.shape:
        raise ValueError("K and value must have the same shape.")

    if not np.all(np.abs(opttype) == 1):
        raise ValueError("opttype must be either 1 or -1.")

    F = float(F)
    T = float(T)

    if T <= 0 or F <= 0:
        return np.full_like(K, np.nan, dtype=float)

    low = IMPVOL_MIN * np.ones_like(K, dtype=float)
    high = IMPVOL_MAX * np.ones_like(K, dtype=float)
    mid = 0.5 * (low + high)

    low_price = black_price(K, T, F, low, opttype)
    high_price = black_price(K, T, F, high, opttype)
    price_tol = TOL * np.maximum(np.abs(value), 1.0)
    valid = (
        (K > 0.0)
        & np.isfinite(value)
        & (value >= low_price - price_tol)
        & (value <= high_price + price_tol)
    )
    impvol = np.full_like(K, np.nan, dtype=float)
    active = valid.copy()

    for _ in range(MAX_ITER):
        if not np.any(active):
            return impvol

        active_idx = np.flatnonzero(active)
        price = black_price(K[active_idx], T, F, mid[active_idx], opttype[active_idx])
        diff = price - value[active_idx]
        converged = (np.abs(diff) <= price_tol[active_idx]) | (
            high[active_idx] - low[active_idx] <= TOL
        )

        converged_idx = active_idx[converged]
        impvol[converged_idx] = mid[converged_idx]
        active[converged_idx] = False

        remaining_idx = active_idx[~converged]
        high_idx = remaining_idx[diff[~converged] > 0.0]
        low_idx = remaining_idx[diff[~converged] <= 0.0]
        high[high_idx] = mid[high_idx]
        low[low_idx] = mid[low_idx]
        mid[remaining_idx] = 0.5 * (low[remaining_idx] + high[remaining_idx])

    warnings.warn(
        "Implied volatility did not converge for all log(K/F) values.",
        RuntimeWarning,
        stacklevel=2,
    )

    return impvol


def black_otm_impvol_mc(
    S: np.ndarray, k: float | np.ndarray, T: float, mc_error: bool = False
) -> dict | np.ndarray:
    """
    Calculate Black implied volatility using Monte Carlo simulated stock prices and
    out-of-the-money (OTM) prices.

    Parameters
    ----------
    S : ndarray
        Array of Monte Carlo simulated stock prices.
    k : float or ndarray
        Log-Forward Moneyness `k=log(K/F)` for which the implied volatility is
        calculated.
    T : float
        Time to maturity of the option.
    mc_error : bool, optional
        If True, computes the 95% confidence interval for the implied volatility.

    Returns
    -------
    dict or ndarray
        If `mc_error` is False, returns an ndarray of OTM implied volatilities.
        If `mc_error` is True, returns a dictionary with the following keys:
        - 'otm_impvol': ndarray of OTM implied volatilities.
        - 'otm_impvol_high': ndarray of upper bounds of the 95% confidence interval.
        - 'otm_impvol_low': ndarray of lower bounds of the 95% confidence interval.
        - 'error_95': ndarray of the 95% confidence interval errors for the option
                      prices.
        - 'otm_price': ndarray of the calculated OTM option prices.
    """
    S = np.asarray(S, dtype=float).ravel()
    k = np.atleast_1d(np.asarray(k, dtype=float))

    if S.size == 0:
        raise ValueError("S must contain at least one simulated price.")

    if T <= 0.0:
        raise ValueError("T must be positive.")

    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values.")

    if not np.all(S > 0.0):
        raise ValueError("S must contain only positive simulated prices.")

    F = np.mean(S)
    K = F * np.exp(k)
    # opttype: 1 for call, -1 for put, depending on moneyness
    opttype = 2 * (K >= F) - 1  # 1 if K >= F (call), -1 if K < F (put)
    payoff = np.maximum(opttype[None, :] * (S[:, None] - K[None, :]), 0.0)
    otm_price = np.mean(payoff, axis=0)
    otm_impvol = black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

    if mc_error:
        error_95 = 1.96 * np.std(payoff, axis=0) / S.shape[0] ** 0.5
        otm_impvol_high = black_impvol(
            K=K, T=T, F=F, value=otm_price + error_95, opttype=opttype
        )
        otm_impvol_low = black_impvol(
            K=K, T=T, F=F, value=otm_price - error_95, opttype=opttype
        )
        return {
            "otm_impvol": otm_impvol,
            "otm_impvol_high": otm_impvol_high,
            "otm_impvol_low": otm_impvol_low,
            "error_95": error_95,
            "otm_price": otm_price,
        }

    return otm_impvol


def implied_vol_from_paths(
    k: float | np.ndarray,
    T: float,
    int_v_dt: np.ndarray,
    int_sqrt_v_dw: np.ndarray,
    s0: float,
    conditioning: bool = False,
    return_skew: bool = False,
    rho_cond: float | None = None,
):
    """
    Estimate the implied volatility (and optionally its skew) from simulated paths.

    Parameters
    ----------
    k : float or np.ndarray, optional
        Log-strike k = log(S/S_0).
    T : float
        Maturity.
    int_v_dt : np.ndarray
        Array of time-integrated variances (shape: n_samples,).
    int_sqrt_v_dw : np.ndarray
        Array of stochastic integrals `int sqrt(v) dW` (shape: n_samples,).
    s0 : float
        Initial stock price.
    conditioning : bool, optional
        If True, use conditioning technique. See Bergomi (2016) - Stochastic Volatility
        Modeling - Chapter 8 - Appendix A.
    return_skew : bool, optional
        If True, also return the estimated implied volatility skew.
        Default is False. Monte Carlo estimation is used, see Bourgey et al. (2024)
        - Local volatility under rough volatility - Section 4.
    rho_cond : float, optional
        Correlation parameter for conditioning. Required if `conditioning` is True.

    Returns
    -------
    float or tuple
        Estimated implied volatility. If `return_skew` is True, returns a
        tuple (implied_vol, implied_vol_skew).
    """
    if T <= 0.0:
        raise ValueError("T must be positive.")

    if s0 <= 0.0:
        raise ValueError("s0 must be positive.")

    if int_v_dt.shape != int_sqrt_v_dw.shape:
        raise ValueError("int_v_dt and int_sqrt_v_dw must have the same shape.")

    k = np.atleast_1d(np.asarray(k, dtype=float))
    int_v_dt = np.asarray(int_v_dt).flatten()
    int_sqrt_v_dw = np.asarray(int_sqrt_v_dw).flatten()

    if conditioning:
        if rho_cond is None:
            raise ValueError("rho_cond must be provided when conditioning is True.")

        if not (-1.0 <= rho_cond <= 1.0):
            raise ValueError("rho_cond must be between -1 and 1.")

        s0_cond = s0 * np.exp(-0.5 * rho_cond**2 * int_v_dt + rho_cond * int_sqrt_v_dw)
        vol_cond = np.sqrt((1.0 - rho_cond**2) * int_v_dt / T)
        F = s0_cond.mean()
        K = F * np.exp(k)
        opttype = 2.0 * (K >= F) - 1.0
        price_cond = black_price(
            K=K[None, :],
            T=T,
            F=s0_cond[:, None],
            vol=vol_cond[:, None],
            opttype=opttype[None, :],
        ).mean(axis=0)
        impvol = black_impvol(K=K, T=T, F=F, value=price_cond, opttype=opttype)

        if return_skew:
            # Control variate to compute digital prices.
            w_cond = vol_cond * T**0.5
            d2_cond = (
                np.log(s0_cond[:, None] / K[None, :]) / w_cond[:, None]
                - 0.5 * w_cond[:, None]
            )
            digit = norm.cdf(d2_cond).mean(axis=0)
    else:
        S = s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
        F = S.mean()
        K = F * np.exp(k)
        opttype = 2.0 * (K >= F) - 1.0
        payoff = np.maximum(opttype[None, :] * (S[:, None] - K[None, :]), 0.0)
        otm_price = np.mean(payoff, axis=0)
        impvol = black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

        if return_skew:
            digit = np.mean(S[:, None] >= K[None, :], axis=0)

    if return_skew:
        w = impvol * T**0.5
        d2 = -k / w - 0.5 * w
        skew = (norm.cdf(d2) - digit) / (norm.pdf(d2) * T**0.5)
        return impvol, skew

    return impvol


# Initial forward variance curve for the Heston model with different shapes
def xi0_heston(
    t,
    shape="flat",
    lbd_xi0=7.0,
    v0_flat=0.025,
    v0_contango=0.005,
    v0_backwardation=0.045,
):
    """
    Compute the initial forward variance curve for the Heston model with
    different shapes ("flat", "contango", "backwardation").

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the initial volatility.
    shape : str, optional
        Term structure shape: "flat", "contango", or "backwardation". Default is "flat".
    lbd_xi0 : float, optional
        Decay rate for the term structure. Default is 7.0.
    v0_flat : float, optional
        Flat volatility level. Default is 0.025.
    v0_contango : float, optional
        Initial volatility for contango shape. Default is 0.005.
    v0_backwardation : float, optional
        Initial volatility for backwardation shape. Default is 0.045.

    Returns
    -------
    ndarray
        Initial volatility value(s) evaluated at time(s) t.

    Raises
    ------
    ValueError
        If shape is not "flat", "contango", or "backwardation".
    """
    if shape == "flat":
        return v0_flat + 0.0 * t
    elif shape == "contango":
        return v0_flat + (v0_contango - v0_flat) * np.exp(-lbd_xi0 * t)
    elif shape == "backwardation":
        return v0_flat + (v0_backwardation - v0_flat) * np.exp(-lbd_xi0 * t)
    else:
        raise ValueError("Unknown shape")


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
    # Second moment of X
    m2_x = (
        lbd**2 * np.exp(2 * mu_1 + 2 * sig_1**2)
        + (1 - lbd) ** 2 * np.exp(2 * mu_2 + 2 * sig_2**2)
        + 2 * lbd * (1 - lbd) * np.exp(mu_1 + mu_2 + 0.5 * (sig_1**2 + sig_2**2))
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


def sum_lognorm_shifted_lognorm_approx(lbd, mu_1, mu_2, sig_1, sig_2):
    """
    X = lbd * exp(mu_1 + sig_1 * Z) + (1 - lbd) * exp(mu_2 + sig_2 * Z)
    Approximate X by Y = c_y + exp(mu_y + sig_y * Z)
    Returns c_y, mu_y, sig_y.
    """

    # moments of X
    E1 = np.exp(mu_1 + 0.5 * sig_1**2)
    E2 = np.exp(mu_2 + 0.5 * sig_2**2)

    E1_2 = np.exp(2 * mu_1 + 2 * sig_1**2)
    E2_2 = np.exp(2 * mu_2 + 2 * sig_2**2)
    E12 = np.exp(mu_1 + mu_2 + 0.5 * (sig_1 + sig_2) ** 2)

    E1_3 = np.exp(3 * mu_1 + 4.5 * sig_1**2)
    E2_3 = np.exp(3 * mu_2 + 4.5 * sig_2**2)
    E1_2E2 = np.exp(2 * mu_1 + mu_2 + 0.5 * (2 * sig_1 + sig_2) ** 2)
    E1E2_2 = np.exp(mu_1 + 2 * mu_2 + 0.5 * (sig_1 + 2 * sig_2) ** 2)

    m1_x = lbd * E1 + (1 - lbd) * E2

    m2_x = lbd**2 * E1_2 + (1 - lbd) ** 2 * E2_2 + 2 * lbd * (1 - lbd) * E12

    m3_x = (
        lbd**3 * E1_3
        + (1 - lbd) ** 3 * E2_3
        + 3 * lbd**2 * (1 - lbd) * E1_2E2
        + 3 * lbd * (1 - lbd) ** 2 * E1E2_2
    )

    # infer shifted lognormal parameters
    var_x = m2_x - m1_x**2
    kappa3_x = m3_x - 3 * m1_x * m2_x + 2 * m1_x**3
    skew_x = kappa3_x / var_x**1.5

    # solve u^3 + 3u = skew_x
    disc = np.sqrt(skew_x**2 / 4 + 1)
    u = np.cbrt(skew_x / 2 + disc) + np.cbrt(skew_x / 2 - disc)

    t = u**2 + 1
    sig_y = np.sqrt(np.log(t))
    mu_y = 0.5 * np.log(var_x / (t * (t - 1)))

    c_y = m1_x - np.exp(mu_y) * np.sqrt(t)

    # moments of Y
    EW = np.exp(mu_y) * np.sqrt(t)
    EW2 = np.exp(2 * mu_y) * t**2
    EW3 = np.exp(3 * mu_y) * t ** (9 / 2)

    m1_y = c_y + EW
    m2_y = c_y**2 + 2 * c_y * EW + EW2
    m3_y = c_y**3 + 3 * c_y**2 * EW + 3 * c_y * EW2 + EW3

    assert np.isclose(m1_x, m1_y), "First moments do not match!"
    assert np.isclose(m2_x, m2_y), "Second moments do not match!"
    assert np.isclose(m3_x, m3_y), "Third moments do not match!"

    return {
        "c_y": c_y,
        "mu_y": mu_y,
        "sig_y": sig_y,
    }


def sqrt_sum_lognorm_shifted_lognorm_approx(lbd, mu_1, mu_2, sig_1, sig_2, n_quad=30):
    """
    V^2 = lbd * exp(mu_1 + sig_1 * Z) + (1 - lbd) * exp(mu_2 + sig_2 * Z)
    with Z = N(0,1).
    Approximate V by Y = c_y + exp(mu_y + sig_y * Z)
    Returns c_y, mu_y, sig_y.
    """

    x_herm, w_herm = gauss_hermite(n_quad)

    def moment_v(p):
        """Compute the p-th moment of V."""
        integrand = (
            lbd * np.exp(mu_1 + sig_1 * x_herm)
            + (1 - lbd) * np.exp(mu_2 + sig_2 * x_herm)
        ) ** (p / 2)
        return np.sum(w_herm * integrand)

    # compute first three moments of V
    m1_v, m2_v, m3_v = (moment_v(p) for p in [1, 2, 3])

    def func_a(d):
        return m1_v - d

    def func_b(d):
        return m2_v - 2 * d * m1_v + d**2

    def func_c(d):
        return m3_v - 3 * d * m2_v + 3 * d**2 * m1_v - d**3

    c_y = root_scalar(
        lambda x: func_c(x) * func_a(x) ** 3 - func_b(x) ** 3, bracket=[-10, 10]
    ).root
    a_y = func_a(c_y)
    b_y = func_b(c_y)
    sig_y = np.log(b_y / a_y**2) ** 0.5
    mu_y = np.log(a_y) - 0.5 * sig_y**2

    def moment_y(p):
        """Compute the p-th moment of Y."""
        return np.sum(w_herm * (np.exp(mu_y + sig_y * x_herm) + c_y) ** p)

    m1_y, m2_y, m3_y = (moment_y(p) for p in [1, 2, 3])
    assert np.isclose(m1_v, m1_y), "First moments do not match!"
    assert np.isclose(m2_v, m2_y), "Second moments do not match!"
    assert np.isclose(m3_v, m3_y), "Third moments do not match!"

    return {
        "c_y": c_y,
        "mu_y": mu_y,
        "sig_y": sig_y,
    }


# Non-uniform grids and relative error calculations
def non_uniform_grid(a: float, b: float, n: int, power: float = 3.0) -> np.ndarray:
    """
    Create a non-uniform grid on the interval [a, b] with clustering towards 'a'.

    Parameters
    ----------
    a : float
        Start of the interval.
    b : float
        End of the interval.
    n : int
        Number of grid points.
    power : float, optional
        Power for clustering (default is 3.0). Higher values cluster more towards 'a'.

    Returns
    -------
    np.ndarray
        1-D array of grid points on [a, b].
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    return a + (b - a) * np.linspace(0.0, 1.0, n) ** power


def relative_error(true_value, approx_value):
    """
    Calculate the relative error between a true value and an approximate value.

    Parameters
    ----------
    true_value : float
        The true value.
    approx_value : float
        The approximate value.

    Returns
    -------
    float
        The relative error, defined as (approx_value - true_value) / true_value.
        Returns NaN if true_value is zero to avoid division by zero.
    """
    return np.where(true_value == 0, np.nan, (approx_value - true_value) / true_value)


# Plotting helpers
def set_matplotlib_style():
    """Set a consistent style for matplotlib plots."""
    plt.rcParams.update({"figure.figsize": (9, 7)})
    sns.set_style(
        style="ticks",
        rc={"axes.grid": True, "axes.spines.top": False, "axes.spines.right": False},
    )
    sns.set_context(
        context="poster",
        rc={
            "grid.linewidth": 1.0,
            "legend.fontsize": "x-small",
            "legend.title_fontsize": "xx-small",
        },
    )
