"""Implied volatility expansions for mixed-kernel models.

Two expansions of the paper are implemented. Both take a dict ``params`` of
proxy coefficients (``lbd``, ``meanp_j``, ``sigp_j``, ``gamma_i``), as built by
``scripts/common.py:mixed_params``:

* the explicit Hermite expansion of Section 3.2 (``hermite_weights``,
  ``hermite_order`` and ``hermite_expansion``), around one component j;
* the mixed-proxy smile Sigma0 + Sigma1 of Section 3.3
  (``mixed_proxy_prices`` and ``proxy_smile_expansion``), whose leading term
  requires a numerical Black inversion and whose correction is explicit.

All strikes are absolute. Kernel amplitudes must already use the model's
convention.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, root_scalar
from scipy.special import eval_hermitenorm, factorial, ndtr


def _inverse_mixture_lognormal(y, lbd, mu_1, mu_2, sig_1, sig_2):
    """
    Solve for x in the mixture of lognormals equation.

    Finds x such that:
        lbd * exp(mu_1 + sig_1 * x) + (1 - lbd) * exp(mu_2 + sig_2 * x) = y

    Parameters
    ----------
    y : float
        Target value.
    lbd : float
        Mixing weight in [0, 1].
    mu_1, mu_2 : float
        Location parameters of the two lognormal components.
    sig_1, sig_2 : float
        Scale parameters of the two lognormal components.

    Returns
    -------
    float
        Solution x to the mixture equation.
    """
    return root_scalar(
        lambda x: (
            lbd * np.exp(mu_1 + sig_1 * x) + (1 - lbd) * np.exp(mu_2 + sig_2 * x) - y
        ),
        bracket=[-100, 100],
    ).root


@lru_cache(maxsize=16)
def _rule(nodes):
    return np.polynomial.legendre.leggauss(nodes)


def mixed_proxy_prices(params, strikes, nodes=128):
    """Return F0, F1, C0, C1 using Gaussian integration by parts.

    Gauss--Legendre quadrature resolves the smooth integrands on either side
    of the exercise boundary. Gaussian tails are truncated at least twelve
    standard deviations beyond the largest exponential tilt. Numerical
    quadrature/truncation errors are not certified weak-error bounds.
    """
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    if strikes.ndim != 1 or np.any(~np.isfinite(strikes)) or np.any(strikes <= 0):
        raise ValueError("strikes must be a one-dimensional array of positive values")
    if not isinstance(nodes, (int, np.integer)) or nodes < 2:
        raise ValueError("nodes must be an integer >= 2")
    weight = params["lbd"]
    if not 0 < weight < 1:
        raise ValueError("lbd must be strictly between zero and one")
    mu = np.array([params[f"meanp_{j}"] for j in (1, 2)])
    sig = np.array([params[f"sigp_{j}"] for j in (1, 2)])
    gam = np.array([params.get(f"gamma_{i}", (0.0, 0.0)) for i in (1, 2, 3)])
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sig)) or np.any(sig <= 0):
        raise ValueError("proxy means must be finite and standard deviations positive")
    if gam.shape != (3, 2) or np.any(~np.isfinite(gam)):
        raise ValueError("gamma coefficients must be three finite pairs")
    log_weights = np.log([weight, 1 - weight])
    x, w = _rule(nodes)
    bound = max(14.0, float(sig.max()) / 2 + 12.0)

    def integrand(z):
        logs = log_weights + mu + sig * z[..., None]
        log_h = np.logaddexp(logs[..., 0], logs[..., 1])
        root = np.exp(log_h / 2)
        pi = np.exp(logs - log_h[..., None])
        zz = z[..., None]
        q = (pi * (gam[0] + gam[1] * zz / sig + gam[2] * (zz**2 - 1) / sig**2)).sum(
            axis=-1
        )
        return root, q

    z = bound * x
    root, q = integrand(z)
    density = bound * w * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
    f0 = float(density @ root)
    f1 = float(density @ (root * q / 2))
    a = np.array([_inverse_mixture_lognormal(k**2, weight, *mu, *sig) for k in strikes])
    if np.any(np.abs(a) >= bound):
        raise ValueError("exercise boundary outside resolved Gaussian interval")
    half = (bound - a[:, None]) / 2
    z = a[:, None] + half * (x + 1)
    root, q = integrand(z)
    density = half * w * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
    c0 = (density * (root - strikes[:, None])).sum(axis=1)
    c1 = (density * root * q / 2).sum(axis=1)
    return f0, f1, c0, c1


def proxy_smile_expansion(f0, f1, c0, c1, strikes, maturity):
    """Return Sigma0 and Sigma1 at fixed absolute strikes.

    Reject unresolved/inadmissible proxy prices instead of silently clipping
    them. The small-window theorem is conditional on the mixed call/futures
    weak-error rate and a positive vega bound on the strike domain.
    """
    if not np.isfinite(maturity) or maturity <= 0 or not np.isfinite(f0) or f0 <= 0:
        raise ValueError("maturity and proxy forward must be positive and finite")
    strikes, c0, c1 = np.broadcast_arrays(
        np.atleast_1d(np.asarray(strikes, dtype=float)), c0, c1
    )
    if (
        np.any(~np.isfinite(strikes))
        or np.any(strikes <= 0)
        or np.any(~np.isfinite(c0))
        or np.any(~np.isfinite(c1))
        or not np.isfinite(f1)
    ):
        raise ValueError(
            "strikes and price corrections must be finite; strikes positive"
        )
    if np.any(c0 <= np.maximum(f0 - strikes, 0)) or np.any(c0 >= f0):
        raise ValueError("proxy call price must lie strictly within Black bounds")
    sigma0 = np.empty(strikes.shape)
    sqrt_t = np.sqrt(maturity)
    for index in np.ndindex(strikes.shape):
        strike, price = strikes[index], c0[index]
        log_moneyness = np.log(f0 / strike)

        def discrepancy(vol):
            if vol == 0:
                return max(f0 - strike, 0.0) - price
            total = vol * sqrt_t
            d1 = log_moneyness / total + total / 2
            return f0 * ndtr(d1) - strike * ndtr(d1 - total) - price

        upper = 1.0 / sqrt_t
        while discrepancy(upper) < 0:
            upper *= 2
        sigma0[index] = brentq(discrepancy, 0.0, upper, xtol=2e-14, rtol=1e-14)
    total = sigma0 * np.sqrt(maturity)
    d0 = np.log(f0 / strikes) / total + total / 2
    vega = f0 * np.sqrt(maturity) * np.exp(-(d0**2) / 2) / np.sqrt(2 * np.pi)
    sigma1 = (c1 - ndtr(d0) * f1) / vega
    if np.any(~np.isfinite(sigma1)) or np.any(sigma0 + sigma1 <= 0):
        raise ValueError("proxy-smile expansion is outside its positive finite domain")
    return sigma0, sigma1


def _phi(x):
    return np.exp(-(np.asarray(x) ** 2) / 2) / np.sqrt(2 * np.pi)


def _components(params):
    weight = params["lbd"]
    lam = np.array([weight, 1 - weight])
    mu = np.array([params["meanp_1"], params["meanp_2"]])
    sig = np.array([params["sigp_1"], params["sigp_2"]])
    gam = np.array([params[f"gamma_{i}"] for i in (1, 2, 3)])  # shape (3, 2)
    return lam, mu, sig, gam


def _hermite_factors(params, j):
    """Return g^j_0, g^j_1 of Proposition 2 and ||g^j_0||^2 = 1 + C_j e^{dsig^2/2}."""
    if j not in (1, 2):
        raise ValueError("j must be either 1 or 2.")
    lam, mu, sig, gam = _components(params)
    a, b = j - 1, 2 - j  # indices of j and of the other component
    dsig = sig[b] - sig[a]
    c_j = lam[b] / lam[a] * np.exp(mu[b] - mu[a] + sig[a] * dsig / 2)

    def q(z):  # first-order correction factor of Lemma 2
        logs = np.log(lam)[:, None] + mu[:, None] + sig[:, None] * z[None, :]
        pi = np.exp(logs - np.logaddexp(logs[0], logs[1]))
        q_l = (
            gam[0][:, None]
            + gam[1][:, None] * z[None, :] / sig[:, None]
            + gam[2][:, None] * (z[None, :] ** 2 - 1) / sig[:, None] ** 2
        )
        return (pi * q_l).sum(axis=0)

    def g0(y):
        return np.sqrt(1 + c_j * np.exp(dsig * y))

    def g1(y):
        return 0.5 * q(y + sig[a] / 2) * g0(y)

    return g0, g1, 1 + c_j * np.exp(dsig**2 / 2)


def hermite_weights(params, j, n_max, nodes=4000, bound=40.0):
    """Return the Hermite weights omega^{(i)}_{n,j}, i = 0, 1, n <= n_max, and
    ||g^j_0||^2.

    The Gaussian integrals are computed by Gauss--Legendre quadrature with
    ``nodes`` nodes on [-bound, bound].
    """
    g0, g1, norm2 = _hermite_factors(params, j)
    x, w = _rule(nodes)
    y, w = bound * x, bound * w * _phi(bound * x)
    he = np.empty((n_max + 1, y.size))
    he[0] = 1.0
    if n_max:
        he[1] = y
    for n in range(1, n_max):
        he[n + 1] = y * he[n] - n * he[n - 1]
    nfact = factorial(np.arange(n_max + 1))
    weights = np.array([(he * g(y)) @ w / nfact for g in (g0, g1)])
    return weights, norm2


def hermite_order(params, j, tol=1e-6, n_max=60, **quadrature):
    """Smallest N with MSE_j(N) = ||g^j_0||^2 - sum_{n<=N} n! (omega^{(0)}_{n,j})^2
    <= tol, computed exactly by Parseval's identity."""
    weights, norm2 = hermite_weights(params, j, n_max, **quadrature)
    mse = norm2 - np.cumsum(factorial(np.arange(n_max + 1)) * weights[0] ** 2)
    hits = np.flatnonzero(mse <= tol)
    if hits.size == 0:
        raise ValueError("Tolerance not reached; increase n_max.")
    return int(hits[0])


def hermite_expansion(params, j, weights, strikes, maturity, theta=None):
    """Explicit Hermite expansion around component j, at absolute strikes.

    ``weights`` are the Hermite weights of ``hermite_weights`` truncated at
    the order N. Returns a dict with the Hermite futures price F_P, the
    Nth-order call prices C^j_{P,N} of Proposition 3, the implied volatility
    expansion (60) with x_theta = ln F_P + theta * delta_j (``theta=None``
    uses theta* = 1{delta_j > 0} of Remark 8), the extended expansion (61), the
    shift and volatility residuals of Proposition 4 (so that price = Black price
    at (x_theta, k_theta, iv) + shift_residual + vol_residual) and the volatility
    correction iv - s_j / (2 sqrt(T)).
    """
    if theta is not None and not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in the interval [0, 1].")
    lam, mu, sig, _ = _components(params)
    s_j = sig[j - 1]
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    a = np.array([_inverse_mixture_lognormal(k**2, lam[0], *mu, *sig) for k in strikes])
    a_j = a - s_j / 2
    scale = np.sqrt(lam[j - 1]) * np.exp(mu[j - 1] / 2 + s_j**2 / 8)
    order = weights.shape[1] - 1
    he = eval_hermitenorm(np.arange(order)[:, None], a_j[None, :])
    tail = weights[:, 1:].sum(axis=0) @ he  # sum_n sum_i omega^{(i)}_n He_{n-1}
    forward = scale * weights[:, 0].sum()  # eq:vix_fut_hermite
    price = forward * ndtr(-a_j) + scale * tail * _phi(a_j) - strikes * ndtr(-a)
    delta_j = np.log(strikes / forward) - a * s_j / 2 + s_j**2 / 8
    shift = np.maximum(delta_j, 0.0) if theta is None else theta * delta_j
    root_t = np.sqrt(maturity)
    iv = s_j / (2 * root_t) + scale * tail / (forward * np.exp(shift) * root_t)
    d1 = np.log(forward / strikes) / (s_j / 2) + s_j / 4
    vega = forward * root_t * _phi(d1)
    iv_extended = s_j / (2 * root_t) + scale * _phi(a_j) * tail / vega
    # Residuals of Proposition 4 at the auxiliary inputs (x_theta, k_theta).
    log_strike = np.log(strikes)
    theta_delta = shift
    x_theta = np.log(forward) + theta_delta
    k_theta = log_strike - (delta_j - theta_delta)
    shift_residual = (forward - np.exp(x_theta)) * ndtr(-a_j) + (
        np.exp(k_theta) - strikes
    ) * ndtr(-a)
    total = iv * root_t
    d1_hat = (x_theta - k_theta) / total + total / 2
    black_hat = np.exp(x_theta) * ndtr(d1_hat) - np.exp(k_theta) * ndtr(d1_hat - total)
    vol_residual = price - black_hat - shift_residual
    return dict(
        forward=forward,
        price=price,
        iv=iv,
        iv_extended=iv_extended,
        shift_residual=shift_residual,
        vol_residual=vol_residual,
        vol_correction=iv - s_j / (2 * root_t),
    )
