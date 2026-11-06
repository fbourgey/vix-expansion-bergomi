import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import factorial as scipy_factorial, ndtr

from mixed_iv import mixed_proxy_prices, proxy_smile_expansion
from rbergomi import RoughBergomiModel, get_params_rbergomi
from scripts.common import mixed_params
from utils import black_price


def test_single_kernel_limit_has_flat_leading_smile_and_forward_cancellation():
    mu, sig, gamma, maturity = -3.0, 0.8, 0.015, 0.25
    params = dict(
        lbd=0.3, meanp_1=mu, meanp_2=mu, sigp_1=sig, sigp_2=sig, gamma_1=(gamma, gamma)
    )
    forward = np.exp(mu / 2 + sig**2 / 8)
    strikes = forward * np.exp(np.array([-0.1, 0.0, 0.4]))
    f0, f1, c0, c1 = mixed_proxy_prices(params, strikes)
    vol = sig / (2 * np.sqrt(maturity))
    d1 = np.log(forward / strikes) / (vol * np.sqrt(maturity)) + sig / 4
    assert f0 == pytest.approx(forward, abs=1e-14)
    assert c0 == pytest.approx(black_price(strikes, maturity, forward, vol), abs=1e-14)
    assert f1 == pytest.approx(gamma * forward / 2)
    assert c1 == pytest.approx(gamma * forward * ndtr(d1) / 2)
    sigma0, sigma1 = proxy_smile_expansion(f0, f1, c0, c1, strikes, maturity)
    assert sigma0 == pytest.approx(np.full(3, vol), abs=2e-12)
    assert sigma1 == pytest.approx(np.zeros(3), abs=2e-13)


def test_gaussian_price_integrals_match_independent_adaptive_quadrature():
    params = dict(
        lbd=0.4,
        meanp_1=-4.0,
        meanp_2=-3.0,
        sigp_1=2.0,
        sigp_2=0.3,
        gamma_1=(0.01, 0.003),
        gamma_2=(0.004, -0.002),
        gamma_3=(0.001, 0.0002),
    )
    strikes = np.array([0.12, 0.2, 0.3])
    f0, f1, c0, c1 = mixed_proxy_prices(params, strikes)

    def integrands(z):
        sig = np.array([2.0, 0.3])
        parts = np.array([0.4, 0.6]) * np.exp(np.array([-4.0, -3.0]) + sig * z)
        root = np.sqrt(sum(parts))
        q = (
            np.array(params["gamma_1"])
            + np.array(params["gamma_2"]) * z / sig
            + np.array(params["gamma_3"]) * (z * z - 1) / sig**2
        )
        density = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        return root, sum(parts * q) / (2 * root), density

    assert f0 == pytest.approx(
        quad(lambda z: integrands(z)[0] * integrands(z)[2], -14, 14, epsabs=1e-13)[0],
        abs=1e-12,
    )
    assert f1 == pytest.approx(
        quad(lambda z: integrands(z)[1] * integrands(z)[2], -14, 14, epsabs=1e-13)[0],
        abs=1e-12,
    )
    for idx, strike in enumerate(strikes):
        from scipy.optimize import brentq

        a = brentq(lambda z: integrands(z)[0] - strike, -14, 14)
        assert c0[idx] == pytest.approx(
            quad(
                lambda z: (integrands(z)[0] - strike) * integrands(z)[2],
                a,
                14,
                epsabs=1e-13,
            )[0],
            abs=1e-12,
        )
        assert c1[idx] == pytest.approx(
            quad(lambda z: integrands(z)[1] * integrands(z)[2], a, 14, epsabs=1e-13)[0],
            abs=1e-12,
        )
    swapped = dict(params, lbd=0.6, meanp_1=-3.0, meanp_2=-4.0, sigp_1=0.3, sigp_2=2.0)
    for key in ("gamma_1", "gamma_2", "gamma_3"):
        swapped[key] = params[key][::-1]
    for actual, expected in zip(mixed_proxy_prices(swapped, strikes), (f0, f1, c0, c1)):
        assert actual == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("scenario, amplitudes", [(1, (1.4, 0.7)), (2, (2.0, 0.2))])
def test_rough_helpers_reproduce_both_physical_kernel_amplitudes(scenario, amplitudes):
    params, xi0, _, second = get_params_rbergomi(scenario, opt="mixed")
    assert params["H"] == 0.1
    first_model = RoughBergomiModel(params, xi0)
    second_model = first_model._clone_with_params(eta=second)
    for model, amplitude in zip((first_model, second_model), amplitudes):
        assert model.kernel(0.4, 0.1) == pytest.approx(amplitude * 0.3 ** (-0.4))


def _scenario_params(maturity=0.25):
    params, xi0, weight, second = get_params_rbergomi(2, opt="mixed")
    model = RoughBergomiModel(params, xi0)
    model.delta_vix = 30 / 365
    models = (model, model._clone_with_params(eta=second))
    return model, weight, second, mixed_params(models, maturity, weight)


@pytest.mark.parametrize("j", [1, 2])
def test_hermite_expansion_converges_to_weak_prices(j):
    from mixed_iv import hermite_expansion, hermite_order, hermite_weights

    *_, params = _scenario_params()
    f0, f1, c0, c1 = mixed_proxy_prices(params, [])
    strikes = (f0 + f1) * np.exp(np.linspace(-0.1, 0.4, 5))
    _, _, c0, c1 = mixed_proxy_prices(params, strikes)
    weights, norm2 = hermite_weights(params, j, 60)
    # Parseval: the squared norm is the sum of the squared Hermite coefficients.
    assert np.isclose(
        norm2, np.sum(scipy_factorial(np.arange(61)) * weights[0] ** 2), rtol=1e-10
    )
    point = hermite_expansion(params, j, weights, strikes, 0.25)
    assert point["forward"] == pytest.approx(f0 + f1, rel=1e-10)
    assert np.allclose(point["price"], c0 + c1, rtol=1e-8)
    assert 0 < hermite_order(params, j) < 60


@pytest.mark.parametrize("j", [1, 2])
@pytest.mark.parametrize("theta", [None, 0.0, 1.0])
def test_hermite_residuals_satisfy_proposition_4(j, theta):
    from mixed_iv import _inverse_mixture_lognormal, hermite_expansion, hermite_weights

    *_, params = _scenario_params()
    weights, _ = hermite_weights(params, j, 20)
    maturity = 0.25
    forward = hermite_expansion(params, j, weights, [], maturity)["forward"]
    strikes = forward * np.exp(np.linspace(-0.1, 0.4, 5))
    point = hermite_expansion(params, j, weights, strikes, maturity, theta=theta)

    s_j = params[f"sigp_{j}"]
    base = s_j / (2 * np.sqrt(maturity))
    assert np.allclose(point["iv"] - point["vol_correction"], base, rtol=0, atol=1e-15)

    lam = np.array([params["lbd"], 1 - params["lbd"]])
    mu = np.array([params["meanp_1"], params["meanp_2"]])
    sig = np.array([params["sigp_1"], params["sigp_2"]])
    a = np.array([_inverse_mixture_lognormal(k**2, lam[0], *mu, *sig) for k in strikes])
    delta_j = np.log(strikes / forward) - a * s_j / 2 + s_j**2 / 8
    shift = np.maximum(delta_j, 0.0) if theta is None else theta * delta_j
    x_theta = np.log(forward) + shift
    k_theta = np.log(strikes) - (delta_j - shift)
    a_j = a - s_j / 2
    # Shift residual: delta_x d_x C^BS + delta_k d_k C^BS at (x_theta, k_theta).
    expected_shift = (np.exp(-shift) - 1) * np.exp(x_theta) * ndtr(-a_j) - (
        np.exp(delta_j - shift) - 1
    ) * np.exp(k_theta) * ndtr(-a)
    assert np.allclose(point["shift_residual"], expected_shift, rtol=1e-12, atol=1e-16)

    # Volatility residual bound (eq:vol_residual_bound) with Vomma of (eq:vomma).
    def vomma(sigma):
        d1 = (x_theta - k_theta) / (sigma * np.sqrt(maturity)) + sigma * np.sqrt(
            maturity
        ) / 2
        vega = np.exp(x_theta) * np.sqrt(maturity) * np.exp(-(d1**2) / 2)
        vega /= np.sqrt(2 * np.pi)
        return vega * (
            (x_theta - k_theta) ** 2 / (sigma**3 * maturity) - sigma * maturity / 4
        )

    grid = np.linspace(0, 1, 2001)[:, None]
    sigmas = base + grid * point["vol_correction"]
    bound = 0.5 * point["vol_correction"] ** 2 * np.abs(vomma(sigmas)).max(axis=0)
    assert np.all(np.abs(point["vol_residual"]) <= bound * (1 + 1e-9) + 1e-15)
