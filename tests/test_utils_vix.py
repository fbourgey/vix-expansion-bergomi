import numpy as np
import pytest

from utils_vix import (
    _deriv_vix_payoff_mixed,
    _hermite_polynomial_weights,
    _inner_mixed_func,
    _inverse_mixture_lognormal,
    _inverse_x_inner_mixed_func,
    _log_inner_mixed_func,
    _vix_payoff,
)


def test_log_inner_mixed_func_matches_inner_without_overflow():
    x = np.array([-1.0, 0.0, 1.0])
    args = {
        "lbd": 0.35,
        "mu_2": -2.8,
        "volvol_1": 1.4,
        "volvol_2": 0.7,
        "fvix2": 0.24**2,
    }

    log_inner = _log_inner_mixed_func(x, **args)

    assert np.exp(log_inner) == pytest.approx(_inner_mixed_func(x, **args))


def test_inner_mixed_inverse_recovers_root():
    args = {
        "lbd": 0.35,
        "mu_2": -2.8,
        "volvol_1": 1.4,
        "volvol_2": 0.7,
        "fvix2": 0.24**2,
    }
    x = 0.4
    z = _inner_mixed_func(x, **args)

    recovered = _inverse_x_inner_mixed_func(z=z, **args)

    assert recovered == pytest.approx(x)


def test_inverse_mixture_lognormal_recovers_gaussian_driver():
    x = -0.25
    args = {
        "lbd": 0.4,
        "mu_1": -3.0,
        "mu_2": -2.5,
        "sig_1": 0.7,
        "sig_2": 0.25,
    }
    y = args["lbd"] * np.exp(args["mu_1"] + args["sig_1"] * x) + (
        1.0 - args["lbd"]
    ) * np.exp(args["mu_2"] + args["sig_2"] * x)

    recovered = _inverse_mixture_lognormal(y=y, **args)

    assert recovered == pytest.approx(x)


def test_vix_payoff_factory_prices_futures_calls_and_puts():
    vix_squared = np.array([0.04, 0.09])

    assert _vix_payoff("fut")(vix_squared) == pytest.approx(np.array([0.2, 0.3]))
    assert _vix_payoff("call", K=0.25)(vix_squared) == pytest.approx(
        np.array([0.0, 0.05])
    )
    assert _vix_payoff("put", K=0.25)(vix_squared) == pytest.approx(
        np.array([0.05, 0.0])
    )

    with pytest.raises(ValueError, match="opt_payoff"):
        _vix_payoff("straddle")


def test_mixed_payoff_derivative_applies_payoff_gates():
    args = {
        "lbd": 0.35,
        "mu_2": -2.8,
        "volvol_1": 1.4,
        "volvol_2": 0.7,
        "fvix2": 0.24**2,
    }
    x = 0.4
    sqrt_inner = np.sqrt(_inner_mixed_func(x, **args))
    base = args["fvix2"] * args["lbd"] * np.exp(x) / (2.0 * sqrt_inner)

    assert sqrt_inner > 0.25
    assert _deriv_vix_payoff_mixed("fut", K=0.25)(x, **args) == pytest.approx(base)
    assert _deriv_vix_payoff_mixed("call", K=0.25)(x, **args) == pytest.approx(base)
    assert _deriv_vix_payoff_mixed("put", K=0.25)(x, **args) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="opt_payoff"):
        _deriv_vix_payoff_mixed("straddle")


def test_hermite_polynomial_weights_have_expected_shape_and_validate_order():
    weights = _hermite_polynomial_weights(n_trunc=3, b=0.5, c=0.2, n_quad=8)

    assert weights.shape == (4, 4)
    assert np.all(np.isfinite(weights))

    with pytest.raises(ValueError, match="positive integer"):
        _hermite_polynomial_weights(n_trunc=3, b=0.5, c=0.2, n_quad=0)
