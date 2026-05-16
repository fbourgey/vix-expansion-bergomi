import numpy as np
import pytest

from bergomi import OneFactorBergomiModel, get_params_one_bergomi


def flat_xi0(level: float = 0.04):
    return lambda t: level * np.ones_like(t)


def test_one_factor_bergomi_model_defaults_rho_to_negative_half():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"w": 4.0, "k": 1.0},
    )

    assert model.rho == pytest.approx(-0.5)
    assert model.params["rho"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"w": 0.0, "k": 1.0}, "w"),
        ({"w": 4.0, "k": -0.1}, "k"),
        ({"w": 4.0, "k": 1.0, "rho": 1.1}, "rho"),
    ],
)
def test_one_factor_bergomi_model_validates_parameters(params, match):
    with pytest.raises(ValueError, match=match):
        OneFactorBergomiModel(s0=1.0, xi0=flat_xi0(), params=params)


def test_one_factor_kernel_and_factor_variance_match_closed_forms():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"w": 2.0, "k": 0.5, "rho": -0.7},
    )
    u = np.array([0.25, 0.5])
    t = np.array([0.1, 0.2])

    assert model.kernel(u, t) == pytest.approx(2.0 * np.exp(-0.5 * (u - t)))
    assert model.var_x(u) == pytest.approx((1.0 - np.exp(-u)) / 1.0)


def test_one_factor_price_vix_fut_wrapper_matches_price_vix_future_payoff():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 2.0, "k": 0.25, "rho": -0.7},
    )

    future = model.price_vix_fut(T=0.25, n_quad=24)
    direct = model.price_vix(T=0.25, n_quad=24, opt_payoff="fut")

    assert future == pytest.approx(direct)
    assert future > 0


def test_one_factor_implied_vol_vix_returns_vector_for_scalar_and_array_inputs():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 2.0, "k": 0.25, "rho": -0.7},
    )

    scalar_impvol = model.implied_vol_vix(k=0.0, T=0.25, n_quad=24)
    vector_impvol = model.implied_vol_vix(k=np.array([-0.05, 0.05]), T=0.25, n_quad=24)

    assert scalar_impvol.shape == (1,)
    assert vector_impvol.shape == (2,)
    assert np.all(np.isfinite(scalar_impvol))
    assert np.all(np.isfinite(vector_impvol))


@pytest.mark.parametrize(
    ("id", "expected_params"),
    [
        (1, {"w": pytest.approx(2.0), "k": pytest.approx(0.25)}),
        (2, {"w": pytest.approx(8.0), "k": pytest.approx(10.0)}),
    ],
)
def test_one_factor_single_params_are_defined_for_supported_ids(id, expected_params):
    params, xi0 = get_params_one_bergomi(id=id, opt="single")

    assert params == expected_params
    assert xi0(np.array([0.0, 0.25])) == pytest.approx(np.array([0.24**2, 0.24**2]))


@pytest.mark.parametrize(
    ("id", "xi0_level", "w_1", "w_2", "lbd"),
    [
        (3, 1.445e-2, 6.1970, 0.6586, 0.3021),
        (4, 2.065e-2, 5.3118, 0.4301, 0.4790),
        (5, 2.533e-2, 4.5273, 0.4238, 0.5497),
        (6, 2.862e-2, 3.6860, 0.3226, 0.6426),
    ],
)
def test_one_factor_mixed_term_bucket_params(id, xi0_level, w_1, w_2, lbd):
    params, xi0, actual_lbd, actual_w_2 = get_params_one_bergomi(
        id=id,
        opt="mixed",
    )

    assert params == {"w": pytest.approx(w_1), "k": pytest.approx(1.0)}
    assert xi0(np.array([0.0, 0.25])) == pytest.approx(np.array([xi0_level, xi0_level]))
    assert actual_w_2 == pytest.approx(w_2)
    assert actual_lbd == pytest.approx(lbd)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"id": 1, "opt": "bad"}, "Invalid opt"),
        ({"id": 3, "opt": "single"}, "single case"),
        ({"id": 7, "opt": "mixed"}, "mixed case"),
    ],
)
def test_one_factor_parameter_helper_rejects_invalid_public_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        get_params_one_bergomi(**kwargs)
