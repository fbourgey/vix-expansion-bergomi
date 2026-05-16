import warnings

import numpy as np
import pytest

from bergomi import OneFactorBergomiModel, get_params_one_bergomi
from rbergomi import RoughBergomiModel, get_params_rbergomi


def flat_xi0(level: float = 0.04):
    return lambda t: level * np.ones_like(t)


def test_rough_bergomi_model_constructs_with_params_dict():
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )

    assert model.s0 == 1.0
    assert model.eta == pytest.approx(1.4)
    assert model.H == pytest.approx(0.3)
    assert model.rho == pytest.approx(-0.7)


def test_rough_bergomi_model_defaults_rho_to_negative_half():
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3},
    )

    assert model.rho == pytest.approx(-0.5)
    assert model.params["rho"] == pytest.approx(-0.5)


def test_rough_bergomi_model_requires_eta_and_h():
    with pytest.raises(ValueError, match="Missing parameters"):
        RoughBergomiModel(
            s0=1.0,
            xi0=flat_xi0(),
            params={"eta": 1.4},
        )


def test_one_factor_bergomi_model_defaults_rho_to_negative_half():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"w": 4.0, "k": 1.0},
    )

    assert model.rho == pytest.approx(-0.5)
    assert model.params["rho"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("model_cls", "params", "match"),
    [
        (RoughBergomiModel, {"eta": 0.0, "H": 0.3}, "eta"),
        (RoughBergomiModel, {"eta": 1.4, "H": 1.0}, "H"),
        (RoughBergomiModel, {"eta": 1.4, "H": 0.3, "rho": -1.1}, "rho"),
        (OneFactorBergomiModel, {"w": 0.0, "k": 1.0}, "w"),
        (OneFactorBergomiModel, {"w": 4.0, "k": -0.1}, "k"),
        (OneFactorBergomiModel, {"w": 4.0, "k": 1.0, "rho": 1.1}, "rho"),
    ],
)
def test_model_constructors_validate_public_parameters(model_cls, params, match):
    with pytest.raises(ValueError, match=match):
        model_cls(s0=1.0, xi0=flat_xi0(), params=params)


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
    ("id", "h", "eta", "rho"),
    [
        (1, 0.1, 1.0, -0.7),
        (2, 0.23, 1.02, -0.5),
    ],
)
def test_rough_single_params_are_defined_for_supported_ids(id, h, eta, rho):
    params, xi0 = get_params_rbergomi(id=id, opt="single")

    assert params["H"] == pytest.approx(h)
    assert params["eta"] == pytest.approx(eta / np.sqrt(2 * h))
    assert params.get("rho", -0.5) == pytest.approx(rho)
    assert xi0(np.array([0.0, 0.25])) == pytest.approx(np.array([0.24**2, 0.24**2]))


@pytest.mark.parametrize(
    ("id", "h", "eta", "eta_2", "lbd"),
    [
        (1, 0.1, 1.4, 0.7, 0.3),
        (2, 0.23, 2.0, 0.2, 0.4),
    ],
)
def test_rough_mixed_params_are_defined_for_supported_ids(id, h, eta, eta_2, lbd):
    params, xi0, actual_lbd, actual_eta_2 = get_params_rbergomi(id=id, opt="mixed")

    assert params == {"H": pytest.approx(h), "eta": pytest.approx(eta / np.sqrt(2 * h))}
    assert xi0(np.array([0.0, 0.25])) == pytest.approx(np.array([0.24**2, 0.24**2]))
    assert actual_eta_2 == pytest.approx(eta_2)
    assert actual_lbd == pytest.approx(lbd)


@pytest.mark.parametrize("id", [3, 4, 5, 6])
def test_rough_mixed_params_reject_unsupported_ids(id):
    with pytest.raises(ValueError, match="Must be 1 or 2"):
        get_params_rbergomi(id=id, opt="mixed")


@pytest.mark.parametrize(
    ("factory", "kwargs", "match"),
    [
        (get_params_one_bergomi, {"id": 1, "opt": "bad"}, "Invalid opt"),
        (get_params_one_bergomi, {"id": 3, "opt": "single"}, "single case"),
        (get_params_one_bergomi, {"id": 7, "opt": "mixed"}, "mixed case"),
        (get_params_rbergomi, {"id": 1, "opt": "bad"}, "Invalid opt"),
        (get_params_rbergomi, {"id": 3, "opt": "single"}, "single case"),
    ],
)
def test_parameter_helpers_reject_invalid_public_inputs(factory, kwargs, match):
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)


@pytest.mark.parametrize("volvol_2", [0.0, -0.1])
def test_mixed_approximation_rejects_non_positive_volvol_2(volvol_2):
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )

    with pytest.raises(ValueError, match="volvol_2"):
        model.implied_vol_vix_approx_mixed(
            T=0.25,
            k=np.array([0.0]),
            order=0,
            lbd=0.5,
            volvol_2=volvol_2,
            n_quad=12,
        )


def test_implied_vol_vix_approx_mixed_reuses_precomputed_params(monkeypatch):
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )
    original = model._get_params_mixed
    call_count = 0

    def counted_get_params_mixed(T, lbd, volvol_2, order, n_quad=30):
        nonlocal call_count
        call_count += 1
        return original(T, lbd, volvol_2, order, n_quad=n_quad)

    monkeypatch.setattr(model, "_get_params_mixed", counted_get_params_mixed)

    impvol = model.implied_vol_vix_approx_mixed(
        T=0.25,
        k=np.array([-0.1, 0.0, 0.1]),
        order=0,
        lbd=0.5,
        volvol_2=1.2,
        n_quad=12,
    )

    assert impvol.shape == (3,)
    assert call_count == 1


def test_implied_vol_vix_approx_mixed_return_all_includes_future_and_impvol():
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )

    future, impvol = model.implied_vol_vix_approx_mixed(
        T=0.25,
        k=np.array([-0.1, 0.0, 0.1]),
        order=0,
        lbd=0.5,
        volvol_2=1.2,
        n_quad=12,
        return_opt="all",
    )

    assert future > 0
    assert impvol.shape == (3,)
    assert np.all(np.isfinite(impvol))


def test_vix_approx_future_wrappers_are_consistent():
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )

    fut_from_future = model.price_vix_fut_approx(T=0.25, order=2, n_quad=12)
    fut_from_option = model.price_vix_approx(
        k=0.0,
        T=0.25,
        order=2,
        return_fut=True,
        n_quad=12,
    )

    assert fut_from_future == pytest.approx(fut_from_option)


def test_implied_vol_vix_approx_reuses_proxy_state(monkeypatch):
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )
    original = model._proxy_approx_state
    call_count = 0

    def counted_proxy_state(T, order, *, meanp=None, tot_varp=None, n_quad=30):
        nonlocal call_count
        call_count += 1
        return original(
            T,
            order,
            meanp=meanp,
            tot_varp=tot_varp,
            n_quad=n_quad,
        )

    monkeypatch.setattr(model, "_proxy_approx_state", counted_proxy_state)

    impvol = model.implied_vol_vix_approx(
        T=0.25,
        k=np.array([-0.1, 0.0, 0.1]),
        order=2,
        n_quad=12,
    )

    assert impvol.shape == (3,)
    assert call_count == 1


def test_implied_vol_vix_expansion_reuses_proxy_state(monkeypatch):
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )
    original = model._proxy_approx_state
    calls = []

    def counted_proxy_state(T, order, *, meanp=None, tot_varp=None, n_quad=30):
        calls.append((T, order, meanp, tot_varp, n_quad))
        return original(
            T,
            order,
            meanp=meanp,
            tot_varp=tot_varp,
            n_quad=n_quad,
        )

    monkeypatch.setattr(model, "_proxy_approx_state", counted_proxy_state)

    impvol = model.implied_vol_vix_expansion(
        k=np.array([-0.1, 0.0, 0.1]),
        T=0.25,
        order=1,
        n_quad=12,
    )

    assert impvol.shape == (3,)
    assert calls == [(0.25, 3, None, None, 12)]


def test_zero_order_implied_vol_vix_expansion_uses_zero_order_state(monkeypatch):
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )
    original = model._proxy_approx_state
    calls = []

    def counted_proxy_state(T, order, *, meanp=None, tot_varp=None, n_quad=30):
        calls.append((order, n_quad))
        return original(
            T,
            order,
            meanp=meanp,
            tot_varp=tot_varp,
            n_quad=n_quad,
        )

    monkeypatch.setattr(model, "_proxy_approx_state", counted_proxy_state)

    impvol = model.implied_vol_vix_expansion(k=0.0, T=0.25, order=0, n_quad=12)

    assert impvol.shape == (1,)
    assert calls == [(0, 12)]


def test_implied_vol_vix_expansion_tracks_legacy_formula():
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )
    T = 0.25
    k = np.array([-0.1, 0.0, 0.1])
    meanp = model.mean_proxy(T)
    tot_varp = model.var_proxy(T)
    vol_proxy = np.sqrt(tot_varp / T)
    gamma_2 = model.gamma_2_proxy(T)
    gamma_3 = model.gamma_3_proxy(T)
    xp = 0.5 * meanp + tot_varp / 8
    F = model.price_vix_fut_approx(T=T, order=3, meanp=meanp, tot_varp=tot_varp)
    log_strike = np.log(F * np.exp(k))
    expected = (
        0.5 * vol_proxy
        + gamma_2 / (2 * vol_proxy * T)
        + 3 * gamma_3 / (8 * vol_proxy * T)
        - gamma_3 * (xp - log_strike) / (vol_proxy**3 * T**2)
    )

    actual = model.implied_vol_vix_expansion(k=k, T=T, order=2, n_quad=80)

    assert actual == pytest.approx(expected, rel=5e-7, abs=5e-7)


def test_implied_vol_vix_expansion_rejects_invalid_n_quad():
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )

    with pytest.raises(ValueError, match="n_quad"):
        model.implied_vol_vix_expansion(k=0.0, T=0.25, order=1, n_quad=0)


def test_one_factor_mixed_approximation_collapses_to_single_future():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 4.0, "k": 1.0, "rho": -0.7},
    )

    single_future = model.price_vix_fut_approx(T=0.25, order=3, n_quad=60)
    mixed_future = model.price_vix_approx_mixed(
        T=0.25,
        lbd=0.35,
        volvol_2=4.0,
        opt_payoff="fut",
        order=3,
        n_quad=60,
    )

    assert mixed_future == pytest.approx(single_future, abs=1e-12)


def test_one_factor_mixed_approximation_collapses_to_single_put():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 4.0, "k": 1.0, "rho": -0.7},
    )
    T = 0.25
    log_moneyness = -0.1
    n_quad = 120

    single_future = model.price_vix_fut_approx(T=T, order=3, n_quad=n_quad)
    single_put = model.price_vix_approx(
        k=log_moneyness,
        T=T,
        opttype=-1,
        order=3,
        n_quad=n_quad,
    )
    mixed_put = model.price_vix_approx_mixed(
        T=T,
        lbd=0.35,
        volvol_2=4.0,
        opt_payoff="put",
        order=3,
        n_quad=n_quad,
        K=single_future * np.exp(log_moneyness),
    )

    assert mixed_put == pytest.approx(single_put, abs=2e-5)


def test_one_factor_mixed_approximation_collapses_to_single_call():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 4.0, "k": 1.0, "rho": -0.7},
    )
    T = 0.25
    log_moneyness = 0.1
    n_quad = 120

    single_future = model.price_vix_fut_approx(T=T, order=3, n_quad=n_quad)
    single_call = model.price_vix_approx(
        k=log_moneyness,
        T=T,
        opttype=1,
        order=3,
        n_quad=n_quad,
    )
    mixed_call = model.price_vix_approx_mixed(
        T=T,
        lbd=0.35,
        volvol_2=4.0,
        opt_payoff="call",
        order=3,
        n_quad=n_quad,
        K=single_future * np.exp(log_moneyness),
    )

    assert mixed_call == pytest.approx(single_call, abs=2e-5)


def test_mixed_implied_vol_expansion_validates_theorem_inputs():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 10.0, "k": 0.1, "rho": -0.7},
    )

    with pytest.raises(ValueError, match="j"):
        model.implied_vol_vix_expansion_mixed(k=0.0, T=0.25, lbd=0.2, volvol_2=2.0, j=3)

    with pytest.raises(ValueError, match="theta"):
        model.implied_vol_vix_expansion_mixed(
            k=0.0, T=0.25, lbd=0.2, volvol_2=2.0, theta=1.5
        )

    with pytest.raises(ValueError, match="order"):
        model.implied_vol_vix_expansion_mixed(
            k=0.0, T=0.25, lbd=0.2, volvol_2=2.0, order=2
        )


def test_mixed_implied_vol_expansion_reuses_precomputed_params(monkeypatch):
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 10.0, "k": 0.1, "rho": -0.7},
    )
    original = model._get_params_mixed
    calls = []

    def counted_get_params_mixed(T, lbd, volvol_2, order, n_quad=30):
        calls.append((T, lbd, volvol_2, order, n_quad))
        return original(T, lbd, volvol_2, order, n_quad=n_quad)

    monkeypatch.setattr(model, "_get_params_mixed", counted_get_params_mixed)

    impvol = model.implied_vol_vix_expansion_mixed(
        k=np.array([-0.1, 0.0, 0.1]),
        T=0.25,
        lbd=0.2,
        volvol_2=2.0,
        n_quad=12,
        n_trunc_herm=8,
    )

    assert impvol.shape == (3,)
    assert calls == [(0.25, 0.2, 2.0, 3, 12)]


def test_mixed_implied_vol_expansion_defaults_to_lower_proxy_vol_component():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 10.0, "k": 0.1, "rho": -0.7},
    )
    k = np.array([-0.1, 0.0, 0.1])

    auto = model.implied_vol_vix_expansion_mixed(
        k=k,
        T=0.25,
        lbd=0.2,
        volvol_2=2.0,
        n_quad=40,
        n_trunc_herm=8,
    )
    component_2 = model.implied_vol_vix_expansion_mixed(
        k=k,
        T=0.25,
        lbd=0.2,
        volvol_2=2.0,
        j=2,
        n_quad=40,
        n_trunc_herm=8,
    )

    assert auto.shape == k.shape
    assert np.all(np.isfinite(auto))
    assert auto == pytest.approx(component_2)


def test_mixed_implied_vol_expansion_uses_optimal_theta_by_default():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 10.0, "k": 0.1, "rho": -0.7},
    )
    k = np.array([-0.1, 1.0])

    auto = model.implied_vol_vix_expansion_mixed(
        k=k,
        T=0.25,
        lbd=0.2,
        volvol_2=2.0,
        n_quad=40,
        n_trunc_herm=8,
    )
    theta_0 = model.implied_vol_vix_expansion_mixed(
        k=k[0],
        T=0.25,
        lbd=0.2,
        volvol_2=2.0,
        theta=0.0,
        n_quad=40,
        n_trunc_herm=8,
    )
    theta_1 = model.implied_vol_vix_expansion_mixed(
        k=k[1],
        T=0.25,
        lbd=0.2,
        volvol_2=2.0,
        theta=1.0,
        n_quad=40,
        n_trunc_herm=8,
    )

    assert auto[0] == pytest.approx(theta_0)
    assert auto[1] == pytest.approx(theta_1)


def test_mixed_implied_vol_expansion_tracks_quadrature_reference():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 10.0, "k": 0.1, "rho": -0.7},
    )
    k = np.linspace(-0.1, 0.4, 6)
    T = 0.25
    lbd = 0.2
    volvol_2 = 2.0

    reference = model.implied_vol_vix(k=k, T=T, n_quad=80, lbd=lbd, w_2=volvol_2)
    expansion = model.implied_vol_vix_expansion_mixed(
        k=k,
        T=T,
        lbd=lbd,
        volvol_2=volvol_2,
        n_quad=80,
        n_trunc_herm=10,
    )
    relative_error = np.max(np.abs((expansion - reference) / reference))

    assert np.all(np.isfinite(expansion))
    assert relative_error < 0.02


def test_mixed_weak_approximation_scenario_2_has_no_exp_overflow_warning():
    model = OneFactorBergomiModel(
        s0=1.0,
        xi0=flat_xi0(0.24**2),
        params={"w": 0.5, "k": 10.0, "rho": -0.7},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        impvol = model.implied_vol_vix_approx_mixed(
            T=1.0 / 12.0,
            k=np.linspace(-0.1, 0.4, 4),
            order=3,
            lbd=0.3,
            volvol_2=6.0,
            n_quad=24,
        )

    assert np.all(np.isfinite(impvol))


def test_price_vix_control_variate_accepts_vector_opttype():
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )

    price = model.price_vix_control_variate(
        T=0.25,
        k=np.array([-0.1, 0.1]),
        n_disc=12,
        opttype=np.array([-1, 1]),
    )

    assert price.shape == (2,)
    assert np.all(np.isfinite(price))


def test_price_vix_control_variate_return_fut_is_positive_scalar():
    model = RoughBergomiModel(
        s0=1.0,
        xi0=flat_xi0(),
        params={"eta": 1.4, "H": 0.3, "rho": -0.7},
    )

    future = model.price_vix_control_variate(T=0.25, n_disc=12, return_fut=True)

    assert np.isscalar(future)
    assert future > 0
