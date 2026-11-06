import numpy as np
import pytest
from scipy.integrate import quad

from bergomi import OneFactorBergomiModel
from rbergomi import RoughBergomiModel

DELTA = 30 / 365
COEFFICIENT_KEYS = ("meanp", "tot_varp", "gamma_1", "gamma_2", "gamma_3")


def flat_xi0(t):
    return np.full_like(np.asarray(t, dtype=float), 0.24**2)


def coefficients(model, T, n_quad):
    state = model._proxy_approx_state(T, 3, n_quad=n_quad)
    return np.array([state[key] for key in COEFFICIENT_KEYS])


def test_standard_kernel_and_factor_variance():
    model = OneFactorBergomiModel({"w": 2.0, "k": 0.5}, flat_xi0)
    u = np.array([0.25, 0.5])
    t = np.array([0.1, 0.2])

    assert model.kernel(u, t) == pytest.approx(2.0 * np.exp(-0.5 * (u - t)))
    assert model.var_x(u) == pytest.approx(1.0 - np.exp(-u))


@pytest.mark.parametrize("H", [0.1, 0.3])
def test_rough_covariance_matches_quadrature(H):
    model = RoughBergomiModel({"H": H, "eta": 1.0}, flat_xi0)
    T = 0.2
    u = np.linspace(0.2, 0.9, 20)
    v = np.linspace(1.1, 1.8, 20)
    expected = [
        2 * H * quad(lambda s, a=a, b=b: ((a - s) * (b - s)) ** (H - 0.5), 0, T)[0]
        for a, b in zip(u, v)
    ]

    assert model.covariance_levy_fbm_vix(T=T, u=u, v=v) == pytest.approx(
        expected, rel=1e-6, abs=1e-9
    )


@pytest.mark.parametrize("H, eta", [(0.1, 1.0), (0.23, 1.02)])
@pytest.mark.parametrize("T", [1 / 12, 6 / 12])
def test_rough_coefficients_converged_at_thirty_nodes(H, eta, T):
    # The code kernel includes sqrt(2H), the paper's amplitude does not.
    model = RoughBergomiModel({"H": H, "eta": eta / np.sqrt(2 * H)}, flat_xi0)
    model.delta_vix = DELTA

    np.testing.assert_allclose(
        coefficients(model, T, 30), coefficients(model, T, 240), rtol=1e-6, atol=0
    )


@pytest.mark.parametrize("w, k", [(2.0, 0.25), (8.0, 10.0), (10.0, 1.0)])
@pytest.mark.parametrize("T", [1 / 12, 6 / 12])
def test_standard_proxy_moments_match_adaptive_quadrature(w, k, T):
    model = OneFactorBergomiModel({"w": w, "k": k}, flat_xi0)
    model.delta_vix = DELTA
    fvix2 = model.fut_vix2(T)

    def window(f, t):
        return quad(lambda u: model.xi0(u) * f(u, t), T, T + DELTA)[0] / DELTA

    kernel = model.kernel
    kernel_sq = quad(lambda t: window(lambda u, t: kernel(u, t) ** 2, t), 0, T)[0]
    mean = np.log(fvix2) - 0.5 * kernel_sq / fvix2
    var = quad(lambda t: (window(kernel, t) / fvix2) ** 2, 0, T)[0]

    grid = model._proxy_grid_data(T, 30)
    assert model._mean_proxy_from_grid(grid) == pytest.approx(mean, rel=1e-10)
    assert model._var_proxy_from_grid(grid) == pytest.approx(var, rel=1e-10)


def test_single_kernel_expansion_matches_closed_form():
    model = RoughBergomiModel({"H": 0.3, "eta": 1.4}, flat_xi0)
    T = 0.25
    k = np.array([-0.1, 0.0, 0.1])
    state = model._proxy_approx_state(T, 3, n_quad=240)
    meanp, tot_varp = state["meanp"], state["tot_varp"]
    gamma_2, gamma_3 = state["gamma_2"], state["gamma_3"]
    vol = np.sqrt(tot_varp / T)
    xp = 0.5 * meanp + tot_varp / 8
    F = model.price_vix_fut_approx(T=T, order=3, meanp=meanp, tot_varp=tot_varp)
    expected = (
        0.5 * vol
        + gamma_2 / (2 * vol * T)
        + 3 * gamma_3 / (8 * vol * T)
        - gamma_3 * (xp - np.log(F) - k) / (vol**3 * T**2)
    )

    actual = model.implied_vol_vix_expansion(k=k, T=T, order=2, n_quad=80)

    assert actual == pytest.approx(expected, rel=5e-7, abs=5e-7)
