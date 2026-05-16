import numpy as np
import pytest
from scipy import integrate

from rbergomi import RoughBergomiModel


@pytest.fixture(
    params=[
        {
            "s0": 1.0,
            "xi0": lambda t: np.ones_like(t) * 0.2**2,
            "params": {"H": 0.3, "eta": 1.4, "rho": -0.7},
        },
        {
            "s0": 2.0,
            "xi0": lambda t: np.ones_like(t) * 0.1**2,
            "params": {"H": 0.1, "eta": 0.8, "rho": 0.5},
        },
    ]
)
def rbergomi(request):
    return RoughBergomiModel(**request.param)


@pytest.mark.parametrize(
    ("T", "u", "v"),
    [
        (0.03, np.linspace(0.05, 1.0, 60), np.linspace(0.03, 0.8, 60)),
        (0.2, np.linspace(0.2, 0.9, 80), np.linspace(1.1, 1.8, 80)),
    ],
)
def test_covariance_levy_fbm_matches_pairwise_quadrature(T, u, v, rbergomi):
    H = rbergomi.H

    def integrand(s, ui, vi):
        return ((ui - s) * (vi - s)) ** (H - 0.5)

    integral = np.array(
        [
            integrate.quad(integrand, 0.0, T, args=(u[i], v[i]))[0]
            for i in range(u.shape[0])
        ]
    )
    expected = 2.0 * H * integral

    assert np.allclose(
        expected, rbergomi.covariance_levy_fbm_vix(T=T, u=u, v=v), rtol=1e-6, atol=1e-9
    )


def test_covariance_matrix_matches_pairwise_quadrature(rbergomi):
    tab_u = np.linspace(0.5, 0.75, 31)
    T = tab_u[0]
    u = np.tile(tab_u, (tab_u.shape[0], 1)).T
    v = u.T

    def integrand(s, ui, vi):
        return 2.0 * rbergomi.H * ((ui - s) * (vi - s)) ** (rbergomi.H - 0.5)

    cov_quad = np.zeros_like(u)
    for i in range(tab_u.shape[0]):
        for j in range(tab_u.shape[0]):
            cov_quad[i, j] = integrate.quad(
                integrand,
                0.0,
                T,
                args=(tab_u[i], tab_u[j]),
            )[0]

    cov_model = rbergomi.covariance_levy_fbm_vix(T=T, u=u, v=v)

    assert np.allclose(cov_quad, cov_model, rtol=1e-6, atol=1e-9)
    assert np.allclose(cov_model, cov_model.T, atol=1e-12)


@pytest.mark.parametrize("T", [1e-3, 0.3, 1.0])
def test_flat_xi0_proxy_methods_match_flat_closed_forms(T, rbergomi):
    n_quad = 40
    atol = 2e-4

    assert np.allclose(
        rbergomi.mean_proxy(T, quad_scipy=True),
        rbergomi.mean_proxy_flat(T),
        atol=atol,
    )
    assert np.allclose(
        rbergomi.var_proxy(T, quad_scipy=True),
        rbergomi.var_proxy_flat(T),
        atol=atol,
    )
    assert np.allclose(
        rbergomi.gamma_1_proxy(T, n_quad=n_quad),
        rbergomi.gamma_1_proxy_flat(T),
        atol=atol,
    )
    assert np.allclose(
        rbergomi.gamma_2_proxy(T, n_quad=n_quad),
        rbergomi.gamma_2_proxy_flat(T, n_quad=n_quad, quad_scipy=False),
        atol=atol,
    )
    assert np.allclose(
        rbergomi.gamma_3_proxy(T, n_quad=n_quad),
        rbergomi.gamma_3_proxy_flat(T, n_quad=n_quad, quad_scipy=False),
        atol=atol,
    )


def test_simulate_vix_is_reproducible_with_seed(rbergomi):
    vix_1 = rbergomi.simulate_vix(T=0.25, n_mc=64, n_disc=16, seed=1234)
    vix_2 = rbergomi.simulate_vix(T=0.25, n_mc=64, n_disc=16, seed=1234)

    assert np.allclose(vix_1, vix_2)


def test_simulate_vix_seed_does_not_mutate_global_rng_state(rbergomi):
    np.random.seed(2024)
    expected = np.random.random(5)

    np.random.seed(2024)
    _ = rbergomi.simulate_vix(T=0.25, n_mc=16, n_disc=8, seed=7)
    after = np.random.random(5)

    assert np.allclose(after, expected)
