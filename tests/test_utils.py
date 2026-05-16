import numpy as np
import pytest

from utils import (
    black_impvol,
    black_otm_impvol_mc,
    black_price,
    cholesky_from_svd,
    gauss_hermite,
    gauss_legendre,
    non_uniform_grid,
    relative_error,
    sqrt_sum_lognorm_shifted_lognorm_approx,
    xi0_heston,
)


@pytest.mark.parametrize("quad", [gauss_legendre, lambda n: gauss_hermite(n)])
@pytest.mark.parametrize("n", [0, -1, True])
def test_quadrature_helpers_reject_invalid_orders(quad, n):
    with pytest.raises(ValueError, match="positive integer"):
        if quad is gauss_legendre:
            quad(0.0, 1.0, n)
        else:
            quad(n)


def test_gauss_legendre_integrates_polynomial_on_custom_interval():
    nodes, weights = gauss_legendre(2.0, 5.0, 3)

    assert np.sum(weights * nodes**4) == pytest.approx((5.0**5 - 2.0**5) / 5.0)


def test_gauss_hermite_integrates_standard_normal_moments():
    nodes, weights = gauss_hermite(5)

    assert np.sum(weights) == pytest.approx(1.0)
    assert np.sum(weights * nodes) == pytest.approx(0.0, abs=1e-15)
    assert np.sum(weights * nodes**2) == pytest.approx(1.0)


def test_cholesky_from_svd_reconstructs_positive_semidefinite_matrix():
    matrix = np.array([[2.0, 2.0], [2.0, 2.0]])

    factor = cholesky_from_svd(matrix)

    assert factor.shape == matrix.shape
    assert factor @ factor.T == pytest.approx(matrix)


def test_cholesky_from_svd_rejects_non_square_and_indefinite_matrices():
    with pytest.raises(ValueError, match="square"):
        cholesky_from_svd(np.ones((2, 3)))

    with pytest.raises(np.linalg.LinAlgError, match="positive semi-definite"):
        cholesky_from_svd(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_black_price_put_call_parity_and_implied_vol_inversion():
    F = 100.0
    K = np.array([90.0, 100.0, 110.0])
    T = 0.75
    vol = 0.32

    call = black_price(K=K, T=T, F=F, vol=vol, opttype=1)
    put = black_price(K=K, T=T, F=F, vol=vol, opttype=-1)
    implied = black_impvol(K=K, T=T, F=F, value=call, opttype=1)

    assert call - put == pytest.approx(F - K)
    assert implied == pytest.approx(np.full_like(K, vol), abs=1e-9)


def test_black_helpers_validate_public_inputs():
    with pytest.raises(ValueError, match="opttype"):
        black_price(K=100.0, T=1.0, F=100.0, vol=0.2, opttype=0)

    with pytest.raises(ValueError, match="same shape"):
        black_impvol(K=np.array([90.0, 100.0]), T=1.0, F=100.0, value=np.array([1.0]))


def test_black_otm_impvol_mc_returns_error_report_and_validates_paths():
    paths = np.array([90.0, 95.0, 100.0, 105.0, 110.0])

    report = black_otm_impvol_mc(paths, k=np.array([-0.05, 0.05]), T=0.5, mc_error=True)

    assert set(report) == {
        "otm_impvol",
        "otm_impvol_high",
        "otm_impvol_low",
        "error_95",
        "otm_price",
    }
    assert report["otm_impvol"].shape == (2,)

    with pytest.raises(ValueError, match="positive simulated prices"):
        black_otm_impvol_mc(np.array([100.0, 0.0]), k=0.0, T=0.5)


def test_sqrt_sum_lognorm_shifted_lognorm_approx_matches_first_three_moments(
    capsys,
):
    lbd = 0.35
    mu_1 = -3.0
    mu_2 = -2.6
    sig_1 = 0.7
    sig_2 = 0.25
    n_quad = 40

    params = sqrt_sum_lognorm_shifted_lognorm_approx(
        lbd=lbd,
        mu_1=mu_1,
        mu_2=mu_2,
        sig_1=sig_1,
        sig_2=sig_2,
        n_quad=n_quad,
    )

    assert capsys.readouterr().out == ""
    assert set(params) == {"c_y", "mu_y", "sig_y"}
    assert np.all(np.isfinite(list(params.values())))

    x_herm, w_herm = gauss_hermite(n_quad)
    v = (
        lbd * np.exp(mu_1 + sig_1 * x_herm)
        + (1.0 - lbd) * np.exp(mu_2 + sig_2 * x_herm)
    ) ** 0.5
    y = np.exp(params["mu_y"] + params["sig_y"] * x_herm) + params["c_y"]

    for power in [1, 2, 3]:
        assert np.sum(w_herm * y**power) == pytest.approx(
            np.sum(w_herm * v**power),
            rel=1e-8,
            abs=1e-10,
        )


@pytest.mark.parametrize(
    ("shape", "expected_at_zero", "terminal_relation"),
    [
        ("flat", 0.025, "flat"),
        ("contango", 0.005, "increasing"),
        ("backwardation", 0.045, "decreasing"),
    ],
)
def test_xi0_heston_supported_shapes(shape, expected_at_zero, terminal_relation):
    t = np.array([0.0, 1.0])

    values = xi0_heston(t, shape=shape)

    assert values[0] == pytest.approx(expected_at_zero)
    if terminal_relation == "flat":
        assert values[1] == pytest.approx(values[0])
    elif terminal_relation == "increasing":
        assert values[1] > values[0]
    else:
        assert values[1] < values[0]


def test_grid_and_relative_error_helpers():
    grid = non_uniform_grid(2.0, 10.0, 5, power=2.0)

    assert grid[0] == pytest.approx(2.0)
    assert grid[-1] == pytest.approx(10.0)
    assert np.all(np.diff(grid) > 0)
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        error = relative_error(np.array([2.0, 0.0]), np.array([3.0, 1.0]))
    assert error == pytest.approx(np.array([0.5, np.nan]), nan_ok=True)

    with pytest.raises(ValueError, match="positive"):
        non_uniform_grid(0.0, 1.0, 0)
