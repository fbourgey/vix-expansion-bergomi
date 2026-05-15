import numpy as np
import pytest

from utils import (
    gauss_hermite,
    sqrt_sum_lognorm_shifted_lognorm_approx,
)


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
