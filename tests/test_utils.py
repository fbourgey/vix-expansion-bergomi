import numpy as np
import pytest

from utils import (
    black_impvol_brent,
    black_price,
    sum_lognorm_single_lognorm_approx,
)


def test_black_price_put_call_parity_and_implied_vol_inversion():
    F = 100.0
    K = np.array([90.0, 100.0, 110.0])
    T = 0.75
    vol = 0.32

    call = black_price(K=K, T=T, F=F, vol=vol, opttype=1)
    put = black_price(K=K, T=T, F=F, vol=vol, opttype=-1)

    assert call - put == pytest.approx(F - K)
    assert black_impvol_brent(K, T, F, call, opttype=1) == pytest.approx(
        np.full_like(K, vol), abs=1e-12
    )
    assert black_impvol_brent(K, T, F, put, opttype=-1) == pytest.approx(
        np.full_like(K, vol), abs=1e-12
    )


def test_sum_lognorm_single_lognorm_approx_matches_comonotone_moments():
    lbd, mu_1, mu_2, sig_1, sig_2 = 0.3, -3.0, -2.6, 0.7, 0.25
    params = sum_lognorm_single_lognorm_approx(lbd, mu_1, mu_2, sig_1, sig_2)
    z, w = np.polynomial.hermite_e.hermegauss(60)
    w = w / np.sqrt(2 * np.pi)
    x = lbd * np.exp(mu_1 + sig_1 * z) + (1 - lbd) * np.exp(mu_2 + sig_2 * z)
    y = np.exp(params["mu_y"] + params["sig_y"] * z)
    for power in [1, 2]:
        assert np.sum(w * y**power) == pytest.approx(np.sum(w * x**power), rel=1e-10)
