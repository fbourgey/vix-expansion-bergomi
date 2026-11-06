"""Independent checks of the reference values used by the figure drivers."""

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from scripts.common import implied, reference, rough_reference, standard_reference
from rbergomi import RoughBergomiModel, get_params_rbergomi


def _flat(level):
    return lambda u: np.full_like(u, level**2)


def test_rough_reference_recovers_exact_lognormal_smile():
    eta, maturity = 1.2, 0.25
    model = SimpleNamespace(
        eta=eta,
        xi0=_flat(0.24),
        covariance_levy_fbm_vix=lambda t, u, v: np.full_like(u, t),
    )
    forward = 0.24 * np.exp(-(eta**2) * maturity / 8)
    strikes = forward * np.exp(np.array([-0.1, 0.0, 0.4]))
    args = SimpleNamespace(reference_nodes=16, power=8, replicates=4, seed=1)
    result = reference("rough", (model, model), maturity, 0.5, strikes, args)
    assert_allclose(result["forward"], forward, rtol=1e-12)
    assert_allclose(result["iv"], eta / 2, atol=1e-10)
    assert np.max(result["iv_se"]) < 1e-10


def test_rough_reference_batches_preserve_complete_sobol_net():
    params, xi0 = get_params_rbergomi(1, opt="single")
    model = RoughBergomiModel(params, xi0)
    strikes = np.array([0.18, 0.24, 0.3])
    args = ((model, model), 0.25, 0.5, strikes, 32, 10, 4)
    batched = rough_reference(*args, batch_power=7)
    full = rough_reference(*args, batch_power=10)
    for actual, expected in zip(batched, full):
        assert_allclose(actual, expected, rtol=1e-8, atol=2e-13)


def test_standard_reference_recovers_lognormal_constant_kernel():
    amplitude, maturity = 2.0, 0.25
    model = SimpleNamespace(
        xi0=_flat(0.24),
        var_x=lambda t: t,
        _f_xi=lambda t, u, x: np.full_like(
            u, np.exp(amplitude * x - amplitude**2 * t / 2)
        ),
    )
    forward = 0.24 * np.exp(-(amplitude**2) * maturity / 8)
    strikes = forward * np.exp(np.array([-0.25, 0.0, 0.5]))
    ref_forward, calls, *_ = standard_reference(
        (model, model), maturity, 0.5, strikes, 16
    )
    assert_allclose(ref_forward, forward, rtol=1e-12)
    assert_allclose(
        implied(ref_forward, strikes, maturity, calls), amplitude / 2, rtol=1e-11
    )
