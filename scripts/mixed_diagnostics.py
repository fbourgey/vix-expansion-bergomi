"""Diagnostics of the mixed-kernel expansions quoted in Section 3.2 and Remark 10.

* Residuals of Proposition 4 of the explicit Hermite expansion (60), with
  N = 60 so that the Hermite truncation residual is negligible: the volatility
  residual and the net shift error, i.e. the shift residual minus the change of
  Black inputs from (x_theta, k_theta) to (ln F_P, k). Their sum is the price
  error C^j_{P,N} - C^BS(ln F_P, k, iv) of the expansion at the true inputs.
  Also their behaviour as the VIX window Delta shrinks at one point of
  Scenario 2.
* The maximum relative difference between the Hermite and weak call prices
  (effect of the Hermite truncation at the orders N of Tables 1-2).
* The moment-matched single log-normal baseline of Remark 10 (zeroth order,
  i.e. the flat smile sig_Y / (2 sqrt(T))), against the saved references.

Reads results/section3/results.csv (written by reproduce_mixed_figures.py) and
writes results/section3/diagnostics.json.

Run from the repository root:
  uv run python scripts/mixed_diagnostics.py
"""

import numpy as np
from scipy.special import ndtr

import common as c
from mixed_iv import hermite_expansion, hermite_weights, mixed_proxy_prices
from utils import sum_lognorm_single_lognorm_approx

SECTION = "section3"
N_HERMITE = 60
LOWER = {1: 2, 2: 1, 3: 2, 4: 2}
DELTAS = (30 / 365, 1e-2, 1e-3, 1e-5)
POINT = dict(scenario=2, months=6, j=1, index=5)  # log-moneyness 0.178


def point_values(params, j, strikes, maturity):
    weights, _ = hermite_weights(params, j, N_HERMITE)
    return hermite_expansion(params, j, weights, strikes, maturity)


def black(forward, strikes, vol, maturity):
    total = vol * np.sqrt(maturity)
    d1 = np.log(forward / strikes) / total + total / 2
    return forward * ndtr(d1) - strikes * ndtr(d1 - total)


def net_shift(point, strikes, maturity):
    """Shift residual minus the change of Black inputs to (ln F_P, k)."""
    error = point["price"] - black(point["forward"], strikes, point["iv"], maturity)
    return error - point["vol_residual"]


def strikes_for(params):
    f0, f1, _, _ = mixed_proxy_prices(params, [])
    return (f0 + f1) * np.exp(c.MONEYNESS)


def residuals():
    out = {}
    for scenario in (1, 2, 3, 4):
        kind, models, weight = c.mixed_models(scenario)
        for j in (1, 2):
            for months in c.MATURITIES:
                maturity = months / 12
                params = c.mixed_params(models, maturity, weight)
                strikes = strikes_for(params)
                point = point_values(params, j, strikes, maturity)
                price = point["price"]
                net = net_shift(point, strikes, maturity)
                base = params[f"sigp_{j}"] / (2 * np.sqrt(maturity))
                out[f"scenario_{scenario}_j{j}_{months}M"] = dict(
                    lower=j == LOWER[scenario],
                    max_abs_shift_over_price=float(
                        np.max(np.abs(point["shift_residual"] / price))
                    ),
                    max_abs_vol_over_price=float(
                        np.max(np.abs(point["vol_residual"] / price))
                    ),
                    max_abs_net_shift_over_price=float(np.max(np.abs(net / price))),
                    max_abs_vol_correction_over_base=float(
                        np.max(np.abs(point["vol_correction"] / base))
                    ),
                )
    return out


def shrinking_window():
    kind, models, weight = c.mixed_models(POINT["scenario"])
    maturity = POINT["months"] / 12
    rows = []
    for delta in DELTAS:
        for model in models:
            model.delta_vix = delta
        params = c.mixed_params(models, maturity, weight)
        strikes = strikes_for(params)[[POINT["index"]]]
        point = point_values(params, POINT["j"], strikes, maturity)
        rows.append(
            dict(
                delta=delta,
                price=float(point["price"][0]),
                shift_residual=float(point["shift_residual"][0]),
                net_shift=float(net_shift(point, strikes, maturity)[0]),
                vol_residual=float(point["vol_residual"][0]),
                vol_correction=float(point["vol_correction"][0]),
                base_vol=params[f"sigp_{POINT['j']}"] / (2 * np.sqrt(maturity)),
            )
        )
    for model in models:
        model.delta_vix = c.DELTA
    return dict(point=POINT | dict(m=float(c.MONEYNESS[POINT["index"]])), rows=rows)


def moment_matched(rows):
    out = {}
    for scenario in (1, 2, 3, 4):
        kind, models, weight = c.mixed_models(scenario)
        errors = {}
        for months in c.MATURITIES:
            maturity = months / 12
            params = c.mixed_params(models, maturity, weight)
            matched = sum_lognorm_single_lognorm_approx(
                weight,
                params["meanp_1"],
                params["meanp_2"],
                params["sigp_1"],
                params["sigp_2"],
            )
            flat = matched["sig_y"] / (2 * np.sqrt(maturity))
            ref = np.array(
                [
                    r["iv_reference"]
                    for r in rows
                    if r["scenario"] == scenario and r["months"] == months
                ]
            )
            errors[f"{months}M"] = float(100 * np.max(np.abs(flat / ref - 1)))
        out[scenario] = dict(max_abs_relative_error_percent=errors)
    return out


def hermite_vs_weak(rows):
    return {
        scenario: float(
            100
            * max(
                abs(r[f"price_hermite_j{LOWER[scenario]}"] / r["price_weak"] - 1)
                for r in rows
                if r["scenario"] == scenario
            )
        )
        for scenario in (1, 2, 3, 4)
    }


def main():
    rows = c.read_rows(c.RESULTS / SECTION / "results.csv")
    result = dict(
        n_hermite=N_HERMITE,
        hermite_vs_weak_price_percent=hermite_vs_weak(rows),
        residuals=residuals(),
        shrinking_window=shrinking_window(),
        moment_matched=moment_matched(rows),
    )
    c.write_json(c.RESULTS / SECTION / "diagnostics.json", result)
    print(result["shrinking_window"])
    print(result["moment_matched"])
    print(result["hermite_vs_weak_price_percent"])
    for key, value in result["residuals"].items():
        print(key, {k: round(v, 4) for k, v in value.items()})


if __name__ == "__main__":
    main()
