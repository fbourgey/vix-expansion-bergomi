"""Reproduce the mixed-kernel results of Section 3 and Appendix C.

For Scenarios 1-4, maturities 1, 3, 6 months and strikes F_P * exp(m),
m in MONEYNESS, this driver computes, from the root module ``mixed_iv``:

* Approx: Brent inversion of the weak price approximation (Theorem 3);
* Section 3.2 / Appendix C: the explicit Hermite expansion (60) with theta*,
  the extended expansion (61) and the Nth-order Hermite call price
  (Proposition 3), for both components j, with the smallest N such that
  MSE_j(N) <= 1e-6 (Tables 1-2);
* Section 3.3: the mixed-proxy expansion Sigma0 + Sigma1 (Theorem 4).

Figures 3-15 go to figures/; raw values, summary, validation checks
and timings to results/section3.

Run from the repository root:
  uv run python scripts/reproduce_mixed_figures.py [--timings | --plot-only]
"""

import argparse

import matplotlib
import numpy as np

import common as c
from mixed_iv import (
    hermite_expansion,
    hermite_order,
    hermite_weights,
    mixed_proxy_prices,
    proxy_smile_expansion,
)

SECTION = "section3"
SCENARIOS = (1, 2, 3, 4)
HERMITE_TOL = 1e-6  # tolerance on MSE_j(N)
HERMITE_N_MAX = 60
INTEGRATION_NODES = 128  # Gauss--Legendre nodes of the Gaussian integrals
# Components shown in Section 3.2.1 (first entry: the lower proxy volatility).
COMPONENTS = {1: (2,), 2: (1, 2), 3: (2,), 4: (1, 2)}
LOWER = {1: 2, 2: 1, 3: 2, 4: 2}


def weak_and_proxy(params, strikes, maturity, nodes=INTEGRATION_NODES):
    """Approx and Sigma0 + Sigma1 (and Sigma0, and the weak prices)."""
    f0, f1, c0, c1 = mixed_proxy_prices(params, strikes, nodes=nodes)
    sigma0, sigma1 = proxy_smile_expansion(f0, f1, c0, c1, strikes, maturity)
    weak = c.implied(f0 + f1, strikes, maturity, c0 + c1)
    return dict(weak=weak, sigma0=sigma0, new=sigma0 + sigma1, price=c0 + c1)


def hermite(params, j, strikes, maturity, order=None):
    if order is None:
        order = hermite_order(params, j, HERMITE_TOL, HERMITE_N_MAX)
    weights, _ = hermite_weights(params, j, order)
    return order, hermite_expansion(params, j, weights, strikes, maturity)


def calculate(args, scenarios=SCENARIOS, validate=True):
    rows, checks = [], []
    for scenario in scenarios:
        kind, models, weight = c.mixed_models(scenario)
        for months in c.MATURITIES:
            maturity = months / 12
            params = c.mixed_params(models, maturity, weight)
            f0, f1, _, _ = mixed_proxy_prices(params, [], nodes=INTEGRATION_NODES)
            strikes = (f0 + f1) * np.exp(c.MONEYNESS)
            ref = c.reference(kind, models, maturity, weight, strikes, args)
            values = weak_and_proxy(params, strikes, maturity)
            expansions = {j: hermite(params, j, strikes, maturity) for j in (1, 2)}

            if validate:
                coarse = c.reference(
                    kind,
                    models,
                    maturity,
                    weight,
                    strikes,
                    args,
                    args.reference_nodes // 2,
                )
                finer = weak_and_proxy(params, strikes, maturity, 2 * INTEGRATION_NODES)
                fine_params = c.mixed_params(
                    models, maturity, weight, 2 * c.COEFFICIENT_NODES
                )
                fine = weak_and_proxy(fine_params, strikes, maturity)
                check = dict(
                    scenario=scenario,
                    months=months,
                    halved_reference_nodes_relative_iv_change_percent=float(
                        np.max(100 * np.abs(coarse["iv"] / ref["iv"] - 1))
                    ),
                    doubled_integration_nodes_iv_change=float(
                        np.max(np.abs(finer["new"] - values["new"]))
                    ),
                    doubled_coefficient_nodes_iv_change=float(
                        np.max(np.abs(fine["new"] - values["new"]))
                    ),
                )
                for j, (order, point) in expansions.items():
                    big = hermite(params, j, strikes, maturity, HERMITE_N_MAX)[1]
                    check[f"N_j{j}"] = order
                    check[f"hermite_futures_vs_weak_j{j}"] = float(
                        abs(point["forward"] - f0 - f1)
                    )
                    check[f"hermite_price_N_max_vs_weak_relative_j{j}"] = float(
                        np.max(np.abs(big["price"] / values["price"] - 1))
                    )
                checks.append(check)

            for i, money in enumerate(c.MONEYNESS):
                row = dict(
                    scenario=scenario,
                    months=months,
                    log_moneyness=float(money),
                    strike=float(strikes[i]),
                    forward_reference=float(ref["forward"]),
                    forward_weak=float(f0 + f1),
                    iv_reference=float(ref["iv"][i]),
                    iv_reference_se=float(ref["iv_se"][i]),
                    price_reference=float(ref["calls"][i]),
                    price_reference_se=float(ref["price_se"][i]),
                    iv_weak=float(values["weak"][i]),
                    iv_sigma0=float(values["sigma0"][i]),
                    iv_new=float(values["new"][i]),
                    price_weak=float(values["price"][i]),
                )
                for j, (order, point) in expansions.items():
                    row[f"N_j{j}"] = order
                    row[f"iv_hermite_j{j}"] = float(point["iv"][i])
                    row[f"iv_extended_j{j}"] = float(point["iv_extended"][i])
                    row[f"price_hermite_j{j}"] = float(point["price"][i])
                rows.append(row)
            print(f"Scenario {scenario}, {months}M done", flush=True)
    if validate:
        c.write_json(c.RESULTS / SECTION / "validation.json", checks)
    return rows


def summarize(rows, confidence):
    def max_error(group, key, reference="iv_reference"):
        return max(abs(100 * (r[key] / r[reference] - 1)) for r in group)

    summary = {}
    for scenario in SCENARIOS:
        group = [r for r in rows if r["scenario"] == scenario]
        entry = {
            f"max_abs_relative_error_percent_{key}": max_error(group, key)
            for key in ("iv_weak", "iv_sigma0", "iv_new")
        }
        for j in (1, 2):
            for months in c.MATURITIES:
                sub = [r for r in group if r["months"] == months]
                entry[f"N_j{j}_{months}M"] = int(sub[0][f"N_j{j}"])
                for key in (f"iv_hermite_j{j}", f"iv_extended_j{j}"):
                    entry[f"max_abs_relative_error_percent_{key}_{months}M"] = (
                        max_error(sub, key)
                    )
        for key in ("price_weak", f"price_hermite_j{LOWER[scenario]}"):
            entry[f"max_abs_relative_error_percent_{key}"] = max_error(
                group, key, "price_reference"
            )
        entry["max_reference_95_halfwidth_percent"] = max(
            100 * confidence * r["iv_reference_se"] / r["iv_reference"] for r in group
        )
        summary[scenario] = entry
    return summary


def benchmark(args):
    """Ten-strike 3M smiles: medians of 51 interleaved runs of Approx, of
    Sigma0 + Sigma1 and of the Hermite expansion (60) around the lower proxy
    volatility, and of 5 reference runs, after warm-up. Each run recomputes
    the coefficients (and, for (60), the Hermite weights at the selected N) and
    the exercise boundary of every strike; it excludes model construction and
    the selection of N."""
    rng = np.random.default_rng(20261005)
    results = {}
    for scenario in SCENARIOS:
        kind, models, weight = c.mixed_models(scenario)
        maturity, j = 0.25, LOWER[scenario]
        params = c.mixed_params(models, maturity, weight)
        f0, f1, _, _ = mixed_proxy_prices(params, [], nodes=INTEGRATION_NODES)
        strikes = (f0 + f1) * np.exp(c.MONEYNESS)
        order = hermite_order(params, j, HERMITE_TOL, HERMITE_N_MAX)

        def run(method):
            def evaluate():
                p = c.mixed_params(models, maturity, weight)
                if method == "hermite":
                    return hermite(p, j, strikes, maturity, order)[1]["iv"]
                f0, f1, c0, c1 = mixed_proxy_prices(p, strikes, INTEGRATION_NODES)
                if method == "weak":
                    return c.implied(f0 + f1, strikes, maturity, c0 + c1)
                return sum(proxy_smile_expansion(f0, f1, c0, c1, strikes, maturity))

            return evaluate

        times = c.median_times(
            {name: run(name) for name in ("weak", "new", "hermite")}, 51, rng
        )
        times |= c.median_times(
            {
                "reference": lambda: c.reference(
                    kind, models, maturity, weight, strikes, args
                )["iv"]
            },
            5,
            rng,
        )
        results[scenario] = dict(j=j, N=order) | times
        print(scenario, {k: v["median_ms"] for k, v in times.items()}, flush=True)
    return dict(protocol=" ".join(benchmark.__doc__.split()), results=results)


def plot(rows, confidence, save=True):
    """Draw the figures of the scenarios present in ``rows``; save=False keeps
    the figures open instead of writing them to figures/ (for the notebooks)."""
    for scenario in SCENARIOS:
        group = [r for r in rows if r["scenario"] == scenario]
        if not group:
            continue
        j = LOWER[scenario]
        common = dict(confidence=confidence, save=save)
        # Figure 7: one legend for the four scenarios, in Scenario 1.
        c.plot_comparison(
            group,
            c.FIGURES / f"scenario_{scenario}",
            [("iv_weak", "Approx", "x"), ("iv_new", "Expansion", "+")],
            panels=("smile",),
            legend=scenario == 1,
            **common,
        )
        # Figures 3-6: one legend per figure, in its top-left panel.
        for index, component in enumerate(COMPONENTS[scenario]):
            c.plot_comparison(
                group,
                c.FIGURES / f"scenario_{scenario}_j_{component}",
                [
                    ("iv_weak", "Approx", "x"),
                    (f"iv_hermite_j{component}", "Expansion", "+"),
                ],
                legend=index == 0,
                **common,
            )
        # Appendix C.2: (61) can blow up away from the money (Remark 9).
        reference = np.array([r["iv_reference"] for r in group])
        c.plot_comparison(
            group,
            c.FIGURES / f"scenario_{scenario}_extended",
            [
                (f"iv_hermite_j{j}", "Expansion", "+"),
                (f"iv_extended_j{j}", "Expansion (extended)", "x"),
            ],
            ylims={
                "smile": (50 * reference.min(), 150 * reference.max()),
                "error": (-100, 100),
            },
            **common,
        )
        # Appendix C.1: call prices.
        c.plot_comparison(
            group,
            c.FIGURES / f"scenario_{scenario}_price",
            [("price_weak", "Approx", "x"), (f"price_hermite_j{j}", "Hermite", "+")],
            reference_key="price_reference",
            se_key="price_reference_se",
            value_label="VIX call price (VIX points)",
            legend_loc="upper right",
            **common,
        )


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    c.add_reference_arguments(parser)
    args = parser.parse_args()
    out = c.RESULTS / SECTION
    if args.plot_only:
        rows = c.read_rows(out / "results.csv")
        confidence = c.saved_confidence_factor(out, args.replicates)
    else:
        confidence = c.confidence_factor(args.replicates)
        rows = calculate(args)
        c.write_rows(out / "results.csv", rows)
        c.write_json(out / "metadata.json", c.settings(args))
        summary = summarize(rows, confidence)
        c.write_json(out / "summary.json", summary)
        print(summary)
    if args.timings:
        c.write_json(out / "timings.json", c.settings(args) | benchmark(args))
    plot(rows, confidence)


if __name__ == "__main__":
    main()
