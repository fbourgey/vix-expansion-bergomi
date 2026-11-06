"""Reproduce the single-kernel results of Section 2.3 (Figures 1-2).

For each model, scenario and maturity, every method uses the strikes
F_P * exp(m), m in MONEYNESS. Approx and Expansion are the root model methods
``implied_vol_vix_approx`` and ``implied_vol_vix_expansion``; the references
are those of ``common.py``. Figures go to figures/ and the
raw values, summary, validation checks and timings to results/section2.

Run from the repository root:
  uv run python scripts/reproduce_single_figures.py [--timings | --plot-only]
"""

import argparse

import matplotlib
import numpy as np

import common as c

SECTION = "section2"
CASES = [(kind, scenario) for kind in ("standard", "rough") for scenario in (1, 2)]


def weak_and_expansion(model, maturity, money, nodes=c.COEFFICIENT_NODES):
    weak = model.implied_vol_vix_approx(k=money, T=maturity, order=3, n_quad=nodes)
    expansion = model.implied_vol_vix_expansion(
        k=money, T=maturity, order=1, n_quad=nodes
    )
    return weak, expansion


def calculate(args, cases=CASES, validate=True):
    rows, checks = [], []
    for kind, scenario in cases:
        model = c.single_model(kind, scenario)
        models = (model, model)  # degenerate mixture used by the references
        for months in c.MATURITIES:
            maturity = months / 12
            fp = model.price_vix_fut_approx(
                T=maturity, order=3, n_quad=c.COEFFICIENT_NODES
            )
            strikes = fp * np.exp(c.MONEYNESS)
            ref = c.reference(kind, models, maturity, 0.5, strikes, args)
            weak, expansion = weak_and_expansion(model, maturity, c.MONEYNESS)

            if validate:
                # Validation: coarser/finer reference and doubled coefficient nodes.
                check = dict(model=kind, scenario=scenario, months=months)
                if kind == "standard":
                    finer = c.standard_reference(
                        models, maturity, 0.5, strikes, 2 * args.reference_nodes
                    )
                    wider = c.standard_reference(
                        models, maturity, 0.5, strikes, args.reference_nodes, tail=16
                    )
                    for name, (forward, calls, *_) in (
                        ("doubled_reference_nodes", finer),
                        ("wider_gaussian_domain", wider),
                    ):
                        iv = c.implied(forward, strikes, maturity, calls)
                        check[f"{name}_iv_change"] = float(
                            np.max(np.abs(iv - ref["iv"]))
                        )
                else:
                    coarse = c.reference(
                        kind,
                        models,
                        maturity,
                        0.5,
                        strikes,
                        args,
                        args.reference_nodes // 2,
                    )
                    check["halved_reference_nodes_relative_iv_change_percent"] = float(
                        np.max(100 * np.abs(coarse["iv"] / ref["iv"] - 1))
                    )
                fine_fp = model.price_vix_fut_approx(
                    T=maturity, order=3, n_quad=2 * c.COEFFICIENT_NODES
                )
                fine = weak_and_expansion(
                    model, maturity, np.log(strikes / fine_fp), 2 * c.COEFFICIENT_NODES
                )
                check["doubled_coefficient_nodes_iv_change"] = float(
                    max(
                        np.max(np.abs(weak - fine[0])),
                        np.max(np.abs(expansion - fine[1])),
                    )
                )
                checks.append(check)

            for i, money in enumerate(c.MONEYNESS):
                rows.append(
                    dict(
                        model=kind,
                        scenario=scenario,
                        months=months,
                        log_moneyness=float(money),
                        strike=float(strikes[i]),
                        forward_reference=float(ref["forward"]),
                        forward_weak=float(fp),
                        iv_reference=float(ref["iv"][i]),
                        iv_reference_se=float(ref["iv_se"][i]),
                        iv_weak=float(weak[i]),
                        iv_expansion=float(expansion[i]),
                    )
                )
            print(f"{kind} {scenario}, {months}M done", flush=True)
    if validate:
        c.write_json(c.RESULTS / SECTION / "validation.json", checks)
    return rows


def summarize(rows, confidence):
    summary = {}
    for kind, scenario in CASES:
        group = [r for r in rows if r["model"] == kind and r["scenario"] == scenario]
        summary[f"{kind}_{scenario}"] = {
            f"max_abs_relative_error_percent_{key}": max(
                abs(100 * (r[key] / r["iv_reference"] - 1)) for r in group
            )
            for key in ("iv_weak", "iv_expansion")
        }
        summary[f"{kind}_{scenario}"]["max_reference_95_halfwidth_percent"] = max(
            100 * confidence * r["iv_reference_se"] / r["iv_reference"] for r in group
        )
    return summary


def benchmark(args, rows):
    """Ten-strike 3M smiles: medians of 51 interleaved Approx/Expansion runs and
    of 5 reference runs, after warm-up. Include the coefficients, prices and
    Brent inversions (and the IV standard errors of the rough reference);
    exclude model and strike construction."""
    rng = np.random.default_rng(20261006)
    results = {}
    for kind, scenario in CASES:
        model = c.single_model(kind, scenario)
        strikes = np.array(
            [
                r["strike"]
                for r in rows
                if r["model"] == kind and r["scenario"] == scenario and r["months"] == 3
            ]
        )
        approx = c.median_times(
            {
                "approx": lambda: model.implied_vol_vix_approx(
                    k=c.MONEYNESS, T=0.25, order=3, n_quad=c.COEFFICIENT_NODES
                ),
                "expansion": lambda: model.implied_vol_vix_expansion(
                    k=c.MONEYNESS, T=0.25, order=1, n_quad=c.COEFFICIENT_NODES
                ),
            },
            51,
            rng,
        )
        ref = c.median_times(
            {
                "reference": lambda: c.reference(
                    kind, (model, model), 0.25, 0.5, strikes, args
                )["iv"]
            },
            5,
            rng,
        )
        results[f"{kind}_{scenario}"] = approx | ref
        print("Timed", kind, scenario, flush=True)
    return dict(protocol=" ".join(benchmark.__doc__.split()), results=results)


def plot(rows, confidence, save=True):
    """Draw the panels of the cases present in ``rows``; save=False keeps the
    figures open instead of writing them to figures/ (for the notebooks)."""
    for kind, scenario in CASES:
        group = [r for r in rows if r["model"] == kind and r["scenario"] == scenario]
        if not group:
            continue
        c.plot_comparison(
            group,
            c.FIGURES / f"single_{kind}_{scenario}",
            [("iv_weak", "Approx", "x"), ("iv_expansion", "Expansion", "+")],
            confidence=confidence,
            legend=scenario == 1,  # one legend per four-panel figure
            save=save,
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
        c.write_json(out / "timings.json", c.settings(args) | benchmark(args, rows))
    plot(rows, confidence)


if __name__ == "__main__":
    main()
