"""Shared settings, models, references, plotting and I/O of the figure drivers.

The numerical methods (proxy coefficients, weak approximations, expansions and
Black inversion) come from the root modules; this module only adds the
reference values and the manuscript settings. Figures are written to
``figures/`` and the raw values to ``results/<section>``.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtri
from scipy.stats import qmc, t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bergomi import OneFactorBergomiModel, get_params_one_bergomi  # noqa: E402
from mixed_iv import mixed_proxy_prices  # noqa: E402
from rbergomi import RoughBergomiModel, get_params_rbergomi  # noqa: E402
from utils import black_impvol_brent  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

DELTA = 30 / 365
MATURITIES = (1, 3, 6)  # months
MONEYNESS = np.linspace(-0.1, 0.4, 10)  # m, with strikes F_P * exp(m)
COEFFICIENT_NODES = 30  # graded Gauss--Legendre nodes for mu_P, sigma_P, gamma_i
IV_XTOL = 1e-13  # absolute volatility tolerance of every Brent inversion
REFERENCE_NODES = 256  # delivery-date nodes of both references
QMC_POWER = 18  # 2**18 Sobol points per scramble
QMC_REPLICATES = 16  # independent scrambles
QMC_SEED = 20261005  # scramble r uses QMC_SEED + r
QMC_BATCH_POWER = 14  # batches of 2**14 points preserve the complete Sobol net
TAIL = 14.0  # Gaussian truncation of the standard reference


def confidence_factor(replicates=QMC_REPLICATES):
    """Student-t factor of a pointwise 95% interval across scrambles."""
    if replicates < 2:
        raise ValueError("At least two independent scrambles are required.")
    return float(t.ppf(0.975, replicates - 1))


def _with_delta(model):
    model.delta_vix = DELTA
    return model


def single_model(kind, scenario):
    """Single-kernel model of Section 2.3 ("standard" or "rough", scenario 1-2)."""
    if kind == "standard":
        params, xi0 = get_params_one_bergomi(scenario, opt="single")
        return _with_delta(OneFactorBergomiModel(params=params, xi0=xi0))
    params, xi0 = get_params_rbergomi(scenario, opt="single")
    return _with_delta(RoughBergomiModel(params=params, xi0=xi0))


def mixed_models(scenario):
    """Scenario 1-4 of Section 3: (kind, (model_1, model_2), lambda).

    Scenarios 1-2 are mixed standard Bergomi, 3-4 mixed rough Bergomi; the
    rough amplitudes are converted to code units by the parameter helpers.
    """
    if scenario in (1, 2):
        params, xi0, weight, second = get_params_one_bergomi(scenario, opt="mixed")
        first = _with_delta(OneFactorBergomiModel(params=params, xi0=xi0))
        return "standard", (first, first._clone_with_params(w=second)), weight
    params, xi0, weight, second = get_params_rbergomi(scenario - 2, opt="mixed")
    first = _with_delta(RoughBergomiModel(params=params, xi0=xi0))
    return "rough", (first, first._clone_with_params(eta=second)), weight


def components(models, weight):
    """(model, fraction) pairs; a single kernel is the mixture (model, model)."""
    if models[0] is models[1]:
        return [(models[0], 1.0)]
    return [(models[0], weight), (models[1], 1 - weight)]


def mixed_params(models, maturity, weight, nodes=COEFFICIENT_NODES):
    """Proxy coefficients of the two-kernel mixture, in the format of ``mixed_iv``."""
    states = [m._proxy_approx_state(maturity, 3, n_quad=nodes) for m in models]
    params = {"lbd": weight}
    for index, state in enumerate(states, 1):
        params[f"meanp_{index}"] = float(state["meanp"])
        params[f"sigp_{index}"] = float(np.sqrt(state["tot_varp"]))
    for order in (1, 2, 3):
        params[f"gamma_{order}"] = tuple(float(s[f"gamma_{order}"]) for s in states)
    return params


def implied(forward, strikes, maturity, prices):
    """Brent Black inversion of call prices, as for every method of the paper."""
    return black_impvol_brent(strikes, maturity, forward, prices, 1, xtol=IV_XTOL)


def standard_reference(models, maturity, weight, strikes, nodes, tail=TAIL):
    """(Mixed) standard Bergomi futures and call prices by quadrature.

    Gauss--Legendre nodes integrate the delivery date; the Gaussian factor is
    integrated adaptively, split at each exercise boundary. A single kernel is
    the degenerate mixture (model, model).
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    u = maturity + DELTA * (x + 1) / 2
    w = w / 2 * models[0].xi0(u)
    std = np.sqrt(models[0].var_x(maturity))

    def vix(z):
        return np.sqrt(
            w
            @ sum(
                f * m._f_xi(maturity, u, std * z) for m, f in components(models, weight)
            )
        )

    def integrate(function, lower):
        return quad(
            lambda z: function(z) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
            lower,
            tail,
            epsabs=2e-13,
            epsrel=2e-12,
        )[0]

    forward = integrate(vix, -tail)
    calls = np.array(
        [
            integrate(
                lambda z: vix(z) - strike,
                brentq(lambda z: vix(z) - strike, -tail, tail, xtol=IV_XTOL),
            )
            for strike in strikes
        ]
    )
    return forward, calls, np.zeros(len(strikes)), np.zeros(len(strikes))


def rough_reference(
    models,
    maturity,
    weight,
    strikes,
    nodes=REFERENCE_NODES,
    power=QMC_POWER,
    replicates=QMC_REPLICATES,
    seed=QMC_SEED,
    batch_power=QMC_BATCH_POWER,
):
    """(Mixed) rough Bergomi futures and call prices by randomized QMC.

    Delivery dates u = T + Delta v^4 with Gauss--Legendre nodes in v; PCA
    factorization of the covariance; geometric proxy control variate with
    coefficient one and exact expectation. Returns the means across the
    scrambles and the standard errors of the IVs (estimated jointly with the
    futures) and of the call prices.
    """
    if nodes < 2 or power < 0 or replicates < 2 or batch_power < 0:
        raise ValueError(
            "Require nodes >= 2, power >= 0, replicates >= 2, batch_power >= 0."
        )
    x, w = np.polynomial.legendre.leggauss(nodes)
    v = (x + 1) / 2
    u = maturity + DELTA * v**4
    weights = w / 2 * 4 * v**3 * models[0].xi0(u)
    left, right = np.meshgrid(u, u, indexing="ij")
    covariance = models[0].covariance_levy_fbm_vix(maturity, left, right)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-10 * eigenvalues.max():
        raise ValueError("Covariance is not positive semidefinite to tolerance")
    active = np.flatnonzero(eigenvalues > 0)[::-1]  # dominant factor first
    factor = eigenvectors[:, active] * np.sqrt(eigenvalues[active])
    factor *= np.where(factor[-1] >= 0, 1.0, -1.0)  # reproducible signs
    variance = np.diag(covariance)
    level = weights.sum()
    mean_weights = weights / level
    average_variance = mean_weights @ variance
    proxy_std = np.sqrt(np.sum((mean_weights @ factor) ** 2))
    mixture = components(models, weight)
    proxy_params = {"lbd": weight}
    for index, model in enumerate(models, 1):
        proxy_params[f"meanp_{index}"] = np.log(level) - (
            model.eta**2 * average_variance / 2
        )
        proxy_params[f"sigp_{index}"] = model.eta * proxy_std
    f0, _, c0, _ = mixed_proxy_prices(proxy_params, strikes)
    exact_proxy = np.r_[f0, c0]
    estimates = []
    for rep in range(replicates):
        sampler = qmc.Sobol(len(active), scramble=True, seed=seed + rep)
        batch = 2 ** min(batch_power, power)
        total = np.zeros(len(strikes) + 1)
        for _ in range(2**power // batch):
            gaussian = ndtri(np.clip(sampler.random(batch), 1e-15, 1 - 1e-15))
            gaussian = gaussian @ factor.T
            average = gaussian @ mean_weights
            vix = np.sqrt(
                sum(
                    f * np.exp(m.eta * gaussian - m.eta**2 * variance / 2)
                    for m, f in mixture
                )
                @ weights
            )
            proxy = np.sqrt(
                level
                * sum(
                    f * np.exp(m.eta * average - m.eta**2 * average_variance / 2)
                    for m, f in mixture
                )
            )
            total += np.r_[
                (vix - proxy).sum(),
                (
                    np.maximum(vix[:, None] - strikes, 0)
                    - np.maximum(proxy[:, None] - strikes, 0)
                ).sum(axis=0),
            ]
        estimates.append(exact_proxy + total / 2**power)
    estimates = np.array(estimates)
    mean = estimates.mean(axis=0)
    ivs = np.array([implied(row[0], strikes, maturity, row[1:]) for row in estimates])
    iv_se = ivs.std(axis=0, ddof=1) / np.sqrt(replicates)
    price_se = estimates[:, 1:].std(axis=0, ddof=1) / np.sqrt(replicates)
    return mean[0], mean[1:], iv_se, price_se


def reference(kind, models, maturity, weight, strikes, args, nodes=None):
    """Reference futures, calls, IV and IV/price standard errors."""
    nodes = args.reference_nodes if nodes is None else nodes
    if kind == "standard":
        forward, calls, iv_se, price_se = standard_reference(
            models, maturity, weight, strikes, nodes
        )
    else:
        forward, calls, iv_se, price_se = rough_reference(
            models,
            maturity,
            weight,
            strikes,
            nodes,
            args.power,
            args.replicates,
            args.seed,
        )
    iv = implied(forward, strikes, maturity, calls)
    return dict(forward=forward, calls=calls, iv=iv, iv_se=iv_se, price_se=price_se)


def add_reference_arguments(parser):
    parser.add_argument("--reference-nodes", type=int, default=REFERENCE_NODES)
    parser.add_argument("--power", type=int, default=QMC_POWER)
    parser.add_argument("--replicates", type=int, default=QMC_REPLICATES)
    parser.add_argument("--seed", type=int, default=QMC_SEED)
    parser.add_argument("--plot-only", action="store_true", help="redraw saved values")
    parser.add_argument("--timings", action="store_true", help="benchmark at 3M")


def default_args():
    """The reference settings of the paper, as the scripts parse them."""
    parser = argparse.ArgumentParser()
    add_reference_arguments(parser)
    return parser.parse_args([])


def settings(args):
    """Settings and source revision recorded in metadata.json."""

    def git(*command):
        return subprocess.check_output(["git", "-C", str(ROOT), *command], text=True)

    import platform

    import scipy

    return dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        source_revision=git("rev-parse", "HEAD").strip(),
        source_status=git("status", "--short", "--", "*.py"),
        delta=DELTA,
        log_moneyness=MONEYNESS.tolist(),
        coefficient_nodes=COEFFICIENT_NODES,
        iv_xtol=IV_XTOL,
        reference_nodes=args.reference_nodes,
        qmc_power=args.power,
        qmc_replicates=args.replicates,
        qmc_seed=args.seed,
        confidence_factor=confidence_factor(args.replicates),
    )


def median_times(methods, runs, rng):
    """Median/quartile runtimes in ms of interleaved evaluations, after warm-up."""
    times = {name: [] for name in methods}
    for function in methods.values():
        function()
    for _ in range(runs):
        for name in rng.permutation(list(methods)):
            start = perf_counter()
            values = methods[name]()
            times[name].append(1000 * (perf_counter() - start))
            assert np.all(np.isfinite(values))
    return {
        name: dict(
            median_ms=float(np.median(sample)),
            q25_ms=float(np.quantile(sample, 0.25)),
            q75_ms=float(np.quantile(sample, 0.75)),
            runs=runs,
        )
        for name, sample in times.items()
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def saved_confidence_factor(out, replicates):
    """Student-t factor of the run that wrote ``out/results.csv``.

    Falls back to ``replicates`` scrambles when ``metadata.json`` is missing.
    """
    metadata = out / "metadata.json"
    if not metadata.exists():
        return confidence_factor(replicates)
    return json.loads(metadata.read_text())["confidence_factor"]


def read_rows(path):
    with path.open() as handle:
        return [
            {k: v if k == "model" else float(v) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


def plot_comparison(
    rows,
    stem,
    methods,
    *,
    confidence,
    reference_key="iv_reference",
    se_key="iv_reference_se",
    panels=("smile", "error"),
    legend=True,
    legend_loc=None,
    value_label="VIX implied volatility in %",
    value_scale=100.0,
    ylims=None,
    save=True,
):
    """Draw 8x6 smile and error panels, saved as ``<stem>_<panel>.pdf``.

    Seaborn talk/whitegrid, tab10 maturity colours, references as dots and
    ``methods`` as (column, label, marker) triples, against log-moneyness.
    Error panels show signed relative errors in percent. Shaded bands are
    pointwise 95% intervals of the reference (invisible when tiny); on error
    panels, the interval endpoints are mapped through each relative error.
    With ``legend``, the smile panel carries the legend, room being reserved
    above the data; ``ylims`` maps a panel to its y-axis limits. With
    ``save=False`` the figures are left open, so a notebook shows them.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcdefaults()
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 6)
    colours = sns.color_palette("tab10")
    if save:
        stem.parent.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        fig, ax = plt.subplots()
        for colour, months in zip(colours, MATURITIES):
            group = sorted(
                (r for r in rows if int(r["months"]) == months),
                key=lambda r: r["log_moneyness"],
            )
            x = np.array([r["log_moneyness"] for r in group])
            ref = np.array([r[reference_key] for r in group])
            half = confidence * np.array([r[se_key] for r in group])
            if panel == "smile":
                ax.plot(x, value_scale * ref, ".", color=colour, label=f"{months}M")
                if np.any(half):
                    ax.fill_between(
                        x,
                        value_scale * (ref - half),
                        value_scale * (ref + half),
                        color=colour,
                        alpha=0.12,
                    )
            for key, _, marker in methods:
                values = np.array([r[key] for r in group])
                if panel == "smile":
                    ax.plot(x, value_scale * values, marker, color=colour)
                    continue
                if np.any(half):
                    ax.fill_between(
                        x,
                        100 * (values / (ref + half) - 1),
                        100 * (values / (ref - half) - 1),
                        color=colour,
                        alpha=0.12,
                    )
                ax.plot(x, 100 * (values / ref - 1), marker, color=colour)
        ax.set_xlabel("Log-moneyness")
        ax.set_ylabel(value_label if panel == "smile" else "Relative error (in %)")
        if ylims and panel in ylims:
            ax.set_ylim(*ylims[panel])
        if legend and panel == "smile":
            for _, label, marker in methods:
                ax.plot([], [], "k" + marker, label=label)
            loc = legend_loc
            if loc is None:
                lower, upper = ax.get_ylim()
                ax.set_ylim(lower, upper + 0.25 * (upper - lower))
                loc = "upper left"
            ax.legend(loc=loc, ncol=3 if len(methods) == 2 else 1)
        if save:
            fig.savefig(stem.with_name(f"{stem.name}_{panel}.pdf"), bbox_inches="tight")
            plt.close(fig)
