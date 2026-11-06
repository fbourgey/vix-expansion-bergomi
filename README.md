# VIX expansions in Bergomi models

Code for the figures and tables in:

> Y. Liao, A. Agarwal and F. Bourgey (2026). *Implied Volatility Expansions for
> VIX Options in Forward Variance Models*.
> [arXiv:2604.25123](https://arxiv.org/abs/2604.25123).

Covers standard and rough Bergomi models with one or two kernels.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync --locked
```

## Reproduce the paper

Run from the repository root, in order:

```bash
uv run python scripts/reproduce_single_figures.py --timings
uv run python scripts/reproduce_mixed_figures.py --timings
uv run python scripts/mixed_diagnostics.py
```

| Script | Paper |
| --- | --- |
| `reproduce_single_figures.py` | Figures 1–2 |
| `reproduce_mixed_figures.py` | Figures 3–15, Tables 1–2 |
| `mixed_diagnostics.py` | Proposition 4 residuals, Remark 10, Appendix C.1 |

PDFs go to `figures/`; raw values, validation checks and timings go to
`results/section2/` and `results/section3/` (git-ignored).

For the first two scripts, `--timings` includes the Table 4 benchmarks and
`--plot-only` redraws saved results. Use `--help` for reference settings.
The diagnostics script also requires saved results from the figure scripts.

## Notebooks

- [Bergomi](vix_weak_approx_bergomi.ipynb): Figure 1 and Scenarios 1–2.
- [Rough Bergomi](vix_weak_approx_rbergomi.ipynb): Figure 2 and Scenarios 3–4.

Both use the figure scripts and display the plots inline.

```bash
uv run --with jupyter jupyter lab
```

## Code

Single-kernel approximations are in `model.py`; two-kernel expansions are in
`mixed_iv.py`. Kernels and model parameters are in `bergomi.py` and `rbergomi.py`.
Shared numerical settings and reference pricers are in `scripts/common.py`.

## Tests

```bash
uv run pytest
```

## License

[MIT](LICENSE).
