# VIX Expansions in Bergomi Models

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-6340ac.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Python research code for reproducing weak-approximation and implied-volatility
expansion results for VIX options in Bergomi and rough Bergomi forward variance
models. The repository contains model implementations, numerical utilities,
notebooks, tests, and a locked Python environment for reproducible runs.

## References

This repository reproduces results from:

- Liao, Y., Agarwal, A., & Bourgey, F. (2026). _Implied Volatility Expansions
  for VIX Options in Forward Variance Models_. arXiv:2604.25123.
  [https://arxiv.org/abs/2604.25123](https://arxiv.org/abs/2604.25123).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and currently targets Python
3.13 or newer. To match the locked environment exactly, install from
`uv.lock`:

```bash
uv sync --locked
```

Useful commands:

```bash
uv run pytest
uv run pytest tests/test_api_contracts.py tests/test_rbergomi_core.py tests/test_utils.py
uv run ruff check .
uv run ruff format --check .
uv run jupyter lab
```

If you prefer a standard virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
.venv/bin/python -m pytest
```

## Reproducing Results

Open the notebooks from the repository root with Jupyter so local imports resolve
against the checked-out source files.

| Notebook | Purpose |
| --- | --- |
| `vix_weak_approx_bergomi.ipynb` | Weak-approximation experiments for the one-factor Bergomi model |
| `vix_weak_approx_rbergomi.ipynb` | Weak-approximation experiments for the rough Bergomi model |

For deterministic Monte Carlo checks, use the explicit `seed` arguments exposed
by the simulation routines. Generated plots should be written under `figures/`
when they are intentionally part of a reproducibility run.

## Repository Map

| File | Contents |
| --- | --- |
| `model.py` | Shared `ForwardVarianceModel` base class, VIX futures, proxy moments, weak approximations, and implied-volatility expansion helpers |
| `bergomi.py` | One-factor Bergomi model, quadrature VIX pricing, implied-volatility routines, and parameter helpers |
| `rbergomi.py` | Rough Bergomi kernel, VIX covariance construction, simulation, approximation, and implied-volatility routines |
| `utils.py` | Black pricing, implied volatility, quadrature, plotting, and linear algebra utilities |
| `utils_vix.py` | Internal VIX payoff, mixed-lognormal, and Hermite expansion helpers |
| `tests/` | Pytest coverage for API contracts, rough-Bergomi numerical routines, and utilities |
| `pyproject.toml` | Project metadata and runtime dependencies |
| `uv.lock` | Locked dependency versions for reproducible installs |

## License

This project is distributed under the [MIT License](LICENSE).
