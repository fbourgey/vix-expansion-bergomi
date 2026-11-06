from collections.abc import Callable

import numpy as np
from scipy.special import hyp2f1

from model import (
    ForwardVarianceModel,
    require_params,
    validate_interval,
    validate_positive,
)


class RoughBergomiModel(ForwardVarianceModel):
    """
    Rough Bergomi model.
    """

    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray],
        s0: float = 1.0,
    ) -> None:
        """
        Initialize Rough Bergomi model.

        Parameters
        ----------
        xi0 : callable
            Initial forward variance curve function.
        params : dict
            Dictionary containing model parameters:
            - eta: volatility of volatility
            - H: Hurst parameter
            - rho: correlation between Brownian motions, by default -0.5
        s0 : float, optional
            Initial stock price, by default 1.0
        """
        params = {**params, "rho": params.get("rho", -0.5)}
        self.eta, self.H, self.rho = require_params(params, ("eta", "H", "rho"))
        super().__init__(params=params, xi0=xi0, s0=s0)
        self._check_params()
        self.name = "rough_bergomi"

    def _check_params(self):
        """Validate Rough Bergomi parameters."""
        validate_positive("eta", self.eta)
        validate_interval("rho", self.rho, -1, 1)
        validate_interval("H", self.H, 0, 1, closed=False)

    def kernel(self, u, t):
        """
        Compute the rough Bergomi kernel function eta * sqrt(2H) * (u - t)^{H - 0.5}.

        Parameters
        ----------
        u : float or np.ndarray
            Upper time(s) (must satisfy u > t).
        t : float or np.ndarray
            Lower time(s).

        Returns
        -------
        float or np.ndarray
            Value(s) of the kernel.
        """
        return self.eta * np.sqrt(2.0 * self.H) * (u - t) ** (self.H - 0.5)

    def covariance_levy_fbm_vix(self, T, u, v):
        r"""
        Compute the covariance matrix for the VIX integrals of Levy's fractional
        Brownian motion.

        Specifically, computes the covariance:
            Cov(Y_T^u, Y_T^v)
        where
            Y_T^u = sqrt(2H) * \int_0^T (u - s)^{H - 1/2} dW_s,
        for u, v >= T, and W is a standard Brownian motion.

        This is used for simulating the VIX under the rough Bergomi model.

        Parameters
        ----------
        T : float
            Lower limit of the integral (typically the VIX start time).
        u : np.ndarray
            First set of time points (must satisfy u >= T).
        v : np.ndarray
            Second set of time points (must satisfy v >= T).

        Returns
        -------
        np.ndarray
            Covariance matrix evaluated at (u, v), with the same shape as u and v.
        """

        def func(x, y):
            tmp1 = x ** (self.H + 0.5) * hyp2f1(
                0.5 - self.H, 0.5 + self.H, 1.5 + self.H, -x / (y - x)
            )
            tmp2 = (x - T) ** (self.H + 0.5) * hyp2f1(
                0.5 - self.H, 0.5 + self.H, 1.5 + self.H, -(x - T) / (y - x)
            )
            return (
                (y - x) ** (self.H - 0.5)
                * (tmp1 - tmp2)
                * 2.0
                * self.H
                / (self.H + 0.5)
            )

        cov = np.zeros_like(u)
        cov[u == v] = u[u == v] ** (2.0 * self.H) - (u[u == v] - T) ** (2.0 * self.H)
        cov[u < v] = func(u[u < v], v[u < v])
        cov[u > v] = func(v[u > v], u[u > v])
        return np.real(cov)

    ####################################################################################
    # Weak approximation methods for VIX pricing
    ####################################################################################


def get_params_rbergomi(id: int, opt: str = "single") -> tuple:
    """Get Rough Bergomi parameters for a given id."""
    if opt not in ["single", "mixed"]:
        raise ValueError("Invalid opt. Must be 'single' or 'mixed'.")

    def flat_xi0(level):
        def xi0(t):
            return level * np.ones_like(t)

        return xi0

    if opt == "single":
        if id not in [1, 2]:
            raise ValueError("Invalid id for single case. Must be 1 or 2.")

        if id == 1:
            params = {"H": 0.1, "eta": 1.0, "rho": -0.7}
            xi0 = flat_xi0(0.24**2)

        if id == 2:
            params = {"H": 0.23, "eta": 1.02}
            xi0 = flat_xi0(0.24**2)

        params["eta"] /= np.sqrt(2 * params["H"])  # convention is different here
        return params, xi0

    mixed_params = {
        1: (0.24**2, 0.1, 1.4, 0.7, 0.3),
        2: (0.24**2, 0.1, 2.0, 0.2, 0.4),
    }
    if id not in mixed_params:
        raise ValueError("Invalid id for mixed case. Must be 1 or 2.")

    xi0_level, h, eta, eta_2, lbd = mixed_params[id]
    params = {"H": h, "eta": eta / np.sqrt(2 * h)}
    return params, flat_xi0(xi0_level), lbd, eta_2 / np.sqrt(2 * h)
