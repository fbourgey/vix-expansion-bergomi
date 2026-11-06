from collections.abc import Callable

import numpy as np

from model import (
    ForwardVarianceModel,
    require_params,
    validate_interval,
    validate_positive,
)


class OneFactorBergomiModel(ForwardVarianceModel):
    """
    One-factor Bergomi model.
    """

    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray],
        s0: float = 1.0,
    ) -> None:
        """
        Initialize One-factor Bergomi model.

        Parameters
        ----------
        xi0 : callable
            Initial forward variance curve function.
        params : dict
            Dictionary containing model parameters:
            - w: volatility of volatility
            - k: mean reversion
            - rho: correlation between Brownian motions, by default -0.5
        s0 : float, optional
            Initial stock price, by default 1.0
        """
        params = {**params, "rho": params.get("rho", -0.5)}
        self.w, self.k, self.rho = require_params(params, ("w", "k", "rho"))
        super().__init__(params=params, xi0=xi0, s0=s0)
        self._check_params()
        self.name = "one_factor_bergomi"

    def _check_params(self):
        """Validate One-factor Bergomi parameters."""
        validate_positive("w", self.w)
        validate_positive("k", self.k)
        validate_interval("rho", self.rho, -1, 1)

    def kernel(self, u, t):
        """
        Compute the one-factor Bergomi kernel function w * exp(-k * (u - t)).

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
        return self.w * np.exp(-self.k * (u - t))

    def var_x(self, t):
        """
        Compute the variance of the OU factor X_t.

        Parameters
        ----------
        t : float or np.ndarray
            Time point(s).

        Returns
        -------
        float or np.ndarray
            Variance of X_t at time t.
        """
        if self.k == 0:
            return t
        else:
            return (1 - np.exp(-2 * self.k * t)) / (2 * self.k)

    def _f_xi(self, t, u, x):
        """
        Helper function to compute the forward variance at time t for maturity u
        given the OU factor X_t = x where xi_t(u) = xi_0(u) * f^u(t, x).
        """
        _exp = np.exp(-self.k * (u - t))
        return np.exp(self.w * _exp * x - 0.5 * self.w**2 * _exp**2 * self.var_x(t))


def get_params_one_bergomi(id: int, opt: str = "single") -> tuple:
    """Get One-factor Bergomi parameters for a given id."""
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
            params = {"w": 2.0, "k": 0.25}
            xi0 = flat_xi0(0.24**2)

        if id == 2:
            # Guyon, "The VIX Future in Bergomi Models", SIFIN.
            params = {"w": 8.0, "k": 10}
            xi0 = flat_xi0(0.24**2)

        return params, xi0

    mixed_params = {
        1: (0.24**2, 10.0, 2.0, 0.2),
        2: (0.24**2, 0.5, 6.0, 0.3),
        3: (1.445e-2, 6.1970, 0.6586, 0.3021),
        4: (2.065e-2, 5.3118, 0.4301, 0.4790),
        5: (2.533e-2, 4.5273, 0.4238, 0.5497),
        6: (2.862e-2, 3.6860, 0.3226, 0.6426),
    }
    if id not in mixed_params:
        raise ValueError("Invalid id for mixed case. Must be 1, 2, 3, 4, 5 or 6.")

    xi0_level, w_1, w_2, lbd = mixed_params[id]
    return {"w": w_1, "k": 1.0}, flat_xi0(xi0_level), lbd, w_2
