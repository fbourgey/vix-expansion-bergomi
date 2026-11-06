from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

import utils

# Grading power of the coefficient grid near the kernel corner u = t = T.
COEFFICIENT_GRADING = 4


def require_params(params: dict, required: tuple[str, ...]) -> tuple:
    """Return required parameter values or raise a consistent error."""
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}.")
    return tuple(params[p] for p in required)


def validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def validate_interval(
    name: str,
    value: float,
    lower: float,
    upper: float,
    *,
    closed: bool = True,
) -> None:
    if closed:
        valid = lower <= value <= upper
        bounds = f"[{lower}, {upper}]"
    else:
        valid = lower < value < upper
        bounds = f"({lower}, {upper})"
    if not valid:
        raise ValueError(f"{name} must be in {bounds}.")


class ForwardVarianceModel(ABC):
    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray] = lambda t: 0.2**2 * np.ones_like(t),
        s0: float = 1.0,
        name: str = "forward_variance_model",
    ) -> None:
        """
        Initialize ForwardVarianceModel.

        Parameters
        ----------
        params : dict
            Model-specific parameters.
        xi0 : callable, optional
            Forward variance curve function xi0(t), must return positive values for
            t >= 0.
        s0 : float
            Initial spot price (must be positive).
        """
        if s0 <= 0.0:
            raise ValueError("Initial spot price s0 must be positive.")

        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")

        self.xi0 = xi0
        self._is_xi0_positive()
        self.xi0_0 = self.xi0(np.zeros(1))[0]
        self.xi0_flat = self._is_xi0_flat()
        self.params = params
        self.s0 = s0
        self.delta_vix = 30.0 / 365.0  # VIX window of 30 days expressed in years
        self.name = name

    @abstractmethod
    def kernel(self, u, t) -> float | np.ndarray:
        """
        Compute the model-specific kernel function.

        Parameters
        ----------
        u : float or np.ndarray
            Upper time(s) (must satisfy u >= t).
        t : float or np.ndarray
            Lower time(s).

        Returns
        -------
        float or np.ndarray
            Value(s) of the kernel function evaluated at (u, t).
        """
        pass

    def _clone_with_params(self, **updates):
        """Return a new model instance with updated params."""
        params = self.params.copy()
        params.update(updates)
        clone = self.__class__(s0=self.s0, xi0=self.xi0, params=params)
        clone.delta_vix = self.delta_vix
        return clone

    def _is_xi0_flat(self) -> bool:
        """Check if the forward variance curve xi0 is flat."""
        t_test = np.linspace(1e-10, 10, 1000)
        return np.allclose(self.xi0(t_test), self.xi0_0)

    def _is_xi0_positive(self):
        """Check if the forward variance curve xi0 is positive for t >= 0."""
        t_test = np.linspace(1e-10, 10, 1000)
        if not np.all(self.xi0(t_test) > np.array([0.0])):
            raise ValueError("xi0 must be positive for all t >= 0.")

    def fut_vix2(self, T: float) -> float:
        r"""
        Compute the fair value of a VIX squared futures contract at maturity T. It
        corresponds to:

            E[VIX_T^2] = 1/delta \int_{T}^{T+delta} \xi_0^u du

        where delta is the VIX window (30/365 years by default) and xi0(u) is the
        forward variance curve.

        Parameters
        ----------
        T : float
            Maturity of the VIX future (must be non-negative).

        Returns
        -------
        float
            Fair value of the VIX squared futures contract at time T.

        Raises
        ------
        ValueError
            If T is negative.

        Notes
        -----
        This is a model-free quantity, depending only on the forward variance curve.
        """
        if T < 0:
            raise ValueError("Maturity T must be non-negative.")

        integral, _ = quad(lambda u: self.xi0(u), T, T + self.delta_vix)
        return integral / self.delta_vix

    ####################################################################################
    # Weak approximation methods for VIX pricing
    ####################################################################################

    def _validate_proxy_inputs(self, T: float, n_quad: int) -> None:
        """Validate shared inputs for quadrature-based proxy computations."""
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

    def _proxy_grid_data(self, T: float, n_quad: int) -> dict[str, np.ndarray | float]:
        """Precompute quadrature data shared by proxy and gamma computations.

        Gauss--Legendre nodes s in [0, 1] are graded towards the kernel corner
        u = t = T through t = T*(1-(1-s)**p) and u = T + delta*s**p, with
        p = COEFFICIENT_GRADING. This resolves the singularity of power-law
        kernels and leaves smooth kernels accurate. ``t_weights`` integrate
        over [0, T] divided by T; ``u_weights`` average over [T, T + delta].

        Kernels depend on u - t only. The lag u - t is formed from the graded
        distances to T, which avoids cancellation when nodes approach T.
        """
        self._validate_proxy_inputs(T, n_quad)

        s_nodes, s_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
        p = COEFFICIENT_GRADING
        t_gap = T * (1.0 - s_nodes) ** p
        u_gap = self.delta_vix * s_nodes**p
        t_nodes = T - t_gap
        t_weights = s_weights * p * (1.0 - s_nodes) ** (p - 1)
        u_nodes = T + u_gap
        u_weights = s_weights * p * s_nodes ** (p - 1)
        fvix2 = self.fut_vix2(T)
        xi0_u = self.xi0(u_nodes)
        lag = u_gap[:, None] + t_gap[None, :]
        kernel_ut = self.kernel(u=lag, t=np.zeros_like(lag))
        weighted_xi0 = u_weights * xi0_u / fvix2
        int_kernel = weighted_xi0 @ kernel_ut
        int_kernel_squared = weighted_xi0 @ (kernel_ut**2)

        return {
            "T": T,
            "fvix2": fvix2,
            "t_nodes": t_nodes,
            "t_weights": t_weights,
            "u_nodes": u_nodes,
            "u_weights": u_weights,
            "xi0_u": xi0_u,
            "kernel_ut": kernel_ut,
            "int_kernel": int_kernel,
            "int_kernel_squared": int_kernel_squared,
        }

    def _mean_proxy_from_grid(self, grid_data: dict[str, np.ndarray | float]) -> float:
        """Compute the proxy mean from shared quadrature data."""
        t_weights = np.asarray(grid_data["t_weights"])
        int_kernel_squared = np.asarray(grid_data["int_kernel_squared"])
        fvix2 = float(grid_data["fvix2"])
        T = float(grid_data["T"])
        correction = -0.5 * T * np.sum(t_weights * int_kernel_squared)
        return np.log(fvix2) + correction

    def _var_proxy_from_grid(self, grid_data: dict[str, np.ndarray | float]) -> float:
        """Compute the proxy variance from shared quadrature data."""
        T = float(grid_data["T"])
        t_weights = np.asarray(grid_data["t_weights"])
        int_kernel = np.asarray(grid_data["int_kernel"])
        return T * np.sum(t_weights * int_kernel**2)

    def _gamma_1_proxy_from_grid(
        self, grid_data: dict[str, np.ndarray | float]
    ) -> float:
        """Compute the first proxy gamma from shared quadrature data."""
        T = float(grid_data["T"])
        t_weights = np.asarray(grid_data["t_weights"])
        u_weights = np.asarray(grid_data["u_weights"])
        xi0_u = np.asarray(grid_data["xi0_u"])
        kernel_ut = np.asarray(grid_data["kernel_ut"])
        int_kernel = np.asarray(grid_data["int_kernel"])
        int_kernel_squared = np.asarray(grid_data["int_kernel_squared"])
        fvix2 = float(grid_data["fvix2"])

        kernel_sq_avg = np.sum(
            t_weights[None, :] * (kernel_ut**2 - int_kernel_squared[None, :]),
            axis=1,
        )
        kernel_centered_var = np.sum(
            t_weights[None, :] * (kernel_ut - int_kernel[None, :]) ** 2,
            axis=1,
        )
        integrand = 0.125 * T**2 * kernel_sq_avg**2
        integrand += 0.5 * T * kernel_centered_var

        return np.sum(u_weights * xi0_u * integrand) / fvix2

    def _gamma_2_proxy_from_grid(
        self, grid_data: dict[str, np.ndarray | float]
    ) -> float:
        """Compute the second proxy gamma from shared quadrature data."""
        T = float(grid_data["T"])
        t_weights = np.asarray(grid_data["t_weights"])
        u_weights = np.asarray(grid_data["u_weights"])
        xi0_u = np.asarray(grid_data["xi0_u"])
        kernel_ut = np.asarray(grid_data["kernel_ut"])
        int_kernel = np.asarray(grid_data["int_kernel"])
        int_kernel_squared = np.asarray(grid_data["int_kernel_squared"])
        fvix2 = float(grid_data["fvix2"])

        first_term = np.sum(
            t_weights[None, :]
            * int_kernel[None, :]
            * (kernel_ut - int_kernel[None, :]),
            axis=1,
        )
        second_term = np.sum(
            t_weights[None, :] * (kernel_ut**2 - int_kernel_squared[None, :]),
            axis=1,
        )
        integrand = first_term * second_term

        return -0.5 * T**2 * np.sum(u_weights * xi0_u * integrand) / fvix2

    def _gamma_3_proxy_from_grid(
        self, grid_data: dict[str, np.ndarray | float]
    ) -> float:
        """Compute the third proxy gamma from shared quadrature data."""
        T = float(grid_data["T"])
        t_weights = np.asarray(grid_data["t_weights"])
        u_weights = np.asarray(grid_data["u_weights"])
        xi0_u = np.asarray(grid_data["xi0_u"])
        kernel_ut = np.asarray(grid_data["kernel_ut"])
        int_kernel = np.asarray(grid_data["int_kernel"])
        fvix2 = float(grid_data["fvix2"])

        kernel_centered_avg = np.sum(
            t_weights[None, :]
            * int_kernel[None, :]
            * (kernel_ut - int_kernel[None, :]),
            axis=1,
        )
        integrand = kernel_centered_avg**2

        return 0.5 * T**2 * np.sum(u_weights * xi0_u * integrand) / fvix2

    def _proxy_approx_state(
        self,
        T: float,
        order: int,
        *,
        meanp: float | None = None,
        tot_varp: float | None = None,
        n_quad: int = 30,
    ) -> dict[str, float]:
        """Build shared approximation state for proxy pricing methods."""
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        grid_data = self._proxy_grid_data(T, n_quad)
        state = {
            "T": T,
            "order": order,
            "meanp": meanp
            if meanp is not None
            else self._mean_proxy_from_grid(grid_data),
            "tot_varp": tot_varp
            if tot_varp is not None
            else self._var_proxy_from_grid(grid_data),
            "gamma_1": 0.0,
            "gamma_2": 0.0,
            "gamma_3": 0.0,
        }
        if order >= 1:
            state["gamma_1"] = self._gamma_1_proxy_from_grid(grid_data)
        if order >= 2:
            state["gamma_2"] = self._gamma_2_proxy_from_grid(grid_data)
        if order >= 3:
            state["gamma_3"] = self._gamma_3_proxy_from_grid(grid_data)

        meanp_value = float(state["meanp"])
        tot_varp_value = float(state["tot_varp"])
        state["volp"] = np.sqrt(tot_varp_value / T)
        state["spot_proxy"] = np.exp(0.5 * meanp_value + 0.125 * tot_varp_value)
        state["fut_proxy"] = state["spot_proxy"] * (
            1.0
            + 0.5 * state["gamma_1"]
            + 0.25 * state["gamma_2"]
            + 0.125 * state["gamma_3"]
        )
        return state

    def _price_vix_approx_from_state(
        self,
        k: float | np.ndarray,
        opttype: float | np.ndarray,
        state: dict[str, float],
        *,
        return_fut: bool = False,
    ):
        """Price a VIX option or future from shared proxy approximation state."""
        if np.any(np.abs(np.atleast_1d(np.asarray(opttype))) != 1):
            raise ValueError("opttype must be either -1 (put) or 1 (call).")

        F = float(state["fut_proxy"])
        if return_fut:
            return F

        T = float(state["T"])
        volp = float(state["volp"])
        S = float(state["spot_proxy"])
        gamma_1 = float(state["gamma_1"])
        gamma_2 = float(state["gamma_2"])
        gamma_3 = float(state["gamma_3"])
        order = int(state["order"])
        k = np.atleast_1d(np.asarray(k, dtype=float))
        opttype = np.atleast_1d(np.asarray(opttype, dtype=float))
        K = F * np.exp(k)

        price_0 = utils.black_price(K=K, T=T, F=S, vol=0.5 * volp, opttype=opttype)
        if order == 0:
            return price_0

        price_1 = (
            0.5 * S * utils.black_delta(K=K, T=T, F=S, vol=0.5 * volp, opttype=opttype)
        )
        if order == 1:
            return price_0 + gamma_1 * price_1

        price_2 = 0.5 * price_1
        price_2 += 0.25 * S**2 * utils.black_gamma(K=K, T=T, F=S, vol=0.5 * volp)
        if order == 2:
            return price_0 + gamma_1 * price_1 + gamma_2 * price_2

        price_3 = -0.5 * price_1 + 1.5 * price_2
        price_3 += 0.125 * S**3 * utils.black_speed(K=K, T=T, F=S, vol=0.5 * volp)
        return price_0 + gamma_1 * price_1 + gamma_2 * price_2 + gamma_3 * price_3

    def price_vix_fut_approx(self, T, order=3, meanp=None, tot_varp=None, n_quad=30):
        """
        Approximate the price of a VIX futures contract at maturity T using a proxy
        expansion.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.

        Returns
        -------
        float
            Approximated VIX futures price.

        Raises
        ------
        ValueError
            If order is invalid or T <= 0.
        """
        state = self._proxy_approx_state(
            T,
            order,
            meanp=meanp,
            tot_varp=tot_varp,
            n_quad=n_quad,
        )
        return self._price_vix_approx_from_state(
            0.0,
            1.0,
            state,
            return_fut=True,
        )

    def implied_vol_vix_approx(
        self, T, k, order=3, meanp=None, tot_varp=None, n_quad=30
    ):
        """
        Approximate the implied volatility of a VIX option at a given log-moneyness
        using the proxy expansion.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        k : float or np.ndarray
            Log-moneyness (typically 0 for ATM). Can be a scalar or array.
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.
        meanp : float or None, optional
            Precomputed mean proxy value. If None, it will be computed internally.
        tot_varp : float or None, optional
            Precomputed total variance proxy value. If None, it will be computed
            internally.

        Returns
        -------
        np.ndarray
            Approximated implied volatility for each log-moneyness value.

        Raises
        ------
        ValueError
            If T <= 0.
        """
        state = self._proxy_approx_state(
            T,
            order,
            meanp=meanp,
            tot_varp=tot_varp,
            n_quad=n_quad,
        )
        k = np.atleast_1d(np.asarray(k))
        F = float(state["fut_proxy"])
        K = F * np.exp(k)
        opttype = 2 * (K >= F) - 1
        otm_price = np.asarray(self._price_vix_approx_from_state(k, opttype, state))
        return utils.black_impvol_brent(K=K, T=T, F=F, value=otm_price, opttype=opttype)

    def implied_vol_vix_expansion(self, k, T, order: int = 0, n_quad: int = 30):
        """
        Compute VIX implied volatility expansion.

        Parameters
        ----------
        k : float or array_like
            Log-moneyness (k = log(K/F)).
        T : float
            Time to maturity (T > 0).
        order : {0, 1, 2}, optional
            Expansion order (default 0).
        n_quad : int, optional
            Number of quadrature points for proxy-state computations. Default is 30.

        Returns
        -------
        float or ndarray
            Approximated implied volatility, same shape as `k`.

        Raises
        ------
        ValueError
            If `order` not in {0,1,2} or if `T <= 0`.
        """
        if order not in [0, 1, 2]:
            raise ValueError("order must be one of 0, 1, or 2.")
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        state_order = 0 if order == 0 else 3
        state = self._proxy_approx_state(T, state_order, n_quad=n_quad)
        meanp = float(state["meanp"])
        tot_varp = float(state["tot_varp"])
        vol_proxy = float(state["volp"])
        gamma_2 = float(state["gamma_2"])
        gamma_3 = float(state["gamma_3"])
        xp = 0.5 * meanp + tot_varp / 8

        k = np.atleast_1d(np.asarray(k, dtype=float))
        log_strike = np.log(float(state["fut_proxy"])) + k

        if order == 0:
            return 0.5 * vol_proxy + 0.0 * log_strike

        return (
            0.5 * vol_proxy
            + gamma_2 / (2 * vol_proxy * T)
            + 3 * gamma_3 / (8 * vol_proxy * T)
            - gamma_3 * (xp - log_strike) / (vol_proxy**3 * T**2)
        )

    ####################################################################################
    # Mixed case approximation methods for VIX pricing
    ####################################################################################
