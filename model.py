from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_hermitenorm
from scipy.stats import norm

import utils
from utils_vix import (
    _compute_coeff_mixed_case,
    _deriv_vix_payoff_mixed,
    _hermite_polynomial_weights,
    _inverse_mixture_lognormal,
    _inverse_x_inner_mixed_func,
    _vix_payoff,
)


def require_params(params: dict, required: tuple[str, ...]) -> tuple:
    """Return required parameter values or raise a consistent error."""
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}.")
    return tuple(params[p] for p in required)


def validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def validate_positive_n_quad(n_quad: int) -> None:
    if n_quad <= 0:
        raise ValueError("n_quad must be > 0.")


def validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


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
        self.delta_vix = 1.0 / 12.0  # VIX window of 30 days expressed in years
        self.name = name

    def __repr__(self) -> str:
        """Return a string representation of the model with its parameters."""
        msg = f"{self.__class__.__name__} with parameters:\n"
        for key in self.params:
            msg += f"{key}={self.params[key]}, "
        msg += f"s0={self.s0}"
        return msg

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
        return self.__class__(s0=self.s0, xi0=self.xi0, params=params)

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

    def mean_proxy(self, T, n_quad=30, quad_scipy=True):
        r"""
        Compute the mean of the VIX proxy (log-variance process) at maturity T.

        The mean is defined as:
        log(F_{VIX^2}) - 1/2 * \int_0^T {
            1/delta * \int_{T}^{T+delta} \xi_0(u) * kernel(u, t)^2 du / F_{VIX^2}
        } dt

        where delta is the VIX window (30/365), xi0(u) is the forward variance curve,
        kernel(u, t) is the rough Bergomi kernel, and F_{VIX^2} is the VIX^2 futures
        price, i.e., F_{VIX^2} = E[VIX_T^2].

        Parameters
        ----------
        T : float
            Maturity.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 30.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. Default is True.

        Returns
        -------
        float
            Mean of the VIX proxy at maturity T.
        """
        if quad_scipy:

            def integrand(t):
                integral = quad(
                    lambda u: self.xi0(u) * self.kernel(u, t) ** 2,
                    T,
                    T + self.delta_vix,
                )
                return integral[0] / self.delta_vix

            out = -0.5 * quad(integrand, 0, T)[0] / self.fut_vix2(T)
            return out + np.log(self.fut_vix2(T))

        return self._mean_proxy_from_grid(self._proxy_grid_data(T, n_quad))

    def var_proxy(self, T, n_quad=30, quad_scipy: bool = True):
        r"""
        Compute the variance of the VIX proxy at maturity T.

        The variance is defined as:
        \int_0^T {
            1/delta * \int_{T}^{T+delta} \xi_0(u) * kernel(u, t)^2 du / F_{VIX^2}
        }^2 dt

        where delta is the VIX window (30/365), xi0(u) is the forward variance curve,
        kernel(u, t) is the rough Bergomi kernel, and F_{VIX^2} is the VIX^2 futures
        price.

        Parameters
        ----------
        T : float
            Maturity.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 30.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. Default is True.

        Returns
        -------
        float
            Variance of the VIX proxy at maturity T.
        """
        if quad_scipy:
            fvix2 = self.fut_vix2(T)

            def integrand(t):
                integral = (
                    quad(
                        lambda u: self.xi0(u) * self.kernel(u, t),
                        T,
                        T + self.delta_vix,
                    )[0]
                    / fvix2
                )
                return (integral / self.delta_vix) ** 2

            return quad(integrand, 0, T)[0]

        return self._var_proxy_from_grid(self._proxy_grid_data(T, n_quad))

    def integral_kernel(self, t, T):
        r"""
        Compute the normalized integral of the kernel over the VIX proxy time interval.

        Specifically, computes:
            (1/delta) * \int_{T}^{T+delta} xi0(u) * kernel(u, t) du / F_{VIX^2}

        This is used in the calculation of gamma coefficients for the VIX proxy
        expansion.

        Parameters
        ----------
        t : float
            Lower time of the kernel.
        T : float
            Start of the VIX window.

        Returns
        -------
        float
            Value of the normalized kernel integral.
        """
        return (
            quad(lambda u: self.xi0(u) * self.kernel(u, t), T, T + self.delta_vix)[0]
            / self.delta_vix
        ) / self.fut_vix2(T)

    def integral_kernel_squared(self, t, T):
        r"""
        Compute the normalized integral of the squared kernel over the VIX proxy time
        interval.

        Specifically, computes:
            (1/delta) * \int_{T}^{T+delta} xi0(u) * kernel(u, t)^2 du / F_{VIX^2}

        This is used in the calculation of gamma coefficients for the VIX proxy
        expansion.

        Parameters
        ----------
        t : float
            Lower time of the kernel.
        T : float
            Start of the VIX window.

        Returns
        -------
        float
            Value of the normalized squared kernel integral.
        """
        return (
            quad(lambda u: self.xi0(u) * self.kernel(u, t) ** 2, T, T + self.delta_vix)[
                0
            ]
            / self.delta_vix
        ) / self.fut_vix2(T)

    def _validate_proxy_inputs(self, T: float, n_quad: int) -> None:
        """Validate shared inputs for quadrature-based proxy computations."""
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

    def _proxy_grid_data(self, T: float, n_quad: int) -> dict[str, np.ndarray | float]:
        """Precompute quadrature data shared by proxy and gamma computations."""
        self._validate_proxy_inputs(T, n_quad)

        v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)
        t_nodes = T * v_nodes
        u_nodes = T + self.delta_vix * v_nodes
        fvix2 = self.fut_vix2(T)
        xi0_u = self.xi0(u_nodes)
        kernel_ut = self.kernel(u=u_nodes[:, None], t=t_nodes[None, :])
        weighted_xi0 = v_weights * xi0_u / fvix2
        int_kernel = weighted_xi0 @ kernel_ut
        int_kernel_squared = weighted_xi0 @ (kernel_ut**2)

        return {
            "T": T,
            "fvix2": fvix2,
            "v_nodes": v_nodes,
            "v_weights": v_weights,
            "u_nodes": u_nodes,
            "xi0_u": xi0_u,
            "kernel_ut": kernel_ut,
            "int_kernel": int_kernel,
            "int_kernel_squared": int_kernel_squared,
        }

    def _mean_proxy_from_grid(self, grid_data: dict[str, np.ndarray | float]) -> float:
        """Compute the proxy mean from shared quadrature data."""
        v_weights = np.asarray(grid_data["v_weights"])
        int_kernel_squared = np.asarray(grid_data["int_kernel_squared"])
        fvix2 = float(grid_data["fvix2"])
        T = float(grid_data["T"])
        correction = -0.5 * T * np.sum(v_weights * int_kernel_squared)
        return np.log(fvix2) + correction

    def _var_proxy_from_grid(self, grid_data: dict[str, np.ndarray | float]) -> float:
        """Compute the proxy variance from shared quadrature data."""
        T = float(grid_data["T"])
        v_weights = np.asarray(grid_data["v_weights"])
        int_kernel = np.asarray(grid_data["int_kernel"])
        return T * np.sum(v_weights * int_kernel**2)

    def _gamma_1_proxy_from_grid(
        self, grid_data: dict[str, np.ndarray | float]
    ) -> float:
        """Compute the first proxy gamma from shared quadrature data."""
        T = float(grid_data["T"])
        v_weights = np.asarray(grid_data["v_weights"])
        xi0_u = np.asarray(grid_data["xi0_u"])
        kernel_ut = np.asarray(grid_data["kernel_ut"])
        int_kernel = np.asarray(grid_data["int_kernel"])
        int_kernel_squared = np.asarray(grid_data["int_kernel_squared"])
        fvix2 = float(grid_data["fvix2"])

        kernel_sq_avg = np.sum(
            v_weights[None, :] * (kernel_ut**2 - int_kernel_squared[None, :]),
            axis=1,
        )
        kernel_centered_var = np.sum(
            v_weights[None, :] * (kernel_ut - int_kernel[None, :]) ** 2,
            axis=1,
        )
        integrand = 0.125 * T**2 * kernel_sq_avg**2
        integrand += 0.5 * T * kernel_centered_var

        return np.sum(v_weights * xi0_u * integrand) / fvix2

    def _gamma_2_proxy_from_grid(
        self, grid_data: dict[str, np.ndarray | float]
    ) -> float:
        """Compute the second proxy gamma from shared quadrature data."""
        T = float(grid_data["T"])
        v_weights = np.asarray(grid_data["v_weights"])
        xi0_u = np.asarray(grid_data["xi0_u"])
        kernel_ut = np.asarray(grid_data["kernel_ut"])
        int_kernel = np.asarray(grid_data["int_kernel"])
        int_kernel_squared = np.asarray(grid_data["int_kernel_squared"])
        fvix2 = float(grid_data["fvix2"])

        first_term = np.sum(
            v_weights[None, :]
            * int_kernel[None, :]
            * (kernel_ut - int_kernel[None, :]),
            axis=1,
        )
        second_term = np.sum(
            v_weights[None, :] * (kernel_ut**2 - int_kernel_squared[None, :]),
            axis=1,
        )
        integrand = first_term * second_term

        return -0.5 * T**2 * np.sum(v_weights * xi0_u * integrand) / fvix2

    def _gamma_3_proxy_from_grid(
        self, grid_data: dict[str, np.ndarray | float]
    ) -> float:
        """Compute the third proxy gamma from shared quadrature data."""
        T = float(grid_data["T"])
        v_weights = np.asarray(grid_data["v_weights"])
        xi0_u = np.asarray(grid_data["xi0_u"])
        kernel_ut = np.asarray(grid_data["kernel_ut"])
        int_kernel = np.asarray(grid_data["int_kernel"])
        fvix2 = float(grid_data["fvix2"])

        kernel_centered_avg = np.sum(
            v_weights[None, :]
            * int_kernel[None, :]
            * (kernel_ut - int_kernel[None, :]),
            axis=1,
        )
        integrand = kernel_centered_avg**2

        return 0.5 * T**2 * np.sum(v_weights * xi0_u * integrand) / fvix2

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

    def gamma_1_proxy(self, T, n_quad=30):
        """
        Compute the first-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 30.

        Returns
        -------
        float
            First-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        return self._gamma_1_proxy_from_grid(self._proxy_grid_data(T, n_quad))

    def gamma_2_proxy(self, T, n_quad=30):
        """
        Compute the second-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 30.

        Returns
        -------
        float
            Second-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        return self._gamma_2_proxy_from_grid(self._proxy_grid_data(T, n_quad))

    def gamma_3_proxy(self, T, n_quad=30):
        """
        Compute the third-order gamma coefficient of the VIX proxy using numerical
        quadrature.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_quad : int, optional
            Number of quadrature points for numerical integration. Default is 30.

        Returns
        -------
        float
            Third-order gamma coefficient.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        return self._gamma_3_proxy_from_grid(self._proxy_grid_data(T, n_quad))

    def price_vix_approx(
        self,
        k,
        T,
        opttype=1,
        order=3,
        return_fut=False,
        meanp=None,
        tot_varp=None,
        n_quad=30,
    ):
        """
        Approximate the price of a VIX option using a proxy expansion.

        Parameters
        ----------
        k : float
            Log-moneyness (typically 0 for ATM).
        T : float
            Maturity of the VIX option.
        opttype : int, optional
            Option type: 1 for call, -1 for put. Default is 1 (call).
        order : int, optional
            Order of the expansion (0, 1, 2, or 3). Default is 3.
        return_fut : bool, optional
            If True, return the proxy for the VIX future instead of the option price.
            Default is False.
        meanp : float, optional
            Mean parameter for the proxy expansion. If None, it will be computed
            internally.
        tot_varp : float, optional
            Total variance parameter for the proxy expansion. If None, it will be
            computed internally.

        Returns
        -------
        float
            Approximated VIX option price (or VIX future if return_fut is True).

        Raises
        ------
        ValueError
            If order or opttype is invalid, or if T <= 0.
        """
        state = self._proxy_approx_state(
            T,
            order,
            meanp=meanp,
            tot_varp=tot_varp,
            n_quad=n_quad,
        )
        return self._price_vix_approx_from_state(
            k,
            opttype,
            state,
            return_fut=return_fut,
        )

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
        return utils.black_impvol(K=K, T=T, F=F, value=otm_price, opttype=opttype)

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

    def _validate_mixed_params(self, T, lbd, volvol_2, opt_payoff, order, n_quad):
        """Validate parameters for price_vix_approx_mixed."""
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if not 0.0 <= lbd <= 1.0:
            raise ValueError("lbd must be in the interval [0, 1].")
        if volvol_2 <= 0.0:
            raise ValueError("volvol_2 must be positive.")
        if opt_payoff not in ["fut", "call", "put"]:
            raise ValueError("opt_payoff must be one of 'fut', 'call', or 'put'.")
        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")
        if n_quad is not None and n_quad <= 0:
            raise ValueError("n_quad must be a positive integer or None.")

    def _get_params_mixed(self, T, lbd, volvol_2, order, n_quad=30):
        """Compute parameters for price_vix_approx_mixed."""
        # Create model with vol-of-vol=volvol_2
        if self.name == "rough_bergomi":
            model_2 = self._clone_with_params(eta=volvol_2)
        elif self.name == "one_factor_bergomi":
            model_2 = self._clone_with_params(w=volvol_2)
        else:
            raise ValueError(
                "model name not one of 'rough_bergomi' or 'one_factor_bergomi'."
            )

        # Parameters common to both regimes
        fvix2 = self.fut_vix2(T)

        proxy_n_quad = 30 if n_quad is None else n_quad
        state_1 = self._proxy_approx_state(T, 3, n_quad=proxy_n_quad)
        state_2 = model_2._proxy_approx_state(T, 3, n_quad=proxy_n_quad)
        mean_1 = state_1["meanp"]
        mean_2 = state_2["meanp"]
        sig_1 = state_1["tot_varp"] ** 0.5
        sig_2 = state_2["tot_varp"] ** 0.5
        volvol_1 = (
            self.params["eta"] if self.name == "rough_bergomi" else self.params["w"]
        )

        # Initialize params dictionary
        params = {
            "fvix2": fvix2,
            "volvol_1": volvol_1,
            "volvol_2": volvol_2,
            "lbd": lbd,
            "meanp_1": mean_1,
            "meanp_2": mean_2,
            "sigp_1": sig_1,
            "sigp_2": sig_2,
            "gamma_1": (state_1["gamma_1"], state_2["gamma_1"]),
            "gamma_2": (state_1["gamma_2"], state_2["gamma_2"]),
            "gamma_3": (state_1["gamma_3"], state_2["gamma_3"]),
        }

        return params

    def price_vix_approx_mixed(
        self,
        T: float,
        lbd: float,
        volvol_2: float,
        opt_payoff: str,
        order: int,
        n_quad: int | None = 50,
        K: float = 0.0,
        n_trunc_herm: int = 0,
    ):
        """
        Price a VIX option in the mixed case using the weak approximation.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        lbd : float
            Mixing parameter between the two regimes.
        volvol_2 : float
            Volatility of volatility parameter for the second regime.
        opt_payoff : str
            Payoff function of the option, e.g., "call" for a call option. Use "put"
            for a put option, or "fut" for a future payoff.
        order : int
            Order of the approximation expansion.
        n_quad : int | None = 50
            Number of quadrature points for numerical integration. If n_quad is None,
            scipy integrate.quad is used.
        K : float, optional (default is 0.0)
            Strike of the VIX option.
        n_trunc_herm : int optional (default is 0)
            If greater than 0, use Hermite series expansion with truncation order
            n_trunc_herm for the order 0 approximation.

        Returns
        -------
        float
            Approximated price of the VIX option using the mixed method.
        """
        # Validation
        self._validate_mixed_params(T, lbd, volvol_2, opt_payoff, order, n_quad)
        params = self._get_params_mixed(T, lbd, volvol_2, order, n_quad=n_quad)

        return self._price_vix_approx_mixed_from_params(
            params=params,
            opt_payoff=opt_payoff,
            order=order,
            n_quad=n_quad,
            K=K,
            n_trunc_herm=n_trunc_herm,
        )

    def _price_vix_approx_mixed_from_params(
        self,
        params,
        opt_payoff: str,
        order: int,
        n_quad: int | None = 50,
        K: float = 0.0,
        n_trunc_herm: int = 0,
    ):
        """Price a mixed VIX payoff from precomputed mixed-model parameters."""
        if n_trunc_herm > 0:
            return _price_vix_approx_mixed_series_from_params(
                params=params,
                opt_payoff=opt_payoff,
                order=order,
                n_quad=n_quad,
                n_trunc_herm=n_trunc_herm,
                K=K,
            )

        total_price = 0.0
        for current_order in range(order + 1):
            total_price += _compute_price_mixed(
                n_quad, K, opt_payoff, params, current_order
            )

        return total_price

    def price_vix_approx_mixed_series(
        self, T, lbd, volvol_2, opt_payoff, order, n_quad, n_trunc_herm, K=0.0
    ):
        """
        Approximate the price a VIX option in the mixed case using Hermite
        series expansion.
        """
        params = self._get_params_mixed(T, lbd, volvol_2, order)
        return _price_vix_approx_mixed_series_from_params(
            params=params,
            opt_payoff=opt_payoff,
            order=order,
            n_quad=n_quad,
            n_trunc_herm=n_trunc_herm,
            K=K,
        )

    def implied_vol_vix_approx_mixed(
        self,
        T: float,
        k: float | np.ndarray,
        order: int,
        lbd: float,
        volvol_2: float,
        n_quad: int | None = 50,
        return_opt="impvol",
        n_trunc_herm: int = 0,
    ):
        """
        Compute the implied volatility of a VIX option using a mixed
        approximation method.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        k : float | np.ndarray
            Log-moneyness of the VIX option.
        order : int
            Order of the approximation expansion.
        lbd : float
            Mixing parameter between the two regimes.
        volvol_2 : float
            Volatility of volatility parameter for the second regime.
        n_quad : int | None
            Number of quadrature points for numerical integration.
        return_opt : str, optional
            If 'impvol', return only the implied volatility.
            If 'all', return both the futures price and the implied volatility.
        n_trunc_herm : int, optional (default is 0)
            If greater than 0, use Hermite series expansion with truncation order
            n_trunc_herm for the order 0 approximation.

        Returns
        -------
        float or tuple
            Approximated Black-Scholes implied volatility for the VIX option, or a tuple
            containing the futures price and the implied volatility if return_opt is
            'all'.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        if lbd < 0 or lbd > 1:
            raise ValueError("lbd must be in the interval [0, 1].")

        if volvol_2 <= 0:
            raise ValueError("volvol_2 must be positive.")

        if n_quad is not None and n_quad <= 0:
            raise ValueError("n_quad must be a positive integer or None.")

        if order not in [0, 1, 2, 3]:
            raise ValueError("order must be one of 0, 1, 2, or 3.")

        if return_opt not in ["impvol", "all"]:
            raise ValueError("return_opt must be either 'impvol' or 'all'.")

        params = self._get_params_mixed(T, lbd, volvol_2, order, n_quad=n_quad)
        F = self._price_vix_approx_mixed_from_params(
            params=params,
            opt_payoff="fut",
            order=order,
            n_quad=n_quad,
            n_trunc_herm=n_trunc_herm,
        )
        k = np.atleast_1d(np.asarray(k))
        K = F * np.exp(k)
        opttype = 2 * (K >= F) - 1
        otm_price = np.array(
            [
                self._price_vix_approx_mixed_from_params(
                    params=params,
                    K=K_i,
                    opt_payoff="call" if opttype_i == 1 else "put",
                    order=order,
                    n_quad=n_quad,
                    n_trunc_herm=n_trunc_herm,
                )
                for K_i, opttype_i in zip(K, opttype, strict=True)
            ]
        )
        impvol_approx = utils.black_impvol(
            K=K, T=T, F=F, value=otm_price, opttype=opttype
        )
        if return_opt == "all":
            return F, impvol_approx
        else:
            return impvol_approx

    def implied_vol_vix_lognorm_approx_mixed(
        self, T: float, k: float | np.ndarray, order: int, lbd: float, volvol_2: float
    ):
        """
        Compute the implied volatility of a VIX option approximating the sum of two
        lognormal distributions with a single lognormal distribution in the mixed case.
        """
        if order != 0:
            raise NotImplementedError(
                "Lognormal approximation is only implemented for order=0."
            )
        params = self._get_params_mixed(T, lbd, volvol_2, order)
        lbd = params["lbd"]
        meanp_1 = params["meanp_1"]
        meanp_2 = params["meanp_2"]
        sigp_1 = params["sigp_1"]
        sigp_2 = params["sigp_2"]
        params_lognorm = utils.sum_lognorm_single_lognorm_approx(
            lbd=lbd,
            mu_1=meanp_1,
            mu_2=meanp_2,
            sig_1=sigp_1,
            sig_2=sigp_2,
        )
        meanp = params_lognorm["mu_y"]
        tot_varp = params_lognorm["sig_y"] ** 2
        return self.implied_vol_vix_approx(
            T=T,
            k=k,
            order=order,
            meanp=meanp,
            tot_varp=tot_varp,
        )

    def implied_vol_vix_expansion_mixed(
        self,
        k,
        T,
        lbd,
        volvol_2,
        j=None,
        theta=None,
        order=3,
        n_quad=30,
        n_trunc_herm=10,
    ):
        """
        Compute VIX implied volatility expansion in the mixed case.

        Parameters
        ----------
        k : float or array_like
            Log-moneyness, defined as log(K / F_P).
        T : float
            Time to maturity (T > 0).
        lbd : float
            Mixing weight in [0, 1].
        volvol_2 : float
            Volatility of volatility parameter for the second regime.
        j : {1, 2} or None, optional
            Component used as the baseline in Theorem 4. If None, use the
            component with the lower proxy volatility.
        theta : float or None, optional
            Auxiliary log-spot/log-strike interpolation parameter in [0, 1]. If None,
            use the paper's optimal theta rule strike by strike.
        order : {3}
            Expansion order. The mixed implied-volatility expansion is implemented
            for the third-order weak price expansion.

        Returns
        -------
        float or ndarray
            Approximated implied volatility, same shape as `k`.

        Raises
        ------
        ValueError
            If inputs are outside their admissible ranges.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if not 0.0 < lbd < 1.0:
            raise ValueError("lbd must be in the interval (0, 1).")
        if volvol_2 <= 0.0:
            raise ValueError("volvol_2 must be positive.")
        if order != 3:
            raise ValueError("order must be 3 for the mixed implied-vol expansion.")
        if theta is not None and not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be in the interval [0, 1].")
        if n_quad <= 0:
            raise ValueError("n_quad must be a positive integer.")
        if n_trunc_herm <= 0:
            raise ValueError("n_trunc_herm must be a positive integer.")

        params = self._get_params_mixed(T, lbd, volvol_2, order, n_quad=n_quad)
        meanp_1 = params["meanp_1"]
        meanp_2 = params["meanp_2"]
        sigp_1 = params["sigp_1"]
        sigp_2 = params["sigp_2"]

        if j is None:
            j = 1 if sigp_1 <= sigp_2 else 2
        if j not in [1, 2]:
            raise ValueError("j must be either 1, 2, or None.")

        lbd_j = lbd if j == 1 else 1.0 - lbd
        meanp_j = meanp_1 if j == 1 else meanp_2
        sigp_j = sigp_1 if j == 1 else sigp_2
        xp_j = 0.5 * meanp_j + 0.125 * sigp_j**2

        c_j, delta_sig_j = _mixed_hermite_factor(params, j)
        weights_herm = _hermite_polynomial_weights(
            n_trunc_herm, c_j, delta_sig_j, n_quad
        )
        weights_herm[1] *= (-1.0) ** j
        weights_herm[3] *= (-1.0) ** j

        c_vec = _compute_coeff_mixed_case_component(params, j)
        fut_hermite = np.sqrt(lbd_j) * np.exp(xp_j) * np.dot(c_vec, weights_herm[:, 0])
        log_fut = np.log(fut_hermite)

        k_input = np.asarray(k, dtype=float)
        scalar_input = k_input.ndim == 0
        k_flat = np.atleast_1d(k_input).ravel()
        herm_orders = np.arange(n_trunc_herm)
        log_strike = log_fut + k_flat
        a_root = np.array(
            [
                _inverse_mixture_lognormal(
                    np.exp(2.0 * log_strike_i),
                    lbd,
                    meanp_1,
                    meanp_2,
                    sigp_1,
                    sigp_2,
                )
                for log_strike_i in log_strike
            ]
        )
        b_root = a_root - 0.5 * sigp_j
        delta_j = k_flat - 0.5 * a_root * sigp_j + 0.125 * sigp_j**2
        if theta is None:
            theta_value = np.where(delta_j > 0.0, 1.0, 0.0)
        else:
            theta_value = theta
        x_theta = log_fut + theta_value * delta_j
        correction_weights = c_vec @ weights_herm[:, 1:]
        correction = correction_weights @ eval_hermitenorm(
            herm_orders[:, None],
            b_root[None, :],
        )
        out = 0.5 * sigp_j / np.sqrt(T) + (
            np.sqrt(lbd_j) * np.exp(xp_j) * correction / (np.exp(x_theta) * np.sqrt(T))
        )

        if scalar_input:
            return float(out[0])
        return out.reshape(k_input.shape)


def _mixed_hermite_factor(params, j):
    """Return C_j and Delta sigma_{P,j} from Theorem 4."""
    lbd = params["lbd"]
    meanp_1 = params["meanp_1"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]

    if j == 1:
        delta_sig = sigp_2 - sigp_1
        c_j = ((1.0 - lbd) / lbd) * np.exp(meanp_2 - meanp_1 + 0.5 * sigp_1 * delta_sig)
    elif j == 2:
        delta_sig = sigp_1 - sigp_2
        c_j = (lbd / (1.0 - lbd)) * np.exp(meanp_1 - meanp_2 + 0.5 * sigp_2 * delta_sig)
    else:
        raise ValueError("j must be either 1 or 2.")

    return c_j, delta_sig


def _compute_coeff_mixed_case_component(params, j):
    """Return dimensionless c_{i,j} coefficients from Theorem 4."""
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    gamma_1 = params["gamma_1"]
    gamma_2 = params["gamma_2"]
    gamma_3 = params["gamma_3"]
    g11, g12 = gamma_1
    g21, g22 = gamma_2
    g31, g32 = gamma_3

    if j == 1:
        c0 = 1.0 + 0.5 * g11 + 0.25 * g21 + 0.125 * g31
        c1 = (
            g11
            + (1.0 - sigp_2 / (2.0 * sigp_1)) * g21
            + (0.75 - sigp_2 / (2.0 * sigp_1)) * g31
            - g12
            - sigp_1 * g22 / (2.0 * sigp_2)
            - sigp_1**2 * g32 / (4.0 * sigp_2**2)
        )
        c2 = (
            (1.0 - sigp_2 / sigp_1) * (g21 + g31)
            + 0.5 * (1.0 - sigp_2 / sigp_1) ** 2 * g31
            + (1.0 - sigp_1 / sigp_2) * (g22 + sigp_1 * g32 / sigp_2)
        )
    elif j == 2:
        c0 = 1.0 + 0.5 * g12 + 0.25 * g22 + 0.125 * g32
        c1 = (
            g11
            + sigp_2 * g21 / (2.0 * sigp_1)
            + sigp_2**2 * g31 / (4.0 * sigp_1**2)
            - g12
            - (1.0 - sigp_1 / (2.0 * sigp_2)) * g22
            - (0.75 - sigp_1 / (2.0 * sigp_2)) * g32
        )
        c2 = (
            (1.0 - sigp_2 / sigp_1) * (g21 + sigp_2 * g31 / sigp_1)
            + (1.0 - sigp_1 / sigp_2) * (g22 + g32)
            + 0.5 * (1.0 - sigp_1 / sigp_2) ** 2 * g32
        )
    else:
        raise ValueError("j must be either 1 or 2.")

    c3 = (1.0 - sigp_2 / sigp_1) ** 2 * g31
    c3 -= (1.0 - sigp_1 / sigp_2) ** 2 * g32
    return np.array([c0, c1, c2, c3])


def _compute_price_0_mixed(n_quad, K, opt_payoff, params):
    """Compute order 0 vix option price in the mixed case."""
    lbd = params["lbd"]
    meanp_1 = params["meanp_1"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    payoff = _vix_payoff(opt_payoff, K)

    if n_quad is None:
        price_0 = quad(
            lambda x: payoff(
                lbd * np.exp(meanp_1 + sigp_1 * norm.ppf(x))
                + (1.0 - lbd) * np.exp(meanp_2 + sigp_2 * norm.ppf(x))
            ),
            0,
            1,
        )[0]
    else:
        nodes, weights = _get_nodes_weights(n_quad, K, opt_payoff, params, order=0)
        # order 0
        price_0 = np.sum(
            weights
            * payoff(
                lbd * np.exp(meanp_1 + sigp_1 * nodes)
                + (1.0 - lbd) * np.exp(meanp_2 + sigp_2 * nodes)
            )
        )

    return price_0


def _price_vix_approx_mixed_series_from_params(
    params,
    opt_payoff: str,
    order: int,
    n_quad: int | None,
    n_trunc_herm: int,
    K: float = 0.0,
):
    """Approximate a mixed VIX payoff using precomputed mixed-model parameters."""
    meanp_1 = params["meanp_1"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    lbd = params["lbd"]

    if order not in [0, 3]:
        raise ValueError("Hermite series expansion only implemented for order 0 and 3.")

    a = (1 - lbd) ** 0.5 * np.exp(meanp_2 / 2 + sigp_2**2 / 8)
    b = (lbd / (1 - lbd)) * np.exp(meanp_1 - meanp_2 + (sigp_1 - sigp_2) * sigp_2 / 2)
    weights_herm = _hermite_polynomial_weights(n_trunc_herm, b, sigp_1 - sigp_2, n_quad)
    c_vec = _compute_coeff_mixed_case(params)

    if opt_payoff == "fut":
        if order == 0:
            return a * weights_herm[0, 0]
        return np.dot(c_vec, weights_herm[:, 0])

    A = _inverse_mixture_lognormal(K**2, lbd, meanp_1, meanp_2, sigp_1, sigp_2)
    B = A - sigp_2 / 2
    i_n_vec = np.array(
        [
            weights_herm[current_order, 0] * norm.cdf(-B)
            + np.sum(
                weights_herm[current_order, 1:]
                * eval_hermitenorm(np.arange(n_trunc_herm), B)
                * norm.pdf(B)
            )
            for current_order in [0, 1, 2, 3]
        ]
    )

    if order == 0:
        if opt_payoff == "call":
            return a * i_n_vec[0] - K * norm.cdf(-A)
        return K * norm.cdf(A) - a * (weights_herm[0, 0] - i_n_vec[0])

    i_n_tot = np.dot(c_vec, i_n_vec)
    if opt_payoff == "call":
        return i_n_tot - K * norm.cdf(-A)
    return K * norm.cdf(A) - (a * weights_herm[0, 0] - i_n_tot)


def _compute_price_mixed(n_quad, K, opt_payoff, params, order):
    """Compute order 1, 2, or 3 price contribution for vix option in the mixed case."""
    if order not in [0, 1, 2, 3]:
        raise ValueError("order must be one of 0, 1, 2, or 3.")

    if order == 0:
        return _compute_price_0_mixed(n_quad, K, opt_payoff, params)

    # Unpack parameters
    lbd = params["lbd"]
    meanp_1 = params["meanp_1"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    volvol_1 = params["volvol_1"]
    volvol_2 = params["volvol_2"]
    fvix2 = params["fvix2"]
    # Gamma values for this order
    gammas = params[f"gamma_{order}"]
    # Derivative of payoff function
    dpayoff_mixed_dy = _deriv_vix_payoff_mixed(opt_payoff, K)

    def _func_psi(x, idx=1):
        """Compute psi function for given index (1 or 2)."""
        sig_idx = sigp_1 if idx == 1 else sigp_2
        mean_idx = meanp_1 - np.log(fvix2) if idx == 1 else meanp_2 - np.log(fvix2)
        return dpayoff_mixed_dy(
            x=mean_idx + sig_idx * x,
            lbd=lbd if idx == 1 else 1 - lbd,
            volvol_1=volvol_1 if idx == 1 else volvol_2,
            volvol_2=volvol_2 if idx == 1 else volvol_1,
            fvix2=fvix2,
            mu_2=meanp_2 if idx == 1 else meanp_1,
        ) / (sig_idx ** (order - 1))

    def _get_order_weight(x, order):
        """Get the order-dependent weight function."""
        if order == 1:
            return 1.0
        elif order == 2:
            return x
        else:  # order == 3
            return x**2 - 1

    def psi_1(x):
        return _func_psi(x, idx=1)

    def psi_2(x):
        return _func_psi(x, idx=2)

    def order_weight(x):
        return _get_order_weight(x, order)

    if n_quad is None:
        # Use scipy.quad for continuous integration
        def integrand_1(u):
            x = norm.ppf(u)
            return order_weight(x) * gammas[0] * psi_1(x)

        def integrand_2(u):
            x = norm.ppf(u)
            return order_weight(x) * gammas[1] * psi_2(x)

        left_1, right_1 = _get_nodes_weights(
            n_quad, K, opt_payoff, params, order, idx=1
        )
        left_2, right_2 = _get_nodes_weights(
            n_quad, K, opt_payoff, params, order, idx=2
        )

        price_1 = quad(integrand_1, left_1, right_1)[0]
        price_2 = quad(integrand_2, left_2, right_2)[0]
    else:
        # Use Gauss quadrature for discrete approximation
        nodes_1, weights_1 = _get_nodes_weights(
            n_quad, K, opt_payoff, params, order, idx=1
        )
        nodes_2, weights_2 = _get_nodes_weights(
            n_quad, K, opt_payoff, params, order, idx=2
        )

        # Compute integrands with order-dependent weights
        weight_fn = order_weight(nodes_1)
        integrand_1 = weight_fn * psi_1(nodes_1)

        weight_fn = order_weight(nodes_2)
        integrand_2 = weight_fn * psi_2(nodes_2)

        price_1 = gammas[0] * np.sum(weights_1 * np.asarray(integrand_1))
        price_2 = gammas[1] * np.sum(weights_2 * np.asarray(integrand_2))

    return price_1 + price_2


def _get_nodes_weights(n_quad, K, opt_payoff, params, order, idx=1):
    """
    Get quadrature nodes and weights for mixed VIX pricing.

    Parameters
    ----------
    n_quad : int
        Quadrature points.
    K : float
        Strike price.
    opt_payoff : str
        Option payoff type.
    params : dict
        Model parameters.
    order : int
        Approximation order.
    idx : int, optional
        Index for psi function, by default 1

    Returns
    -------
    tuple
        Quadrature nodes, weights, left, right.
    """
    lbd = params["lbd"]
    meanp_1 = params["meanp_1"]
    meanp_2 = params["meanp_2"]
    sigp_1 = params["sigp_1"]
    sigp_2 = params["sigp_2"]
    volvol_1 = params["volvol_1"]
    volvol_2 = params["volvol_2"]
    fvix2 = params["fvix2"]

    if opt_payoff == "fut" and n_quad is None:
        left = 0.0
        right = 1.0

    if opt_payoff in ["call", "put"]:
        if order == 0:
            endpoint = _inverse_mixture_lognormal(
                K**2, lbd, meanp_1, meanp_2, sigp_1, sigp_2
            )
        else:
            endpoint_x = _inverse_x_inner_mixed_func(
                z=K**2,
                mu_2=meanp_2 if idx == 1 else meanp_1,
                lbd=lbd if idx == 1 else 1 - lbd,
                volvol_1=volvol_1 if idx == 1 else volvol_2,
                volvol_2=volvol_2 if idx == 1 else volvol_1,
                fvix2=fvix2,
            )
            sig_idx = sigp_1 if idx == 1 else sigp_2
            mean_idx = meanp_1 - np.log(fvix2) if idx == 1 else meanp_2 - np.log(fvix2)
            endpoint = (endpoint_x - mean_idx) / sig_idx

        left = norm.cdf(endpoint) if opt_payoff == "call" else 0.0
        right = 1.0 if opt_payoff == "call" else norm.cdf(endpoint)

    if n_quad is None:
        return left, right

    if opt_payoff == "fut":
        # Gauss-Hermite quadrature for future payoff
        nodes, weights = utils.gauss_hermite(n_quad)
    else:
        # Gauss-Legendre quadrature for call/put payoff
        nodes, weights = utils.gauss_legendre(float(left), float(right), n_quad)
        nodes = norm.ppf(nodes)

    return nodes, weights
