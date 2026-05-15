from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1

import utils
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

    def cholesky_cov_matrix_vix(
        self, T: float, n_disc: int, return_cov: bool = False
    ) -> np.ndarray:
        r"""
        Compute the lower-triangular Cholesky factor of the covariance matrix of the
        Gaussian vector (Y_{T}^{u_i}) for 0 <= i <= n, where u_i = T + delta * i / n
        and delta = 30 / 365 (30 days in years).

        Here, W is a standard Brownian motion and
        Y_T^u = \sqrt{2H} \int_0^T (u-s)^{H-1/2} dW_s, for any u >= T.

        Parameters
        ----------
        T : float
            Maturity of interest.
        n_disc : int
            Number of time discretization steps (must be positive).
        return_cov : bool, optional
            If True, return the full covariance matrix instead of its Cholesky factor.
            Default is False.

        Returns
        -------
        np.ndarray
            Lower-triangular Cholesky factor of the covariance matrix, or the covariance
            matrix itself if `return_cov` is True.
        """
        tab_u = np.linspace(T, T + self.delta_vix, n_disc + 1)
        u = np.tile(tab_u, (n_disc + 1, 1)).T
        v = u.T

        cov = self.covariance_levy_fbm_vix(T=T, u=u, v=v)

        if return_cov:
            return cov
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = utils.cholesky_from_svd(cov)
        except Exception as e:
            print(f"Error in Cholesky decomposition: {e}")
            raise

        return chol

    def simulate_vix(
        self,
        T: float,
        n_mc: int,
        n_disc: int,
        rule="trap",
        seed=None,
        control_variate: bool = False,
        lbd: float | None = None,
        eta_2: float | None = None,
        return_xi: bool = False,
    ):
        """
        Simulate sample paths of the VIX.

        Parameters
        ----------
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'left'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        control_variate : bool, optional
            If True, use control variate technique to reduce variance of the
            simulation. Default is False.
        lbd : float or None, optional
            If provided, use a mixed model with two different eta values.
            lbd is the weight for the first eta value.
        eta_2 : float or None, optional
            If provided, use a mixed model with two different eta values.
            This is the second eta value. Must be provided if `lbd` is not None.
        return_xi : bool, optional
            If True, return the simulated xi values instead of VIX values.
            Default is False.
        Returns
        -------
        np.ndarray
            Simulated VIX values (shape: n_mc,) or simulated xi values if
            `return_xi` is True (shape: n_disc + 1, n_mc).
        """
        if rule not in ["left", "right", "trap"]:
            raise ValueError("rule must be one of 'left', 'right', or 'trap'.")

        tab_u = np.linspace(T, T + self.delta_vix, n_disc + 1)
        n_disc = tab_u.shape[0] - 1
        rng = np.random.default_rng(seed)
        y = self.cholesky_cov_matrix_vix(T, n_disc) @ rng.standard_normal(
            (n_disc + 1, n_mc)
        )
        var_y = tab_u ** (2.0 * self.H) - (tab_u - T) ** (2.0 * self.H)

        is_mixed = lbd is not None and eta_2 is not None
        if is_mixed:
            assert eta_2 is not None and lbd is not None
            exp_1 = np.exp(self.eta * y - 0.5 * self.eta**2 * var_y[:, None])
            eta_2 = float(eta_2)
            exp_2 = np.exp(eta_2 * y - 0.5 * eta_2**2 * var_y[:, None])
            xi = self.xi0(tab_u[:, None]) * (lbd * exp_1 + (1.0 - lbd) * exp_2)
        else:
            xi = self.xi0(tab_u[:, None]) * np.exp(
                self.eta * y - 0.5 * self.eta**2 * var_y[:, None]
            )

        if return_xi:
            return xi

        if control_variate:
            if is_mixed:
                raise ValueError("Control variate not implemented for the mixed case.")
            log_cv = (
                np.log(self.xi0(tab_u[:, None]))
                + self.eta * y
                - 0.5 * self.eta**2 * var_y[:, None]
            )

        if rule == "left":
            vix = np.sqrt(xi[:-1, :].mean(axis=0))
            if control_variate:
                cv = np.exp(log_cv[:-1, :].mean(axis=0)) ** 0.5

        elif rule == "right":
            vix = np.sqrt(xi[1:, :].mean(axis=0))
            if control_variate:
                cv = np.exp(log_cv[1:, :].mean(axis=0)) ** 0.5

        else:
            vix = np.sqrt(0.5 * (xi[:-1, :].mean(axis=0) + xi[1:, :].mean(axis=0)))
            if control_variate:
                cv = np.exp(log_cv[:-1, :].mean(axis=0)) ** 0.5
            # if control_variate:
            #     cv_left = np.exp(log_cv[:-1, :].sum(axis=0) / n_disc) ** 0.5
            #     cv_right = np.exp(log_cv[1:, :].sum(axis=0) / n_disc) ** 0.5
            #     cv = (cv_left, cv_right)

        return vix if not control_variate else (vix, cv)

    def implied_vol_vix(
        self, k, T, n_mc, n_disc, rule="trap", seed=None, lbd=None, eta_2=None
    ) -> np.ndarray:
        """
        Compute the implied volatility of a VIX option at a given log-moneyness
        using Monte Carlo simulation.

        Parameters
        ----------
        k : float or np.ndarray
            Log-moneyness of the VIX option (typically 0 for ATM). Can be a scalar
            or array.
        T : float
            Maturity of the VIX option.
        n_mc : int
            Number of Monte Carlo simulation paths.
        n_disc : int
            Number of time discretization steps for the VIX simulation.
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'trap'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        lbd : float or None, optional
            If provided, use a mixed model with two different eta values.
            lbd is the weight for the first eta value.
        eta_2 : float or None, optional
            If provided, use a mixed model with two different eta values.
            This is the second eta value. Must be provided if `lbd` is not None.

        Returns
        -------
        np.ndarray
            Implied volatility values for the VIX option(s) at the specified
            log-moneyness.
        """
        vix = self.simulate_vix(
            T=T, n_mc=n_mc, n_disc=n_disc, rule=rule, seed=seed, lbd=lbd, eta_2=eta_2
        )
        return np.asarray(utils.black_otm_impvol_mc(S=np.asarray(vix), k=k, T=T))

    def price_vix(
        self,
        T: float,
        n_mc: int,
        n_disc: int,
        k: float | np.ndarray = 0.0,
        rule: str = "trap",
        seed: int | None = None,
        opttype: float | np.ndarray = 1.0,
        lbd: float | None = None,
        eta_2: float | None = None,
        return_opt: str = "price",
        control_variate: bool = False,
    ):
        """
        Compute a VIX option price at a given log-moneyness by Monte Carlo simulation.

        Parameters
        ----------
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        k : float or np.ndarray, optional
            Log-moneyness of the VIX option (typically 0 for ATM). Can be a scalar
            or array. Default is 0.0 (ATM).
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'trap'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        opttype : float or np.ndarray, optional
            Option type: 1 for call, -1 for put. Default is 1 (call).
        lbd : float or None, optional
            If provided, use a mixed model with two different eta values.
            lbd is the weight for the first eta value.
        eta_2 : float or None, optional
            If provided, use a mixed model with two different eta values.
            This is the second eta value. Must be provided if `lbd` is not None.
        return_opt : str, optional
            Specifies what to return:
            - 'price': return the option price.
            - 'fut': return the VIX future price.
            - 'both': return both the option price and the VIX future price.
            Default is 'price'.
        control_variate : bool, optional
            If True, use control variate technique to reduce variance. Default is False.

        Returns
        -------
        float or np.ndarray
            Estimated VIX option price.
        """
        if return_opt not in ["price", "fut", "both"]:
            raise ValueError("return_opt must be either 'price' or 'fut' or 'both'.")

        vix = self.simulate_vix(
            T=T,
            n_mc=n_mc,
            n_disc=n_disc,
            rule=rule,
            seed=seed,
            lbd=lbd,
            eta_2=eta_2,
            control_variate=control_variate,
        )
        if control_variate:
            vix_mc, vix_mc_cv = vix
            F = np.mean(vix_mc - vix_mc_cv) + self.price_vix_control_variate(
                T=T, n_disc=n_disc, rule=rule, return_fut=True
            )
        else:
            vix_mc = np.asarray(vix)
            F = np.mean(vix_mc)

        if return_opt == "fut":
            return F

        k = np.atleast_1d(np.asarray(k))
        opttype = np.atleast_1d(np.asarray(opttype))
        K = F * np.exp(k)
        payoff = np.maximum(opttype[None, :] * (vix_mc[:, None] - K[None, :]), 0.0)
        if control_variate:
            payoff_cv = np.maximum(
                opttype[None, :] * (vix_mc_cv[:, None] - K[None, :]), 0.0
            )
            price = np.mean(
                payoff - payoff_cv, axis=0
            ) + self.price_vix_control_variate(T=T, k=k, n_disc=n_disc, rule=rule)
        else:
            price = np.mean(payoff, axis=0)

        return price if return_opt == "price" else (price, F)

    def price_vix_fut(
        self,
        T,
        n_mc,
        n_disc,
        rule="trap",
        seed=None,
        control_variate: bool = False,
        lbd=None,
        eta_2=None,
    ):
        """
        Estimate the price of a VIX futures contract at maturity T using Monte Carlo
        simulation.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        n_mc : int
            Number of Monte Carlo simulation paths.
        n_disc : int
            Number of time discretization steps for the VIX simulation.
        rule : str, optional
            Integration rule for the VIX simulation ('left', 'right', or 'trap').
            Default is 'trap'.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        control_variate : bool, optional
            If True, use control variate technique to reduce variance.
            Default is False.

        Returns
        -------
        float
            Estimated VIX futures price at maturity T.
        """
        return self.price_vix(
            T=T,
            n_mc=n_mc,
            n_disc=n_disc,
            rule=rule,
            seed=seed,
            lbd=lbd,
            eta_2=eta_2,
            return_opt="fut",
            control_variate=control_variate,
        )

    def price_vix_control_variate(
        self,
        T: float,
        n_disc: int,
        k: float | np.ndarray = 0.0,
        rule: str = "left",
        opttype: float | np.ndarray = 1.0,
        return_fut: bool = False,
    ):
        """
        Price a VIX option using a control variate.

        This method computes the price of a VIX option with maturity `T` and strike `K`
        using a control variate technique to reduce variance in the Monte Carlo
        estimator. The control variate is typically chosen as an analytically tractable
        approximation or proxy for the VIX payoff.

        Parameters
        ----------
        T : float
            Maturity of the VIX option.
        k : float or np.ndarray, optional
            Log-moneyness (typically 0 for ATM).
        n_disc : int
            Number of time discretization steps for the simulation.
        rule : str, optional
            Numerical integration rule for the VIX calculation ('left', 'trap', etc.).
            Default is 'left'.
        opttype : int, optional
            Option type: 1 for call, -1 for put. Default is 1 (call).
        return_fut : bool, optional
            If True, return the proxy for the VIX future instead of the option price.

        Returns
        -------
        float
            Estimated price of the VIX option using the control variate method.
        """
        if T < 0:
            raise ValueError("Maturity T must be non-negative.")
        if rule not in ["left", "right", "trap"]:
            raise ValueError("rule must be one of 'left', 'right', or 'trap'.")
        opttype = np.atleast_1d(np.asarray(opttype))
        if np.any(np.abs(opttype) != 1):
            raise ValueError("opttype must be either -1 (put) or 1 (call).")
        n_disc = int(n_disc)
        if n_disc <= 0:
            raise ValueError("n_disc must be a positive integer.")

        tab_u = np.linspace(T, T + self.delta_vix, n_disc + 1)
        cov_matrix = self.cholesky_cov_matrix_vix(T, n_disc, return_cov=True)
        cov_matrix *= self.eta**2
        mean_vec = np.log(self.xi0(tab_u)) - 0.5 * self.eta**2 * (
            tab_u ** (2.0 * self.H) - (tab_u - T) ** (2.0 * self.H)
        )
        if rule == "left":
            mean = np.mean(mean_vec[:-1])
            std = (np.sum(cov_matrix[:-1, :-1]) / n_disc**2) ** 0.5
        elif rule == "right":
            mean = np.mean(mean_vec[1:])
            std = (np.sum(cov_matrix[1:, 1:]) / n_disc**2) ** 0.5
        else:
            mean = np.mean(mean_vec)
            std = np.mean(cov_matrix.flatten()) ** 0.5

        F_cv = np.exp(0.5 * mean + 0.5 * (0.5 * std) ** 2)

        if return_fut:
            return F_cv

        k = np.atleast_1d(np.asarray(k))
        K = F_cv * np.exp(k)

        return utils.black_price(
            K=K, T=T, F=F_cv, vol=0.5 * std / T**0.5, opttype=opttype
        )

    ####################################################################################
    # Weak approximation methods for VIX pricing
    ####################################################################################

    def mean_proxy_flat(self, T):
        """
        Compute the mean of the VIX proxy when the forward variance curve xi0 is flat.

        Parameters
        ----------
        T : float
            Maturity.

        Returns
        -------
        float
            Mean of the VIX proxy at maturity T for flat xi0.
        """
        mean = (
            (T + self.delta_vix) ** (2.0 * self.H + 1.0)
            - self.delta_vix ** (2.0 * self.H + 1.0)
            - T ** (2.0 * self.H + 1.0)
        )
        mean *= -(self.eta**2) / (2.0 * self.delta_vix * (2.0 * self.H + 1.0))
        return mean + np.log(self.xi0_0)

    def var_proxy_flat(self, T):
        """
        Compute the variance of the VIX proxy when the forward variance curve xi0 is
        flat.

        Parameters
        ----------
        T : float
            Maturity.

        Returns
        -------
        float
            Variance of the VIX proxy at maturity T for flat xi0.
        """
        var = (
            (T + self.delta_vix) ** (2.0 * self.H + 2.0)
            + T ** (2 * self.H + 2)
            - self.delta_vix ** (2 * self.H + 2)
        ) / (2 * self.H + 2)
        var -= (
            2
            * self.delta_vix ** (self.H + 0.5)
            * T ** (self.H + 1.5)
            * hyp2f1(-self.H - 0.5, self.H + 1.5, self.H + 2.5, -T / self.delta_vix)
            / (self.H + 1.5)
        )
        var *= (
            self.eta * np.sqrt(2.0 * self.H) / (self.delta_vix * (self.H + 0.5))
        ) ** 2

        return var

    def gamma_1_proxy_flat(self, T):
        """
        Compute the first-order gamma coefficient of the VIX proxy when xi0 is flat.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.

        Returns
        -------
        float
            First-order gamma coefficient for flat xi0.

        Raises
        ------
        ValueError
            If T <= 0.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")

        tmp0 = (
            (T + self.delta_vix) ** (4.0 * self.H + 1.0)
            + self.delta_vix ** (4.0 * self.H + 1.0)
            - T ** (4.0 * self.H + 1.0)
        )
        tmp0 /= (4.0 * self.H + 1.0) * self.delta_vix
        tmp1 = (
            T ** (2 * self.H + 1)
            + self.delta_vix ** (2 * self.H + 1)
            - (T + self.delta_vix) ** (2 * self.H + 1)
        )
        tmp1 /= self.delta_vix * (2 * self.H + 1)
        tmp1 = -(tmp1**2)
        tmp2 = (
            -2
            * T ** (2 * self.H)
            * self.delta_vix ** (2 * self.H)
            * hyp2f1(-2 * self.H, 2 * self.H + 1, 2 * self.H + 2, -self.delta_vix / T)
            / (2.0 * self.H + 1.0)
        )
        gamma_10 = self.eta**4 * (tmp0 + tmp1 + tmp2) / 8.0

        gamma_11 = (
            (T + self.delta_vix) ** (2 * self.H + 1)
            - T ** (2 * self.H + 1)
            - self.delta_vix ** (2 * self.H + 1)
        )
        gamma_11 *= self.eta**2 / (2.0 * (2.0 * self.H + 1.0) * self.delta_vix)
        gamma_11 -= 0.5 * self.var_proxy(T)

        return gamma_10 + gamma_11

    def gamma_2_proxy_flat(self, T, quad_scipy: bool = False, n_quad: int = 30):
        """
        Compute the second-order gamma coefficient of the VIX proxy when xi0 is flat.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. If False, use Gauss-Legendre
            quadrature. Default is False.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 30.

        Returns
        -------
        float
            Second-order gamma coefficient for flat xi0.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        def g2(t, a):
            tmp = (
                (T * t + self.delta_vix) ** (self.H + 0.5) - (T * t) ** (self.H + 0.5)
            ) * (
                (T + a * self.delta_vix) ** (2.0 * self.H)
                - (a * self.delta_vix) ** (2.0 * self.H)
            )
            tmp *= (T * t + a * self.delta_vix) ** (self.H - 0.5)
            return tmp

        if quad_scipy:

            def inner_integral(a):
                return quad(lambda t: g2(t, a), 0.0, 1.0)[0]

            integral = quad(inner_integral, 0.0, 1.0)[0]

        else:
            v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)

            def inner_integral(a):
                return np.sum(v_weights * g2(v_nodes, a))

            integral = np.sum(
                v_weights * np.array([inner_integral(a) for a in v_nodes])
            )

        gamma_2 = (
            -integral * T * self.eta**4 * self.H / (self.delta_vix * (self.H + 0.5))
        )
        gamma_2 += (
            self.eta**2
            * (
                (T + self.delta_vix) ** (2.0 * self.H + 1)
                - T ** (2.0 * self.H + 1.0)
                - self.delta_vix ** (2.0 * self.H + 1.0)
            )
            * self.var_proxy_flat(T)
        ) / (2.0 * (2.0 * self.H + 1.0) * self.delta_vix)

        return gamma_2

    def gamma_3_proxy_flat(self, T, quad_scipy: bool = False, n_quad: int = 30):
        """
        Compute the third-order gamma coefficient of the VIX proxy when xi0 is flat.

        Parameters
        ----------
        T : float
            Maturity of the VIX future.
        quad_scipy : bool, optional
            If True, use scipy's quad for integration. If False, use Gauss-Legendre
            quadrature. Default is False.
        n_quad : int, optional
            Number of quadrature points for numerical integration (if not using scipy).
            Default is 30.

        Returns
        -------
        float
            Third-order gamma coefficient for flat xi0.

        Raises
        ------
        ValueError
            If T <= 0 or n_quad < 1.
        """
        if T <= 0:
            raise ValueError("Maturity T must be positive.")
        if n_quad < 1:
            raise ValueError("n_quad must be at least 1.")

        def g3(t, a):
            d = self.delta_vix / T
            tmp0 = (
                (a * d) ** (self.H - 0.5)
                * hyp2f1(-self.H + 0.5, self.H + 1.5, self.H + 2.5, -1.0 / (a * d))
                / (self.H + 1.5)
            )
            tmp1 = (1 + a * d) ** (self.H + 0.5) * hyp2f1(
                -self.H - 0.5,
                self.H + 0.5,
                self.H + 1.5,
                -(1.0 + a * d) / (d - a * d),
            )
            tmp1 -= (a * d) ** (self.H + 0.5) * hyp2f1(
                -self.H - 0.5, self.H + 0.5, self.H + 1.5, -(a * d) / (d - a * d)
            )
            tmp1 *= (1.0 - a) ** (self.H + 0.5) * d ** (self.H + 0.5) / (self.H + 0.5)
            tmps = tmp1 - tmp0
            return (
                T ** (4.0 * self.H)
                * ((t + d) ** (self.H + 0.5) - t ** (self.H + 0.5))
                * (t + a * d) ** (self.H - 0.5)
                * tmps
            )

        if quad_scipy:

            def inner_integral(a):
                return quad(lambda t: g3(t, a), 0.0, 1.0)[0]

            integral = quad(inner_integral, 0.0, 1.0)[0]

        else:
            v_nodes, v_weights = utils.gauss_legendre(0.0, 1.0, n_quad)

            def inner_integral(a):
                return np.sum(v_weights * g3(v_nodes, a))

            integral = np.sum(
                v_weights * np.array([inner_integral(a) for a in v_nodes])
            )

        gamma_3 = (
            2.0
            * self.eta**4
            * self.H**2
            * T**2
            / (self.delta_vix**2 * (self.H + 0.5) ** 2)
        )
        gamma_3 *= integral
        gamma_3 -= 0.5 * self.var_proxy_flat(T) ** 2

        return gamma_3


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
        2: (0.24**2, 0.23, 2.0, 0.2, 0.4),
    }
    if id not in mixed_params:
        raise ValueError("Invalid id for mixed case. Must be 1 or 2.")

    xi0_level, h, eta, eta_2, lbd = mixed_params[id]
    params = {"H": h, "eta": eta / np.sqrt(2 * h)}
    return params, flat_xi0(xi0_level), lbd, eta_2
