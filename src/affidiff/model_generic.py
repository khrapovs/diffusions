"""Generic model class."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
from mygmm import GMM

from affidiff.helper_functions import ajd_diff, ajd_drift, columnwise_prod, instruments, nice_errors, rolling_window

if TYPE_CHECKING:
    from affidiff.param_generic import GenericParam

try:
    from affidiff.simulate import simulate  # type: ignore
except Exception:
    simulate = None


class SDE(object):
    """Generic Model.

    Attributes
    ----------
    param : parameter instance
        True parameters used for simulation of the data

    Methods
    -------
    simulate
        Simulate observations from the model
    sim_realized
        Simulate realized returns and variance
    sim_realized_pq
        Simulate realized returns and variance under both P and Q
    gmmest
        Estimate model parameters using GMM
    integrated_gmm
        Estimate model parameters using Integrated GMM

    """

    def __init__(self, param: Any = None) -> None:  # noqa: ANN401
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        self.nsub: int | None = None
        self.ndiscr: int | None = None
        self.param: Any = param
        self.errors: np.ndarray | None = None

    def update_theta(self, param: GenericParam | object) -> None:
        """Update model parameters.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        self.param = param

    def get_start(self) -> np.ndarray | list[float]:
        """Return starting values for simulation."""
        raise NotImplementedError("Must be overridden")

    @staticmethod
    def realized_depvar(data: Any, subset: Any = None) -> Any:  # noqa: PLR0917, ANN401
        """Realized dependent variables."""
        raise NotImplementedError("Must be overridden")

    def mat_a(self, param: Any, subset: Any = None) -> Any:  # noqa: PLR0917, ANN401
        """Matrix A in integrated moments."""
        raise NotImplementedError("Must be overridden")

    def realized_const(  # noqa: PLR0917
        self,
        param: Any = None,  # noqa: ANN401
        aggh: Any = 1,  # noqa: ANN401
        subset: Any = None,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Realized constant in integrated moments."""
        raise NotImplementedError("Must be overridden")

    def euler_loc(self, state: np.ndarray, theta: GenericParam | object) -> np.ndarray:  # noqa: PLR0917
        """Euler location.

        Parameters
        ----------
        state : (nsim, nvars) array
            Current value of the process
        theta : parameter instance
            Model parameters

        Returns
        -------
        (nsim, nvars) array
            Location term in Euler discretization

        """
        return ajd_drift(state, theta)

    def euler_scale(self, state: np.ndarray, theta: GenericParam | object) -> np.ndarray:  # noqa: PLR0917
        """Euler scale.

        Parameters
        ----------
        state : (nsim, nvars) array
            Current value of the process
        theta : parameter instance
            Model parameters

        Returns
        -------
        (nsim, nvars, nvars) array
            Scale term in Euler discretization

        """
        return ajd_diff(state, theta)

    def loc(self, state: np.ndarray, theta: GenericParam | object) -> np.ndarray:  # noqa: PLR0917
        """Location.

        Parameters
        ----------
        state : (nsim, nvars) array
            Current value of the process
        theta : parameter instance
            Model parameters

        Returns
        -------
        (nsim, nvars) array
            Location term in Euler discretization

        """
        return self.euler_loc(state, theta)

    def scale(self, state: np.ndarray, theta: GenericParam | object) -> np.ndarray:  # noqa: PLR0917
        """Scale.

        Parameters
        ----------
        state : (nsim, nvars) array
            Current value of the process
        theta : parameter instance
            Model parameters

        Returns
        -------
        (nsim, nvars, nvars) array
            Scale term in Euler discretization

        """
        return self.euler_scale(state, theta)

    def exact_loc(self, state: np.ndarray, theta: GenericParam | object) -> np.ndarray:  # noqa: PLR0917
        """Exact location."""
        return self.euler_loc(state, theta)

    def exact_scale(self, state: np.ndarray, theta: GenericParam | object) -> np.ndarray:  # noqa: PLR0917
        """Exact scale."""
        return self.euler_scale(state, theta)

    @staticmethod
    def mean_vol(param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Unconditional mean of volatility."""
        raise NotImplementedError

    @staticmethod
    def mean_vol2(param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Unconditional mean of squared volatility."""
        raise NotImplementedError

    @staticmethod
    def mean_ret(param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Unconditional mean of returns."""
        raise NotImplementedError

    @staticmethod
    def mean_cross(param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Unconditional mean of returns times volatility."""
        raise NotImplementedError

    def momcond(  # noqa: PLR0917
        self,
        theta: Any,  # noqa: ANN401
        data: Any = None,  # noqa: ANN401
        instrlag: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Moment conditions."""
        raise NotImplementedError

    def depvar_unc_mean(self, param: GenericParam | object, aggh: float) -> np.ndarray:  # noqa: PLR0917
        """Unconditional means of realized data.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        array

        """
        return np.array(
            [
                self.mean_vol(param, aggh),
                self.mean_vol2(param, aggh),
                self.mean_ret(param, aggh),
                self.mean_cross(param, aggh),
            ]
        )

    def update(self, state: np.ndarray, error: np.ndarray) -> np.ndarray:  # noqa: PLR0917
        """Euler update function.

        Parameters
        ----------
        state : (nsim, nvars) array_like
            Current value of the process
        error : (nsim, nvars) array_like
            Random shocks

        Returns
        -------
        (nsim, nvars) array
            Update of the process value. Same shape as the input.

        """
        # (nsim, nvars) array_like
        loc = self.euler_loc(state, self.param)
        # (nsim, nvars, nvars) array_like
        scale = self.euler_scale(state, self.param)

        assert self.ndiscr is not None
        new_state = loc / self.ndiscr + (np.transpose(scale, axes=[1, 2, 0]) * error.T).sum(1).T / self.ndiscr**0.5

        return new_state

    def simulate(  # noqa: PLR0917
        self,
        start: Any = None,  # noqa: ANN401
        nsub: int = 80,
        ndiscr: int = 1,
        nobs: int = 500,
        nsim: int = 1,
        diff: Any = None,  # noqa: ANN401
        new_innov: bool = True,
        cython: bool = False,
    ) -> Any:  # noqa: ANN401
        """Simulate observations from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        nsub : int
            Interval length
        ndiscr : int
            Number of Euler discretization points inside a subinterval
        nobs : int
            Number of points to simulate in one series
        nsim : int
            Number of time series to simulate
        diff : int
            Dimensions which should be differentiated,
            i.e. return = price[1:] - price[:-1]
        new_innov : bool
            Whether to generate new innovations (True),
            or use already stored (False)
        cython : bool
            Whether to use cython-optimized simulation (True) or not (False)

        Returns
        -------
        paths : (nobs, 2*nsim, nvars) array
            Simulated data

        """
        if start is None:
            start = self.get_start()
        if np.size(self.param.mat_k0) != np.size(start):
            raise ValueError("Start for paths is of wrong dimension!")
        self.nsub = nsub
        self.ndiscr = ndiscr
        nvars = np.size(start)
        npoints = nobs * ndiscr

        if self.errors is None or new_innov:
            # Generate new errors
            self.errors = np.random.normal(size=(npoints, nsim, nvars))
            # Standardize the errors
            self.errors = nice_errors(self.errors, 1)

        if cython:
            assert simulate is not None
            dt = 1 / ndiscr / nsub

            paths = simulate(
                self.errors,
                np.atleast_1d(start).astype(float),
                np.atleast_1d(self.param.mat_k0).astype(float),
                np.atleast_2d(self.param.mat_k1).astype(float),
                np.atleast_2d(self.param.mat_h0).astype(float),
                np.atleast_3d(self.param.mat_h1).astype(float),
                float(dt),
            )
        else:
            nsim = self.errors.shape[1]
            paths = start * np.ones((npoints + 1, nsim, nvars))

            for i in range(npoints):
                # (nsim, nvars)
                paths[i + 1] = paths[i] + self.update(paths[i], self.errors[i])

        # (nobs+1, nsim, nvars)
        paths = paths[::ndiscr]
        if diff is not None:
            paths[1:, :, diff] = paths[1:, :, diff] - paths[:-1, :, diff]
        return paths[1:]

    def sim_realized(  # noqa: PLR0917
        self,
        start: Any = None,  # noqa: ANN401
        nsub: int = 80,
        ndiscr: int = 10,
        aggh: int = 1,
        nperiods: int = 500,
        nsim: int = 1,
        diff: Any = None,  # noqa: ANN401
        new_innov: bool = True,
        cython: bool = True,
    ) -> tuple[Any, Any]:  # noqa: ANN401
        """Simulate realized returns and variance from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        nsub : int
            Number of subintervals for latent simulation (fractions of the day)
        ndiscr : int
            Number of Euler discretization points inside unit interval
        aggh : int
            Number of intervals (days) to aggregate over using rolling mean
        nperiods : int
            Number of points to simulate in one series (days)
        nsim : int
            Number of time series to simulate
        diff : int
            Dimensions which should be differentiated,
            i.e. return = price[1:] - price[:-1]
        new_innov : bool
            Whether to generate new innovations (True),
            or use already stored (False)
        cython : bool
            Whether to use cython-optimized simulation (True) or not (False)

        Returns
        -------
        returns : (nperiods, ) array
            Simulated returns
        rvar : (nperiods, ) array
            Simulated realized variance

        """
        if start is None:
            start = self.get_start()
        nobs = nperiods * nsub
        paths = self.simulate(
            start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=diff, new_innov=new_innov, cython=cython
        )
        returns = paths[:, 0, 0].reshape((nperiods, nsub))
        # Compute realized var and returns over one day
        rvar = (returns**2).sum(1)
        returns = returns.sum(1)
        # Aggregate over arbitrary number of days
        rvar = rolling_window(np.mean, rvar, window=aggh)
        returns = rolling_window(np.mean, returns, window=aggh)
        return returns, rvar

    def sim_realized_pq(  # noqa: PLR0917
        self,
        start_p: np.ndarray | Sequence[float] | None = None,
        start_q: np.ndarray | Sequence[float] | None = None,
        aggh: list[int] | tuple[int, int] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Simulate realized data from the model under both P and Q.

        Parameters
        ----------
        start_p : array_like
            Starting value for simulation under P
        start_q : array_like
            Starting value for simulation under Q
        aggh : list
            Aggregation windows for P and Q respectively
        kwargs : dict
            Anything that needs to go through sim_realized

        Returns
        -------
        data_p : tuple
            Returns and realized variance under P
        data_q : tuple
            Returns and realized variance under Q

        Notes
        -----
        For argumentsts see sim_realized

        """
        if aggh is None:
            aggh = [1, 1]
        if start_p is None:
            start_p = self.get_start()
        data_p = self.sim_realized(start_p, aggh=aggh[0], new_innov=True, **kwargs)
        self.param.convert_to_q()
        if start_q is None:
            start_q = self.get_start()
        data_q = self.sim_realized(start_q, aggh=aggh[1], new_innov=False, **kwargs)
        return data_p, data_q

    def gmmest(self, theta_start: Any, **kwargs: object) -> object:  # noqa: ANN401
        """Estimate model parameters using GMM.

        Parameters
        ----------
        theta_start : array
            Initial parameter values for estimation
        kwargs : dict
            Anything that needs to go through mygmm

        Notes
        -----
        For arguments see momcond

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta_start.get_theta(), **kwargs)

    def integrated_gmm(  # noqa: PLR0917
        self,
        param_start: GenericParam | object,
        subset: str = "all",
        measure: str = "P",
        names: list[str] | None = None,
        bounds: list[tuple[float | None, float | None]] | None = None,
        constraints: object = (),
        **kwargs: object,
    ) -> object:
        """Estimate model parameters using Integrated GMM.

        Parameters
        ----------
        param_start : parameter class
            Initial parameter values for estimation
        subset : str

            Which parameters to estimate. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

        measure : str

            Under which measure to estimate:
                - 'P' : physical measure
                - 'Q' : risk-neutral
                - 'PQ' : both

        names : list of str
            Parameter names
        bounds : list of tuples
            Parameter bounds
        constraints : dict or sequence of dict
            Equality and inequality constraints. See scipy.optimize.minimize
        kwargs : dict
            Anything that needs to go through mygmm

        Notes
        -----
        For arguments see integrated_mom

        """
        estimator = GMM(self.integrated_mom)
        self.param: Any = param_start
        theta_start = self.param.get_theta(subset=subset, measure=measure)
        if names is None:
            names = self.param.get_names(subset=subset, measure=measure)
        if bounds is None:
            bounds = self.param.get_bounds(subset=subset, measure=measure)
        if constraints == ():
            constraints = self.param.get_constraints()
        return estimator.gmmest(
            theta_start, names=names, subset=subset, measure=measure, bounds=bounds, constraints=constraints, **kwargs
        )

    def integrated_mom(  # noqa: PLR0917
        self,
        theta: np.ndarray | Sequence[float],
        data: object = None,
        instr_data: object = None,
        instr_choice: str = "const",
        aggh: object = 1,
        subset: str = "all",
        instrlag: int = 1,
        measure: str = "P",
        **kwargs: object,  # noqa: ARG002
    ) -> tuple[np.ndarray, Any]:
        """Integrated moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : (2, nobs) array
            Returns and realized variance
        instr_data : (ninstr, nobs) array
            Instruments (no lags)
        instrlag : int
            Number of lags for the instruments
        instr_choice : str {'const', 'var'}
            Choice of instruments.
                - 'const' : just a constant (unconditional moments)
                - 'var' : lags of instrument data
        aggh : int
            Number of intervals (days) to aggregate over using rolling mean
        subset : str
            Which parameters to estimate. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility
        measure : str
            Under which measure to estimate:
                - 'P' : physical measure
                - 'Q' : risk-neutral
                - 'PQ' : both
        kwargs : dict
            Anything that needs to go through mygmm

        Returns
        -------
        moments : (nobs - instrlag - 2, 3 * ninstr = nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        subset_sl = None
        if subset == "vol":
            subset_sl = slice(2)

        self.param.update(theta=theta, subset=subset, measure=measure)
        lag = 2

        if measure == "PQ":
            error = []
            data_list = list(cast(Iterable, data)) if data is not None else []
            aggh_list = list(aggh) if isinstance(aggh, (list, tuple)) else [aggh, aggh]  # type: ignore[arg-type]
            measure_list = list(measure)
            for data_x, agg, meas in zip(data_list, aggh_list, measure_list, strict=False):
                if meas == "Q":
                    self.param.convert_to_q()
                depvar = self.realized_depvar(data_x)[lag:]
                # (nobs - lag, 4) array
                error.append(
                    depvar.dot(self.mat_a(self.param, subset_sl).T) - self.realized_const(self.param, agg, subset_sl)
                )

            error = np.hstack(error)

        else:
            depvar = self.realized_depvar(data)[lag:]  # type: ignore[arg-type]
            # (nobs - lag, 4) array
            error = depvar.dot(self.mat_a(self.param, subset_sl).T) - self.realized_const(self.param, aggh, subset_sl)

        nobs = error.shape[0] + lag
        # self.instruments(data, instrlag=instrlag): (nobs, ninstr*instrlag+1)
        # (nobs-lag, ninstr*instrlag+1)
        instr = instruments(instr_data, nobs=nobs, instrlag=instrlag, instr_choice=instr_choice)[:-lag]
        # (nobs - instrlag - lag, 4 * (ninstr*instrlag + 1))
        moms = columnwise_prod(error, instr)

        return moms, None
