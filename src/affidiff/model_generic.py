"""Generic model class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
from mygmm import GMM, Results

from affidiff.helper_functions import ajd_diff, ajd_drift, columnwise_prod, instruments, nice_errors, rolling_window

if TYPE_CHECKING:
    from affidiff.param_generic import GenericParam

try:
    from affidiff.simulate import simulate  # type: ignore
except Exception:
    simulate = None


class SDE(ABC):
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

    def __init__(self, param: GenericParam | None = None) -> None:
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

    def update_theta(self, param: GenericParam) -> None:
        """Update model parameters.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        self.param = param

    @abstractmethod
    def get_start(self) -> np.ndarray | list[float]:
        """Return starting values for simulation."""
        raise NotImplementedError("Must be overridden")

    @staticmethod
    def realized_depvar(*, data: np.ndarray | Sequence[np.ndarray], subset: slice | None = None) -> np.ndarray:
        """Realized dependent variables."""
        raise NotImplementedError("Must be overridden")

    def mat_a(self, *, param: GenericParam, subset: slice | None = None) -> np.ndarray:
        """Matrix A in integrated moments."""
        raise NotImplementedError("Must be overridden")

    def realized_const(
        self,
        *,
        param: GenericParam | None = None,
        aggh: float = 1,
        subset: slice | None = None,
    ) -> np.ndarray:
        """Realized constant in integrated moments."""
        raise NotImplementedError("Must be overridden")

    def euler_loc(self, *, state: np.ndarray, theta: GenericParam) -> np.ndarray:
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
        return ajd_drift(state=state, theta=theta)

    def euler_scale(self, *, state: np.ndarray, theta: GenericParam) -> np.ndarray:
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
        return ajd_diff(state=state, theta=theta)

    def loc(self, *, state: np.ndarray, theta: GenericParam) -> np.ndarray:
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
        return self.euler_loc(state=state, theta=theta)

    def scale(self, *, state: np.ndarray, theta: GenericParam) -> np.ndarray:
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
        return self.euler_scale(state=state, theta=theta)

    def exact_loc(self, *, state: np.ndarray, theta: GenericParam) -> np.ndarray:
        """Exact location."""
        return self.euler_loc(state=state, theta=theta)

    def exact_scale(self, *, state: np.ndarray, theta: GenericParam) -> np.ndarray:
        """Exact scale."""
        return self.euler_scale(state=state, theta=theta)

    @staticmethod
    def mean_vol(*, param: GenericParam, aggh: float) -> float:
        """Unconditional mean of volatility."""
        raise NotImplementedError

    @staticmethod
    def mean_vol2(*, param: GenericParam, aggh: float) -> float:
        """Unconditional mean of squared volatility."""
        raise NotImplementedError

    @staticmethod
    def mean_ret(*, param: GenericParam, aggh: float) -> float:
        """Unconditional mean of returns."""
        raise NotImplementedError

    @staticmethod
    def mean_cross(*, param: GenericParam, aggh: float) -> float:
        """Unconditional mean of returns times volatility."""
        raise NotImplementedError

    def momcond(
        self,
        *,
        theta: GenericParam | np.ndarray | Sequence[float],
        data: np.ndarray | Sequence[np.ndarray] | None = None,
        instrlag: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Moment conditions."""
        raise NotImplementedError

    def depvar_unc_mean(self, *, param: GenericParam, aggh: float) -> np.ndarray:
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
                self.mean_vol(param=param, aggh=aggh),
                self.mean_vol2(param=param, aggh=aggh),
                self.mean_ret(param=param, aggh=aggh),
                self.mean_cross(param=param, aggh=aggh),
            ]
        )

    def update(self, *, state: np.ndarray, error: np.ndarray) -> np.ndarray:
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
        loc = self.euler_loc(state=state, theta=self.param)
        # (nsim, nvars, nvars) array_like
        scale = self.euler_scale(state=state, theta=self.param)

        assert self.ndiscr is not None
        new_state = loc / self.ndiscr + (np.transpose(scale, axes=[1, 2, 0]) * error.T).sum(1).T / self.ndiscr**0.5

        return new_state

    def simulate(
        self,
        *,
        start: np.ndarray | float | Sequence[float] | None = None,
        nsub: int = 80,
        ndiscr: int = 1,
        nobs: int = 500,
        nsim: int = 1,
        diff: int | Sequence[int] | slice | None = None,
        new_innov: bool = True,
        cython: bool = False,
    ) -> np.ndarray:
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
            self.errors = nice_errors(errors=self.errors, sdim=1)

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
                paths[i + 1] = paths[i] + self.update(state=paths[i], error=self.errors[i])

        # (nobs+1, nsim, nvars)
        paths = paths[::ndiscr]
        if diff is not None:
            paths[1:, :, diff] = paths[1:, :, diff] - paths[:-1, :, diff]
        return paths[1:]

    def sim_realized(
        self,
        *,
        start: np.ndarray | float | Sequence[float] | None = None,
        nsub: int = 80,
        ndiscr: int = 10,
        aggh: int = 1,
        nperiods: int = 500,
        nsim: int = 1,
        diff: int | Sequence[int] | slice | None = None,
        new_innov: bool = True,
        cython: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            start=start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=diff, new_innov=new_innov, cython=cython
        )
        returns = paths[:, 0, 0].reshape((nperiods, nsub))
        # Compute realized var and returns over one day
        rvar = (returns**2).sum(1)
        returns = returns.sum(1)
        # Aggregate over arbitrary number of days
        rvar = rolling_window(fun=np.mean, mat=rvar, window=aggh)
        returns = rolling_window(fun=np.mean, mat=returns, window=aggh)
        return returns, rvar

    def sim_realized_pq(
        self,
        *,
        start_p: np.ndarray | Sequence[float] | None = None,
        start_q: np.ndarray | Sequence[float] | None = None,
        aggh: list[int] | tuple[int, int] | None = None,
        nsub: int = 80,
        ndiscr: int = 10,
        nperiods: int = 500,
        nsim: int = 1,
        diff: int | Sequence[int] | slice | None = None,
        new_innov: bool = True,
        cython: bool = False,
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
        nsub : int
            Number of subintervals for latent simulation (fractions of the day)
        ndiscr : int
            Number of Euler discretization points inside unit interval
        nperiods : int
            Number of points to simulate in one series (days)
        nsim : int
            Number of time series to simulate
        diff : int
            Dimensions which should be differentiated
        new_innov : bool
            Whether to generate new innovations (True), or use already stored (False)
        cython : bool
            Whether to use cython-optimized simulation (True) or not (False)

        Returns
        -------
        data_p : tuple
            Returns and realized variance under P
        data_q : tuple
            Returns and realized variance under Q

        """
        if aggh is None:
            aggh = [1, 1]
        if start_p is None:
            start_p = self.get_start()
        data_p = self.sim_realized(
            start=start_p,
            aggh=aggh[0],
            new_innov=new_innov,
            nsub=nsub,
            ndiscr=ndiscr,
            nperiods=nperiods,
            nsim=nsim,
            diff=diff,
            cython=cython,
        )
        self.param.convert_to_q()
        if start_q is None:
            start_q = self.get_start()
        data_q = self.sim_realized(
            start=start_q,
            aggh=aggh[1],
            new_innov=new_innov,
            nsub=nsub,
            ndiscr=ndiscr,
            nperiods=nperiods,
            nsim=nsim,
            diff=diff,
            cython=cython,
        )
        return data_p, data_q

    def gmmest(
        self,
        *,
        theta_start: GenericParam,
        data: np.ndarray | Sequence[np.ndarray] | None = None,
        instrlag: int = 1,
        iter: int = 2,
        method: str = "BFGS",
        kernel: str = "Bartlett",
        band: int | None = None,
    ) -> Results:
        """Estimate model parameters using GMM.

        Parameters
        ----------
        theta_start : parameter instance
            Initial parameter values for estimation
        data : array_like, optional
            Data passed to the moment condition function
        instrlag : int
            Number of lags for the instruments
        iter : int
            Number of GMM iterations
        method : str
            Optimization method passed to scipy.optimize.minimize
        kernel : str
            HAC kernel for weighting matrix ('Bartlett', 'Parzen', etc.)
        band : int, optional
            HAC bandwidth. If None, chosen automatically.

        Notes
        -----
        For moment condition arguments see momcond.

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(
            theta_start.get_theta(),
            data=data,
            instrlag=instrlag,
            iter=iter,
            method=method,
            kernel=kernel,
            band=band,
        )

    def integrated_gmm(
        self,
        *,
        param_start: GenericParam,
        data: np.ndarray | Sequence[Any] | tuple[Any, ...] | None = None,
        instr_data: np.ndarray | None = None,
        instr_choice: str = "const",
        aggh: float | Sequence[float] = 1,
        instrlag: int = 1,
        subset: str = "all",
        measure: str = "P",
        names: list[str] | None = None,
        bounds: list[tuple[float | None, float | None]] | None = None,
        constraints: Sequence[dict[str, object]] | dict[str, object] | tuple[()] = (),
        iter: int = 2,
        method: str = "BFGS",
        kernel: str = "Bartlett",
        band: int | None = None,
    ) -> Results:
        """Estimate model parameters using Integrated GMM.

        Parameters
        ----------
        param_start : parameter class
            Initial parameter values for estimation
        data : array_like, optional
            Returns and realized variance used in moment conditions
        instr_data : array_like, optional
            Instruments (no lags)
        instr_choice : str {'const', 'var'}
            Choice of instruments
        aggh : int or list of int
            Number of intervals (days) to aggregate over using rolling mean
        instrlag : int
            Number of lags for the instruments
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
        iter : int
            Number of GMM iterations
        method : str
            Optimization method passed to scipy.optimize.minimize
        kernel : str
            HAC kernel for weighting matrix ('Bartlett', 'Parzen', etc.)
        band : int, optional
            HAC bandwidth. If None, chosen automatically.

        """
        estimator = GMM(self.integrated_mom)
        self.param = param_start
        theta_start = self.param.get_theta(subset=subset, measure=measure)  # type: ignore[call-arg]
        if names is None:
            names = self.param.get_names(subset=subset, measure=measure)  # type: ignore[call-arg]
        if bounds is None:
            bounds = self.param.get_bounds(subset=subset, measure=measure)  # type: ignore[call-arg]
        if constraints == ():
            constraints = self.param.get_constraints()
        return estimator.gmmest(
            theta_start,
            names=names,
            data=data,
            instr_data=instr_data,
            instr_choice=instr_choice,
            aggh=aggh,
            instrlag=instrlag,
            subset=subset,
            measure=measure,
            bounds=bounds,
            constraints=constraints,
            iter=iter,
            method=method,
            kernel=kernel,
            band=band,
        )

    def integrated_mom(
        self,
        *,
        theta: np.ndarray | Sequence[float],
        data: np.ndarray | Sequence[np.ndarray] | None = None,
        instr_data: np.ndarray | None = None,
        instr_choice: str = "const",
        aggh: float | Sequence[float] = 1,
        subset: str = "all",
        instrlag: int = 1,
        measure: str = "P",
    ) -> tuple[np.ndarray, np.ndarray | None]:
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

        assert self.param is not None
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
                depvar = self.realized_depvar(data=data_x)[lag:]
                # (nobs - lag, 4) array
                error.append(
                    depvar.dot(self.mat_a(param=self.param, subset=subset_sl).T)
                    - self.realized_const(param=self.param, aggh=float(cast(float, agg)), subset=subset_sl)
                )

            error = np.hstack(error)

        else:
            assert data is not None
            depvar = self.realized_depvar(data=data)[lag:]
            aggh_val = aggh[0] if isinstance(aggh, Sequence) and not isinstance(aggh, (str, bytes)) else float(aggh)  # type: ignore[arg-type]
            # (nobs - lag, 4) array
            error = depvar.dot(self.mat_a(param=self.param, subset=subset_sl).T) - self.realized_const(
                param=self.param, aggh=float(cast(float, aggh_val)), subset=subset_sl
            )

        nobs = error.shape[0] + lag
        # self.instruments(data, instrlag=instrlag): (nobs, ninstr*instrlag+1)
        # (nobs-lag, ninstr*instrlag+1)
        instr = instruments(data=instr_data, nobs=nobs, instrlag=instrlag, instr_choice=instr_choice)[:-lag]
        # (nobs - instrlag - lag, 4 * (ninstr*instrlag + 1))
        moms = columnwise_prod(left=error, right=instr)

        return moms, None
