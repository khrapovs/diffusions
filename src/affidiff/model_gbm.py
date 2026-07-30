"""GBM model class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, cast

import numdifftools as nd
import numpy as np
from statsmodels.tsa.tsatools import lagmat

from affidiff.helper_functions import columnwise_prod
from affidiff.model_generic import SDE
from affidiff.param_gbm import GBMparam

if TYPE_CHECKING:
    from affidiff.param_generic import GenericParam


class GBM(SDE):
    """Geometric Brownian Motion."""

    def __init__(self, param: GBMparam | None = None) -> None:
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        super().__init__(param)

    def get_start(self) -> list[float]:
        """Return starting values for simulation."""
        return [1.0]

    @staticmethod
    def drift(*, state: np.ndarray | float, theta: GBMparam | np.ndarray | Sequence[float]) -> np.ndarray | float:  # noqa: ARG004
        """Drift function.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        scalar
            Drift value

        """
        if isinstance(theta, GBMparam):
            return theta.mean - theta.sigma**2 / 2
        return float(theta[0]) - float(theta[1]) ** 2 / 2

    @staticmethod
    def diff(*, state: np.ndarray | float, theta: GBMparam | np.ndarray | Sequence[float]) -> np.ndarray | float:  # noqa: ARG004
        """Diffusion (instantaneous volatility) function.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        scalar
            Diffusion value

        """
        if isinstance(theta, GBMparam):
            return theta.sigma
        return float(theta[1])

    def betamat(self, theta: np.ndarray | Sequence[float]) -> np.ndarray:
        """Coefficients in linear representation of the first moment.

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        float
            Constant coefficient

        """
        param = GBMparam.from_theta(theta=theta)
        loc = float(self.exact_loc(state=np.array(0), theta=param))
        return np.array([loc, 0], dtype=float)

    def gammamat(self, theta: np.ndarray | Sequence[float]) -> np.ndarray:
        """Coefficients in linear representation of the second moment.

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        float
            Constant coefficient

        """
        param = GBMparam.from_theta(theta=theta)
        loc = float(self.exact_loc(state=np.array(0), theta=param))
        scale = float(self.exact_scale(state=np.array(0), theta=param))
        return np.array([loc**2 + scale**2, 0], dtype=float)

    def dbetamat(self, theta: GenericParam | np.ndarray | Sequence[float]) -> np.ndarray:
        """Calculate derivative of the first moment coefficients (numerical).

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Derivatives of the coefficient

        """
        with np.errstate(divide="ignore"):
            return nd.Jacobian(self.betamat)(theta)

    def dgammamat(self, theta: GenericParam | np.ndarray | Sequence[float]) -> np.ndarray:
        """Calculate derivative of the second moment coefficients (numerical).

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Derivatives of the coefficient

        """
        with np.errstate(divide="ignore"):
            return nd.Jacobian(self.gammamat)(theta)

    def dbetamat_exact(self, theta: np.ndarray | Sequence[float]) -> np.ndarray:
        """Calculate derivative of the first moment coefficients (exact).

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Derivatives of the coefficient

        """
        _mean, sigma = float(theta[0]), float(theta[1])
        assert self.nsub is not None
        return np.array([[1 / self.nsub, -sigma / self.nsub], [0, 0]])

    def dgammamat_exact(self, theta: np.ndarray | Sequence[float]) -> np.ndarray:
        """Calculate derivative of the second moment coefficients (exact).

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Derivatives of the coefficient

        """
        mean, sigma = float(theta[0]), float(theta[1])
        assert self.nsub is not None
        return np.array(
            [
                [
                    2 / self.nsub**2 * (mean - sigma**2 / 2),
                    2 * sigma / self.nsub - 2 * sigma / self.nsub**2 * (mean - sigma**2 / 2),
                ],
                [0, 0],
            ]
        )

    @staticmethod
    def realized_depvar(*, data: np.ndarray | Sequence[np.ndarray], subset: slice | None = None) -> np.ndarray:  # noqa: ARG004
        """Array of the left-hand side variables in realized moment conditions.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance
        subset : slice
            Which moments to use

        Returns
        -------
        (3, nobs) array
            Dependend variables

        """
        ret, rvar = data
        return np.vstack([ret, rvar, rvar**2])

    def realized_const(
        self,
        *,
        param: GenericParam | np.ndarray | Sequence[float] | None = None,
        aggh: float = 1,  # noqa: ARG002
        subset: slice | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        param : array
            Parameters
        aggh : int
            Interval length
        subset : slice
            Which moments to use

        Returns
        -------
        array
            Intercept

        """
        if param is None:
            param = self.param
        assert param is not None
        if isinstance(param, GBMparam):
            mean, sigma = param.mean, param.sigma
        elif isinstance(param, (np.ndarray, Sequence)):
            param_arr = np.asarray(param)
            mean, sigma = float(param_arr[0]), float(param_arr[1])
        else:
            raise TypeError("Invalid param type for realized_const")
        return np.array([mean - sigma**2 / 2, sigma**2, sigma**4])

    def drealized_const(self, theta: GenericParam | np.ndarray | Sequence[float]) -> np.ndarray:
        """Calculate derivative of the intercept in the realized moment conditions.

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        (nparams, nintercepts) array
            Derivatives of the coefficient

        """

        def _realized_const_wrapper(theta: GenericParam | np.ndarray | Sequence[float]) -> np.ndarray:
            return self.realized_const(param=theta)

        with np.errstate(divide="ignore"):
            return nd.Jacobian(_realized_const_wrapper)(theta)

    @staticmethod
    def instruments(*, data: np.ndarray | Sequence[np.ndarray], instrlag: int = 1) -> np.ndarray:
        """Create an array of instruments.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance
        instrlag : int
            Number of lags for the instruments

        Returns
        -------
        (ninstr, nobs - instrlag) array
            Derivatives of the coefficient

        """
        data_arr = np.asarray(data)
        lmat = cast(np.ndarray, lagmat(data_arr.T, maxlag=instrlag))
        return np.vstack([np.ones_like(data_arr[0]), lmat.T])[:, instrlag:]

    def integrated_mom(
        self,
        *,
        theta: GenericParam | np.ndarray | Sequence[float],
        data: np.ndarray | Sequence[np.ndarray] | None = None,
        instr_data: np.ndarray | None = None,  # noqa: ARG002
        instr_choice: str = "const",  # noqa: ARG002
        aggh: float | Sequence[float] = 1,  # noqa: ARG002
        subset: str = "all",  # noqa: ARG002
        instrlag: int = 1,
        measure: str = "P",  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrated moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : (2, nobs) array
            Returns and realized variance
        instr_data : array_like, optional
            Instrument data
        instr_choice : str
            Instrument choice
        aggh : int
            Aggregation horizon
        subset : str
            Subset
        instrlag : int
            Number of lags for the instruments
        measure : str
            Measure

        Returns
        -------
        moments : (nobs, nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        assert data is not None
        # (nobs - instrlag, 3) array
        error = self.realized_depvar(data=data).T[instrlag:] - self.realized_const(param=theta)
        # (nobs - instrlag, ninstr)
        instr = self.instruments(data=data, instrlag=instrlag).T
        # (nobs - instrlag, 3 * ninstr = nmoms)
        moms = columnwise_prod(left=error, right=instr)
        # (nintercepts, nparams)
        dmoms = -self.drealized_const(theta)
        dmoments = []
        for minstr in instr.mean(0):
            dmoments.append(dmoms * minstr)
        dmoments_arr = np.vstack(dmoments)

        return moms, dmoments_arr

    def momcond(
        self,
        *,
        theta: GenericParam | np.ndarray | Sequence[float],
        data: np.ndarray | Sequence[np.ndarray] | None = None,
        instrlag: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : array
            Whatever data is necessary to compute moment function
        instrlag : int
            Number of lags for the instruments

        Returns
        -------
        moments : (nobs, nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        assert data is not None
        data_arr = np.asarray(data)
        datalag = 1
        lagdata = cast(np.ndarray, lagmat(data_arr, maxlag=datalag))[datalag:]
        nobs = lagdata.shape[0]
        datamat = np.hstack([np.ones((nobs, 1)), lagdata])

        # Coefficients in the first moment (mean)
        if isinstance(theta, GenericParam):
            theta_vec = theta.get_theta()
        else:
            theta_vec = theta
        linearcoef = [self.betamat(theta_vec), self.gammamat(theta_vec)]
        # Coefficients in the second moment (variance)
        dlinearcoef = [self.dbetamat(theta), self.dgammamat(theta)]

        modelerror = []
        for i in range(len(linearcoef)):
            # Difference between data and model prediction
            error = data_arr[datalag:] ** (i + 1) - datamat.dot(linearcoef[i])
            modelerror.append(error)
        modelerror_arr = np.vstack(modelerror)

        instruments = np.hstack([np.ones((nobs, 1)), cast(np.ndarray, lagmat(data_arr[:-datalag], maxlag=instrlag))]).T

        mom, dmom = [], []
        for instr in instruments:
            mom.append(modelerror_arr * instr)
            meandata = (datamat.T * instr).mean(1)
            dtheta = []
            for coef in dlinearcoef:
                dtheta.append(meandata.dot(coef))
            dtheta_arr = -np.vstack(dtheta)
            dmom.append(dtheta_arr)

        mom_arr = np.vstack(mom).T
        dmom_arr = np.vstack(dmom)

        return mom_arr, dmom_arr
