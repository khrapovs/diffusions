# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""GBM model class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import numdifftools as nd
import numpy as np
from statsmodels.tsa.tsatools import lagmat

from .helper_functions import columnwise_prod
from .model_generic import SDE
from .param_gbm import GBMparam

if TYPE_CHECKING:
    pass

__all__ = ["GBM"]


class GBM(SDE):
    """Geometric Brownian Motion."""

    def __init__(self, param: Any = None) -> None:  # noqa: ANN401
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        super().__init__(param)

    @staticmethod
    def drift(state: np.ndarray | float, theta: Any) -> np.ndarray | float:  # noqa: PLR0917, ARG004, ANN401
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
        return theta.mean - theta.sigma**2 / 2

    @staticmethod
    def diff(state: np.ndarray | float, theta: Any) -> np.ndarray | float:  # noqa: PLR0917, ARG004, ANN401
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
        return theta.sigma

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
        param = GBMparam.from_theta(theta)
        loc = float(self.exact_loc(np.array(0), param))
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
        param = GBMparam.from_theta(theta)
        loc = float(self.exact_loc(np.array(0), param))
        scale = float(self.exact_scale(np.array(0), param))
        return np.array([loc**2 + scale**2, 0], dtype=float)

    def dbetamat(self, theta: np.ndarray | Sequence[float]) -> np.ndarray:
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

    def dgammamat(self, theta: np.ndarray | Sequence[float]) -> np.ndarray:
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
    def realized_depvar(data: np.ndarray, subset: slice | None = None) -> np.ndarray:  # noqa: ARG004, PLR0917
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

    def realized_const(  # noqa: PLR0917
        self,
        param: Any = None,  # noqa: ANN401
        aggh: Any = 1,  # noqa: ARG002, ANN401
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
        theta = param
        mean, sigma = float(theta[0]), float(theta[1])
        return np.array([mean - sigma**2 / 2, sigma**2, sigma**4])

    def drealized_const(self, theta: Any) -> np.ndarray:  # noqa: ANN401
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
        with np.errstate(divide="ignore"):
            return nd.Jacobian(self.realized_const)(theta)

    @staticmethod
    def instruments(data: Any, instrlag: int = 1) -> np.ndarray:  # noqa: PLR0917, ANN401
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

    def integrated_mom(  # noqa: PLR0917
        self,
        theta: Any,  # noqa: ANN401
        data: Any = None,  # noqa: ANN401
        instr_data: Any = None,  # noqa: ARG002, ANN401
        instr_choice: str = "const",  # noqa: ARG002
        aggh: Any = 1,  # noqa: ARG002, ANN401
        subset: str = "all",  # noqa: ARG002
        instrlag: int = 1,
        measure: str = "P",  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrated moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : (2, nobs) array
            Returns and realized variance
        instr_data : object
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
        kwargs : dict
            Keyword args

        Returns
        -------
        moments : (nobs, nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        assert data is not None
        # (nobs - instrlag, 3) array
        error = self.realized_depvar(data).T[instrlag:] - self.realized_const(theta)
        # (nobs - instrlag, ninstr)
        instr = self.instruments(data, instrlag=instrlag).T
        # (nobs - instrlag, 3 * ninstr = nmoms)
        moms = columnwise_prod(error, instr)
        # (nintercepts, nparams)
        dmoms = -self.drealized_const(theta)
        dmoments = []
        for minstr in instr.mean(0):
            dmoments.append(dmoms * minstr)
        dmoments_arr = np.vstack(dmoments)

        return moms, dmoments_arr

    def momcond(  # noqa: PLR0917
        self,
        theta: np.ndarray | Sequence[float],
        data: Any = None,  # noqa: ANN401
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
        linearcoef = [self.betamat(theta), self.gammamat(theta)]
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
