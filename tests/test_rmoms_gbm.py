"""Test suite for realized moments of GBM."""

import numpy as np

from affidiff import GBM, GBMparam


class TestRealizedMomentsGBM:
    """Test realized moments for GBM."""

    def test_gbm_relized_mom(self) -> None:
        """Test realized moments of GBM model."""
        mean, sigma = 1.5, 0.2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        gbm.nsub = 2

        nperiods = 10
        data = np.ones((2, nperiods))
        instrlag = 2

        depvar = gbm.realized_depvar(data)
        # Test shape of dependent variables
        assert depvar.shape == (3, nperiods)

        const = gbm.realized_const(param.get_theta())
        # Test shape of the intercept
        assert const.shape == (3,)

        instr = gbm.instruments(data, instrlag=instrlag)
        ninstr = 1 + data.shape[0] * instrlag
        # Test shape of instrument matrix
        assert instr.shape == (ninstr, nperiods - instrlag)

        rmom, drmom = gbm.integrated_mom(param.get_theta(), data=data, instrlag=instrlag)
        nmoms = 3 * ninstr
        # Test shape of moments and gradients
        assert rmom.shape == (nperiods - instrlag, nmoms)
        assert drmom.shape == (nmoms, np.size(param.get_theta()))
