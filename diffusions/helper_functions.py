#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions

"""
from __future__ import print_function, division

import numpy as np

__all__ = ['nice_errors']


def nice_errors(errors, sdim):
    """Normalize the errors and apply antithetic sampling.

    Parameters
    ----------
    errors : array
        Innovations to be standardized
    sdim : int
        Which dimention corresponds to simulation instances?

    Returns
    -------
    errors : array
        Standardized innovations

    """
    errors -= errors.mean(sdim, keepdims=True)
    errors /= errors.std(sdim, keepdims=True)
    errors = np.concatenate((errors, -errors), axis=sdim)
    return errors
