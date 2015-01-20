#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometric Brownian Motion

"""
from __future__ import print_function, division

__all__ = ['GBM']


class GBM(object):

    r"""Geometric Brownian Motion.

    Suppose that :math:`S_{t}` evolves according to

    .. math::
        \frac{dS_{t}}{S_{t}}=\mu dt+\sigma dW_{t}.

    In logs:

    .. math::
        d\log S_{t}=\left(\mu-\frac{1}{2}\sigma^{2}\right)dt+\sigma dW_{t}.

    After integration on the interval :math:`\left[t,t+h\right]`:

    .. math::
        r_{t,h}=\log\frac{S_{t+h}}{S_{t}}
            =\left(\mu-\frac{1}{2}\sigma^{2}\right)h
            +\sigma\sqrt{h}\varepsilon_{t+h}.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self):
        # Parameter names
        self.names = ['mu', 'sigma']

    def drift(self, x, theta):
        mu, sigma = theta
        return mu - .5 * sigma ** 2

    def diff(self, x, theta):
        mu, sigma = theta
        return sigma

    def exact_loc(self, x, theta):
        return self.drift(x, theta) * self.h

    def exact_scale(self, x, theta):
        return self.diff(x, theta) * self.h ** .5


if __name__ == '__main__':
    pass
