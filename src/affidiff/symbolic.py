#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Symbolic calculations

"""

import sympy as sp

mu = sp.Symbol('mu', positive=True)
kappa_s = sp.Symbol('kappa_s', positive=True)
kappa_y = sp.Symbol('kappa_y', positive=True)
eta_s = sp.Symbol('eta_s', positive=True)
eta_y = sp.Symbol('eta_y', positive=True)
h = sp.Symbol('aggh', positive=True)
u = sp.Symbol('u', positive=True)

#%%
big_as = sp.exp(-kappa_s * h)
big_ay = sp.exp(-kappa_y * h)
big_bs = kappa_s / (kappa_s - kappa_y) * (big_ay - big_as)

small_as = sp.integrate(big_as, (h, 0, h))
small_bs = sp.integrate(big_bs, (h, 0, h)).simplify()

small_ay = sp.integrate(big_ay, (h, 0, h))

assert (small_bs - kappa_s / (kappa_s - kappa_y)
    * (small_ay - small_as)).simplify() == 0

small_bs = kappa_s / (kappa_s - kappa_y) * (small_ay - small_as)

#%%
temp1 = (big_as.subs(h, -u)**2).simplify()
temp2 = (big_bs.subs(h, -u)**2).simplify()

var_sigma = mu * (eta_s**2 * sp.integrate(temp1, (u, -sp.oo, 0))
    + eta_y**2 * sp.integrate(temp2, (u, -sp.oo, 0)))

var_sigma = var_sigma.simplify()

print(var_sigma)

#%%
temp1 = (small_as.subs(h, h-u)**2)
temp2 = (small_bs.subs(h, h-u)**2)

var_error = mu * (eta_s**2 * sp.integrate(temp1, (u, 0, h))
    + eta_y**2 * sp.integrate(temp2, (u, 0, h))) / h**2

var_error = var_error.simplify()

print(var_error)
