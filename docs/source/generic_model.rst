General Affine Diffusions
-------------------------

A jump-diffusion process is a Markov process solving the stochastic
differential equationd

.. math::
    Y_{t}=\mu\left(Y_{t},\theta_{0}\right)dt
        +\sigma\left(Y_{t},\theta_{0}\right)dW_{t}.

A discount-rate function :math:`R:D\to\mathbb{R}` is an affine function of the
state

.. math::
    R\left(Y\right)=\rho_{0}+\rho_{1}\cdot Y,

for :math:`\rho=\left(\rho_{0},\rho_{1}\right)\in\mathbb{R}
\times\mathbb{R}^{N}`.
The affine dependence of the drift and diffusion coefficients of :math:`Y` are
determined by coefficients :math:`\left(K,H\right)` defined by:

:math:`\mu\left(Y\right)=K_{0}+K_{1}Y`,
for :math:`K=\left(K_{0},K_{1}\right)
\in\mathbb{R}^{N}\times\mathbb{R}^{N\times N}`,

and

:math:`\left[\sigma\left(Y\right)\sigma\left(Y\right)^{\prime}\right]_{ij}
=\left[H_{0}\right]_{ij}+\left[H_{1}\right]_{ij}\cdot Y`,
for :math:`H=\left(H_{0},H_{1}\right)\in\mathbb{R}^{N\times N}
\times\mathbb{R}^{N\times N\times N}`.

Here

.. math::
    \left[H_{1}\right]_{ij}\cdot Y=\sum_{k=1}^{N}\left[H_{1}\right]_{ijk}Y_{k}.


A characteristic :math:`\chi=\left(K,H,\rho\right)`
captures both the distribution
of :math:`Y` as well as the effects of any discounting.


.. automodule:: diffusions.generic_param

.. autoclass:: diffusions.generic_param.GenericParam

.. automodule:: diffusions.generic_model

.. autoclass:: diffusions.generic_model.SDE
	:members: simulate, sim_realized, gmmest, integrated_gmm
