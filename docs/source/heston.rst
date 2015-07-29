Heston
------

The model is

.. math::
    dp_{t}&=\left(r+\left(\lambda_r-\frac{1}{2}\sigma_{t}^{2}\right)\right)dt
    +\sigma_{t}dW_{t}^{r},\\
    d\sigma_{t}^{2}&=\kappa\left(\mu-\sigma_{t}^{2}\right)dt
    +\eta\sigma_{t}dW_{t}^{\sigma},

with :math:`p_{t}=\log S_{t}`,
and :math:`Corr\left[dW_{s}^{r},dW_{s}^{\sigma}\right]=\rho`,
or in other words

.. math::
    W_{t}^{\sigma}=\rho W_{t}^{r}+\sqrt{1-\rho^{2}}W_{t}^{v}.

Feller condition for positivity of the volatility process is
:math:`\kappa\mu>\frac{1}{2}\eta^{2}`.


.. automodule:: diffusions.heston_param

.. autoclass:: diffusions.heston_param.HestonParam

.. automodule:: diffusions.heston

.. autoclass:: diffusions.heston.Heston
