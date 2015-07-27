Central Tendency (CT) model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is

.. math::
    dp_{t}&=\left(r+\left(\lambda-\frac{1}{2}\right)
        \sigma_{t}^{2}\right)dt+\sigma_{t}dW_{t}^{r},\\
    d\sigma_{t}^{2}&=\kappa_{\sigma}\left(v_{t}^{2}-\sigma_{t}^{2}\right)dt
        +\eta_{\sigma}\sigma_{t}dW_{t}^{\sigma},\\
    dv_{t}^{2}&=\kappa_{v}\left(\mu-v_{t}^{2}\right)dt+\eta_{v}v_{t}dW_{t}^{v},

with :math:`p_{t}=\log S_{t}`,
and :math:`Corr\left[dW_{s}^{r},dW_{s}^{\sigma}\right]=\rho`,
or in other words
:math:`W_{t}^{\sigma}=\rho W_{t}^{r}+\sqrt{1-\rho^{2}}W_{t}^{v}`.
Also let :math:`R\left(Y_{t}\right)=r`.


.. automodule:: diffusions.central_tendency

.. autoclass:: diffusions.central_tendency.CentTend

.. automodule:: diffusions.central_tendency_param

.. autoclass:: diffusions.central_tendency_param.CentTendParam
