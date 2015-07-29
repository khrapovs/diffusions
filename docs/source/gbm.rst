Geometric Brownian Motion (GBM)
-------------------------------

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
        +\sigma\sqrt{h}\varepsilon_{t+h},

where :math:`\varepsilon_{t}\sim N\left(0,1\right)`.


.. automodule:: diffusions.gbm_param

.. autoclass:: diffusions.gbm_param.GBMparam

.. automodule:: diffusions.gbm

.. autoclass:: diffusions.gbm.GBM
