Cox-Ingersoll-Ross (CIR)
------------------------

Suppose that :math:`r_{t}` evolves according to

.. math::
    dr_{t}=\kappa\left(\mu-r_{t}\right)dt+\eta\sqrt{r_{t}}dW_{t}.

Feller condition for positivity of the process is
:math:`\kappa\mu>\frac{1}{2}\eta^{2}`.


.. automodule:: diffusions.param_cir

.. autoclass:: diffusions.param_cir.CIRparam

.. automodule:: diffusions.model_cir

.. autoclass:: diffusions.model_cir.CIR
