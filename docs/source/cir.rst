Cox-Ingersoll-Ross (CIR) model
------------------------------

Suppose that :math:`r_{t}` evolves according to

.. math::
    dr_{t}=\kappa\left(\mu-r_{t}\right)dt+\eta\sqrt{r_{t}}dW_{t}.

Feller condition for positivity of the process is
:math:`\kappa\mu>\frac{1}{2}\eta^{2}`.


.. automodule:: diffusions.cir

.. autoclass:: diffusions.cir.CIR

.. automodule:: diffusions.cir_param

.. autoclass:: diffusions.cir_param.CIRparam
