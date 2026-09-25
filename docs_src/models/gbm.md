# Geometric Brownian Motion (GBM)

Suppose that $S_{t}$ evolves according to

$$
\frac{dS_{t}}{S_{t}}=\mu dt+\sigma dW_{t}.
$$

In logs:

$$
d\log S_{t}=\left(\mu-\frac{1}{2}\sigma^{2}\right)dt+\sigma dW_{t}.
$$

After integration on the interval $\left[t,t+h\right]$:

$$
r_{t,h}=\log\frac{S_{t+h}}{S_{t}}
    =\left(\mu-\frac{1}{2}\sigma^{2}\right)h
    +\sigma\sqrt{h}\varepsilon_{t+h},
$$

where $\varepsilon_{t}\sim N\left(0,1\right)$.

See the API reference for
[`GBMparam`](../reference/affidiff/param_gbm.md) and
[`GBM`](../reference/affidiff/model_gbm.md).
