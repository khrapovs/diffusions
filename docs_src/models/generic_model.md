# General Affine Diffusions

A jump-diffusion process is a Markov process solving the stochastic
differential equation

$$
Y_{t}=\mu\left(Y_{t},\theta_{0}\right)dt
    +\sigma\left(Y_{t},\theta_{0}\right)dW_{t}.
$$

A discount-rate function $R:D\to\mathbb{R}$ is an affine function of the
state

$$
R\left(Y\right)=\rho_{0}+\rho_{1}\cdot Y,
$$

for $\rho=\left(\rho_{0},\rho_{1}\right)\in\mathbb{R}\times\mathbb{R}^{N}$.

The affine dependence of the drift and diffusion coefficients of $Y$ are
determined by coefficients $\left(K,H\right)$ defined by:

$$
\mu\left(Y\right)=K_{0}+K_{1}Y,
$$

for $K=\left(K_{0},K_{1}\right)\in\mathbb{R}^{N}\times\mathbb{R}^{N\times N}$,

and

$$
\left[\sigma\left(Y\right)\sigma\left(Y\right)^{\prime}\right]_{ij}
=\left[H_{0}\right]_{ij}+\left[H_{1}\right]_{ij}\cdot Y,
$$

for $H=\left(H_{0},H_{1}\right)\in\mathbb{R}^{N\times N}\times\mathbb{R}^{N\times N\times N}$.

Here

$$
\left[H_{1}\right]_{ij}\cdot Y=\sum_{k=1}^{N}\left[H_{1}\right]_{ijk}Y_{k}.
$$

A characteristic $\chi=\left(K,H,\rho\right)$ captures both the distribution
of $Y$ as well as the effects of any discounting.

See the API reference for
[`GenericParam`](../reference/affidiff/param_generic.md) and
[`SDE`](../reference/affidiff/model_generic.md).
