# Heston

The model is

$$
\begin{aligned}
dp_{t}&=\left(r+\left(\lambda_r-\frac{1}{2}\sigma_{t}^{2}\right)\right)dt
+\sigma_{t}dW_{t}^{r},\\
d\sigma_{t}^{2}&=\kappa\left(\mu-\sigma_{t}^{2}\right)dt
+\eta\sigma_{t}dW_{t}^{\sigma},
\end{aligned}
$$

with $p_{t}=\log S_{t}$,
and $Corr\left[dW_{s}^{r},dW_{s}^{\sigma}\right]=\rho$,
or in other words

$$
W_{t}^{\sigma}=\rho W_{t}^{r}+\sqrt{1-\rho^{2}}W_{t}^{v}.
$$

Feller condition for positivity of the volatility process is
$\kappa\mu>\frac{1}{2}\eta^{2}$.

See the API reference for
[`HestonParam`](../reference/affidiff/param_heston.md) and
[`Heston`](../reference/affidiff/model_heston.md).
