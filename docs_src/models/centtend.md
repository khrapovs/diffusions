# Central Tendency (CT)

The model is

$$
\begin{aligned}
dp_{t}&=\left(r+\left(\lambda-\frac{1}{2}\right)
    \sigma_{t}^{2}\right)dt+\sigma_{t}dW_{t}^{r},\\
d\sigma_{t}^{2}&=\kappa_{\sigma}\left(v_{t}^{2}-\sigma_{t}^{2}\right)dt
    +\eta_{\sigma}\sigma_{t}dW_{t}^{\sigma},\\
dv_{t}^{2}&=\kappa_{v}\left(\mu-v_{t}^{2}\right)dt+\eta_{v}v_{t}dW_{t}^{v},
\end{aligned}
$$

with $p_{t}=\log S_{t}$,
and $Corr\left[dW_{s}^{r},dW_{s}^{\sigma}\right]=\rho$,
or in other words
$W_{t}^{\sigma}=\rho W_{t}^{r}+\sqrt{1-\rho^{2}}W_{t}^{v}$.
Also let $R\left(Y_{t}\right)=r$.

See the API reference for
[`CentTendParam`](../reference/affidiff/param_ct.md) and
[`CentTend`](../reference/affidiff/model_ct.md).
