# Cox-Ingersoll-Ross (CIR)

Suppose that $r_{t}$ evolves according to

$$
dr_{t}=\kappa\left(\mu-r_{t}\right)dt+\eta\sqrt{r_{t}}dW_{t}.
$$

Feller condition for positivity of the process is
$\kappa\mu>\frac{1}{2}\eta^{2}$.

See the API reference for
[`CIRparam`](../reference/affidiff/parameters/cir.md) and
[`CIR`](../reference/affidiff/models/cir.md).
