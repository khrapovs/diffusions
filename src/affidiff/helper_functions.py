"""Helper functions."""

from __future__ import annotations

import contextlib
import itertools as it
import time
from typing import Any, Callable, Generator, Sequence

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.tsatools import lagmat


def ajd_drift(*, state: Any, theta: Any) -> np.ndarray:  # noqa: ANN401
    """Instantaneous mean.

    Parameters
    ----------
    state : (nsim, nvars) array_like
        Current value of the process
    theta : parameter instance
        Model parameter

    Returns
    -------
    (nsim, nvars) array_like
        Value of the drift

    """
    state_arr = np.atleast_2d(state)
    return theta.mat_k0 + state_arr.dot(np.transpose(theta.mat_k1))


def ajd_diff(*, state: Any, theta: Any) -> np.ndarray:  # noqa: ANN401
    """Instantaneous volatility.

    Parameters
    ----------
    state : (nsim, nvars) array_like
        Current value of the process
    theta : parameter instance
        Model parameter

    Returns
    -------
    (nsim, nvars, nvars) array_like
        Value of the diffusion

    """
    state_arr = np.atleast_2d(state)
    mat_h1 = np.atleast_3d(theta.mat_h1)
    # (nsim, nvars, nvars)
    var = theta.mat_h0 + np.tensordot(state_arr, mat_h1, axes=(1, 0))
    try:
        return np.linalg.cholesky(var)
    except np.linalg.LinAlgError:
        return np.ones_like(var) * 1e10


def nice_errors(*, errors: np.ndarray, sdim: int) -> np.ndarray:
    """Normalize the errors and apply antithetic sampling.

    Parameters
    ----------
    errors : array
        Innovations to be standardized
    sdim : int
        Which dimension corresponds to simulation instances?

    Returns
    -------
    errors : array
        Standardized innovations

    """
    if errors.shape[sdim] > 10:
        errors = errors - errors.mean(sdim, keepdims=True)
        errors = errors / errors.std(sdim, keepdims=True)
    return np.concatenate((errors, -errors), axis=sdim)


def plot_trajectories(*, paths: Any, nsub: int, names: str | list[str]) -> None:  # noqa: ANN401
    """Plot process realizations.

    Parameters
    ----------
    paths : array
        Process realizations. Shape is either (nobs,) or (nobs, nsim)
    nsub : int
        Number of subintervals inside of unit interval
    names : str or list of strings
        Labels

    """
    if isinstance(paths, list):
        assert isinstance(names, list)
        for path_elem, name in zip(paths, names, strict=False):
            p_arr = np.asarray(path_elem)
            x = np.arange(0, p_arr.shape[0] / nsub, 1 / nsub)
            plt.plot(x, p_arr, label=name)
    else:
        p_arr = np.asarray(paths)
        x = np.arange(0, p_arr.shape[0] / nsub, 1 / nsub)
        plt.plot(x, p_arr, label=names)

    plt.xlabel("$t$")
    plt.ylabel("$x_t$")
    plt.legend()
    plt.show()


def plot_final_distr(*, paths: Any, names: str | list[str]) -> None:  # noqa: ANN401
    """Plot marginal distribution of the process.

    Parameters
    ----------
    paths : array
        Process realizations. Shape is (nobs, nsim)
    names : str or list of strings
        Labels

    """
    if isinstance(paths, list):
        assert isinstance(names, list)
        for path_elem, name in zip(paths, names, strict=False):
            p_arr = np.asarray(path_elem)
            if p_arr.ndim != 2:
                raise ValueError("Simulate more paths!")
            sns.kdeplot(p_arr[-1], label=name)
    else:
        p_arr = np.asarray(paths)
        sns.kdeplot(p_arr[-1], label=names)

    plt.xlabel("x")
    plt.legend()
    plt.show()


def plot_realized(
    *,
    returns: Any,  # noqa: ANN401
    rvar: Any,  # noqa: ANN401
    suffix: list[str] | None = None,
) -> None:
    """Plot realized returns and volatility.

    Parameters
    ----------
    returns : array
        Returns
    rvar : array
        Realized variance
    suffix : list of str
        Label suffixes

    """
    _fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 6))
    if isinstance(returns, list):
        returns_arr = np.vstack(returns).T
        rlabel = ["Returns " + x for x in (suffix or [])]
    else:
        returns_arr = returns
        rlabel = ["Returns"]
    if isinstance(rvar, list):
        rvar_arr = np.vstack(rvar).T
        vlabel = ["Realized volatility " + x for x in (suffix or [])]
    else:
        rvar_arr = rvar
        vlabel = ["Realized volatility"]
    axes[0].plot(returns_arr)
    axes[1].plot(rvar_arr**0.5)
    axes[0].legend(rlabel)
    axes[1].legend(vlabel)
    plt.show()


def columnwise_prod(*, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Columnwise kronker product.

    Parameters
    ----------
    left : (n, m) array
    right : (n, p) array

    Returns
    -------
    (n, m*p) array
        [left * right[:, 0], ..., left * right[:, -1]]

    Example
    -------
    .. doctest::

        >>> left = np.arange(6).reshape((3,2))
        >>> left
        array([[0, 1],
               [2, 3],
               [4, 5]])
        >>> right = np.arange(9).reshape((3,3))
        >>> right
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        >>> columnwise_prod(left, right)
        array([[ 0,  0,  0,  1,  0,  2],
               [ 6,  9,  8, 12, 10, 15],
               [24, 30, 28, 35, 32, 40]])

    """
    prod = left[:, np.newaxis, :] * right[:, :, np.newaxis]
    return prod.reshape((left.shape[0], left.shape[1] * right.shape[1]))


def rolling_window(*, fun: Callable[..., Any], mat: np.ndarray, window: int = 1) -> np.ndarray:
    """Apply function over rolling window.

    Source: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html

    Parameters
    ----------
    fun : function
        Function to apply
    mat : array_like
        Data to transform
    window : int
        Window size

    Returns
    -------
    array

    Examples
    --------
    .. doctest::

        >>> mat = np.arange(10).reshape((2,5))
        >>> mat
        array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]])
        >>> rolling_window(np.mean, mat, window=2)
        array([[ 0.5,  1.5,  2.5,  3.5],
               [ 5.5,  6.5,  7.5,  8.5]])

    """
    shape = mat.shape[:-1] + (mat.shape[-1] - window + 1, window)
    strides = mat.strides + (mat.strides[-1],)
    mat_s = np.lib.stride_tricks.as_strided(mat, shape=shape, strides=strides)
    return np.apply_along_axis(fun, -1, mat_s)


def poly_coef(roots: Sequence[float] | np.ndarray) -> list[float]:
    """Ploynomial coefficients.

    Parameters
    ----------
    roots : list of floats
        Roots of the polynomial, i.e. [a0, a1] for (1 - a0 * x) * (1 - a1 * x)

    Returns
    -------
    coefs : list of floats
        List of all polynomial coefficients, i.e. [p0, p1, p2]
        with p0 = 1, p1 = - a0 - a1, p2 = a0*a1

    Examples
    --------
    .. doctest::

        >>> poly_coef([2, 3])
        [1, -5, 6]

        >>> poly_coef([2, 3, 4])
        [1, -9, 26, -24]

    """
    roots_arr = np.array(roots)
    nroots = roots_arr.size
    coefs: list[float] = [1.0]
    for power in range(nroots):
        comb = it.combinations(range(nroots), power + 1)
        temp = [float(np.prod(roots_arr[[x]])) for x in comb]
        coefs.append(float((-1) ** (power + 1) * np.sum(temp)))
    return coefs


def instruments(
    *,
    data: Any = None,  # noqa: ANN401
    instrlag: int = 1,
    nobs: int | None = None,
    instr_choice: str = "const",
) -> np.ndarray:
    """Create an array of instruments.

    Parameters
    ----------
    data : (ninstr, nobs) array
        Returns and realized variance
    instrlag : int
        Number of lags for the instruments
    nobs : int
        Number of observations in the data to match
    instr_choice : str {'const', 'var'}
        Choice of instruments.
            - 'const' : just a constant (unconditional moments)
            - 'var' : lags of instrument data

    Returns
    -------
    (nobs, ninstr*instrlag + 1) array
        Instrument array

    Examples
    --------
    .. doctest::

        >>> instruments(nobs=3)
        array([[ 1.],
               [ 1.],
               [ 1.]])

        >>> data = np.arange(6).reshape((2,3))
        >>> instruments(data=data)
        array([[ 1.],
               [ 1.],
               [ 1.]])
        >>> instruments(data=data, instr_choice='var')
        array([[ 1.,  0.,  0.],
               [ 1.,  0.,  3.],
               [ 1.,  1.,  4.]])
        >>> instruments(data=data, instr_choice='var', instrlag=2)
        array([[ 1.,  0.,  0.,  0.,  0.],
               [ 1.,  0.,  3.,  0.,  0.],
               [ 1.,  1.,  4.,  0.,  3.]])

    """
    if data is not None:
        nobs = data.shape[-1]

    if instr_choice == "const" or data is None:
        if nobs is None:
            raise ValueError("Specify nobs!")
        return np.ones((nobs, 1))

    else:
        instr = lagmat(np.atleast_2d(data).T, maxlag=instrlag)
        width = ((0, 0), (1, 0))
        return np.pad(instr, width, mode="constant", constant_values=1)


def format_time(t: float) -> str:
    """Format time for nice printing.

    Parameters
    ----------
    t : float
        Time in seconds

    Returns
    -------
    format template

    """
    if t > 60 or t == 0:
        units = "min"
        t /= 60
    elif t > 1:
        units = "s"
    elif t > 1e-3:
        units = "ms"
        t *= 1e3
    elif t > 1e-6:
        units = "us"
        t *= 1e6
    else:
        units = "ns"
        t *= 1e9
    return f"{t:.1f} {units}"


@contextlib.contextmanager
def take_time(desc: str) -> Generator[None, None, None]:
    """Context manager for timing the code.

    Parameters
    ----------
    desc : str
        Description of the code

    Example
    -------
    >>> with take_time('Estimation'):
    >>>    estimate()

    """
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"{desc} took {format_time(dt)}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    doctest.testmod()
