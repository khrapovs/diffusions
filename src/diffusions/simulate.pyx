cimport cython
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_lapack cimport dpotrf

__all__ = ['simulate']


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simulate(double[:, :, :] errors, double[:] start,
             double[:] mat_k0, double[:, :] mat_k1,
             double[:, :] mat_h0, double[:, :, :] mat_h1, double dt):
    """Simulate AJD process.

    Parameters
    ----------
    errors : (npoints, nsim, nvars) array
        Standard Normal innovations
    start : (nvars, ) array
        Initial values for variables
    mat_k0 : (nvars, ) array
        Parameter matrix for drift (intercept)
    mat_k1 : (nvars, nvars) array
        Parameter matrix for drift (slope)
    mat_h0 : (nvars, nvars) array
        Parameter matrix for diffusion (intercept)
    mat_h0 : (nvars, nvars, nvars) array
        Parameter matrix for diffusion (slope)
    dt : float
        Time interval

    Returns
    -------
    paths : (npoints + 1, nsim, nvars) array
        Simulated paths

    """
    cdef:
        Py_ssize_t i, j, s, t
        int info = 0
        int npoints = errors.shape[0]
        int nsim = errors.shape[1]
        int nvars = errors.shape[2]
        double[:] drift = np.empty(nvars, float)
        double[:] diff = np.empty(nvars, float)
        double[:, :] var = np.empty((nvars, nvars), float)
        double[:, :, :] paths = np.empty((npoints + 1, nsim, nvars), float)

    for s in range(nsim):
        for i in range(nvars):
            paths[0, s, i] = start[i]

    for t in range(npoints):
        for s in range(nsim):

            for i in range(nvars):
                for j in range(nvars):
                    var[i, j] = mat_h0[i, j]
                    for k in range(nvars):
                        var[i, j] += mat_h1[k, i, j] * paths[t, s, k]

            # http://www.math.utah.edu/software/lapack/lapack-d/dpotrf.html
            dpotrf('U', &nvars, &var[0, 0], &nvars, &info)

            for i in range(nvars):
                drift[i] = mat_k0[i]
                diff[i] = 0.0
                for j in range(nvars):
                    drift[i] += mat_k1[i, j] * paths[t, s, j]
                for j in range(i+1):
                    diff[i] += var[i, j] * errors[t, s, j]
                paths[t+1, s, i] = paths[t, s, i]
                paths[t+1, s, i] += drift[i] * dt + diff[i] * dt**.5

    return np.asarray(paths)
