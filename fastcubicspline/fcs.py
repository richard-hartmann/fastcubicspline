# python imports
import warnings
import traceback
from typing import Union

# third party imports
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

# fcspline import
try:
    from . import fcs_c
    HAS_FCS_C = True
except ImportError as e:
    warnings.warn("could not import cython extension 'fcs_c' -> use pure Python variant")
    warnings.warn(f"ImportError: {e}")
    traceback.print_exc()
    HAS_FCS_C = False


def _intp(x: float, x_low: float, dx: float, y: NDArray, ypp: NDArray, n: int) -> Union[float, complex]:
    """
    cubic spline interpolation formula, specific for equally spaced x-values

    adapted from, Ch. 3.3 pp 120

        Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P., 2007.
        Numerical Recipes 3rd Edition: The Art of Scientific Computing,
        Auflage: 3. ed. Cambridge University Press, Cambridge, UK; New York.


    Parameters:
        x:
            point at which to evaluate the spline
        x_low:
            lowest value of the x-axes, i.e., f(x_low) = y[0]
        dx:
            the spacing of the x values, i.e., dx =x[i+1] - x[i]
        y:
            the y values y[i] = f(x[i])
        ypp:
            the second derivative of the cubic spline at the points x[i], it follows
            consistently by solving a tri-diagonal eigenvalue equation
        n:
            size of the y values

    Returns:
        the value of the cubic spline at x
    """
    j = int((x-x_low) / dx)
    if j < 0:
        j = 0
    elif j >= n - 1:
        j = n - 2
    x_jp1 = x_low + (j + 1) * dx

    a = (x_jp1 - x) / dx
    b = 1 - a

    c = 1 / 6 * (a ** 3 - a) * dx ** 2
    d = 1 / 6 * (b ** 3 - b) * dx ** 2

    return a * y[j] + b * y[j + 1] + c * ypp[j] + d * ypp[j + 1]


def _intp_array(x: NDArray, x_low: float, dx: float, y: NDArray, ypp: NDArray, n: int) -> NDArray:
    """
    call the interpolation for an array of x values

    Same parameters as in `_intp`

    Parameters:
        x:
            points at which to evaluate the spline
        x_low:
            lowest value of the x-axes, i.e., f(x_low) = y[0]
        dx:
            the spacing of the x values, i.e., dx =x[i+1] - x[i]
        y:
            the y values y[i] = f(x[i])
        ypp:
            the second derivative of the cubic spline at the points x[i], it follows
            consistently by solving a tri-diagonal eigenvalue equation
        n:
            size of the y values

    Returns:
        the values of the cubic spline at x-values
    """
    res = np.empty(shape=x.shape, dtype=y.dtype)
    for i, xi in enumerate(x):
        res[i] = _intp(xi, x_low, dx, y, ypp, n)
    return res


# check https://en.wikipedia.org/wiki/Finite_difference_coefficient#Forward_and_backward_finite_difference
def snd_finite_diff(y, dx, _ord):
    if _ord == 1:
        return (y[0] - 2*y[1] + y[2]) / dx**2
    elif _ord == 2:
        if len(y) < 4:
            raise RuntimeError("need at least 4 data points to estimate curvature of order 2")
        return (2*y[0] - 5*y[1] + 4*y[2] - y[3]) / dx**2
    elif _ord == 3:
        if len(y) < 5:
            raise RuntimeError("need at least 5 data points to estimate curvature of order 3")
        return (35/12*y[0] - 26/3*y[1] + 19/2*y[2] - 14/3*y[3] + 11/12*y[4]) / dx**2
    else:
        raise ValueError("order must be 1, 2 or 3!")
    

class FCS(object):
    def __init__(self, x_low, x_high, y, ypp_specs=None, use_pure_python = False):
        if x_high <= x_low:
            raise ValueError("x_high must be greater that x_low")
        self.x_low = x_low

        if np.iscomplexobj(y[0]):
            self.y = np.asarray(y, dtype=np.complex128)
            self.dtype = np.complex128
        else:
            self.y = np.asarray(y, dtype=np.float64)
            self.dtype = np.float64

        if self.y.ndim > 1:
            raise ValueError("y must be 1D")

        self.n = len(y)
        self.dx = (x_high - x_low) / (self.n-1)

        if ypp_specs is None:
            self.ypp_l = 0
            self.ypp_h = 0
        elif isinstance(ypp_specs, tuple):
            self.ypp_l = ypp_specs[0]
            self.ypp_h = ypp_specs[1]
        elif isinstance(ypp_specs, int):
            self.ypp_l = snd_finite_diff(self.y, self.dx, ypp_specs)
            self.ypp_h = snd_finite_diff(self.y[::-1], self.dx, ypp_specs)
        else:
            raise ValueError("unrecognized ypp_specs of type '{}'".format(type(ypp_specs)))

        self.ypp = self._get_ypp()

        # pad with dummy zero to avoid index error
        self.y = np.hstack((self.y, [0]))
        self.ypp = np.hstack((self.ypp, [0]))

        if HAS_FCS_C and not use_pure_python:
            if self.dtype == np.complex128:
                self.intp = fcs_c.intp_cplx
                self.intp_array = fcs_c.intp_cplx_array
            else:
                self.intp = fcs_c.intp
                self.intp_array = fcs_c.intp_array
        else:
            if HAS_FCS_C:
                warnings.warn("Note: you are using pure python, even though the c extension is avaiable!")
            self.intp = _intp
            self.intp_array = _intp_array

    def _get_ypp(self):
        """
        solve the
        :return:
        """
        ab = np.zeros(shape=(3, self.n))
        ab[0, 2:] = 1
        ab[1, :] = 4
        ab[2, :-2] = 1

        b = np.empty(shape=self.n, dtype=self.dtype)
        b[1:-1] = (self.y[2:] - 2 * self.y[1:-1] + self.y[:-2]) * 6 / self.dx ** 2
        b[0] = 4*self.ypp_l
        b[-1] = 4*self.ypp_h

        return solve_banded((1, 1), ab, b)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            res = np.empty(shape=x.shape, dtype=self.dtype)
            flat_res = res.flat
            flat_res[:] = self.intp_array(x.flatten(), self.x_low, self.dx, self.y, self.ypp, self.n)
            return res
        else:
            return self.intp(x, self.x_low, self.dx, self.y, self.ypp, self.n)


class NPointPoly(object):
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n = len(self.x)

    def __call__(self, x):
        C = self.y
        D = self.y
        res = self.y[0]
        for m in range(self.n-1):
            x_i      = self.x[:-(m + 1)]
            x_i_m_p1 = self.x[m + 1:]
            D_new = (x_i_m_p1 - x)*(C[1:] - D[:-1]) / (x_i - x_i_m_p1)
            C_new = (x_i - x)*(C[1:] - D[:-1]) / (x_i - x_i_m_p1)
            C = C_new
            D = D_new
            res += C_new[0]
        return res