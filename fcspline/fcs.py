import numpy as np
from scipy.linalg import solve_banded
import traceback
import warnings

try:
    from . import fcs_c
    has_fcs_s = True
except ImportError:
    warnings.warn("could not import cython extension 'fcs_c' -> use pure Python variant")
    traceback.print_exc()
    has_fcs_s = False
    

def _intp(x, x_low, dx, y, ypp, n):
    j = int( (x-x_low) / dx)
    if j < 0:
        j = 0
    elif j >= n - 1:
        j = n - 2
    x_jp1 = x_low + (j + 1) * dx

    A = (x_jp1 - x) / dx
    B = 1 - A

    C = 1 / 6 * (A ** 3 - A) * dx ** 2
    D = 1 / 6 * (B ** 3 - B) * dx ** 2

    return A * y[j] + B * y[j + 1] + C * ypp[j] + D * ypp[j + 1]


def _intp_array(x, x_low, dx, y, ypp, n):
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

        if has_fcs_s and not use_pure_python:
            if self.dtype == np.complex128:
                self.intp = fcs_c.intp_cplx
                self.intp_array = fcs_c.intp_cplx_array
            else:
                self.intp = fcs_c.intp
                self.intp_array = fcs_c.intp_array
        else:
            if has_fcs_s:
                warnings.warn("Note: you are using pure python, even though the c extension is avaiable!")
            self.intp = _intp
            self.intp_array = _intp_array


    def _get_ypp(self):
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