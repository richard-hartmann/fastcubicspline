# python imports
import warnings
import traceback
from typing import Union, Iterable

# third party imports
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

# fcspline import
try:
    from . import fcs_c

    HAS_FCS_C = True
except ImportError as e:
    warnings.warn(
        "could not import cython extension 'fcs_c' -> use pure Python variant"
    )
    warnings.warn(f"ImportError: {e}")
    traceback.print_exc()
    HAS_FCS_C = False


def _intp(
    x: float, x_low: float, dx: float, y: NDArray, ypp: NDArray, n: int
) -> Union[float, complex]:
    """
    cubic spline interpolation formula, specific for equally spaced x-values

    adapted from, Ch. 3.3 pp 120

        Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P., 2007.
        Numerical Recipes 3rd Edition: The Art of Scientific Computing,
        edition: 3. ed. Cambridge University Press, Cambridge, UK; New York.


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
    j = int((x - x_low) / dx)
    if j < 0:
        j = 0
    elif j >= n - 1:
        j = n - 2
    x_jp1 = x_low + (j + 1) * dx

    a = (x_jp1 - x) / dx
    b = 1 - a

    c = 1 / 6 * (a**3 - a) * dx**2
    d = 1 / 6 * (b**3 - b) * dx**2

    return a * y[j] + b * y[j + 1] + c * ypp[j] + d * ypp[j + 1]


def _intp_array(
    x: NDArray, x_low: float, dx: float, y: NDArray, ypp: NDArray, n: int
) -> NDArray:
    """
    call the interpolation for an array of x values

    Same parameters as in `_intp`. This is a pure Python implementation.

    In most cases you will not need this function directly.
    Use the FCS class instead.
    It will ensure to use the Cython accelerated version if available.

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


def snd_finite_diff(y: NDArray, dx: float, order: int):
    """
    estimate the curvature of the y-array (real or complex) using forward finite difference formulas
    (see https://en.wikipedia.org/wiki/Finite_difference_coefficient#Forward_finite_difference)
    where `order` controls the accuracy.
    Assume equally spaced x values,  i.e. y_i = f(x_i) with difference dx = x_i - x_(i-1).

    Parameters:
        y:
            the array for which to estimate the curvature
        dx:
            the spacing of the x-values
        order:
            the kind of formula to use, the larger `order` the smaller the error.
            Can take the vales 1, 2 or 3.

    Returns:
        the second derivative estimated using forward finite difference
    """
    if order == 1:
        return (y[0] - 2 * y[1] + y[2]) / dx**2
    elif order == 2:
        if len(y) < 4:
            raise RuntimeError(
                "need at least 4 data points to estimate curvature of order 2"
            )
        return (2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / dx**2
    elif order == 3:
        if len(y) < 5:
            raise RuntimeError(
                "need at least 5 data points to estimate curvature of order 3"
            )
        return (
            35 / 12 * y[0]
            - 26 / 3 * y[1]
            + 19 / 2 * y[2]
            - 14 / 3 * y[3]
            + 11 / 12 * y[4]
        ) / dx**2
    else:
        raise ValueError("order must be 1, 2 or 3!")


class FCS(object):
    """
    High level interface for cubic spline interpolation for equally spaced valued (real of complex)

    Once initialized, it can be called like a regular function to obtain interpolated values.

    Example:

        >>> from fastcubicspline import FCS
        >>> # set up x-limits
        >>> x_low = 1
        >>> x_high = 5

        >>> # set up the y-data, here complex values
        >>> y_data = [9+9j, 4+4j, 0, 6+6j, 2+2j]

        >>> # class init
        >>> fcs = FCS(x_low, x_high, y_data)

        >>> # simply call the FCS-object like a regular function
        >>> # to get interpolated values
        >>> print(fcs(2.5))
        >>> # (0.921875+0.921875j)
    """

    def __init__(
        self,
        x_low: float,
        x_high: float,
        y: Union[NDArray, Iterable],
        ypp_specs: Union[None, tuple, int] = None,
        use_pure_python: bool = False,
    ):
        """
        Parameters:
            x_low:
                The x-value that corresponds to y[0] (first y-value).
            x_high:
                The x-value that correspons to y[-1] (last y-value).
            y:
                The y-values to be interpolated.
            ypp_specs:
                Sets the value for the second derivative at x_low / x_high and, thus, controls the behavior when called
                for x-values outside the interval [x_low, x_high].
                The following values are accepted:
                    None - set ypp to zero, meaning linear extrapolation
                    a tuple (ypp_low, ypp_high) - use these values for the curvature
                    an int (1,2 or 3) - estimate the curvature using finite forward difference formulas of that order
            use_pure_python:
                If set True, do not use the Cython accelerated code, even if it is available
                (mainly for testing and debugging).
        """
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
        self.dx = (x_high - x_low) / (self.n - 1)

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
            raise ValueError(
                "unrecognized ypp_specs of type '{}'".format(type(ypp_specs))
            )

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
                warnings.warn(
                    "Note: you are using pure python, even though the c extension is avaiable!"
                )
            self.intp = _intp
            self.intp_array = _intp_array

    def _get_ypp(self):
        """
        Solve the banded linear equation problem to obtain the second derivatives.

        adapted from, Ch. 3.3 pp 120

        Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P., 2007.
        Numerical Recipes 3rd Edition: The Art of Scientific Computing,
        edition: 3. ed. Cambridge University Press, Cambridge, UK; New York.

        Returns:
            the values of the second derivative
        """
        ab = np.zeros(shape=(3, self.n))
        ab[0, 2:] = 1
        ab[1, :] = 4
        ab[2, :-2] = 1

        b = np.empty(shape=self.n, dtype=self.dtype)
        b[1:-1] = (self.y[2:] - 2 * self.y[1:-1] + self.y[:-2]) * 6 / self.dx**2
        b[0] = 4 * self.ypp_l
        b[-1] = 4 * self.ypp_h

        return solve_banded((1, 1), ab, b)

    def __call__(self, x: Union[float, NDArray]):
        """
        Returns the interpolated values for x (numpy arrays accepted).

        Parameters:
            x:
                The x-value for which to calculate the y-value using cubic spline interpolation.
                Using a numpy array on any shape as input will return a numpy array of y-values of that shape.
        Return:
            The interpolated y-value.
        """
        if isinstance(x, np.ndarray):
            res = np.empty(shape=x.shape, dtype=self.dtype)
            flat_res = res.flat
            flat_res[:] = self.intp_array(
                x.flatten(), self.x_low, self.dx, self.y, self.ypp, self.n
            )
            return res
        else:
            return self.intp(x, self.x_low, self.dx, self.y, self.ypp, self.n)


class NPointPoly(object):
    """
    Polynomial Interpolation (Extrapolation)

    For n given points (x_i,y_i) use the uniquely defined polynomial that passes through
    such points, i.e. y_i = P_n(x_i), for interpolation / extrapolation:

        y = p_n(x) for any value of x

    (based on 'Polynomial Interpolation and Extrapolation' from Press, W.H., Teukolsky, S.A., Vetterling, W.T.,
    Flannery, B.P., 2007. Numerical Recipes 3rd Edition: The Art of Scientific Computing,
    edition: 3. ed. Cambridge University Press, Cambridge, UK; New York.)
    """

    def __init__(self, x: Union[NDArray, Iterable], y: Union[NDArray, Iterable]):
        """
        Parameters:
            x:
                the x-values (1D Array of list)
            y:
                the y-values
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n = len(self.x)

    def __call__(self, x: float):
        """
        evaluate the polynomial P_n at the point x.

        Parameters:
            x:
                The x-value
        Returns:
            The value of the polynomial at x, i.e., P_n(x).
        """
        C = self.y
        D = self.y
        res = self.y[0]
        for m in range(self.n - 1):
            x_i = self.x[: -(m + 1)]
            x_i_m_p1 = self.x[m + 1 :]
            D_new = (x_i_m_p1 - x) * (C[1:] - D[:-1]) / (x_i - x_i_m_p1)
            C_new = (x_i - x) * (C[1:] - D[:-1]) / (x_i - x_i_m_p1)
            C = C_new
            D = D_new
            res += C_new[0]
        return res
