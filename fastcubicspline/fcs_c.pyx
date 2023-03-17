#cython: language_level=3, boundscheck=False

cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np


cpdef double intp(double x, double x_low, double dx,
                  np.ndarray[np.float64_t, ndim=1] y,
                  np.ndarray[np.float64_t, ndim=1] ypp, int n) except *:

    cdef int j = int( (x-x_low) / dx)

    if j < 0:
        j = 0
    elif j >= n - 1:
        j = n - 2

    cdef double x_jp1 = x_low + (j + 1) * dx

    cdef double A = (x_jp1 - x) / dx
    cdef double B = 1 - A

    cdef double C = 1 / 6 * (A ** 3 - A) * dx ** 2
    cdef double D = 1 / 6 * (B ** 3 - B) * dx ** 2

    return A * y[j] + B * y[j + 1] + C * ypp[j] + D * ypp[j + 1]

cpdef np.ndarray[np.float64_t, ndim=1] intp_array(np.ndarray[np.float64_t, ndim=1] x, double x_low, double dx,
                                                  np.ndarray[np.float64_t, ndim=1] y,
                                                  np.ndarray[np.float64_t, ndim=1] ypp, int n):
    cdef int i
    cdef int l = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] res = np.empty(l, dtype=np.float64)
    for i in range(l):
        res[i] = intp(x[i], x_low, dx, y, ypp, n)
    return res


cpdef complex intp_cplx(double x, double x_low, double dx,
                        np.ndarray[np.complex128_t, ndim=1] y,
                        np.ndarray[np.complex128_t, ndim=1] ypp, int n) except *:

    cdef int j = int( (x-x_low) / dx)

    if j < 0:
        j = 0
    elif j >= n - 1:
        j = n - 2

    cdef double x_jp1 = x_low + (j + 1) * dx

    cdef double A = (x_jp1 - x) / dx
    cdef double B = 1 - A

    cdef double C = 1 / 6 * (A ** 3 - A) * dx ** 2
    cdef double D = 1 / 6 * (B ** 3 - B) * dx ** 2

    return A * y[j] + B * y[j + 1] + C * ypp[j] + D * ypp[j + 1]

cpdef np.ndarray[np.complex128_t, ndim=1] intp_cplx_array(np.ndarray[np.float64_t, ndim=1] x, double x_low, double dx,
                                                          np.ndarray[np.complex128_t, ndim=1] y,
                                                          np.ndarray[np.complex128_t, ndim=1] ypp, int n):
    cdef int i
    cdef int l = x.shape[0]
    cdef np.ndarray[np.complex128_t, ndim=1] res = np.empty(l, dtype=np.complex128)
    for i in range(l):
        res[i] = intp_cplx(x[i], x_low, dx, y, ypp, n)
    return res