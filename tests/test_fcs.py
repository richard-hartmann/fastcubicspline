import sys
import pathlib

__fcs_module_path__ = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(__fcs_module_path__))

import numpy as np
import fcSpline

def catch_exception(func, args, exc):
    try:
        func(*args)
    except exc as e:
        print("OK, caught", exc, e)
    except:
        raise
    else:
        assert False, "no Exception was raised but expected {}".format(exc)

def _calls_helper(y):
    xl = 1
    xh = 5

    try:
        fcSpline.FCS(xl, xh, y, ord_bound_apprx=4)
    except ValueError as e:
        print("OK: caught ValueError", e)
    else:
        assert False

    for ord_bound_apprx in [1,2,3]:
        for pp in [True, False]:
            spl = fcSpline.FCS(xl, xh, y, ord_bound_apprx=ord_bound_apprx, use_pure_python=pp)
            spl(xl)
            spl(xh)
            spl((xl+xh)/2)

            spl(xl - 0.001)
            spl(xh + 0.001)

            xfine = np.linspace(xl, xh, 3*(len(y)-1) + 1)
            spl(xfine)

            xfine = np.linspace(xl - 0.001, xh, 5)
            spl(xfine)
            xfine = np.linspace(xl, xh+0.001, 5)
            spl(xfine)

def test_calls():
    # real data
    y = [1,3,2,6,5,8]
    _calls_helper(y)

    # complex data
    y = np.asarray(y)
    y = y + 1j*y[::-1]
    _calls_helper(y)

def second_deriv(y, dx):
    return np.gradient(np.gradient(y, dx), dx)[2:-2]

def test_spline_property():
    xl = 0
    xh = 10
    n = 15
    x = np.linspace(xl, xh, n)
    y = np.sin(x)

    spl = fcSpline.FCS(xl, xh, y, use_pure_python=True)

    # here we check that the spline evaluates exactly to the data points
    for i, xi in enumerate(x):
        d = abs(spl(xi) - y[i])
        assert d < 1e-14, "d={} < 1e-14 failed".format(d)

    # here we check the continuity of the second derivatives
    for fac in [3,5,7]:
        xf, dx = np.linspace(xl, xh, 500, retstep=True)
        yf = spl(xf)
        y_pp = second_deriv(yf, dx)
        d = np.abs(y_pp[1:] - y_pp[:-1])
        d1 = np.max(d)

        xf, dx = np.linspace(xl, xh, fac*500, retstep=True)
        yf = spl(xf)
        y_pp = second_deriv(yf, dx)
        d = np.abs(y_pp[1:] - y_pp[:-1])
        d2 = np.max(d)
        assert abs(fac - d1 / d2) < 0.02

    # here we check convergence for complex function
    xl = 0
    xh = 10

    n = 2**(np.asarray([6,8,10,12]))
    mrd = [5e-5, 5e-8, 6e-11, 3e-13]

    for i, ni in enumerate(n):
        x = np.linspace(xl, xh, ni)
        f = lambda x: np.sin(x) + 1j*np.exp(-(x-5)**2/10)
        y = f(x)

        spl = fcSpline.FCS(xl, xh, y)
        xf = np.linspace(xl, xh, 4*(ni-1)+1)
        yf = spl(xf)
        rd = np.abs(f(xf) - yf)/np.abs(f(xf))
        assert np.max(rd) < mrd[i]


def test_extr():
    xl = 0
    xh = 5
    n = 5
    x = np.linspace(xl, xh, n)


    y = np.sin(x)
    spl = fcSpline.FCS(xl, xh, y, use_pure_python=True)
    x_fine, dx = np.linspace(xl-1,xh+1, 100*(n+2), retstep=True)
    y_fine = spl(x_fine)
    yp_1 = np.gradient(y_fine, dx)
    yp_2 = np.gradient(yp_1, dx)
    yp_3 = np.gradient(yp_2, dx)
    assert abs(yp_3[50] - yp_3[150]) < 1e-10
    assert abs(yp_3[-50] - yp_3[-150]) < 1e-10

    y = np.sin(x)
    spl = fcSpline.FCS(xl, xh, y)
    x_fine, dx = np.linspace(xl - 1, xh + 1, 100 * (n + 2), retstep=True)
    y_fine = spl(x_fine)
    yp_1 = np.gradient(y_fine, dx)
    yp_2 = np.gradient(yp_1, dx)
    yp_3 = np.gradient(yp_2, dx)
    assert abs(yp_3[50] - yp_3[150]) < 1e-10
    assert abs(yp_3[-50] - yp_3[-150]) < 1e-10

    y = np.sin(x) + 1j*np.exp(-x)
    spl = fcSpline.FCS(xl, xh, y, use_pure_python=True)
    x_fine, dx = np.linspace(xl - 1, xh + 1, 100 * (n + 2), retstep=True)
    y_fine = spl(x_fine)
    yp_1 = np.gradient(y_fine, dx)
    yp_2 = np.gradient(yp_1, dx)
    yp_3 = np.gradient(yp_2, dx)
    assert abs(yp_3[50] - yp_3[150]) < 1e-10
    assert abs(yp_3[-50] - yp_3[-150]) < 1e-10


def test_NPointPoly():
    x = [0, 1, 2, 3, 4, 5]
    y = [1, 2, 1, 2, 1, 2]

    poly = fcSpline.NPointPoly(x, y)

    import matplotlib.pyplot as plt
    plt.plot(x,y, ls='', marker='.')

    xx = np.linspace(x[0]-1, x[-1]+1, 500)
    yy = [poly(xi) for xi in xx]
    plt.plot(xx, yy)
    plt.show()

def test_speedup():
    xl = 0
    xh = 10
    n = 11
    x, dx = np.linspace(xl, xh, n, retstep=True)
    y = np.sin(x)

    spl = fcSpline.FCS(xl, xh, y, use_pure_python=True)

    print(spl(0))

    x0 = 2.2
    print(spl(x0))

    from fcSpline import fcs

    tmp = (x0 - xl) / dx
    idxl = int(tmp)-1
    idxh = int(tmp+2)

    res = 0
    for k in range(idxl, idxh+1):
        res += spl.coef[k+1]* fcs._phi(tmp - k)

    print(res)

    res = 0
    if tmp == idxl+1:
        print("tmp is int")
        res = spl.coef[idxl+1] + 4 * spl.coef[idxl+2] + spl.coef[idxl+3]
    else:
        res =   spl.coef[idxl+1] * (2 - (tmp - idxl)) ** 3 \
              + spl.coef[idxl+2] * (4 - 6 * (tmp - (idxl + 1)) ** 2 + 3 * (tmp - (idxl + 1)) ** 3) \
              + spl.coef[idxl+3] * (4 - 6 * (idxl + 2 - tmp) ** 2 + 3 * (idxl + 2 - tmp) ** 3) \
              + spl.coef[idxl+4] * (2 - (idxl + 3 - tmp)) ** 3
    print(res)




if __name__ == "__main__":
    test_calls()
    test_spline_property()
    test_NPointPoly()
    test_extr()
    test_speedup()