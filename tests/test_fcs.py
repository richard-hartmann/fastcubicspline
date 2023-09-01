import numpy as np
import fastcubicspline


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
        fastcubicspline.FCS(xl, xh, y, ypp_specs=4)
    except ValueError as e:
        print("OK: caught ValueError", e)
    else:
        assert False

    for ypp_specs in [None, 1, 2, 3, (10, 20)]:
        for pp in [True, False]:
            spl = fastcubicspline.FCS(
                xl, xh, y, ypp_specs=ypp_specs, use_pure_python=pp
            )
            spl(xl)
            spl(xh)
            spl((xl + xh) / 2)

            spl(xl - 0.001)
            spl(xh + 0.001)

            xfine = np.linspace(xl, xh, 3 * (len(y) - 1) + 1)
            spl(xfine)

            xfine = np.linspace(xl - 0.001, xh, 5)
            spl(xfine)
            xfine = np.linspace(xl, xh + 0.001, 5)
            spl(xfine)


def test_cython_extension():
    # import fastcubicspline.fcs_c
    # print(fcs_c)

    assert fastcubicspline.HAS_FCS_C


def test_calls():
    # real data
    y = [1, 3, 2, 6, 5, 8]
    _calls_helper(y)

    # complex data
    y = np.asarray(y)
    y = y + 1j * y[::-1]
    _calls_helper(y)


def second_deriv(y, dx):
    return np.gradient(np.gradient(y, dx), dx)[2:-2]


def test_spline_property():
    xl = 0
    xh = 10
    n = 15
    x = np.linspace(xl, xh, n)
    y = np.sin(x)

    spl = fastcubicspline.FCS(xl, xh, y, use_pure_python=True)

    # here we check that the spline evaluates exactly to the data points
    for i, xi in enumerate(x):
        d = abs(spl(xi) - y[i])
        assert d < 1e-14, "d={} < 1e-14 failed".format(d)

    # here we check the continuity of the second derivatives
    for fac in [3, 5, 7]:
        xf, dx = np.linspace(xl, xh, 500, retstep=True)
        yf = spl(xf)
        y_pp = second_deriv(yf, dx)
        d = np.abs(y_pp[1:] - y_pp[:-1])
        d1 = np.max(d)

        xf, dx = np.linspace(xl, xh, fac * 500, retstep=True)
        yf = spl(xf)
        y_pp = second_deriv(yf, dx)
        d = np.abs(y_pp[1:] - y_pp[:-1])
        d2 = np.max(d)
        assert abs(fac - d1 / d2) < 0.02

    # here we check convergence for complex function
    xl = 0
    xh = 10

    n = 2 ** (np.asarray([6, 8, 10, 12]))
    mrd = [5e-5, 5e-8, 6e-11, 3e-13]

    for i, ni in enumerate(n):
        x = np.linspace(xl, xh, ni)
        f = lambda x: np.sin(x) + 1j * np.exp(-((x - 5) ** 2) / 10)
        y = f(x)

        spl = fastcubicspline.FCS(xl, xh, y, ypp_specs=3)
        xf = np.linspace(xl, xh, 4 * (ni - 1) + 1)
        yf = spl(xf)
        rd = np.abs(f(xf) - yf) / np.abs(f(xf))
        assert np.max(rd) < mrd[i]


def test_few_points():
    xl = 0
    xh = 1
    n = 5
    x = np.linspace(xl, xh, n)
    y = [1, 2, 1, 2, 1]

    spl = fastcubicspline.FCS(xl, xh, y, use_pure_python=True)
    for i in range(len(y)):
        assert spl(x[i]) == y[i]


def test_cubic_fnc():
    xl = 0
    xh = 4
    n = 5
    x = np.linspace(xl, xh, n)
    y = [0, 1, 8, 27, 64]
    x_fine = np.linspace(xl, xh, 750)

    for use_pure_python in [True, False]:
        # use analytic value for the endpoint curvature
        spl = fastcubicspline.FCS(
            xl, xh, y, ypp_specs=(0, 24), use_pure_python=use_pure_python
        )
        y_spl = spl(x_fine)
        d = np.abs(x_fine**3 - y_spl)
        assert np.max(d) < 1e-14, "{}".format(np.max(d))

        # use third order finite difference approximation for endpoint curvature
        spl = fastcubicspline.FCS(
            xl, xh, y, ypp_specs=3, use_pure_python=use_pure_python
        )
        y_spl = spl(x_fine)
        d = np.abs(x_fine**3 - y_spl)
        assert np.max(d) < 1e-14, "{}".format(np.max(d))


def test_NPointPoly(plot=False):
    x = [0, 1, 2, 3, 4, 5]
    y = [1, 2, 1, 2, 1, 2]

    poly = fastcubicspline.NPointPoly(x, y)
    for i in range(len(x)):
        assert poly(x[i]) == y[i]

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(x, y, ls="", marker=".")

        xx = np.linspace(x[0] - 1, x[-1] + 1, 500)
        yy = [poly(xi) for xi in xx]
        plt.plot(xx, yy)
        plt.show()


if __name__ == "__main__":
    test_calls()
    test_spline_property()
    test_cubic_fnc()
    test_few_points()
    # test_NPointPoly(plot=True))
