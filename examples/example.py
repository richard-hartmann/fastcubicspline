import sys
import pathlib

__fcs_module_path__ = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(__fcs_module_path__))

import numpy as np
import matplotlib.pyplot as plt
import fcSpline

def ex1():
    n = 15
    xl = -10
    xh = 10
    f = lambda x: np.sin(x)
    x = np.linspace(xl, xh, n)
    y = f(x)
    spl = fcSpline.FCS(xl, xh, y)

    xfine, dxfine = np.linspace(xl, xh, 500, retstep=True)
    yfine = spl(xfine)

    plt.plot(x, y, ls='', marker='.', label='data set y=sin(x)')
    plt.plot(xfine, yfine, label='interpol.')


    y_pp = np.gradient(np.gradient(yfine, dxfine), dxfine)
    plt.plot(xfine, y_pp, label='2nd derv.')

    plt.grid()
    plt.legend()
    plt.show()

def ex2():
    # set up x-limits
    x_low = 1
    x_high = 5

    # set up the y-data, here complex values
    y_data = [9+9j, 4+4j, 0, 6+6j, 2+2j]

    # class init
    fcs = fcSpline.FCS(x_low, x_high, y_data)

    # simply call the FCS-object like a regular function
    # to get interpolated values
    print(fcs(2.5))


def compare_speed():
    from scipy.interpolate import InterpolatedUnivariateSpline
    from time import time
    for n in [15, 150 ,1500, 15000]:

        xl = -10
        xh = 10
        f = lambda x: np.sin(x)
        x = np.linspace(xl, xh, n)
        y = f(x)
        t0 = time()
        spl = fcSpline.FCS(xl, xh, y)
        t1 = time()
        spl_scip = InterpolatedUnivariateSpline(x, y, k=3)
        t2 = time()
        print("n", n)
        print("INIT  -  fcs: {:.3e}s, sci {:.3e}s  factor {:.3g}".format(t1-t0, t2-t1, (t2-t1) / (t1-t0) ))
        t_fcs = t_sci = 0

        N = 10000

        for i in range(10000):
            x = np.random.rand()*(xh-xl) + xl
            t0 = time()
            spl(x)
            t1 = time()
            spl_scip(x)
            t2 = time()

            t_fcs += (t1 - t0)
            t_sci += (t2 - t1)

        print("EVAL  -  fcs: {:.3e}s, sci {:.3e}s  factor {:.3g}".format(t_fcs/N, t_sci/N, t_sci / t_fcs))



if __name__ == "__main__":
    # ex1()
    ex2()
    # compare_speed()
