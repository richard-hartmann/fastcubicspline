from scipy.interpolate import CubicSpline
import numpy as np
from fastcubicspline import FCS
import time

# init data
x_low = 1
x_high = 5
x_data = np.linspace(x_low, x_high, x_high)
y_data = [9, 4, 0, 6, 2]

# init Spline interpolators
fcs = FCS(x_low, x_high, y_data)
cs = CubicSpline(x_data, y_data)

# fine x-data we want interpolation for
N = 1000
x = np.linspace(x_low, x_high, N)

print("do {} interpolations in a python loop".format(N))

# use fast cubic spline
t0 = time.perf_counter_ns()
for xi in x:
    fcs(xi)
t1 = time.perf_counter_ns()
dt_fcs = (t1 - t0) / 10**6
print(" fast cubic spline: {:.2f}ms".format(dt_fcs))


# use scipy CubicSpline
t0 = time.perf_counter_ns()
for xi in x:
    cs(xi)
t1 = time.perf_counter_ns()
dt_cs = (t1 - t0) / 10**6
print("scipy cubic spline: {:.2f}ms".format(dt_cs))

print("fcs is {:.2f}x faster".format(dt_cs / dt_fcs))
