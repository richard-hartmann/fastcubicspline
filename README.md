# fcSpline
a fast cubic spline interpolator for equally spaced values

# Why not using scipy's Cubic Spline?

There are two reasons why fcSpline should be used instead 
of [scipy.interpolate.CubicSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy.interpolate.CubicSpline).

1) When called in a loop, fcSpline it 3 to 5 time faster than CubicSpline (see `fcs_timing.py`).
2) Natively handles complex data.

What are the drawbacks? Well, fcSpline works on equally spaced data only.

# Example

    >>> import import fcSpline
    # set up x-limits
    >>> x_low = 1
    >>> x_high = 5
    
    # set up the y-data, here complex values
    >>> y_data = [9+9j, 4+4j, 0, 6+6j, 2+2j]
 
    # class init
    >>> fcs = fcSpline.FCS(x_low, x_high, y_data)
    
    # simply call the FCS-object like a regular function
    # to get interpolated values
    >>> print(fcs(2.5))
    (0.921875+0.921875j)

# Install

Make sure [pip](https://pip.pypa.io/en/stable/installation/) is installed.
After downloading change into the fcSpline directory and run:

    python -m pip install .

See the option `--user` if you don't have the appropriate permission
for an installation to the standard location.

# Source

https://github.com/cimatosa/fcSpline

