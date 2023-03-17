# fastcubicspline
a fast cubic spline interpolator for equally spaced values and complex data

# Why not using scipy's Cubic Spline?

There are two reasons why fcSpline should be used instead 
of [scipy.interpolate.CubicSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy.interpolate.CubicSpline).

1) When called in a loop, fcSpline it 3 to 5 time faster than CubicSpline (see `fcs_timing.py`).
2) Natively handles complex data.

What are the drawbacks? Well, fcSpline works on equally spaced data only.

# Example

```python
>>> from fastcubicspline import FCS
# set up x-limits
>>> x_low = 1
>>> x_high = 5

# set up the y-data, here complex values
>>> y_data = [9+9j, 4+4j, 0, 6+6j, 2+2j]

# class init
>>> fcs = FCS(x_low, x_high, y_data)

# simply call the FCS-object like a regular function
# to get interpolated values
>>> print(fcs(2.5))
(0.921875+0.921875j)
```

# Install

`fcspline` is on PyPi. So you can simply install the latest release with

    pip install fcspline

## From Source

Fetch the latest version (or check any previous stage) 
by cloning from https://github.com/cimatosa/fcSpline.

### pip

Make sure [pip](https://pip.pypa.io/en/stable/installation/) is installed.
Change into the fcSpline package directory and run:

    python -m pip install .

See the option `--user` if you don't have the appropriate permission
for an installation to the standard location.

### poetry

Change into the fcSpline package directory.
Install `fcspline` and its dependencies into a virtual environment with

    poetry install

and spawn a shell using that environment `poetry shell`.
Now you can check if the tests pass with `pytest`.

In case of poetry errors, you might want to get the latest poetry version
with `poetry self update`.

### Manually Build Cython Extension

Some distutils magic is contained in `build_ext.py` so you can simply call

    python3 build_ext.py

to build the Cython extension inplace.
Run `pytest` to verify that the Cython extension is available.

Clean the build files by calling

    python3 build_ext.py clean


# Testing

Run and list all tests with

    pytest -v



### MIT licence
Copyright (c) 2023 Richard Hartmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

