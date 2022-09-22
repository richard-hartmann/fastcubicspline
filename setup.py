from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
import numpy

setup(
        name='fcSpline',
        ext_modules = cythonize(Extension("fcSpline.fcs_c",
                                          ["fcSpline/fcs_c.pyx"],
                                          extra_compile_args=['-O3'],
                                          include_dirs = [numpy.get_include()],
                                          )),
)