from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
import numpy

author      = u"Richard Hartmann"
authors     = [author]
description = 'A fast cubic spline interpolator for real and complex data.'
name        = 'fcSpline'
version     = '1.0'


setup(
        name=name,
        author=author,
        author_email='richard.hartmann@tu-dresden.de',
        url='https://github.com/cimatosa/fcSpline',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
        long_description=description,
        keywords=["fast", "cubic", "spline"],
        classifiers= [
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'License :: OSI Approved :: BSD License',
            'Topic :: Utilities',
            'Intended Audience :: Researcher'],
        platforms=['ALL'],
        setup_requires=['cython>=0.29'],
        install_requires=['numpy>=1.20', 'scipy>=1.6'],
        ext_modules = cythonize(Extension("fcSpline.fcs_c",
                                          ["./fcSpline/fcs_c.pyx"],
                                          extra_compile_args=['-O3'],
                                          include_dirs = [numpy.get_include()],
                                          )),

)