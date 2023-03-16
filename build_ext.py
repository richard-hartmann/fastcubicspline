"""
This script provides the necessary information to build the Cython extension

It implements two things:

a) provides the hook for the poetry build system by implementing `build(setup_kwargs)`

b) if executed as python script, i.e., `python3 build_ext.py`, do distutils magic to
build the extension inplace which corresponds to former call of `setup.py build_ext --inplace`.
"""

from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
from distutils.extension import Extension
from distutils.dist import Distribution
from distutils.errors import *
import numpy

fcs_c_ext = cythonize(Extension(
    "fcspline.fcs_c",
    sources=["./fcspline/fcs_c.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3'],
))

# the following is adapted from https://stackoverflow.com/a/60163996
class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed('File not found. Could not compile C extension.')

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed('Could not compile C extension.')


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    # NOTE that with cythonize, ext_modules must not be a list!
    # so for more than one Cython extension things will need to be adapted
    setup_kwargs.update(
        {"ext_modules": fcs_c_ext, "cmdclass": {"build_ext": ExtBuilder}}
    )


if __name__ == "__main__":
    # see https://stackoverflow.com/a/60525118
    # distutils magic. This is essentially the same as calling
    # python setup.py build_ext --inplace
    setup_kwargs = {}
    build(setup_kwargs)
    dist = Distribution(attrs=setup_kwargs)

    build_ext_cmd = dist.get_command_obj('build_ext')
    build_ext_cmd.ensure_finalized()
    build_ext_cmd.inplace = 1
    build_ext_cmd.run()
