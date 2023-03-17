"""
This script provides the necessary information to build the Cython extension

It implements two things:

a) provides the hook for the poetry build system by implementing `build(setup_kwargs)`

b) if executed as python script, i.e., `python3 build_ext.py`, do distutils magic to
build the extension inplace which corresponds to former call of `setup.py build_ext --inplace`.
"""
from Cython.Build import cythonize
import argparse
from distutils.command.build_ext import build_ext
from distutils.extension import Extension
from distutils.dist import Distribution
from distutils.errors import *
import os
import pathlib
import shutil

import numpy

fcs_c_ext = Extension(
    "fcspline.fcs_c",
    sources=["./fcspline/fcs_c.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3'],
)

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
        {"ext_modules": cythonize(fcs_c_ext), "cmdclass": {"build_ext": ExtBuilder}}
    )


def cmd_build_ext():
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


def cmd_clean():
    root_path = pathlib.Path(__file__).absolute().parent

    dirs_to_remove = [
        root_path / 'build',
        root_path / 'fcspline.egg-info',
        root_path / 'fcspline/__pycache__',
    ]
    for d in dirs_to_remove:
        if d.exists():
            print(f"rm {d}")
            shutil.rmtree(d, ignore_errors=True)
        else:
            print(f"cannot rm {d}, does not exist")

    for f in (root_path / 'fcspline').iterdir():
        name = f.name
        if (
            (name == 'fcs_c.c') or
            (name.startswith('fcs_c.') and name.endswith('.so'))
        ):
            print(f"rm {f}")
            os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        help="what to do, choose between 'build_ext' (default), or 'clear'\n" +
        "  build_ext: triggers Cython inplace build (using distutils magic)\n" +
        "  clean: without asking remove 'build/', 'fcspline.egg-info/', 'fcspline/fcs_c.c', 'fcspline/fcs_c*.so', "+
        "'fcspline/__pycache__'",
        default='build_ext',
        nargs='?',
        type=str,
    )
    args = parser.parse_args()

    if args.command == 'build_ext':
        cmd_build_ext()
    elif args.command == 'clean':
        cmd_clean()
    else:
        raise ValueError(f"unknown command '{args.command}'")
