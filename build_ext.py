"""
This script provides the necessary information to build the Cython extension.

Default execution (just as poetry does), i.e., `python3 build_ext.py` does dome distutils magic to
build the extension inplace which corresponds to former call of `setup.py build_ext --inplace`.

running with the extra argument 'clean' deletes without asking any generated files, i.e,
remove 'build/', 'dist/', 'fastcubicspline.egg-info/', 'fastcubicspline/fcs_c.c',
'fastcubicspline/fcs_c*.so', 'fastcubicspline/__pycache__'.
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

# assume that the build_ext.py script is in the root directory of the package
root_path = pathlib.Path(__file__).absolute().parent

fcs_c_ext = Extension(
    "fastcubicspline.fcs_c",
    sources=["./fastcubicspline/fcs_c.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"],
)

class cd:
    """
    Context manager for changing the current working directory

    taken from https://stackoverflow.com/questions/431684/equivalent-of-shell-cd-command-to-change-the-working-directory/13197763#13197763
    """
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


# the following is adapted from https://stackoverflow.com/a/60163996
class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    def run(self):
        try:
            with cd(root_path):
                build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed("File not found. Could not compile C extension.")

    def build_extension(self, ext):
        try:
            with cd(root_path):
                build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed("Could not compile C extension.")


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    # NOTE that with cythonize, ext_modules must not be a list!
    # so for more than one Cython extension things will need to be adapted
    with cd(root_path):
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

    build_ext_cmd = dist.get_command_obj("build_ext")
    build_ext_cmd.ensure_finalized()
    build_ext_cmd.inplace = 1
    build_ext_cmd.run()


def cmd_clean():
    dirs_to_remove = [
        root_path / "build",
        root_path / "dist",
        root_path / "fastcubicspline.egg-info",
        root_path / "fastcubicspline/__pycache__",
    ]
    for d in dirs_to_remove:
        if d.exists():
            print(f"rm {d}")
            shutil.rmtree(d, ignore_errors=True)
        else:
            print(f"cannot rm {d}, does not exist")

    for f in (root_path / "fastcubicspline").iterdir():
        name = f.name
        if (name == "fcs_c.c") or (name.startswith("fcs_c.") and name.endswith(".so")):
            print(f"rm {f}")
            os.remove(f)


# note that upton poetry install / build, poetry actually triggers
# Command '['... .venv/bin/python', 'build_ext.py']'
# so the follows works such that this call will build the Cython extension
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        help="what to do, choose between 'build_ext' or 'clear'\n"
        + "  build_ext: triggers Cython inplace build (using distutils magic)\n"
        + "  clean: without asking remove 'build/', 'dist/', 'fastcubicspline.egg-info/', 'fastcubicspline/fcs_c.c', "
        + "'fastcubicspline/fcs_c*.so', 'fastcubicspline/__pycache__'",
        default="build_ext",
        nargs="?",
        type=str,
    )
    args = parser.parse_args()

    if args.command == "build_ext":
        cmd_build_ext()
    elif args.command == "clean":
        cmd_clean()
    else:
        parser.print_help()
        raise ValueError(f"unknown command '{args.command}'")
