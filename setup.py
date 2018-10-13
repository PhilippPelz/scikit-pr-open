import build
from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import distutils.unixccompiler
import distutils.command.build
import distutils.command.clean
import platform
import subprocess
import shutil
import sys
import os
from subprocess import call

this_file = os.path.dirname(__file__)

# build_all_cmd = ['bash', './lib/build_all.sh']
# if subprocess.call(build_all_cmd) != 0:
#     print('/lib/build_all.sh failed with error. Please look at the compilation errors and fix them.')
#     sys.exit(1)

setup(
    name="skpr",
    version="0.1",
    description="Phase retrieval algorithms in Python with pytorch",
    long_description = file('README.md','r').read(),
    url="https://github.com/PhilippPelz/scikit-pr",
    author="Philipp Pelz",
    zip_safe = False,
    author_email="philipp.pelz@mpsd.mpg.de",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    package_dir = {'skpr':'skpr'},
    packages=find_packages(exclude=["build"]),
    package_data={'skpr': [ 'lib/*.so*','lib/build/*.so*', 'lib/*.dylib*', 'lib/*.h']},
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="skpr._ext.skpr_thnn",
    # Extensions to compile.
    cffi_modules=[
        "build.py:ffi"
    ]
)
#
#['skpr',
#                'skpr.core',\
#                'skpr.util',\
#                'skpr.simulation',\
#                'skpr.inout',\
#                'skpr.nn',\
#                'skpr.optim',\
#                'skpr.resources' ],