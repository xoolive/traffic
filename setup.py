from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [Extension('traffic.tools.bresenham',
                        ['traffic/tools/bresenham.pyx'],)]

setup(name="traffic",
      version=0.1,
      description="Tools for ATM",
      ext_modules=cythonize(extensions),
      packages=["traffic", "traffic.data", "traffic.so6", "traffic.tools"],
      package_data={'traffic.tools': 'traffic/tools/bresenham.pyx'},
      )
