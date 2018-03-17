from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [Extension('traffic.tools.bresenham',
                        ['traffic/tools/bresenham.pyx'],)]

setup(name="traffic",
      version=0.1,
      description="Tools for ATM",
      entry_points={'console_scripts': [
          "traffic=traffic.console:main"
      ]},
      ext_modules=cythonize(extensions),
      packages=["traffic", "traffic.core", "traffic.data", "traffic.so6",
                "traffic.tools"],
      package_data={'traffic.tools': 'traffic/tools/bresenham.pyx'},
      python_requires='>=3.6',
      )
