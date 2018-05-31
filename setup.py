from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import os.path

bresenham_path = os.path.join('traffic', 'tools', 'bresenham.pyx')

extensions = [Extension('traffic.tools.bresenham',
                        [bresenham_path],)]

setup(name="traffic",
      version=0.1,
      description="Tools for ATM",
      entry_points={'console_scripts': [
          "traffic=traffic.console:main"
      ]},
      ext_modules=cythonize(extensions),
      packages=["traffic", "traffic.core", "traffic.data",
                "traffic.data.adsb", "traffic.so6", "traffic.tools"],
      package_data={'traffic.tools': bresenham_path},
      python_requires='>=3.6',
      )
