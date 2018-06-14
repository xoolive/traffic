from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import os.path

bresenham_path = os.path.join('traffic', 'algorithms', 'bresenham.pyx')
extensions = [Extension('traffic.algorithms.bresenham', [bresenham_path],)]

setup(name="traffic",
      version=0.1,
      description="Tools for ATM",
      entry_points={'console_scripts': [
          "traffic=traffic.console:main"
      ]},
      ext_modules=cythonize(extensions),
      packages=["traffic", "traffic.core",
                "traffic.data", "traffic.data.adsb", "traffic.data.basic",
                "traffic.data.sectors", "traffic.data.so6",
                "traffic.algorithms", "traffic.drawing"],
      package_data={'traffic.data.sectors': ['firs.json']},
      python_requires='>=3.6',
      )
