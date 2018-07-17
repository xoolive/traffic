from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import os.path

bresenham_path = os.path.join("traffic", "algorithms", "bresenham.pyx")
extensions = [Extension("traffic.algorithms.bresenham", [bresenham_path])]

setup(
    name="traffic",
    version=0.1,
    description="A toolbox for manipulating and analysing air traffic data",
    entry_points={"console_scripts": ["traffic=traffic.console:main"]},
    ext_modules=cythonize(extensions),
    packages=[
        "traffic",
        "traffic.algorithms",
        "traffic.core",
        "traffic.data",
        "traffic.data.adsb",
        "traffic.data.basic",
        "traffic.data.airspaces",
        "traffic.data.so6",
        "traffic.drawing",
        "traffic.plugins",
    ],
    package_data={"traffic.data.airspaces": ["firs.json"]},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "cython",
        "geodesy",
        "Cartopy",
        "Shapely",
        "requests",
        "maya",
        "appdirs",
        "paramiko",
        "tqdm",
        "cartotools==1.0",
    ],
    dependency_links=[
        "https://github.com/xoolive/cartotools.git#whl=cartotools-1.0"
    ],
    python_requires=">=3.6",
)
