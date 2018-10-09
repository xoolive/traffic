import os
import os.path

from setuptools import setup

setup(
    name="traffic",
    version=0.1,
    description="A toolbox for manipulating and analysing air traffic data",
    entry_points={"console_scripts": ["traffic=traffic.console:main"]},
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
        "traffic.qtgui",
    ],
    package_data={
        "traffic.data.airspaces": ["firs.json"],
        "traffic": [
            os.path.join("..", "icons", f)
            for f in os.listdir(os.path.join("..", "traffic", "icons"))
        ],
    },
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
        "tqdm>=4.26",
        "cartotools==1.0",
        "pyModeS==2.0",
    ],
    dependency_links=[
        "https://github.com/xoolive/cartotools.git#whl=cartotools-1.0"
    ],
    python_requires=">=3.6",
)
