import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="traffic",
    version="1.1.1",
    author="Xavier Olive",
    author_email="git@xoolive.org",
    url="https://github.com/xoolive/traffic/",
    license="MIT",
    description="A toolbox for manipulating and analysing air traffic data",
    long_description=long_description,
    # https://dustingram.com/articles/2018/03/16/markdown-descriptions-on-pypi
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["traffic=traffic.console:main"]},
    packages=find_packages(),
    package_data={
        "traffic.data.airspaces": ["firs.json"],
        "traffic": [
            os.path.join("..", "icons", f)
            for f in os.listdir(os.path.join("..", "traffic", "icons"))
            if f.startswith("travel")
        ],
    },
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "Cartopy",
        "Shapely",
        "requests",
        "maya",  # human-friendly datetimes
        "appdirs",  # proper configuration directories
        "paramiko",  # ssh connections
        "PyQt5",
        "ipywidgets",  # IPython widgets for traffic
        "tornado",  # dependency for matplotlib with WebAgg
        "ipympl",  # interactive matplotlib in notebooks
        "tqdm>=4.28",  # progressbars
        "cartotools==1.0",
        # restore dependency versionwhen officially released
        "pyModeS",
        # "pyModeS==2.0",
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
