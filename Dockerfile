# This basic Dockerfile may be built and run using the following commands:
#
#    docker build -f Dockerfile -t jupyter/traffic:latest .
#    docker run -it -p 8888:8888 --name jupyter_traffic jupyter/traffic:latest
#
# Then connect to http://localhost:8888 to benefit from a Jupyter notebook
# shipped with a working version of the latest release of the traffic library.

FROM jupyter/minimal-notebook

USER jovyan
RUN conda install cartopy shapely pip
RUN pip install keplergl traffic
