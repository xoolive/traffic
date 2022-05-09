
How to run traffic in a Docker container?
-----------------------------------------

It is quite simple to run the traffic library in a Docker container. The following two examples show how you can use the traffic library in a docker with `jupyter notebook <https://jupyter-docker-stacks.readthedocs.io/en/latest/>`__. If you are completely unfamiliar with Docker and how to modify a Docker image, you can find a good tutorial in the `official documentation <https://docs.docker.com/get-started/>`__. Of course, you can base your Docker container on a different Dockerfile as the one used in the examples.

In the simplest case when you just want to run the traffic library in a Docker container, you can install the library and its dependencies directly into the base environment of the container. In that case, the ``Dockerfile`` could look like the following:

.. code:: dockerfile

    FROM jupyter/minimal-notebook

    USER jovyan
    RUN mamba install -c conda-forge traffic

    # manually set environmet variable for PROJ when running in base environment
    ENV PROJ_LIB=/opt/conda/share/proj

Note the last line, which sets a environment variable for PROJ. This is needed because the conda base environment never gets properly activated and this fixes the issue described `here <https://gis.stackexchange.com/questions/364421/how-to-make-proj-work-via-anaconda-in-google-colab>`__.
To run this Docker, you first have to generate an image with ``docker build``:

.. code:: bash

    docker build -f Dockerfile -t jupyter/traffic:latest .

The Docker container can now be started: 

.. code:: bash

	docker run -it -p 8888:8888 --name jupyter_traffic jupyter/traffic:latest


If you have already a working conda environment that you would like to use, you
can install your existing environment into the Docker container.

The ``Dockerfile`` could look like the following:

.. code:: dockerfile

    FROM jupyter/minimal-notebook

    # copy conda environment file to image
    COPY traffic_env.yml traffic_env.yml

    # install nb_conda into the base python to allow the user to choose the environment in the jupyter notebook and install environment
    USER jovyan
    RUN mamba install -y nb_conda
    RUN mamba env create -f traffic_env.yml

Note that the environment file ``traffic_env.yml`` has to be in the same directory as the ``Dockerfile``. The file ``traffic_env.yml`` could look like the following:

.. code:: yaml

    name: traffic_env
    channels:
      - conda-forge
      - defaults
    dependencies:
      - python
      - traffic
