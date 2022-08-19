FROM jupyter/pyspark-notebook:spark-3.3.0

# Based on https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#add-a-custom-conda-environment-and-jupyter-kernel

# name your environment and choose the python version
ARG conda_env=traffic_spark
ARG py_ver=3.8

RUN mamba create --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" python=${py_ver} ipython ipykernel traffic pyspark && \
    mamba clean --all -f -y

# create Python kernel and link it to jupyter
RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN echo "conda activate ${conda_env}" >> "${HOME}/.bashrc"

# manually set environment variable for PROJ when running in base environment
ENV PROJ_LIB=/opt/conda/share/proj

# Enable insecure writes to solve docker issue "RuntimeError: Permissions assignment failed for secure file:"
# https://github.com/jupyter/notebook/issues/5058
# ENV JUPYTER_ALLOW_INSECURE_WRITES=true