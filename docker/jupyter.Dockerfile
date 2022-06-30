FROM jupyter/minimal-notebook

USER jovyan
RUN mamba install -c conda-forge -y traffic

# manually set environment variable for PROJ when running in base environment
ENV PROJ_LIB=/opt/conda/share/proj
