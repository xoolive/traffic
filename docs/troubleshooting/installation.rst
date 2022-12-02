How to troubleshoot installation issues?
----------------------------------------

There are three installation processes:

- the **Anaconda way (recommended)**, the most simple way, with only one command
  to execute;

  .. code:: bash

     conda create -n traffic -c conda-forge python=3.9 traffic

- the pip way, but you are responsible for non Python dependency management.
  With Linux, you may check how the environment is created for GitHub Actions;

  .. code:: bash

     pip install traffic

- the poetry way, recommended for development, which ensures the same versions
  of all dependencies are installed in a virtual environment on any computer
  running the library: this step is crucial for consistency in continuous
  integration.

  .. code:: bash

     poetry install

The following questions are the most common about installation issues:

- **Installation is really slow**, **conda stalls on the resolution process**

  Before filing a bug report, please try again:

  - to start the installation from a fresh conda environment, please do not
    reuse a potentially corrupted environment (when conda loses track of
    dependency versions);
  - to disable the ``channel_priority`` option with

    .. code:: bash

        conda config --set channel_priority disabled

  If you cannot see the end of the tunnel, consider running traffic with the
  provided `Dockerfile <docker.html>`_.



- **Python crashes when I try to reproduce plots in the documentation**

  Usually, this happens when there is something wrong with your Cartopy and/or
  shapely installation.  These libraries strongly depend on the ``geos`` and
  ``proj`` libraries. You must have shapely and Cartopy versions matching the
  correct versions of these libraries.

  The problem is sometimes hard to understand, and you may end up fixing it
  without really knowing how.

  If you don't know how to install these dependencies, start with a **fresh**
  Anaconda distribution and install the following libraries *the conda way*:

  .. parsed-literal::
     conda install cartopy shapely

  If it still does not work, try something along:

  .. parsed-literal::
     conda uninstall cartopy shapely
     pip uninstall cartopy shapely
     # be sure to remove all previous versions before installing again
     conda install cartopy shapely

  If it still does not work, try again with:

  .. parsed-literal::
     conda uninstall cartopy shapely
     pip uninstall cartopy shapely
     # this forces the recompilation of the packages
     pip install --no-binary :all: cartopy shapely


- **Widgets do not display in Jupyter Lab or Jupyter Notebook**

  After executing a cell in a Jupyter environment, you may see one of the following output:

  .. parsed-literal::
      A Jupyter Widget
      # or
      Error displaying widget
      # or
      HBox(children=(IntProgress(value=0, max=1), HTML(value='')))
      # or
      Map(basemap={'url': 'https://{s}.tile.openstreetmap.org/…

  You will need to activate the widgets extensions:

  - with Jupyter Lab:

      .. parsed-literal::
         jupyter labextension install @jupyter-widgets/jupyterlab-manager
         jupyter labextension install jupyter-leaflet

  - with Jupyter Notebook:

      .. parsed-literal::
         jupyter nbextension enable --py --sys-prefix widgetsnbextension
         jupyter nbextension enable --py --sys-prefix ipyleaflet
