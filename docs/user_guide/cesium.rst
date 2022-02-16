
CesiumJS
~~~~~~~~

`CesiumJS <http://cesiumjs.org/>`__ is a great tool for displaying and
animating geospatial 3D data. The library provides an export of a
Traffic structure to a czml file. A copy of this file is available in
the ``data/`` directory. You may drag and drop it on the
http://cesiumjs.org/ page after you open it on your browser.


.. warning:: 

    The plugin must be `activated <plugins.html>`__ in your configuration file.

.. code:: python

    demo.to_czml('data/sample_cesium.czml')