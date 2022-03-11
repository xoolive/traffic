How to configure EUROCONTROL data files?
========================================

The traffic library supports two sources of information provided, **under
conditions**, by EUROCONTROL:

- | the ``nm_`` prefix refers to data that we download from EUROCONTROL `Network
    Manager Demand Data Repository (DDR) <https://www.eurocontrol.int/ddr>`_,
    under Dataset Files > Airspace Environment Datasets.
  | `Access conditions <https://www.eurocontrol.int/ddr#access-conditions>`_ are
    managed by EUROCONTROL.

  .. code:: python

    from traffic.data import nm_airways, nm_airspaces, nm_navaids, nm_freeroute

- | the ``aixm_`` prefix refers to XML files following the
    `AIXM <http://aixm.aero/>`_ standard and produced by EUROCONTROL. We
    download these files from the `Network Manager B2B web services
    <https://eurocontrol.int/service/network-manager-business-business-b2b-web-services>`_.
  | You have to own a B2B certificate granted by EUROCONTROL to get access to
    this data.

  .. code:: python

    from traffic.data import aixm_airspaces, aixm_navaids

  **See also**: download AIXM data with
  :meth:`~traffic.data.eurocontrol.b2b.NMB2B.aixm_dataset`

The first thing to do is to put the path to a directory containing your files
from EUROCONTROL in your configuration file.

Identify the path to your configuration file here:

.. code:: python

    >>> import traffic
    >>> traffic.config_file
    PosixPath('/home/xo/.config/traffic/traffic.conf')

Then edit the following line accordingly, only set the path to data you have access to:

::

    [global]
    aixm_path = /home/xo/Documents/data/AIRAC_2111
    nm_path = /home/xo/Documents/data/ENV_PostOPS_AIRAC_2111_04NOV2021_With_Airspace_Closure

