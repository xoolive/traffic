How to troubleshoot data access issues?
---------------------------------------

- **What does AIRAC mean?**

  Aeronautical Information Publications are updated every 28 days according to
  fixed calendar. This cycle is known as AIRAC (Aeronautical Information
  Regulation And Control) cycle.
  
- **I want to know more about Eurocontrol NM files**

  We download these files from Eurocontrol `Network Manager Demand Data
  Repository (DDR) <https://www.eurocontrol.int/ddr>`_ under Dataset Files >
  Airspace Environment Datasets. `Access conditions
  <https://www.eurocontrol.int/ddr#access-conditions>`_ are managed by
  EUROCONTROL.
  
  Should you have no such access, basic data are also provided in the library.

- **I want to know more about Eurocontrol AIXM files**

  These are XML files following the `AIXM <http://aixm.aero/>`_ standard and
  produced by Eurocontrol. We download these files from Eurocontrol `Network
  Manager B2B web services
  <https://eurocontrol.int/service/network-manager-business-business-b2b-web-services>`_.
  
  You have to own a B2B certificate granted by EUROCONTROL to get access to this
  data.
  
- **I want to have access to the OpenSky Impala data**

  https://opensky-network.org/impala-guide

- **I queried data from the OpenSky Impala shell but it returns None**

  Please be aware that data you may request from OpenSky is limited by the
  coverage and the storage capacity on OpenSky side. The traffic library
  facilitates the access to the database but is subject to technical issues they
  may encounter on the backend side.
  
  If your query returns None, it usually means that the OpenSky database
  returned empty results.
  
  This can be due to:
  
  - | the **temporal extent** of your query
    | Try a similar request on a more recent day (e.g. January 1st of the current year);
  
  - | the **coverage of the network** at the time of the query
    | If you ask for trajectories landing or taking off from a particular
      airport, do confirm it is well covered by the network;
  
  - | an **error in the generated query** (but consider the two first options before)
    | Use the ``cached=False`` parameter and the ``INFO`` or ``DEBUG`` logging
      mode from your logger to manually check the generated query before it is
      executed.
  
    .. code:: python
  
       import logging
       logging.basicConfig(level=logging.DEBUG)
  
  .. warning::

     If you decide to file a bug with your request, please provide your whole
     set of parameters, without obfuscation, and do not omit the logging
     messages in the issue description.
