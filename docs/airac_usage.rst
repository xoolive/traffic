How to find all the airspaces crossed by a trajectory?
------------------------------------------------------

The provided infrastructure lets you find all elementary sectors crossed
by a trajectory (here from the so6 file)

.. code:: python

    from traffic.data import SO6
    so6 = SO6.from_file('data/sample_m3.so6.7z')
    
    with plt.style.context('traffic'):
        fig = plt.figure()
        ax = plt.axes(projection=Lambert93())
    
        ax.add_feature(countries())
        ax.gridlines()
        ax.set_extent(nm_airspaces['LFFFUIR'])
    
        nm_airspaces['LFFFUIR'].plot(ax, lw=2, alpha=.5, linestyle='dashed')
        so6['DAH1008'].plot(ax, marker='.')
        
        # display elementary sectors (ES) crossed by the trajectory
        for airspace in nm_airspaces.search("LF.*/ES"):
            if so6['DAH1008'].intersects(airspace):
                airspace.plot(ax, alpha=.5, lw=2)




.. image:: _static/airac_traj_sectors.png
   :scale: 70 %
   :alt: Trajectories over Bordeaux AAC
   :align: center


Another use case could be to plot all flights going through an airspace
at noon.

.. code:: python

    # callsigns at noon inside LFBBBDX
    bdx_noon = (
        so6.at("2018-01-01 12:00")
        .inside_bbox(nm_airspaces['LFBBBDX'])
        .intersects(nm_airspaces['LFBBBDX'])
    )
    
    # full so6 limited to flights hereabove
    so6_bdx_noon = so6.select(bdx_noon)

.. code:: python

    from cartes.crs import EuroPP
    from cartes.utils.features import countries
    
    with plt.style.context('traffic'):
        fig = plt.figure()
        ax = plt.axes(projection=EuroPP())
    
        ax.add_feature(countries())
        ax.gridlines()
        ax.set_extent((-10, 15, 35, 55))
    
        nm_airspaces['LFBBBDX'].plot(ax, lw=2, alpha=.5)
    
        for _, flight in so6_bdx_noon:
            flight.plot(ax, color='#aa3a3a', lw=.4, alpha=.5)



.. image:: _static/airac_traj_bdx.png
   :scale: 70 %
   :alt: Trajectories over Bordeaux AAC
   :align: center

