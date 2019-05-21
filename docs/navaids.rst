Navigational beacons
--------------------

.. autoclass:: traffic.data.basic.navaid.Navaid
    :members:
    :no-inherited-members:
    :no-undoc-members:


A navaid database (also available in
`bluesky <https://github.com/ProfHoekstra/bluesky>`__) is provided. All
beacons are accessible by a bracket search. Their representation is a
namedtuple, with fields accessible by the dot notation.

.. code:: python

    >>> from traffic.data import navaids
    >>> navaids['NARAK']

    NARAK (FIX): 44.295278 1.748889



.. code:: python

    >>> navaids['NARAK']._asdict()

    OrderedDict([('id', 'NARAK'),
                 ('type', 'FIX'),
                 ('lat', 44.295278),
                 ('lon', 1.748889),
                 ('alt', None),
                 ('frequency', None),
                 ('magnetic_variation', None),
                 ('description', None)])



.. code:: python

    >>> navaids.search('gaillac')

    [GAI (VOR): 43.954056 1.824167 979
     GAILLAC-CASTELNEAU DE MONTMIRAIL VOR 115.8MHz]



