.. raw:: html
    :file: ../embed_widgets/squawk7700.html

Analysing in-flight emergencies using big data
----------------------------------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3937483.svg
    :target: https://doi.org/10.5281/zenodo.3937483

.. admonition:: Information

    This page complements the following publication:

    | X. Olive, A. Tanner, M. Strohmeier, M. Schäfer, M. Feridun, A. Tart, I. Martinovic and V. Lenders.
    | **OpenSky Report 2020: Analysing in-flight emergencies using big data**.
    | *Proceedings of the 39th Digital Avionics Systems Conference* (DASC), 2020


.. code:: python

    from traffic.data.datasets import squawk7700
    
    squawk7700

.. raw:: html

    <b>Traffic with 832 identifiers</b><style  type="text/css" >
        #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow0_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 100.0%, transparent 100.0%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow1_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 86.8%, transparent 86.8%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow2_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 79.8%, transparent 79.8%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow3_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 79.5%, transparent 79.5%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow4_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 78.8%, transparent 78.8%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow5_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 75.9%, transparent 75.9%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow6_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 72.7%, transparent 72.7%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow7_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 72.0%, transparent 72.0%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow8_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 70.6%, transparent 70.6%);
            }    #T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow9_col0 {
                width:  10em;
                 height:  80%;
                background:  linear-gradient(90deg,#5fba7d 70.4%, transparent 70.4%);
            }</style><table id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19a" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>    </tr>    <tr>        <th class="index_name level0" >flight_id</th>        <th class="blank" ></th>    </tr></thead><tbody>
                    <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row0" class="row_heading level0 row0" >ASR172B_20190715</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow0_col0" class="data row0 col0" >22825</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row1" class="row_heading level0 row1" >UAL275_20191110</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow1_col0" class="data row1 col0" >19812</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row2" class="row_heading level0 row2" >UAE15_20190208</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow2_col0" class="data row2 col0" >18220</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row3" class="row_heading level0 row3" >UAL1827_20190903</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow3_col0" class="data row3 col0" >18150</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row4" class="row_heading level0 row4" >AAL2807_20180428</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow4_col0" class="data row4 col0" >17987</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row5" class="row_heading level0 row5" >AAL343_20191119</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow5_col0" class="data row5 col0" >17319</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row6" class="row_heading level0 row6" >AAL163_20190317</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow6_col0" class="data row6 col0" >16589</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row7" class="row_heading level0 row7" >UAL102_20180919</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow7_col0" class="data row7 col0" >16438</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row8" class="row_heading level0 row8" >UAL1566_20181026</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow8_col0" class="data row8 col0" >16125</td>
                </tr>
                <tr>
                            <th id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19alevel0_row9" class="row_heading level0 row9" >PIA709_20200124</th>
                            <td id="T_9f554fa2_c28d_11ea_9b3f_85d1d741d19arow9_col0" class="data row9 col0" >16060</td>
                </tr>
        </tbody></table>

Metadata
========

Associated metadata is merged into the Traffic structure, but also
available as an attribute. The table includes information about:

-  flight information: ``callsign``, ``number`` (IATA flight number),
   ``origin`` , ``destination`` (where the aircraft intended to land),
   ``landing`` (where the aircraft actually landed, if available),
   ``diverted`` (where the aircraft actually landed, if applicable);
-  aircraft information: ``icao24`` transponder identifier,
   ``registration`` (the tail number) and ``typecode``;
-  information about the nature of the emergency, from Twitter and `The
   Aviation Herald <https://avherald.com/>`__.

.. code:: python

    squawk7700.metadata.iloc[:10, :10]  # just a preview to fit this page


.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="0" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>flight_id</th>
          <th>callsign</th>
          <th>number</th>
          <th>icao24</th>
          <th>registration</th>
          <th>typecode</th>
          <th>origin</th>
          <th>landing</th>
          <th>destination</th>
          <th>diverted</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ARG1511_20180101</td>
          <td>ARG1511</td>
          <td>AR1511</td>
          <td>e06442</td>
          <td>LV-FQB</td>
          <td>B738</td>
          <td>SACO</td>
          <td>SABE</td>
          <td>SABE</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>DAL14_20180101</td>
          <td>DAL14</td>
          <td>DL14</td>
          <td>a14c29</td>
          <td>N183DN</td>
          <td>B763</td>
          <td>KATL</td>
          <td>NaN</td>
          <td>EDDF</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>JBU263_20180108</td>
          <td>JBU263</td>
          <td>B6263</td>
          <td>aa600a</td>
          <td>N768JB</td>
          <td>A320</td>
          <td>KJFK</td>
          <td>NaN</td>
          <td>KSEA</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>DAL65_20180108</td>
          <td>DAL65</td>
          <td>DL65</td>
          <td>ab2855</td>
          <td>N818NW</td>
          <td>A333</td>
          <td>KATL</td>
          <td>KLAX</td>
          <td>KLAX</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>EDW24_20180111</td>
          <td>EDW24</td>
          <td>WK24</td>
          <td>4b1901</td>
          <td>HB-JMF</td>
          <td>A343</td>
          <td>LSZH</td>
          <td>LSZH</td>
          <td>MMUN</td>
          <td>LSZH</td>
        </tr>
        <tr>
          <th>5</th>
          <td>ASA111_20180112</td>
          <td>ASA111</td>
          <td>AS111</td>
          <td>a602ab</td>
          <td>N487AS</td>
          <td>B739</td>
          <td>KLAS</td>
          <td>NaN</td>
          <td>PANC</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>6</th>
          <td>ONE8511_20180113</td>
          <td>ONE8511</td>
          <td>O68511</td>
          <td>e4919d</td>
          <td>PR-OCX</td>
          <td>A332</td>
          <td>KMIA</td>
          <td>NaN</td>
          <td>SBGR</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>7</th>
          <td>BAW882_20180115</td>
          <td>BAW882</td>
          <td>BA882</td>
          <td>406532</td>
          <td>G-EUYM</td>
          <td>A320</td>
          <td>EGLL</td>
          <td>NaN</td>
          <td>UKBB</td>
          <td>EDDT</td>
        </tr>
        <tr>
          <th>8</th>
          <td>ENY3342_20180117</td>
          <td>ENY3342</td>
          <td>AA3342</td>
          <td>a272b8</td>
          <td>N257NN</td>
          <td>E75L</td>
          <td>KDFW</td>
          <td>NaN</td>
          <td>KLIT</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>9</th>
          <td>WJA506_20180122</td>
          <td>WJA506</td>
          <td>WS506</td>
          <td>c0499b</td>
          <td>C-GBWS</td>
          <td>B736</td>
          <td>CYXE</td>
          <td>NaN</td>
          <td>CYYZ</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    squawk7700.metadata.iloc[:10, 10:]  # just a preview to fit this page




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="0" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>tweet_problem</th>
          <th>tweet_result</th>
          <th>tweet_fueldump</th>
          <th>avh_id</th>
          <th>avh_problem</th>
          <th>avh_result</th>
          <th>avh_fueldump</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>engine</td>
          <td>return</td>
          <td>unknown</td>
          <td>4b382175</td>
          <td>engine</td>
          <td>return</td>
          <td>unknown</td>
        </tr>
        <tr>
          <th>5</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>6</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>7</th>
          <td>medical</td>
          <td>diverted</td>
          <td>unknown</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>8</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>9</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>


Data exploration
================

Simple queries provide subsets of the trajectories:

-  diverted aircraft: ``diverted == diverted`` selects flights where
   ``diverted`` is not empty (``NaN``);
-  returning aircraft, when the diversion airport is the origin airport

.. code:: python

    squawk7700.query("diverted == diverted") | squawk7700.query("diverted == origin")


.. raw:: html

    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        vertical-align: top;'>
        <b>Traffic with 295 identifiers</b><style  type="text/css" >
    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow0_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 100.0%, transparent 100.0%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow1_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 98.7%, transparent 98.7%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow2_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 83.1%, transparent 83.1%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow3_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 82.6%, transparent 82.6%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow4_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 80.7%, transparent 80.7%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow5_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 75.5%, transparent 75.5%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow6_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 74.5%, transparent 74.5%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow7_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 74.0%, transparent 74.0%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow8_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 67.4%, transparent 67.4%);
        }    #T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow9_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 63.3%, transparent 63.3%);
        }</style><table id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19a" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>    </tr>    <tr>        <th class="index_name level0" >flight_id</th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row0" class="row_heading level0 row0" >UAE15_20190208</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow0_col0" class="data row0 col0" >18220</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row1" class="row_heading level0 row1" >AAL2807_20180428</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow1_col0" class="data row1 col0" >17987</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row2" class="row_heading level0 row2" >BAW269_20181106</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow2_col0" class="data row2 col0" >15145</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row3" class="row_heading level0 row3" >ROU1901_20190809</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow3_col0" class="data row3 col0" >15051</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row4" class="row_heading level0 row4" >ANA232_20180622</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow4_col0" class="data row4 col0" >14712</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row5" class="row_heading level0 row5" >SWA3708_20191027</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow5_col0" class="data row5 col0" >13757</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row6" class="row_heading level0 row6" >AFR1145_20190820</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow6_col0" class="data row6 col0" >13566</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row7" class="row_heading level0 row7" >ETD87W_20180324</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow7_col0" class="data row7 col0" >13474</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row8" class="row_heading level0 row8" >EDW24_20180111</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow8_col0" class="data row8 col0" >12272</td>
            </tr>
            <tr>
                        <th id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19alevel0_row9" class="row_heading level0 row9" >UAL41_20180730</th>
                        <td id="T_a07b3b76_c28d_11ea_9b3f_85d1d741d19arow9_col0" class="data row9 col0" >11532</td>
            </tr>
    </tbody></table>
    </div>
    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        vertical-align: top;'>
        <b>Traffic with 111 identifiers</b><style  type="text/css" >
    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow0_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 100.0%, transparent 100.0%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow1_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 81.0%, transparent 81.0%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow2_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 71.9%, transparent 71.9%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow3_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 66.8%, transparent 66.8%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow4_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 60.8%, transparent 60.8%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow5_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 52.8%, transparent 52.8%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow6_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 51.5%, transparent 51.5%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow7_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 50.9%, transparent 50.9%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow8_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 49.9%, transparent 49.9%);
        }    #T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow9_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 46.1%, transparent 46.1%);
        }</style><table id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19a" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>    </tr>    <tr>        <th class="index_name level0" >flight_id</th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row0" class="row_heading level0 row0" >BAW269_20181106</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow0_col0" class="data row0 col0" >15145</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row1" class="row_heading level0 row1" >EDW24_20180111</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow1_col0" class="data row1 col0" >12272</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row2" class="row_heading level0 row2" >DLH501_20180626</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow2_col0" class="data row2 col0" >10889</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row3" class="row_heading level0 row3" >NAX7075_20180606</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow3_col0" class="data row3 col0" >10119</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row4" class="row_heading level0 row4" >FDX5392_20190221</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow4_col0" class="data row4 col0" >9215</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row5" class="row_heading level0 row5" >CLX4327_20191028</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow5_col0" class="data row5 col0" >7995</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row6" class="row_heading level0 row6" >AFR090_20180302</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow6_col0" class="data row6 col0" >7801</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row7" class="row_heading level0 row7" >AFR1196_20180303</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow7_col0" class="data row7 col0" >7712</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row8" class="row_heading level0 row8" >BGA114D_20190311</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow8_col0" class="data row8 col0" >7563</td>
            </tr>
            <tr>
                        <th id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19alevel0_row9" class="row_heading level0 row9" >AFR724_20190201</th>
                        <td id="T_a0b64306_c28d_11ea_9b3f_85d1d741d19arow9_col0" class="data row9 col0" >6976</td>
            </tr>
    </tbody></table>
    </div>




An emergency situation can be displayed in the notebook, or included
into a leaflet widget.

.. code:: python

    squawk7700["AFR1196_20180303"]




.. raw:: html

    <b>Flight AFR1196 – AF1196 (AFR1196_20180303)</b><ul><li><b>aircraft:</b> 392af2 · 🇫🇷 F-GKXS (A320)</li><li><b>from:</b> LFPG (2018-03-03 15:22:07+00:00)</li><li><b>to:</b> GMMN (2018-03-03 17:32:24+00:00)</li><li><b>diverted to: LFPG</b></li></ul><div style="white-space: nowrap"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="300" height="300" viewBox="-223627.95561049317 -113937.61875084888 440744.8165950642 234517.58321890776" preserveAspectRatio="xMinYMin meet"><g transform="matrix(1,0,0,-1,0,6642.345717210017)"><polyline fill="none" stroke="#66cc99" stroke-width="2938.298777300428" points="200338.3930614474,104239.89131917899 197988.70380359475,103995.42286520542 189675.0630460206,102955.41427180581 129727.68823495628,95894.60275589055 128493.16390510823,95687.42496627322 127108.36680809128,95304.68977061866 126371.46751559038,95023.34251395545 125304.0256587075,94515.52300592023 116072.65745922575,89862.8727747386 41887.51921374679,52164.8990525608 39448.44185466627,50746.21931203608 37885.71702972868,49698.24106006187 31620.460117324506,44914.34190102379 -54409.295131145525,-21730.07955389291 -55396.35945568289,-22517.910121982793 -57373.10160610266,-24274.232895792993 -59315.558224908,-26294.170817710165 -60095.97152970856,-27217.666157268664 -61212.80873840916,-28645.432081026855 -90241.18218045933,-67595.71041355812 -95460.49814782171,-74560.48795987989 -98225.21777989354,-78150.80186055445 -99241.60892315522,-79365.39438865807 -100627.71060086448,-80742.18230314326 -101761.66869927371,-81693.92326538205 -102844.12470774468,-82458.79265755502 -104296.36179953764,-83277.93067457381 -105933.70281308483,-83976.2673752727 -106961.69406322806,-84309.73110568932 -107760.64730057516,-84517.93140294332 -109522.60264759528,-84862.58958082488 -122947.99614290315,-87131.12046408669 -185069.44402817555,-97361.2605823397 -186078.14835095408,-97510.39982620762 -187379.1717834787,-97613.73665473539 -188936.09405938117,-97602.17765415796 -190759.4825241407,-97442.02917386526 -192623.10971183865,-97111.5136128656 -194173.8877818972,-96692.4324514014 -195593.14882213675,-96178.95522189284 -197168.96778911742,-95450.71038990712 -198731.1047146122,-94547.2861322798 -200158.79461698933,-93530.92108854679 -200816.65267622093,-92988.88315389417 -201603.96456709056,-92266.38869549021 -202782.82437004882,-91021.36166502128 -203722.7875507481,-89831.21394970118 -204804.46377224848,-88154.89059936283 -205549.2757210543,-86718.02034525917 -206162.31272939584,-85239.86958347281 -206462.2461744415,-84348.73592195484 -206769.26369247923,-83239.89889265782 -207076.0357047605,-81708.78476649254 -207268.40186469277,-79965.32087480802 -207304.07351437968,-78344.5579669639 -207216.2839200646,-76901.34180868219 -206931.7502859317,-75032.23933510318 -206502.45775806427,-73336.67315231769 -206001.03105734967,-71902.67711929475 -205694.89540913,-71164.88172387623 -205253.2831819783,-70258.70532730054 -204430.10296054033,-68815.92077214204 -203404.0034554147,-67345.01080177218 -202409.70605940023,-66162.07616999272 -201865.74806230745,-65591.21084946087 -200924.4678690418,-64706.23481762605 -199948.6433895388,-63912.248431341926 -198734.9641765987,-63052.805022593035 -197602.22700538771,-62376.80597600776 -196915.77466071487,-62016.910994125006 -195546.79343428515,-61405.51196785213 -193854.90051043907,-60809.78612222395 -192554.13078758,-60475.08060683673 -191004.70935515568,-60198.35107245239 -172352.9362384445,-57791.610420643665 -56450.35213253483,-43069.87389449679 9521.265200675982,-34740.46892153856 42128.14015764277,-30571.417557034343 45194.60109893661,-30191.003012867073 46232.777574943204,-30099.943740375802 50623.165121251324,-30113.539638131097 52656.619976113354,-30380.39514788947 53995.509830087205,-30658.861850430767 55010.6338486249,-30918.4602162674 56430.302347313045,-31367.893770646504 60563.217229135604,-33290.750901525855 63599.44581935554,-35396.108988259264 65834.74022241704,-37457.304146931354 70000.09270353493,-43436.534780939306 70314.47058777443,-44108.460331562324 71028.6892561964,-45870.1821115301 71529.26374097604,-47432.56607039053 71778.96294958566,-48371.63807067302 71975.25200172832,-49264.77213711332 72219.81010217102,-50734.411889845884 72433.81992542223,-53773.122035470755 70588.62953375834,-63155.14060194002 69627.10498144908,-65176.785899680784 68729.6273553012,-66714.53393423079 51786.94918082859,-85794.0791818545 51094.42909804836,-86151.38404728271 49250.27770860345,-86952.71754728185 39736.00576619013,-88581.18039892578 38197.98124735154,-88479.50179960085 37101.254264138755,-88344.0856845129 35359.539658263115,-88034.28270059949 33580.78351100695,-87571.48827206255 32977.98812077324,-87378.1680872354 30609.81307713904,-86452.90566035676 29635.175479156682,-85946.92723594286 26209.125445232377,-83867.16851286532 23405.012373691115,-81430.96215760539 22416.650746237887,-80363.45550356185 21342.590611009236,-79032.12397538724 20653.15126827605,-78071.90006287357 19722.864607373103,-76568.94384777763 19032.931538149183,-75287.61544375062 18149.739039226803,-73276.99711051365 17674.467999276232,-71909.58929017259 17044.21035445791,-69496.69302563145 16575.96806831275,-63566.26485154652 16703.279503463997,-61961.900324365066 16921.21229213834,-60464.839179461815 17686.87717765538,-57403.3313099729 18327.687079919837,-55641.41716326498 18995.51290650755,-54164.02865291975 20026.87643650718,-52281.55723409144 20759.013515000446,-51147.865074746194 21698.416546990822,-49836.27078814836 27260.464785778968,-42299.67052288577 38498.04426282782,-27956.679463133794 40804.308713517625,-26092.71540672379 41581.96899394362,-25558.60796210541 43438.68386912543,-24465.720407017 44979.00442360783,-23720.862438452434 47107.27475778585,-22897.124855235754 49009.440305545475,-22354.246147769958 51126.49880471684,-21943.86124145142 52522.11199236461,-21778.6668627352 53588.83394789433,-21702.572174513316 54897.88103501475,-21680.05708715597 56244.78752289377,-21733.14780230542 57393.81608715749,-21835.700026405673 58731.64046786445,-22024.02215101183 60228.98921203815,-22324.174299809634 61896.09323306629,-22783.03337514949 63139.94044921313,-23204.221369654602 64090.800907874494,-23576.98576401682 65674.26656866015,-24309.808367294692 66553.02666646976,-24784.632273672945 67862.07146983605,-25573.69064636963 69494.5165891961,-26730.18455622293 70290.01891707258,-27366.710649326345 71185.47282879955,-28163.330332176458 72076.74238987663,-29027.16566138512 73333.75605211429,-30429.808158470907 74238.79334377193,-31583.56680642046 74831.0136841945,-32431.15059450896 75903.6498538363,-34193.78815170174 76223.81760852179,-34797.94739322001 76786.5726310044,-35963.84503678589 77465.29043773419,-37628.743093424695 77978.52987083,-39226.25297619656 78278.71533989259,-40366.72317479429 78487.39302491634,-41337.15068269697 78746.26491857355,-43028.716268232485 78856.10011310296,-44279.50517988515 78901.84555031867,-45374.649963429525 78896.02276821858,-46498.105031992374 78823.40676628627,-47767.418312852395 78556.15685611418,-49903.89632362587 78377.88338927635,-50837.52988307301 76559.23883142926,-56178.9601934882 76041.13774051913,-57211.36682327729 75374.11353768795,-58360.32300044459 68326.05538209784,-67823.0819780537 57911.1192222062,-81733.2538216986 51175.96802414432,-86568.01783261044 49238.109549331406,-87403.12106390609 43944.33941831958,-88759.61831542396 42383.99383795109,-88921.79786038544 38830.655109419225,-88915.51716794117 37743.06805777494,-88808.1959507605 35819.37685814041,-88494.6169571023 34368.3263175537,-88152.18583329381 31444.323874275586,-87160.98905650577 30434.68054348715,-86718.47811829804 26478.284677992826,-84416.99997420711 25347.514035972592,-83503.65671576133 22551.412920200866,-80891.2161164899 22225.733319382744,-80514.28793033102 20706.91422813241,-78536.50327110145 19216.227722749412,-76073.75310925386 18804.30526625037,-75257.30421216736 17848.61745119127,-72967.71714948791 16685.191707198504,-68258.07494231005 16465.660407207717,-63944.43604481246 16548.395361769286,-62655.20040169244 16671.039731516095,-61583.529931680016 16999.691794791164,-59750.21011013942 17522.962799711415,-57775.56854184627 17759.956848555208,-57045.94790022049 18209.118840062303,-55834.175279139716 18870.49114134807,-54350.869763842544 19455.294460839432,-53224.040877382926 20305.725247892446,-51794.01201155652 21282.522761537344,-50377.363561802566 27226.914269440647,-42328.98955184132 38959.75648565283,-27536.764341029775 40850.79342492445,-26052.524995276133 42482.54849163383,-24993.463372008417 43842.79072487159,-24243.76652532325 44840.69709372727,-23767.755708611883 46041.002656362034,-23262.949844386585 48268.11921560173,-22519.95534635907 50196.96369505753,-22058.252710018478 52008.14125874704,-21778.15593257237 53905.754766014616,-21636.455282641407 55538.10201376568,-21626.116353793666 56570.74416495398,-21678.184895568815 57719.87201601952,-21791.024899168526 59214.25419733158,-22019.078767599985 60273.3592715317,-22235.96071022388 61826.18323666767,-22649.18350669471 64242.85841453463,-23513.167859434674 65132.313098973136,-23906.98138107783 66323.73172854264,-24502.197889402152 67440.82064987977,-25140.837116498707 68489.74275450101,-25809.408751175215 69217.91678272621,-26323.58260090349 70555.92392306634,-27373.987753802685 71507.86613704244,-28226.640377097683 72351.28869393558,-29054.78823546097 73233.57270815657,-30016.92238988094 73884.05725873422,-30793.97306934817 74493.28692998635,-31590.121351066075 75576.79878906252,-33205.394497519104 76087.53220400494,-34069.79353130295 76720.8922391725,-35282.56767355307 77374.08549065237,-36734.354200080015 77810.94631852928,-37874.399892670706 78187.04178430144,-39029.988854208415 78599.26413261234,-40626.39272961827 78994.70281457221,-42883.305980651 79110.73648511882,-44052.039322686695 79183.58514630824,-45867.77562106087 79056.41544581515,-48374.67079589095 78897.57434128276,-49588.38399632432 78585.15542383095,-51259.817470031594 76991.05294395704,-55981.69903572447 76498.18455625963,-57001.734087134806 75756.1751201373,-58334.97857785338 74351.67101674271,-60430.020907398255 73599.72959528968,-61462.764828606836 62856.621809461045,-76039.92635632727 58094.85707514628,-82016.10397557651 57291.627020383,-82800.64589421437 50419.5028642201,-87289.49375912029 48980.98430949645,-87860.76519224128 47908.13084509493,-88219.40484169249 44968.87336385892,-88929.93192571543 44137.758903775284,-89064.04549827975 39664.171630689925,-89277.38004026485 38581.923898085515,-89204.77002998511 37308.42327853516,-89065.47115907296 36180.1286443394,-88879.18855786927 35306.457662361245,-88701.3286786832 33709.53116268127,-88283.2737736649 33269.947149113555,-88142.51961781635 25699.36351206428,-84138.58353887148 22782.592742857832,-81443.78493612257 22323.37453613251,-80937.90202157562 21270.335053886894,-79632.78943440359 20723.70154252778,-78882.53713611202 18259.58837176821,-74420.53929296108 17751.08098893406,-73066.44574664721 17047.85777202282,-70738.92663286818 16632.920471005196,-68633.48826436859 16378.477855169585,-64110.75003189792 16457.613690944603,-62838.06146460385 16598.834063462324,-61595.72856374684 16904.6782064296,-59885.068897271514 17461.000613747838,-57785.93553013288 18185.118170807953,-55781.62546324199 19000.315490515288,-54013.90075819599 19879.20926467308,-52442.53189408067 20763.93623618626,-51081.676659396566 25621.39398820151,-44485.769716240284 48311.44891712371,-22395.779036951153 49776.75212599253,-22016.776043742164 51273.48423425845,-21728.749106338506 52776.04333330808,-21531.909162790285 54103.945735538786,-21432.6969112385 55388.13807754642,-21405.557783306176 57149.88329648422,-21479.969518273 58471.17332601013,-21617.781511077257 59715.2923977025,-21810.277243558503 61009.01236646671,-22078.080826520825 62646.50331559932,-22526.423712300115 63830.20743135772,-22926.80496546045 65232.91491895416,-23486.507344609898 66596.91855897872,-24137.605724043482 67967.68777898041,-24888.910153268702 68742.68807073156,-25366.351569807757 70198.91268700095,-26379.015004963538 71289.34661551898,-27247.87495760181 72557.76908237943,-28387.638779534624 73975.14186814314,-29876.47700824396 75046.24123913034,-31181.13112699219 76267.69336876059,-32926.49406823192 76925.34061423875,-34017.35412646718 77675.30158574833,-35447.97415914044 78291.1713366002,-36834.50071028923 78802.81123674239,-38186.57694209967 79229.2851411434,-39561.69113643235 79616.84676288026,-41128.16403829093 79930.16799175976,-42920.83004534068 80032.94881192013,-43806.03713835005 80141.20745000402,-45503.933840342004 80135.42117411566,-47003.92195533823 79937.81629112214,-49441.749599205694 79680.97587580454,-51023.352127385435 79014.49763112076,-53618.4524282623 78392.97175914852,-55383.10224951337 78022.62369055333,-56262.964364819556 77711.97749493022,-56851.987361789055 67225.4577844292,-71834.48557459957 65971.9616405168,-73134.5633011367 65392.43380003905,-73655.20552819535 63820.13749261259,-74843.38287251281 59346.165132047594,-76884.31447484535 51808.55415870576,-77151.52710614789 50831.99152348924,-76920.92870982482 50029.94350035415,-76682.11078718328 47789.58809835536,-75764.09667946072 43234.960294836405,-72309.97608578233 42538.743247347586,-71503.590814112 41190.09283709019,-69525.42605607233 40581.49873127484,-68369.22196857231 40281.83060708534,-67699.44484966528 39108.76580195861,-63012.922550279945 39075.1814420593,-62270.26508451157 39106.97660160494,-60016.573935297594 39270.67322824011,-58731.05725371247 40455.42509214386,-54769.651655883914 51317.85290269218,-40545.9784652568 62282.686418434794,-26094.869564311615 124733.01794519092,54168.38383840253 125383.48223835524,55156.24185131305 126147.130451747,56560.96618125081 129649.90714863392,64050.717672313855 135837.9917792028,77447.25105726875 136076.28217711038,77924.23288496124 136593.40474999262,78734.20615723357 136913.0928605654,79135.51242177513 137648.76871824253,79904.27465580772 139640.69258346394,81703.94294742825 146028.8165279506,87188.76395117425 153572.90386914343,93584.94792107034 154520.1570149077,94271.74372883131 155645.45834076696,95016.78660227312 161181.20760171034,98544.52695392408 162212.48652774762,99143.82419965707 163053.1891992098,99514.31480126356 164352.82621515877,99883.10685661598 166591.53000709467,100239.14255258771 200792.97888845752,104256.0823719454" opacity="0.8" /></g></svg></div>



The ``highlight`` keyword helps identifying parts of the trajectory
where the 7700 squawk code was activated.

.. code:: python

    from ipywidgets import Layout
    
    squawk7700["AFR1196_20180303"].map_leaflet(
        zoom=6,
        highlight=dict(red=lambda f: next(f.emergency())),
        layout=Layout(height="500px", max_width="800px"),
    )


.. raw:: html

    <script type="application/vnd.jupyter.widget-view+json">
    {
        "version_major": 2,
        "version_minor": 0,
        "model_id": "edf5b2498c754ad9a88e8c388186b4ca"
    }
    </script>

Information about the nature of the emergencies have been collected from
two sources of information: Twitter and The Aviation Herald. The
following categories have been created:

-  ``nan`` means no information was found;
-  ``unclear`` means that we found an entry about the flight, but that
   the reason remains unknown;
-  ``misc`` means that the explanation does not fit any category.

.. code:: python

    tweet_issues = set(squawk7700.metadata.tweet_problem)
    avh_issues = set(squawk7700.metadata.avh_problem)
    tweet_issues | avh_issues




.. parsed-literal::

    {'air_condition',
     'bird',
     'bomb_threat',
     'brakes',
     'cabin_pressure',
     'cracked_windshield',
     'door',
     'engine',
     'flaps',
     'fuel_leak',
     'heating',
     'hot_air_leak',
     'hydraulics',
     'instrument',
     'landing_gear',
     'maintenance',
     'medical',
     'misc',
     nan,
     'operational_issue',
     'slats',
     'smoke_burn_smell_flames',
     'technical',
     'tyre',
     'unclear',
     'weather_damage'}

Cabin depressurisation
======================

Since the metadata has been merged into the Traffic structure, we can
select flights meeting certain requirements:

-  we found 31 flights related to cabin pressure or cracked windshields;
-  among them, 27 flights were diverted

.. code:: python

    pressure_pbs = ["cabin_pressure", "cracked_windshield"]
    pressure = squawk7700.query(
        f"tweet_problem in {pressure_pbs} or avh_problem in {pressure_pbs}"
    )
    pressure | pressure.query("diverted == diverted")




.. raw:: html

    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        vertical-align: top;'>
        <b>Traffic with 31 identifiers</b><style  type="text/css" >
    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow0_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 100.0%, transparent 100.0%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow1_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 77.1%, transparent 77.1%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow2_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 56.4%, transparent 56.4%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow3_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 44.1%, transparent 44.1%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow4_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 42.9%, transparent 42.9%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow5_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 36.7%, transparent 36.7%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow6_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 35.5%, transparent 35.5%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow7_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 32.8%, transparent 32.8%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow8_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 31.2%, transparent 31.2%);
        }    #T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow9_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 31.2%, transparent 31.2%);
        }</style><table id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19a" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>    </tr>    <tr>        <th class="index_name level0" >flight_id</th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row0" class="row_heading level0 row0" >ANA232_20180622</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow0_col0" class="data row0 col0" >14712</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row1" class="row_heading level0 row1" >TJK646_20180701</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow1_col0" class="data row1 col0" >11350</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row2" class="row_heading level0 row2" >AUA463_20190714</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow2_col0" class="data row2 col0" >8297</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row3" class="row_heading level0 row3" >UAL135_20180520</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow3_col0" class="data row3 col0" >6485</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row4" class="row_heading level0 row4" >LLX5963_20180526</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow4_col0" class="data row4 col0" >6310</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row5" class="row_heading level0 row5" >ROU1522_20190909</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow5_col0" class="data row5 col0" >5396</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row6" class="row_heading level0 row6" >SVA768_20180427</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow6_col0" class="data row6 col0" >5224</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row7" class="row_heading level0 row7" >JZR306_20190626</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow7_col0" class="data row7 col0" >4823</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row8" class="row_heading level0 row8" >GMI32FG_20180504</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow8_col0" class="data row8 col0" >4591</td>
            </tr>
            <tr>
                        <th id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19alevel0_row9" class="row_heading level0 row9" >EZY64NL_20190904</th>
                        <td id="T_a1e68a92_c28d_11ea_9b3f_85d1d741d19arow9_col0" class="data row9 col0" >4588</td>
            </tr>
    </tbody></table>
    </div>
    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        vertical-align: top;'>
        <b>Traffic with 27 identifiers</b><style  type="text/css" >
    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow0_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 100.0%, transparent 100.0%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow1_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 77.1%, transparent 77.1%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow2_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 44.1%, transparent 44.1%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow3_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 42.9%, transparent 42.9%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow4_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 36.7%, transparent 36.7%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow5_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 35.5%, transparent 35.5%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow6_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 32.8%, transparent 32.8%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow7_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 31.2%, transparent 31.2%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow8_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 31.2%, transparent 31.2%);
        }    #T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow9_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 31.0%, transparent 31.0%);
        }</style><table id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19a" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>    </tr>    <tr>        <th class="index_name level0" >flight_id</th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row0" class="row_heading level0 row0" >ANA232_20180622</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow0_col0" class="data row0 col0" >14712</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row1" class="row_heading level0 row1" >TJK646_20180701</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow1_col0" class="data row1 col0" >11350</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row2" class="row_heading level0 row2" >UAL135_20180520</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow2_col0" class="data row2 col0" >6485</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row3" class="row_heading level0 row3" >LLX5963_20180526</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow3_col0" class="data row3 col0" >6310</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row4" class="row_heading level0 row4" >ROU1522_20190909</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow4_col0" class="data row4 col0" >5396</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row5" class="row_heading level0 row5" >SVA768_20180427</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow5_col0" class="data row5 col0" >5224</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row6" class="row_heading level0 row6" >JZR306_20190626</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow6_col0" class="data row6 col0" >4823</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row7" class="row_heading level0 row7" >GMI32FG_20180504</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow7_col0" class="data row7 col0" >4591</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row8" class="row_heading level0 row8" >EZY64NL_20190904</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow8_col0" class="data row8 col0" >4588</td>
            </tr>
            <tr>
                        <th id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19alevel0_row9" class="row_heading level0 row9" >RPA4599_20190719</th>
                        <td id="T_a200ea7c_c28d_11ea_9b3f_85d1d741d19arow9_col0" class="data row9 col0" >4556</td>
            </tr>
    </tbody></table>
    </div>



These flights are usually characterised by a rapid descent to around
10,000ft.

.. code:: python

    (
        pressure["RPA4599_20190719"].encode("altitude")
        + next(pressure["RPA4599_20190719"].emergency())
        .assign_id("emergency")
        .encode("altitude")
    )

.. raw:: html

    <div id="squawk7700_depressurisation"></div>

    <script type="text/javascript">
      var spec = "../_static/squawk7700_depressurisation.json";
      vegaEmbed('#squawk7700_depressurisation', spec)
      .then(result => console.log(result))
      .catch(console.warn);
    </script>


Dumping fuel to reduce landing weight
=====================================

Emergencies are sometimes associated to dumping fuel in order to reduce
landing weight:

.. code:: python

    tweet_fueldump = set(squawk7700.metadata.tweet_fueldump)
    avh_fueldump = set(squawk7700.metadata.avh_fueldump)
    tweet_fueldump | avh_fueldump

.. parsed-literal::

    {'fueldump', 'hold_to_reduce', nan, 'unknown'}

.. code:: python

    fuel = ["fueldump", "hold_to_reduce"]
    dump_fuel = squawk7700.query(f"tweet_fueldump in {fuel} or avh_fueldump in {fuel}")
    dump_fuel | dump_fuel["AFL2175_20190723"] | dump_fuel["BAW119_20190703"]


.. raw:: html

    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        vertical-align: top;'>
        <b>Traffic with 32 identifiers</b><style  type="text/css" >
    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow0_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 100.0%, transparent 100.0%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow1_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 74.9%, transparent 74.9%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow2_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 60.9%, transparent 60.9%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow3_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 56.9%, transparent 56.9%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow4_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 52.0%, transparent 52.0%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow5_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 49.9%, transparent 49.9%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow6_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 49.4%, transparent 49.4%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow7_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 49.4%, transparent 49.4%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow8_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 45.4%, transparent 45.4%);
        }    #T_a2617572_c28d_11ea_9b3f_85d1d741d19arow9_col0 {
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#5fba7d 42.8%, transparent 42.8%);
        }</style><table id="T_a2617572_c28d_11ea_9b3f_85d1d741d19a" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>    </tr>    <tr>        <th class="index_name level0" >flight_id</th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row0" class="row_heading level0 row0" >BAW269_20181106</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow0_col0" class="data row0 col0" >15145</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row1" class="row_heading level0 row1" >TJK646_20180701</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow1_col0" class="data row1 col0" >11350</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row2" class="row_heading level0 row2" >UAL986_20191215</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow2_col0" class="data row2 col0" >9222</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row3" class="row_heading level0 row3" >BAW57K_20180319</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow3_col0" class="data row3 col0" >8623</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row4" class="row_heading level0 row4" >DAL445_20180303</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow4_col0" class="data row4 col0" >7871</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row5" class="row_heading level0 row5" >BGA114D_20190311</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow5_col0" class="data row5 col0" >7563</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row6" class="row_heading level0 row6" >AAL1897_20180603</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow6_col0" class="data row6 col0" >7484</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row7" class="row_heading level0 row7" >RYR77FJ_20190807</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow7_col0" class="data row7 col0" >7479</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row8" class="row_heading level0 row8" >VIR651Y_20181109</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow8_col0" class="data row8 col0" >6879</td>
            </tr>
            <tr>
                        <th id="T_a2617572_c28d_11ea_9b3f_85d1d741d19alevel0_row9" class="row_heading level0 row9" >UAL135_20180520</th>
                        <td id="T_a2617572_c28d_11ea_9b3f_85d1d741d19arow9_col0" class="data row9 col0" >6485</td>
            </tr>
    </tbody></table>
    </div><br/>
    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        font-size: 85%;
        vertical-align: top;'>
        <b>Flight AFL2175 – SU2175 (AFL2175_20190723)</b><ul><li><b>aircraft:</b> 42434f · 🇧🇲 VP-BFF (A321)</li><li><b>from:</b> ENGM (2019-07-23 11:37:31+00:00)</li><li><b>to:</b> UUEE (2019-07-23 12:23:44+00:00)</li><li><b>diverted to: ENGM</b></li></ul><div style="white-space: nowrap"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="300" height="300" viewBox="-84964.54803647526 -25848.508230967644 169726.03766471182 52570.2727185732" preserveAspectRatio="xMinYMin meet"><g transform="matrix(1,0,0,-1,0,873.2562566379129)"><polyline fill="none" stroke="#66cc99" stroke-width="1131.5069177647456" points="-68565.83075879737,-19192.13558263774 -69150.08132255793,-19404.10097439863 -69610.00848288037,-19512.714788537105 -70216.20285104954,-19562.35868783017 -70741.92184200541,-19531.83977007469 -72772.3290218438,-19133.169790801643 -75695.9141115911,-18441.01625671291 -76479.39060709448,-18118.66118319172 -77247.68138569224,-17599.320473294196 -77619.9827457315,-17250.003694738156 -77860.9180058911,-16954.111347621278 -78292.52488206165,-16230.716403117465 -78441.4864971678,-15846.394512590346 -78585.42664364165,-15194.826883988131 -78678.39849333778,-12936.11286883683 -78642.36440960175,-12272.841203755566 -78453.00875886965,-11212.618300836208 -78252.12712296909,-10564.031260972275 -77965.62728903547,-9880.168706817058 -77666.12590576876,-9347.009749387451 -77208.30896607408,-8724.141961945374 -76666.83715464384,-8179.4973921765 -76320.3519402507,-7902.0291020438735 -75772.48310807702,-7560.493649553773 -74988.73862948366,-7204.276063827841 -74355.68830560455,-7028.474727245733 -73795.57013372132,-6926.74631593849 -72507.15706101924,-6879.934965708728 -70342.9755229948,-7021.275681278943 -68234.08258276094,-7280.650556059904 -61448.49270232613,-8224.061036207539 -55591.84061165452,-8887.561424580535 -11379.262277483618,-13266.830953007926 9955.624246001398,-15428.64144233077 12126.455386702302,-15553.57347056902 13025.83215230814,-15538.293809856254 14696.325717184267,-15402.298829237987 27561.943209593133,-13730.472053209742 36027.483852257625,-12728.766718981906 53727.63622512247,-10976.181239031848 74002.686989301,-8245.436216801992 74738.8397543617,-8102.133410592091 75354.31134672399,-7906.528429354626 75881.76733641778,-7662.153621213365 76386.56575139429,-7358.947285282492 76737.77183042848,-7096.512934092279 77359.55805385389,-6486.524875680724 77711.50881486654,-6019.9202457312485 78040.18685069964,-5432.552995284898 78238.81406906254,-4932.248164721122 78370.11025580717,-4454.9943662766 78475.34008509909,-3498.7570933852826 77650.80611431788,-955.4713721347122 76985.16821660052,-220.37004930569884 76294.99245337087,270.84827026911717 75656.33702330386,576.6722372424746 75144.27827020199,731.5676625779255 74593.98592978722,829.7221742227399 73617.94966499397,889.1544908962926 71298.66654853887,924.6436646464717 68524.45600963659,1125.98945713832 50848.00862843086,2543.101855558969 33050.05867032402,4057.940459875529 9936.719873840917,6157.214568514903 -13285.146530652242,8416.907781505744 -44822.16043885584,11177.852539439611 -45704.9662682384,11276.258015621226 -46683.63418277633,11451.930012073075 -47385.34512198953,11654.84807448339 -47810.93323113069,11823.652255746008 -48132.301047160094,11980.922402665987 -48675.95655341627,12342.721140749116 -49007.00416924322,12632.476353298032 -49466.86912404283,13197.56068431419 -49669.58611611626,13559.638115799216 -49857.215269278495,14013.941506912593 -50023.5576542684,14806.869068376855 -49945.812870444315,15856.237690424696 -49798.81479293763,16341.828436128128 -49598.472682233536,16785.196105733903 -49376.48761279946,17154.61919670343 -48937.10035654496,17674.05524939939 -48605.85131184327,17975.220785304668 -48034.34242177886,18361.841677708726 -47546.080953267185,18578.460338316545 -46874.75222807944,18777.21018439399 -42269.7178726089,19664.664259525 -39108.70628182728,20319.0702756244 -38356.0670668034,20392.895934210763 -37742.39552107522,20350.746650023102 -37316.52292912456,20264.72426106425 -36562.450920118914,19986.44043677021 -35830.301045491586,19510.12335363107 -35453.776559409605,19154.473520738145 -35177.55885196454,18821.189327457 -35002.028585876404,18570.508248024053 -34716.207110633244,18007.551152577907 -34523.11984949813,17385.76117862325 -34435.86014726346,16858.52652416261 -34453.940457403165,16091.006737230344 -34511.50212345189,15712.895405924182 -34667.80901105567,15169.73990190802 -34951.45096020792,14553.898900785307 -35242.476352931866,14107.855920890055 -35616.15800309546,13690.22393782169 -36089.99707692577,13284.77632299986 -36617.10555332681,12963.611975824024 -37257.04287844104,12710.766058726253 -38015.3036841659,12516.32733224278 -45693.54950922037,10939.461734984247 -46501.0317453147,10898.094791910784 -47105.239497191586,10983.688191975556 -47756.5204442847,11200.63423321895 -48072.66214955375,11362.930373590669 -48485.51400611905,11634.699436999379 -48918.16662153822,12008.80020049538 -49207.27732680282,12342.187440441287 -49544.21657164495,12852.447242411818 -49893.048983400986,13682.876007983792 -50001.125896681915,14191.85871089493 -50048.81990214797,14649.003085880198 -50012.53587022905,15322.859007108258 -49948.026337000294,15695.460070134297 -49812.02742642467,16165.639687167242 -49544.735538603556,16768.899581823047 -49308.709011688516,17158.797907960226 -48896.02670320373,17657.50683233691 -48489.81103565805,18014.906399972577 -48016.84245531426,18335.067994889894 -47524.579743228576,18572.99107441285 -46858.57621326656,18777.00104605188 -38859.9450836899,20394.201654954126 -38053.76716841446,20435.61494446808 -37544.33346330806,20379.293264718337 -37020.10680940632,20241.28089432327 -36331.243980783634,19927.056378900663 -36046.67974467852,19736.699362957046 -35679.95792418972,19447.43191762087 -35334.69565766433,19087.001876946186 -34964.7601830228,18578.447996684867 -34720.70620511112,18100.962326021494 -34433.39361164136,16961.55362473863 -34430.52787187058,16165.824533488283 -34486.06826349701,15804.205896463962 -34717.824420696954,15019.79216804413 -35003.851812807996,14467.519977536587 -35374.33665058669,13951.018403318287 -35711.63143821887,13581.973328749937 -36158.54050587909,13208.94846548279 -36646.53735177437,12907.801611991503 -37160.08975047961,12683.836925924585 -38001.894731484106,12462.67304772075 -40078.86603343367,12024.988000325264 -45922.63592675682,10906.299009083272 -54352.45619444809,9778.354670928955 -54783.29244057098,9680.384072649615 -54984.63114910604,9595.238592811078 -55307.27442876635,9356.32522852385 -55532.18355485386,9097.357052555306 -55871.77973826731,8403.726904312327 -56749.05706330756,5729.093697974838 -57764.37577679761,2401.8479480911233" opacity="0.8" /></g></svg></div>
    </div>
    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        font-size: 85%;
        vertical-align: top;'>
        <b>Flight BAW119 – BA119 (BAW119_20190703)</b><ul><li><b>aircraft:</b> 405bfe · 🇬🇧 G-YMMU (B772)</li><li><b>from:</b> EGLL (2019-07-03 14:27:29+00:00)</li><li><b>to:</b> VOBL (2019-07-03 15:39:33+00:00)</li><li><b>diverted to: EGLL</b></li></ul><div style="white-space: nowrap"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="300" height="300" viewBox="-62886.75744350486 -73006.77502732295 126959.83909374714 146210.94094385864" preserveAspectRatio="xMinYMin meet"><g transform="matrix(1,0,0,-1,0,197.39088921275106)"><polyline fill="none" stroke="#66cc99" stroke-width="974.7396062923909" points="-23200.891225762363,66360.57145179376 -21632.194967699914,66365.11560608502 -21080.40410494669,66284.09419546687 -20895.087658149827,66221.18462563501 -20545.857200911207,66043.67798683992 -20286.784622614054,65856.1920339301 -18775.213661407517,64503.81890858845 -18503.63163028978,64167.9903090933 -18262.504080393923,63764.78064904554 -17784.980235871746,62511.43723748379 -17419.207485998813,61312.12252390211 -17328.56240637839,60629.355280259486 -17325.588054016902,60277.58361537589 -17385.99279260977,59682.166665922974 -17552.758562987736,59040.49866533814 -18061.26925092904,57871.73513299333 -19233.349165360618,55457.24464010926 -22163.74377150154,49870.18528045394 -25634.554678911503,42889.01726649897 -27476.33582092852,39096.87588194234 -27783.018913822027,38271.034905100656 -27893.635563556923,37825.001471418196 -27997.038105899628,37065.15093805562 -28002.206812846172,36484.14595530874 -27906.06115748908,35658.9923256274 -27733.47543529963,34939.95120168551 -27294.196078357592,33598.15667080594 -24136.465447828436,24920.026432223567 -23797.213161161602,24087.61801774253 -23333.545407343765,23102.573794296033 -22394.4088386167,21331.18729513274 -19871.7933659442,16920.644371023896 -16753.550357356886,11597.793973851374 -15839.333114322202,9821.75948929804 -15019.525122902658,7856.436227894845 -11949.362979879308,-220.53483465358087 -9479.568528026803,-6518.122364459162 -5231.524586404402,-17177.1792121623 -2432.5929557918753,-24087.804156462793 -1788.7681200910006,-25498.335594899992 -159.70190748765737,-28821.766179546976 1469.838646949205,-31874.162836980136 3306.726571514051,-34947.13634311776 6481.157628378517,-40018.11635476357 16983.54401202409,-56610.48028390929 17755.30003495962,-57760.63994603023 18405.047933879985,-58642.52721986628 19319.83934801461,-59754.40678032794 20296.56849001997,-60809.87387588493 21557.409763084524,-62001.486029733926 22673.447033834203,-62918.4511572429 23890.30787462569,-63789.84330299353 24951.37867105118,-64458.10063002407 26614.77920497997,-65349.25258497476 27714.862730168865,-65842.4195966988 28616.38122571638,-66200.07812677688 29809.356605869212,-66602.54549170926 31921.795501513836,-67142.6892583875 33737.77845639871,-67436.88960727958 35764.35096744799,-67591.55499236521 37654.404596958615,-67572.42733715929 38617.31150115038,-67503.31561757658 39865.39274830283,-67349.02342356043 41527.518587692095,-67034.97255464991 42833.78471821502,-66693.52602861576 43711.19210547567,-66417.47177903303 45414.981713486486,-65760.93069323449 46336.50331321419,-65339.92212030539 47274.970136584154,-64852.91832628893 49037.303484314274,-63781.33874093653 50497.56694690238,-62705.938401197076 51409.00509972567,-61931.986077896545 52240.506586955686,-61149.77600499481 53406.69574887992,-59895.55381787548 54585.32939215504,-58406.687483034126 55274.386965600126,-57386.11675698102 55737.75046626917,-56619.962510594945 56218.92761336204,-55751.64015687345 56717.13673253214,-54737.99626097617 57423.1405457955,-52998.91025649779 58031.681783990934,-50991.75687621784 58408.11953709351,-49127.96787960082 58577.72010366475,-47756.19651523596 58634.86209846307,-46883.44406992468 58657.861615284564,-45738.66329676307 58627.17717607141,-44780.918249438924 58503.24154988756,-43382.54013668608 58392.255199657775,-42618.770023549725 58085.921450257396,-41125.472552695726 57815.74230218708,-40130.915934485034 57462.91912375736,-39071.133506217775 57067.77317981857,-38081.09873092719 56676.82918750036,-37235.48030418789 55736.306642910815,-35539.54484610799 54709.5216624763,-34061.51343363677 53463.763163075746,-32603.563470034438 52691.75759396615,-31838.175669952692 51978.96721524767,-31203.632561319195 50462.975241101216,-30057.35410415756 49701.3385119578,-29569.538920808176 48655.30313689957,-28983.795877240613 47098.714165657,-28272.995442107138 45442.59737357527,-27704.762634634168 43575.423357085325,-27278.120709045776 42107.579095004665,-27085.929176654052 27402.971204319198,-26042.98127187819 10330.65727526231,-24943.015722655877 9007.538447869867,-24929.928024895242 7582.231594831711,-25045.00235659482 6432.152828218469,-25199.3191653075 4503.270237004434,-25557.20667827031 -157.83817805946472,-26629.166426792017 -21412.607751933996,-31217.124400722347 -22243.975981842745,-31405.269290257245 -23001.965232081773,-31629.856717937193 -23560.312139500875,-31870.77367749578 -23957.327803781816,-32099.59812679875 -24647.053117664716,-32658.1170191954 -25140.247640677422,-33239.847772691384 -25470.500489831793,-33762.771734378184 -25695.914380590173,-34232.59030576018 -25904.220469349366,-34873.7270537027 -25995.729313562537,-35292.750723098194 -26049.148214078316,-35817.26282206889 -26017.23546384316,-36611.83688363765 -25952.42031379534,-36973.71883782323 -25858.63044196214,-37359.737429373105 -25666.44303688711,-37878.55272563577 -25254.2324251646,-38626.287686804586 -24715.13112948665,-39286.55097381366 -24017.57407445495,-39869.8058751782 -23358.845793072738,-40245.043123447205 -22683.139343519488,-40497.53637798541 -21914.665496242,-40653.5765359883 -21017.422268026294,-40677.62127630486 -20276.539139298784,-40569.19970187203 -12577.984678962835,-38820.51761514303 -3459.4507630973035,-36883.85647285252 -2407.393860485423,-36584.25679760651 -1874.4550788934257,-36345.13465423884 -1192.9446337826544,-35887.023180591226 -622.8514647377838,-35331.36521395798 -219.54650709452713,-34787.205794909576 113.63642356266764,-34197.27157675185 329.2481086700292,-33648.32197329579 461.51347445882345,-33162.72346542972 558.7169966359556,-32403.948449348572 548.4160772016434,-31813.231805463944 497.1876825262861,-31400.752506897123 338.4728428197207,-30769.30679201644 143.97899884128498,-30270.25928453625 -117.68914087134138,-29769.476647479238 -480.25184454195875,-29256.847259336788 -889.4898751180281,-28813.752213418607 -1231.5005699463152,-28521.25625309071 -1600.4417119430518,-28258.507367497547 -2132.3270573616996,-27962.955881535505 -2555.1389574014684,-27785.39389526804 -3226.746231869518,-27600.831265316 -3684.4024106765987,-27530.960697065224 -4264.975194308558,-27508.423182205075 -4787.539139174203,-27545.603025885037 -5575.618567916379,-27669.112114059077 -8030.453964458015,-28191.475274026263 -16006.087400303331,-30098.087108989886 -21742.63554341759,-31340.31219334759 -22602.06878974747,-31548.73962886821 -23460.446165575027,-31876.401236447175 -24135.295584230455,-32287.632234591667 -24599.87614727019,-32689.411571102806 -24908.83957190344,-33032.16441623221 -25515.87269107542,-33984.89150673352 -25616.892152872413,-34207.083966740945 -25850.129866181855,-34937.26132118056 -25972.302775007287,-35872.884314474395 -25897.91586866134,-36789.88556607056 -25727.44098535107,-37438.06408580664 -25378.588953228664,-38216.57040896987 -24934.279877688936,-38871.21843086073 -24289.716383427494,-39511.223413912194 -23475.98387530026,-40037.97756926067 -22692.78715873059,-40357.32585004563 -21808.834529613403,-40531.97470093376 -20878.388874072298,-40530.49953697164 -19934.356803199375,-40381.40601658036 -17645.862903167723,-39880.380015577066 -11018.63429956645,-38294.427785249056 -3743.5147394747837,-36738.658935945175 -2812.094353204038,-36466.93124937313 -2254.075084921825,-36200.932074473094 -1746.206491917601,-35851.221441991845 -1259.4224027938862,-35388.21609789454 -777.8220796965056,-34761.69079759342 -475.447818931618,-34217.965936270055 -212.43688309659774,-33560.29032768817 -54.65897372953417,-32871.528254278455 -12.591323499997849,-32457.232586052614 -50.49537960765086,-31599.379548349167 -117.72165801299174,-31214.341045645637 -275.669757688808,-30672.554676232652 -674.7512077175127,-29841.936711222537 -1021.5040438659709,-29349.900535696113 -1299.9295130728763,-29033.935529196267 -1988.0432093843292,-28448.520816412893 -2728.6249780056496,-28033.886531392116 -3564.423537980511,-27748.292877383923 -4257.0598921010815,-27634.100912869366 -4624.7609208411595,-27618.252242318107 -5481.078993734872,-27684.748382588874 -6878.7628587532745,-27952.39157724906 -15474.122897010257,-29989.984929728616 -22549.454118984835,-31538.609173251687 -23100.75719159653,-31726.609477009177 -23676.606099623496,-32010.075180160817 -24111.819042005125,-32297.493809165695 -24569.357870433916,-32702.760790597258 -24970.937027260556,-33169.3643162683 -25327.018758555405,-33743.098349273736 -25602.050379957906,-34398.76997750257 -25730.629111854367,-34874.583235853104 -25807.28632597524,-35485.29603085292 -25809.711553839556,-35977.26279191305 -25692.534890199302,-36759.82504146622 -25472.971589015575,-37429.295849507704 -25072.94707726399,-38145.54076023471 -24081.98787584192,-39172.37537591497 -23268.834109708507,-39634.96620540526 -22527.474973797114,-39876.42452256454 -21811.79388733291,-39977.84712780403 -21245.314514316757,-39973.96030888486 -20795.331096450696,-39925.01481895302 -19393.655194317813,-39640.38639337513 -13998.467293763755,-38280.06160178177 -11217.798916968279,-37634.2457196074 -4158.7815798484235,-36173.87234283549 -3485.152418573847,-35967.21774857516 -2821.9021205435242,-35652.150477299576 -2232.6087315153236,-35216.99550117541 -1807.3064224627067,-34766.528941949204 -1376.9083581188574,-34135.20645525743 -1074.8976346395004,-33466.96784659049 -885.4499554567661,-32721.271388082092 -833.7099005111546,-31925.226855924175 -920.5989491545679,-31161.35213272168 -1191.674069890796,-30310.86482685247 -1547.072164080899,-29650.138223133632 -2077.577407320762,-28997.433160758366 -2434.7272808622697,-28671.018076405428 -2875.8987671105524,-28354.89029140151 -3301.3087998274145,-28121.595078756098 -3984.0739969270107,-27862.18150874055 -4612.972975673776,-27732.195390681332 -5197.410545914247,-27685.038125553918 -6203.775067312881,-27781.47398526887 -8171.476592665991,-28193.932438862586 -14953.553312094558,-29893.103050518726 -18105.952879878605,-30610.668085238733 -22044.36812090399,-31442.365595289575 -22900.899714463503,-31676.5637237043 -23357.312748776574,-31862.966378577068 -23993.74604605905,-32226.742393929148 -24646.56248263556,-32788.96921308512 -25256.285125076174,-33597.744788735676 -25440.691428508526,-33951.33673391883 -25639.07238408329,-34440.018666343036 -25762.39507602353,-34915.85686002195 -25785.980058769794,-36500.430123198304 -25716.017515593794,-36832.29801343269 -25524.116455910676,-37408.67727627127 -25235.321218883724,-37979.04285856086 -24720.349551753694,-38660.16541352206 -24135.528201729056,-39185.67984164876 -23755.09372177943,-39433.578061405395 -23069.26084336413,-39749.77927612958 -22342.3412492295,-39938.843225676625 -21922.12960334949,-39986.4492921001 -21295.756374770477,-39974.791266107495 -20096.008548702834,-39749.33934689208 -18687.07418849111,-39347.77143737486 -15388.207741447446,-38312.41366097101 -14439.767032304284,-37968.144613763485 -13543.239954999339,-37458.45708319353 -13127.343071183506,-37133.605917370005 -12747.252845677167,-36762.80337824936 -12161.334727061127,-35985.32543707441 -11749.046180070944,-35142.14265478507 -11579.457774715549,-34578.043910986606 -11483.629278607952,-34055.20722087366 -11450.145758977163,-33174.90161502828 -11493.319166938985,-32732.54530094461 -11614.89468261361,-32146.650530628453 -11945.25771215894,-31221.418040076307 -12457.833374863385,-30137.866213490866 -18962.09554545228,-17559.65283194054 -24047.42764753472,-7568.218345684405 -41112.73495540397,26656.137858289036 -41552.03279114836,27627.326659784176 -41810.23055343136,28479.947223500334 -41927.03278883443,29208.164169903677 -41946.54811237909,30093.99703558738 -41871.43062198414,30797.7846013231 -41771.60981813551,31257.94246794001 -41482.03535209832,32084.31231313342 -41200.22225312191,32636.079719498248 -40879.61122127993,33122.46745387809 -40288.917696379365,33815.14539438393 -37801.31628875945,36152.95782224041 -33311.345574526225,40172.293188082076 -28947.94843584755,44195.51848116952 -23592.44435057948,49172.14391334473 -23063.421811031825,49732.48140852038 -22698.90437876568,50240.16200410573 -22490.237554362804,50622.60266960479 -22343.812498801195,50977.09412446457 -22201.806436153845,51511.244957609066 -22142.947818699362,51964.268622215524 -22145.951552220737,52468.48878959537 -22307.809115043645,53314.626405634684 -22502.87655540363,53813.12274477707 -22814.06516158154,54353.143292428715 -23247.47445101228,54878.19583834522 -23746.721117449815,55307.33953009389 -24174.121255178892,55576.497697132356 -24810.362168236774,55854.03207455167 -25325.010863466454,55996.38243999002 -26095.038389422243,56087.76233627636 -27159.811142629063,56098.367563640495 -52219.96272492587,55380.751290853834 -52735.34720590624,55386.00885629339 -53483.49031795418,55492.14437778768 -54683.34425213226,55874.2353963551 -55823.00676664942,56335.37486918869 -56375.56136819966,56661.58685467021 -56750.64795543558,57008.51507722265 -57105.07779195602,57530.39863310799 -57301.84099063081,58076.44589009341 -57418.87952794891,59010.080536735695 -57471.537408547134,60345.97151472131 -57456.026577137025,60809.2865523517 -57378.66666891173,61261.724035885665 -57034.473586055705,62066.94592601374 -55537.59424581973,64363.66316542771 -54990.30856499818,65131.68901434377 -54694.88219354918,65480.74470613653 -54256.16521833017,65843.83931792578 -53711.84818904988,66169.60779330217 -51239.64593998686,67334.16289778553 -50675.52674222651,67568.28795772934 -50359.59956513862,67656.51050965482 -49730.152044808834,67750.68251669624 -48774.94859387331,67783.81444198746 -26582.34880578678,67788.94588157797" opacity="0.8" /></g></svg></div>
    </div>


Landing attempts
================

Also, sometimes emergency situations are associated to several landing
attempts, at the same or at different airports.
``Flight.aligned_on_ils()`` and ``Flight.landing_attempts()`` are two
experimental methods to detect these events:

.. code:: python

    from functools import reduce  # Python standard library
    from operator import or_
    
    attempts = list(
        f.assign(flight_id=f"{f.max('airport')} on ILS {f.max('ILS')}")
        for f in squawk7700["AFR1145_20190820"].landing_attempts()
    )
    
    # reduce(or_, attempts) means attempts[0] | attempts[1] | attempts[2] | etc.
    squawk7700["AFR1145_20190820"].last("45T") | reduce(or_, attempts)




.. raw:: html

    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        font-size: 85%;
        vertical-align: top;'>
        <b>Flight AFR1145 – AF1145 (AFR1145_20190820)</b><ul><li><b>aircraft:</b> 3944e1 · 🇫🇷 F-GRHB (A319)</li><li><b>from:</b> UUEE (2019-08-20 09:04:02+00:00)</li><li><b>to:</b> LFPG (2019-08-20 09:49:01+00:00)</li><li><b>diverted to: ELLX</b></li></ul><div style="white-space: nowrap"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="300" height="300" viewBox="-88076.04666313536 -47330.4147151632 175325.09753451747 95508.94617064831" preserveAspectRatio="xMinYMin meet"><g transform="matrix(1,0,0,-1,0,848.1167403219151)"><polyline fill="none" stroke="#66cc99" stroke-width="1168.8339835634497" points="80755.52874047405,41685.00932457706 -3606.3805048287827,35575.69818036425 -12472.881329181224,35238.817621174756 -13555.887889014539,35103.9588647076 -14623.615243586853,34892.900449170425 -15702.085187675042,34580.25037087984 -16557.93925354107,34251.85173825616 -23338.524429835878,31038.619421819294 -24612.033189028105,30305.98320802661 -25980.18733793354,29375.524212062053 -41424.574445559716,17155.01989937049 -47723.69885916535,12246.108666289041 -55700.25421360731,5606.3446539320375 -56320.61423438546,5115.6631571794605 -57167.60980213031,4523.187005167357 -57791.81631135366,4131.787540825961 -58647.77263039039,3664.601572988678 -59610.51547574449,3234.970766263927 -60195.92124294198,3018.267295603465 -70528.68279313362,31.346773953593956 -72032.06945120747,-493.28674751302646 -73549.96043895304,-1191.9900274006272 -74658.20630615705,-1827.4899762399084 -75625.00416137153,-2473.544093474649 -76649.12550159433,-3277.332846177449 -77396.02980234529,-3948.101258181807 -78328.63975353153,-4922.558884994938 -79104.9887432252,-5899.139933237395 -79660.10152062569,-6726.1048361398225 -80253.00280492818,-7787.2912050106415 -80601.06470911235,-8555.511671180973 -80965.68029773976,-9537.326633371287 -81147.187245289,-10146.258691786354 -81399.20012577686,-11272.905414151557 -81535.03899857697,-12314.721257687223 -81582.52453222731,-13388.411574676724 -81545.45749085746,-14402.265689957743 -81425.08938123363,-15442.825599408736 -81265.09139013659,-16285.387746980612 -81027.56763272088,-17195.29380005065 -80706.7090578985,-18136.97548096899 -80266.59046145208,-19161.86620973113 -79844.09646964271,-19972.59991559537 -79312.29967292974,-20835.810626517927 -78506.32915541921,-21942.243726952678 -78073.72881673572,-22447.465524772593 -77535.981789313,-23019.566877986956 -76824.71330817482,-23692.44822241018 -75191.4062065165,-25009.768433882586 -55949.351412104406,-39288.71094219408 -54900.018427035284,-39925.85917915911 -54565.51342719222,-40094.896818548696 -53731.97098344497,-40419.6889361072 -52966.20183298537,-40630.82377963807 -52409.26330670505,-40738.02406065299 -51805.89709788739,-40809.968884339716 -51109.94967597976,-40836.89258425515 -49964.443493210536,-40744.71956227888 -48746.64098123119,-40456.291088786005 -47667.44713938284,-40014.50821221966 -47055.77581886356,-39666.70440714031 -46458.080652882876,-39258.65370149853 -46008.44252143132,-38879.32184578288 -45504.34465299064,-38356.12572583224 -45064.95585680939,-37815.00431524125 -44718.02179898636,-37308.729654735325 -44501.05352470643,-36934.69720846444 -44025.39649216669,-35892.683120948604 -43784.428920410406,-35081.22187387668 -43667.33893647283,-34440.65517237934 -43515.582218892894,-33178.5297206498 -42277.74081557777,-19717.11499376407 -41713.506736885625,-11666.701124718915 -41102.26866183524,-996.6177482508376 -40091.032661038196,11138.170394119641 -40016.96273665505,11758.984850915513 -39825.06236037219,12420.355803646265 -39527.19869103746,12967.041330173593 -39171.42406998451,13404.584599729327 -38834.90604508281,13693.766191150837 -38317.71381900397,14014.651631036379 -37645.08775087911,14260.71459116995 -37119.2200738889,14348.728806359852 -36349.931732206234,14332.4541798424 -35565.67098714939,14134.614853887719 -35028.249892954074,13883.59672088631 -27393.05522597091,9463.799525429498 -23029.171505712595,6860.032811676626 -22497.43293715541,6386.26291530632 -22109.082028616678,5834.770933061435 -21896.424279964158,5265.720063527816 -21834.857184122247,4613.05852422555 -21886.472962772084,4209.651437277178 -22084.990009627312,3630.276676763484 -22433.652809748062,2923.7222647317226 -23300.08306600384,1399.909911329498 -23677.981817239488,810.9272075378408 -23950.37944276819,516.813977474143 -24389.994521239638,147.09109667711596 -25347.17202887083,-495.1443650962926 -26419.36027719191,-1146.8349829593506 -32376.293228046125,-4478.331079716158 -39243.702873030066,-8414.78087360837 -41532.74085106838,-9655.971512007014 -41883.2744218217,-9891.460558180752 -42110.24203951806,-10101.997505486397 -42457.39404277505,-10566.251269641427 -42664.95643746434,-11050.39426590264 -42765.53865542316,-11557.033320986748 -42736.64982890748,-12193.360013472087 -42586.76477516621,-12713.098575855684 -42244.66895205948,-13444.636691147853 -41528.63909325645,-14735.465246881298 -41191.241843769574,-15153.293653973977 -40766.53639641032,-15507.551192896692 -40392.151417444395,-15717.458403108227 -39968.9439140672,-15865.565231858927 -39555.14124386739,-15949.538467028118 -38520.06544714649,-16046.893153531724 -37435.3975457834,-16090.715970656793 -36925.10758373614,-16024.047582188829 -36484.87138814845,-15884.459107304769 -35190.32408290254,-15272.094055245896 -29648.32767558071,-12348.656101096443 -21623.696781836985,-8192.550529460204 -21272.54973527841,-7976.816967559498 -20854.52300884496,-7610.472360298003 -20628.1942153832,-7321.382048900743 -20379.799872962616,-6861.498522915602 -20248.052846154875,-6305.639306796776 -20186.35690064563,-5308.886447441126 -20185.681089993155,-771.4013144806513 -20231.075240289603,-358.81782620485563 -20406.62167206682,105.17024578666931 -20551.860524016458,304.28918662236526 -20801.868244391986,518.4838320222626 -21311.53398889976,699.27246495776 -21937.02115994628,732.3178084681034 -23194.385359585423,518.6170237383557 -23638.296262962464,347.422354471171 -24112.83848370359,107.62104210450578 -26356.505550236805,-1091.1338653289472 -38507.18820245955,-7926.454885702378" opacity="0.8" /></g></svg></div>
    </div><br/>
    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        font-size: 85%;
        vertical-align: top;'>
        <b>Flight AFR1145 – AF1145 (ELLX on ILS 24)</b><ul><li><b>aircraft:</b> 3944e1 · 🇫🇷 F-GRHB (A319)</li><li><b>from:</b> UUEE (2019-08-20 09:34:02+00:00)</li><li><b>to:</b> LFPG (2019-08-20 09:37:07+00:00)</li><li><b>diverted to: ELLX</b></li></ul><div style="white-space: nowrap"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="300" height="300" viewBox="-6979.629488080615 -4335.528109903207 13950.157925264619 8678.767064186846" preserveAspectRatio="xMinYMin meet"><g transform="matrix(1,0,0,-1,0,7.7108443804318085)"><polyline fill="none" stroke="#66cc99" stroke-width="93.00105283509745" points="6453.855921433461,3826.5664385330974 6419.090259140633,3792.697528734646 6361.797718651541,3741.7159577449993 6223.8345367771635,3624.367256765464 6148.93018100135,3567.327116844205 6080.504436904022,3512.280063741125 6015.164789695252,3468.800824960011 5940.2504164902875,3416.941270693426 5861.676789650481,3359.297039947984 5790.417371079689,3313.2252793397224 5710.575834858043,3252.2163951114467 5648.040595157333,3216.5110990654903 5575.091372193716,3165.5216130261215 5502.1407772682105,3114.5331122762996 5437.210865944305,3074.685674397386 5362.285520325024,3022.8341361110515 5265.951545942615,2955.4297301808124 5199.890411575472,2915.670343454214 5126.932927529556,2864.6869096309047 5062.560141283755,2825.7937483050787 4992.978771841079,2779.130020642678 4913.25328837459,2727.0214154108426 4840.286027053749,2681.133242906551 4762.808434856877,2633.9476709976516 4725.620059350912,2609.7525890304028 4612.922427097766,2540.619708771942 4543.329192276495,2499.1394546927195 4500.502387704262,2473.215306315717 4425.5555509558635,2426.554442020836 4345.109973639736,2380.3216831737805 4281.006034350481,2343.595422036415 4199.147389985078,2298.744437006902 4126.168325588065,2252.865903217941 4058.403395360632,2206.9922550050005 3981.192432559798,2162.1480519887896 3896.7928108178644,2115.2287085325115 3815.2105552019725,2068.830432483964 3776.8833922095296,2048.9558185355013 3672.6109017335816,1992.8789312140557 3604.8355034033066,1952.102354205879 3526.6277798812152,1911.3198337488845 3484.918912301179,1885.8359357914364 3381.488914275746,1825.1915861506443 3306.52034160656,1783.7239126793193 3271.1498249882957,1763.509508324485 3194.0653336261034,1721.524350358096 3114.7256714166924,1676.8638609897682 3044.1300295769756,1623.062202238593 2973.9416151247924,1595.3226771100217 2894.1699425643583,1560.847956775187 2824.5509957457366,1519.389823183978 2760.1443759469507,1483.1992094311854 2701.3734306838155,1452.0172557065325 2631.7516137637776,1410.5616036889417 2583.54994305628,1384.6500415790958 2492.5024083662925,1332.8306645938023 2406.808076080633,1286.192870255046 2332.529954850151,1243.7034720812846 2264.735123972534,1202.9436893786706 2187.2115817892254,1161.8356469956987 2108.2787338710805,1116.3271436538814 2047.9496227770326,1084.1161502879233 2005.0997226032505,1058.2117670327345 1903.327979935931,1001.2206820496833 1860.4749585929053,980.4951180757007 1790.136062053717,938.0145228243497 1721.2036638871016,902.7840317415764 1624.7836923180287,845.7998889822406 1565.8594705342805,810.6606790319171 1524.132930312981,785.19190452355 1425.0296567964037,729.1598104302942 1362.2956128396445,695.5751207318282 1284.1944064912038,652.7539848024129 1216.384314671408,612.0073426046735 1148.5732042461327,571.2615506050472 1080.7600609137946,535.6079845138962 1007.7299616887528,494.8629210510666 933.712096829475,452.1338999970273 872.1006597114553,413.37741858951847 804.2837260785659,377.7273172924191 731.2494252706422,336.98598578427266 663.4312338526086,296.24627557231554 590.8172532978414,260.51197061865827 564.0281707381858,239.79844718212127 459.9701168705665,179.12362015866762 381.8547248706692,131.05154858367308 345.19331204232435,112.92725759459253 247.89866168787296,58.55643921159795 167.80545401222486,11.091429838573042 141.71845048006838,-4.183429086641324 65.71273111830645,-45.00214743912595 -30.741471599037368,-101.95677775340128 -68.25203325257564,-127.84476089178182 -137.91515375795495,-164.08709105879618 -213.0792205716,-207.83592480086617 -286.12877455598397,-243.4721526376538 -346.910761436679,-283.1640298411203 -384.42408444657383,-309.0498196888903 -507.68348943334377,-381.52697160778837 -550.5566410037733,-402.2334420600228 -620.0864677131533,-447.0990172016345 -700.6172542317319,-490.2364016329678 -760.9810487828282,-523.4516022194366 -818.525157717897,-557.5294993062255 -896.6615272948435,-599.8013841884098 -964.5036509911682,-640.5206826076924 -1038.2701932206442,-681.7557660470711 -1100.1899208506204,-716.8653611645043 -1177.6256705685169,-759.3922153389445 -1247.3054998165092,-800.7979239699425 -1319.384451928545,-844.1006276228917 -1381.3063355148884,-873.2526982564885 -1444.6413424405368,-910.2564675889245 -1520.673090417795,-956.0579718776384 -1580.345561509339,-1001.8631968740376 -1653.4132340070864,-1032.3896073414783 -1740.4503410826635,-1080.2557229925067 -1810.1385856047532,-1121.654182884225 -1879.826099904117,-1157.8740888737502 -1935.2694692940684,-1195.2200222750453 -2008.3455111413639,-1235.9243668608501 -2070.9832686448153,-1271.5403570369267 -2136.2314943012725,-1307.8449496205826 -2201.4805930821185,-1344.1487555793149 -2281.8998441230315,-1395.8920887320423 -2342.424518100538,-1429.2621758970777 -2410.287861351788,-1469.9633427467984 -2478.1522226364177,-1510.663658301535 -2546.017601943777,-1551.3631225492497 -2608.9423156688263,-1582.1402772524013 -2678.6461870816247,-1628.7052021608686 -2748.346010639937,-1664.9139250125954 -2807.043119851637,-1698.8838035584486 -2880.1324839567146,-1739.5763753438453 -2941.3731005928908,-1778.7210996398012 -3005.7163073136862,-1814.9294107896621 -3075.4236394153404,-1856.311572134529 -3145.1290342376697,-1892.515185822781 -3224.718308725756,-1943.0368735795998 -3292.5946946239933,-1983.7269731491244 -3360.4720984162363,-2024.4162211694515 -3433.5699972073407,-2065.1013181518338 -3496.2266799782547,-2100.700808936856 -3579.4866035596756,-2145.950971934814 -3643.8418983834817,-2187.3293429791165 -3720.7529198830352,-2227.8354147708283 -3799.362788851529,-2275.2427373419778 -3863.721293324738,-2316.6184925999996 -3906.6225393585173,-2337.2983767732044 -3981.846646227764,-2380.390779648794 -4046.0654261764184,-2420.038587666458 -4117.62273017432,-2461.75113086184 -4196.238520357275,-2507.944570487923 -4258.622821972965,-2543.103981968075 -4326.514465210855,-2583.7811109152876 -4400.0523442920085,-2626.8690496611 -4459.049951723797,-2657.886780559123 -4555.598081453419,-2714.7607120902267 -4618.971744835323,-2746.4633802376775 -4681.64846160046,-2787.140493163778 -4748.695882248841,-2823.3257667318444 -4818.429457333427,-2864.6854714852375 -4880.120249020163,-2904.066751469291 -4957.899796083568,-2947.4021850619433 -5031.588706197379,-2990.481413226433 -5104.712145559361,-3031.143937429757 -5177.836351505135,-3071.460298021206 -5247.576385288093,-3112.8144728999046 -5282.305034260121,-3132.80100892815 -5360.230888857992,-3174.8361148498934 -5418.11943698875,-3214.1287268075607 -5486.023019590752,-3249.6999650322055 -5585.272347961317,-3305.603583062463 -5660.666284775974,-3350.570970185982 -5687.491626096611,-3366.075855698632 -5800.161946073375,-3433.265976853754 -5836.023226496625,-3452.9888720319555 -5907.467459939373,-3495.2819555257615 -5977.218499091058,-3536.626727563804 -6055.439287129531,-3580.0322171725225 -6127.447923579549,-3619.3014815481956 -6191.270659954856,-3656.249193185459 -6259.190711921097,-3696.902072123384 -6320.615377561548,-3732.9880849758847 -6390.372618779119,-3774.3275324700553 -6462.956972330074,-3818.8555941526656" opacity="0.8" /></g></svg></div>
    </div>
    
    <div style='
        margin: 1ex;
        min-width: 250px;
        max-width: 300px;
        display: inline-block;
        font-size: 85%;
        vertical-align: top;'>
        <b>Flight AFR1145 – AF1145 (ELLX on ILS 24)</b><ul><li><b>aircraft:</b> 3944e1 · 🇫🇷 F-GRHB (A319)</li><li><b>from:</b> UUEE (2019-08-20 09:44:53+00:00)</li><li><b>to:</b> LFPG (2019-08-20 09:48:26+00:00)</li><li><b>diverted to: ELLX</b></li></ul><div style="white-space: nowrap"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="300" height="300" viewBox="-7442.8691777764 -4388.016463054885 14875.985618227234 8784.798440759163" preserveAspectRatio="xMinYMin meet"><g transform="matrix(1,0,0,-1,0,8.765514649392571)"><polyline fill="none" stroke="#66cc99" stroke-width="99.17323745484823" points="6882.154010146121,3845.819547399565 6803.979704071869,3820.2638483342853 6772.709775601243,3810.0418845932377 6651.019716739881,3761.911213748706 6579.892426026683,3733.4333716478063 6501.161349623136,3699.5970526047163 6439.190812077746,3672.1675882648187 6381.873680384188,3641.551395506004 6303.141841819656,3606.164649250805 6225.542969712724,3564.99858761 6147.943702615147,3523.143280337152 6078.369580030094,3486.820955270724 6035.552498812121,3466.0623354504405 5965.976891598829,3429.7414556724398 5880.346255772315,3383.0486435209273 5826.828446330089,3351.9246906685976 5751.3317284275345,3314.9961500945574 5671.615743159499,3268.916987677947 5618.090245027999,3242.9727894508633 5521.748331092225,3191.096877878714 5473.57668309917,3165.159565521394 5370.892408603032,3110.9499824153017 5334.4112454176975,3090.5483759448166 5261.448095748295,3049.7459012035865 5178.05960026577,3003.843048570852 5110.303908982406,2968.1389876099583 5045.366686299168,2931.7480388552544 4980.004513610231,2896.7381979581673 4906.187574773195,2859.13264569498 4844.486956354889,2825.335885663416 4767.009288849826,2781.343146401282 4697.416406288488,2745.0385827823807 4627.822592234145,2708.73491427948 4558.232198971707,2667.254455268104 4479.61976272287,2626.457790862732 4411.854760787646,2590.762477673385 4338.873875516825,2555.063838113812 4263.783868786098,2511.6842876544038 4194.184159023722,2475.3861971560964 4151.355849496119,2449.464805021492 4081.7545891300188,2413.168160868121 3979.1851491521447,2356.2236704039 3906.2019287795893,2315.439477778286 3872.948917433124,2299.105916896194 3838.429268467688,2279.7513470953477 3728.950895616373,2218.57827594048 3653.42883532572,2174.68914016049 3583.824139413255,2133.221219663088 3509.9900374185195,2091.147393709259 3433.9019615680218,2045.103591244672 3369.223827130827,2009.5958926673482 3310.7449093990267,1977.717126786152 3241.135066172764,1936.2536169133236 3187.5878405797735,1905.1557853092747 3107.2658509968805,1858.5100319369124 3025.1105121541664,1810.8289129971936 2957.3285272622247,1770.0604360261127 2887.713133931389,1728.9466540430597 2812.7400705437626,1687.485808882333 2748.4795312543447,1646.0314213054396 2705.6351821922844,1625.299158888525 2628.8278015006845,1581.5099933904846 2555.684700884306,1537.2058765133058 2486.061060076295,1500.929763816332 2409.813122846879,1459.215037613592 2330.7465097373697,1412.8402641584687 2287.8987226507716,1392.1113090202796 2196.8492335384985,1340.2968903875721 2117.7781905693964,1296.1688895907023 2076.0583166636516,1270.6958489543854 2041.5253879201882,1252.2156941694286 1935.2474435244178,1194.2730525642348 1859.4152680026375,1148.5966832933082 1768.3557143017258,1102.5712138316378 1704.0788282718936,1065.702850625449 1643.1839858645335,1031.2525231699838 1580.8795641454017,993.1780982105798 1538.8721795467325,970.1253669704149 1473.7454402651626,936.1935158298016 1356.3195889380063,868.2490629320606 1291.6138671193546,832.5936186305196 1257.2169084870354,812.2200233535943 1194.624746181724,776.5662296210273 1093.404202603589,718.6412456260185 1012.059177764948,674.7015415376324 980.9036020368369,651.3100009269933 918.1662599182557,618.6796887397595 808.6221979950765,552.4743579238938 745.177411002711,521.8307200028999 688.6415003180578,486.2699114351817 611.2381098406688,438.9712508397413 541.5873739938847,402.7201618035952 469.53887225115284,364.0535154045318 407.64163360119755,325.043403611594 343.34625026192305,288.79527146802445 271.2949823317938,246.9383692699154 203.4725734624834,211.29578537747238 146.08369584181108,180.74569563117964 78.25971032904427,140.01330769912457 15.652044887590847,104.3731555551716 -52.17387465784057,63.6424031014422 -114.78327826307937,28.00376043233458 -171.04790900544978,-6.339944914197022 -256.7855874240257,-52.93555955782067 -321.08988931309193,-89.17579554064734 -380.0364656130797,-125.41564531022361 -449.7014994625108,-166.83163986912163 -514.0084399143859,-203.06958285443298 -573.9443968274785,-236.71839356796508 -636.5606939667074,-272.35099796097387 -712.293863979166,-316.9557890452093 -771.2455280894015,-353.1913754116641 -835.556825052977,-389.4254965838574 -915.9464986849634,-430.8338260967348 -969.540912966347,-461.89049092444696 -1038.3682525783508,-501.40057978009776 -1108.8886266625054,-539.528737673224 -1142.7386645093402,-557.384648173171 -1200.144379013534,-593.0118266569967 -1294.0808626885246,-643.9039074408562 -1360.797685942203,-684.4459311287488 -1419.7563191861218,-715.4967897163426 -1484.0762415383228,-751.7232031056233 -1549.80835939504,-791.4865168231443 -1602.0001133521107,-824.1769113817088 -1666.3225408524681,-860.4011582209428 -1695.945285582631,-877.9960230686695 -1789.6088827513554,-927.671469876795 -1848.5743769343821,-963.8953152450148 -1886.097465127958,-984.5930375808507 -1945.0642220582588,-1020.8158311703146 -2004.0298541255372,-1051.8603244797573 -2089.7997946081014,-1098.426892825768 -2148.769148524336,-1134.6474662473909 -2186.2945627242934,-1155.3431070927168 -2254.433769300095,-1193.4573015941605 -2304.2343846634412,-1222.6038513163885 -2358.8322538387015,-1254.5092766150274 -2458.012067756651,-1310.4702426818942 -2507.9544219786426,-1336.4219663096737 -2566.9291536576425,-1372.6379817932168 -2625.904672557596,-1408.8533542703965 -2684.8784170623594,-1439.8904297886163 -2739.9046026048222,-1473.258802453182 -2808.1932328613957,-1512.31518932982 -2859.973388442889,-1539.3845405530117 -2915.422918010248,-1569.212842448258 -2974.4028993172574,-1605.4244161671427 -3006.1485247232754,-1620.7672976472602 -3089.681627227419,-1671.6339785726268 -3140.618269974076,-1698.5285384989736 -3209.7590653594057,-1737.7519566504504 -3274.668590281109,-1776.1141047053043 -3329.839429865345,-1803.867270872284 -3376.827919699075,-1829.2950289938747 -3467.705800987688,-1884.7248960936365 -3517.7993316922684,-1910.6672115939698 -3574.9531717853015,-1946.7871326527604 -3633.9381913634447,-1977.8138673945816 -3671.475100972948,-1998.4992092309906 -3747.5393202574423,-2042.8887251337192 -3810.197298942064,-2078.4845948429006 -3867.6366179550305,-2114.0834610523884 -3930.2925196545852,-2144.586586943824 -3982.513274352936,-2179.497657810409 -4068.318017140704,-2226.0328668343695 -4097.389899170839,-2241.1987695498656 -4164.853588366585,-2282.913791642189 -4223.84563479351,-2313.93409817139 -4282.842441255745,-2350.1314073675376 -4357.926644901237,-2391.492829482895 -4416.9209755300435,-2422.511032044585 -4478.601620492132,-2459.8258480982113 -4541.269208915028,-2495.4132534958794 -4593.916778600947,-2525.9170704577227 -4645.71846874064,-2556.4210998032186 -4703.163915779588,-2586.9197465499965 -4765.834444623315,-2622.5045518341035 -4823.285790505715,-2658.0932724075255 -4872.830550473036,-2686.1809000544786 -4937.196882060428,-2722.366259185552 -4996.20288674117,-2758.5557901363427 -5058.309361640359,-2790.254497137629 -5141.879700693411,-2841.089490785062 -5173.215753543167,-2856.7652408670533 -5235.892064830374,-2891.9131331434764 -5291.229573265631,-2923.9608721992217 -5360.968012913955,-2965.313602836795 -5419.974315307828,-2996.32087372948 -5484.347973861574,-3032.4997262135244 -5533.616422613877,-3059.632356496133 -5596.29785602883,-3095.207545509337 -5653.7543491830065,-3125.6961045948774 -5716.437333260673,-3161.269902520934 -5773.9006161420675,-3196.8485296754316 -5838.421222795956,-3234.0585485887295 -5902.800491098242,-3270.232424546405 -5967.183083886482,-3308.649177031791 -6008.976305600856,-3334.059816864463 -6074.490460614019,-3373.59586405652 -6133.50527945715,-3404.595357810632 -6186.594611224565,-3435.6872590482963 -6267.640526707879,-3482.1067708668243 -6326.657629008428,-3513.1041593349146 -6348.123514184157,-3528.6120230848655 -6411.026494442583,-3563.4440311868966 -6473.93028219607,-3598.2753084344695 -6557.526322632063,-3649.088438190853 -6621.772199761114,-3683.614349634373 -6686.155645918001,-3714.601276862879 -6714.262098282948,-3735.449734960269 -6797.856126737154,-3781.167815498041 -6863.236823257899,-3817.933088205706 -6891.906747471688,-3837.054032750173" opacity="0.8" /></g></svg></div>
    </div>




Explanation about this particular situation is available at
https://avherald.com/h?article=4cbcbfb7

.. code:: python

    squawk7700.metadata.query('flight_id == "AFR1145_20190820"').iloc[0]




.. parsed-literal::

    flight_id         AFR1145_20190820
    callsign                   AFR1145
    number                      AF1145
    icao24                      3944e1
    registration                F-GRHB
    typecode                      A319
    origin                        UUEE
    landing                       ELLX
    destination                   LFPG
    diverted                      ELLX
    tweet_problem              unclear
    tweet_result              diverted
    tweet_fueldump             unknown
    avh_id                    4cbcbfb7
    avh_problem                 brakes
    avh_result                diverted
    avh_fueldump               unknown
    Name: 623, dtype: object



