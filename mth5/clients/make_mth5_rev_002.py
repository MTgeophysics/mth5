# -*- coding: utf-8 -*-
"""
Updated on Wed Aug  25 19:57:00 2021

@author: jpeacock + tronan
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import pandas as pd

from obspy.clients import fdsn
from obspy import UTCDateTime
from obspy import read as obsread
from obspy.core.inventory import Inventory

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment

from mth5.mth5 import MTH5
from mth5.clients.helper_functions import augment_streams
from mth5.clients.helper_functions import get_channel_from_row
from mth5.clients.helper_functions import get_trace_start_end_times
from mth5.clients.helper_functions import make_network_inventory
from mth5.clients.helper_functions import make_station_inventory
from mth5.timeseries import RunTS

# =============================================================================


class MakeMTH5:
    def __init__(self, client="IRIS", mth5_version="0.2.0"):
        """
        These column names represent a very important schema that underlies just about
        everything to do with interation with FDSN clients, if not even more general
        than that.  The order matters becuase of how we use them as arguments to sort the
        dataframe.  They may want to be imported from some metadata_model_schema.py or
        similar.

        Suggest self.client be replaced with seld.client_id.  That is a string that
        identifies the client.  Then we can re-use self.client as the object of type
        obspy.clients.fdsn.client.Client.

        Parameters
        ----------
        client
        mth5_version
        """
        self.column_names = [
            "network",
            "station",
            "location",
            "channel",
            "start",
            "end",
        ]
        self.client = client
        self.mth5_version = mth5_version

    def _validate_dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, (str, Path)):
                fn = Path(df)
                if not fn.exists():
                    raise IOError(f"File {fn} does not exist. Check path")
                df = pd.read_csv(fn)
                df = df.fillna("")
            else:
                raise ValueError(f"Input must be a pandas.Dataframe not {type(df)}")

        if df.columns.to_list() != self.column_names:
            raise ValueError(
                f"column names in file {df.columns} are not the expected {self.column_names}"
            )

        return df

    def make_mth5_from_fdsnclient(
        self, df, path=None, client=None, interact=False, inventory=None, streams=None
    ):
        """
        Make an MTH5 file from an FDSN data center

        :param df: DataFrame with columns

            - 'network'   --> FDSN Network code
            - 'station'   --> FDSN Station code
            - 'location'  --> FDSN Location code
            - 'channel'   --> FDSN Channel code
            - 'start'     --> Start time YYYY-MM-DDThh:mm:ss
            - 'end'       --> End time YYYY-MM-DDThh:mm:ss

        :type df: :class:`pandas.DataFrame`
        :param path: Path to save MTH5 file to, defaults to None
        :type path: string or :class:`pathlib.Path`, optional
        :param client: FDSN client name, defaults to "IRIS"
        :type client: string, optional
        :raises AttributeError: If the input DataFrame is not properly
        formatted an Attribute Error will be raised.
        :raises ValueError: If the values of the DataFrame are not correct a
        ValueError will be raised.
        :return: MTH5 file name
        :rtype: :class:`pathlib.Path`


        .. seealso:: https://docs.obspy.org/packages/obspy.clients.fdsn.html#id1

        .. note:: If any of the column values are blank, then any value will
        searched for.  For example if you leave 'station' blank, any station
        within the given start and end time will be returned.



        """
        if path is None:
            path = Path().cwd()
        else:
            path = Path(path)

        if client is not None:
            self.client = client

        df = self._validate_dataframe(df)

        unique_list = self.get_unique_networks_and_stations(df)
        if self.mth5_version in ["0.1.0"]:
            if len(unique_list) != 1:
                raise AttributeError("MTH5 supports one survey/network per container.")

        file_name = path.joinpath(self.make_filename(df))
        print(file_name)

        # initiate MTH5 file
        m = MTH5(file_version=self.mth5_version)
        m.open_mth5(file_name, "w")

        # read in inventory and streams
        if (inventory is None) and (streams is None):  # testing
            inventory, streams = self.get_inventory_from_df(df, self.client)
        # translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
        translator = XMLInventoryMTExperiment()
        experiment = translator.xml_to_mt(inventory)
        m.from_experiment(experiment)

        # TODO: Add survey level when structure allows.
        if self.mth5_version in ["0.1.0"]:
            for station_id in unique_list[0]["stations"]:
                # get the streams for the given station
                msstreams = streams.select(station=station_id)
                trace_start_times, trace_end_times = get_trace_start_end_times(
                    msstreams
                )
                run_list = m.get_station(station_id).groups_list
                n_times = len(trace_start_times)

                # adding logic if there are already runs filled in
                if len(run_list) == n_times:
                    for run_id, start, end in zip(
                        run_list, trace_start_times, trace_end_times
                    ):
                        # add the group first this will get the already filled in
                        # metadata to update the run_ts_obj.
                        run_group = m.stations_group.get_station(station_id).add_run(
                            run_id
                        )
                        # then get the streams an add existing metadata
                        run_stream = msstreams.slice(
                            UTCDateTime(start), UTCDateTime(end)
                        )
                        run_ts_obj = RunTS()
                        run_ts_obj.from_obspy_stream(run_stream, run_group.metadata)
                        run_group.from_runts(run_ts_obj)

                # if there is just one run
                elif len(run_list) == 1:
                    if n_times > 1:
                        for run_id, times in enumerate(
                            zip(trace_start_times, trace_end_times), 1
                        ):
                            run_group = m.stations_group.get_station(
                                station_id
                            ).add_run(f"{run_id:03}")
                            run_stream = msstreams.slice(
                                UTCDateTime(times[0]), UTCDateTime(times[1])
                            )
                            run_ts_obj = RunTS()
                            run_ts_obj.from_obspy_stream(run_stream, run_group.metadata)
                            run_group.from_runts(run_ts_obj)

                    elif n_times == 1:
                        run_group = m.stations_group.get_station(station_id).add_run(
                            run_list[0]
                        )
                        run_stream = msstreams.slice(
                            UTCDateTime(times[0]), UTCDateTime(times[1])
                        )
                        run_ts_obj = RunTS()
                        run_ts_obj.from_obspy_stream(run_stream, run_group.metadata)
                        run_group.from_runts(run_ts_obj)
                else:
                    print(
                        "Could be the case that the run_list indicates several "
                        "runs, but not all of them were explicitly called in the "
                        "dataframe"
                    )
                    raise ValueError("Cannot add Run for some reason.")

        # Version 0.2.0 has the ability to store multiple surveys
        elif self.mth5_version in ["0.2.0"]:
            for survey_dict in unique_list:
                survey_id = survey_dict["network"]
                survey_group = m.get_survey(survey_id)
                for station_id in survey_dict["stations"]:
                    # get the streams for the given station
                    msstreams = streams.select(station=station_id)
                    trace_start_times, trace_end_times = get_trace_start_end_times(
                        msstreams
                    )
                    run_list = m.get_station(station_id, survey_id).groups_list
                    n_times = len(trace_start_times)

                    # adding logic if there are already runs filled in
                    if len(run_list) == n_times:
                        for run_id, start, end in zip(
                            run_list, trace_start_times, trace_end_times
                        ):
                            # add the group first this will get the already filled in
                            # metadata to update the run_ts_obj.
                            run_group = survey_group.stations_group.get_station(
                                station_id
                            ).add_run(run_id)
                            # then get the streams an add existing metadata
                            run_stream = msstreams.slice(
                                UTCDateTime(start), UTCDateTime(end)
                            )
                            run_ts_obj = RunTS()
                            run_ts_obj.from_obspy_stream(run_stream, run_group.metadata)
                            run_group.from_runts(run_ts_obj)

                    # if there is just one run
                    elif len(run_list) == 1:
                        if n_times > 1:
                            for run_id, times in enumerate(
                                zip(trace_start_times, trace_end_times), 1
                            ):
                                run_group = survey_group.stations_group.get_station(
                                    station_id
                                ).add_run(f"{run_id:03}")
                                run_stream = msstreams.slice(
                                    UTCDateTime(times[0]), UTCDateTime(times[1])
                                )
                                run_ts_obj = RunTS()
                                run_ts_obj.from_obspy_stream(
                                    run_stream, run_group.metadata
                                )
                                run_group.from_runts(run_ts_obj)

                        elif n_times == 1:
                            run_group = survey_group.stations_group.get_station(
                                station_id
                            ).add_run(run_list[0])
                            run_stream = msstreams.slice(
                                UTCDateTime(times[0]), UTCDateTime(times[1])
                            )
                            run_ts_obj = RunTS()
                            run_ts_obj.from_obspy_stream(run_stream, run_group.metadata)
                            run_group.from_runts(run_ts_obj)
                    else:
                        raise ValueError("Cannot add Run for some reason.")

        if not interact:
            m.close_mth5()

            return file_name
        if interact:
            return m

    def get_inventory_from_df(self, df, client=None, data=True):
        """
        Get an :class:`obspy.Inventory` object from a
        :class:`pandas.DataFrame`

        :param df: DataFrame with columns

            - 'network'   --> FDSN Network code
            - 'station'   --> FDSN Station code
            - 'location'  --> FDSN Location code
            - 'channel'   --> FDSN Channel code
            - 'start'     --> Start time YYYY-MM-DDThh:mm:ss
            - 'end'       --> End time YYYY-MM-DDThh:mm:ss

        :type df: :class:`pandas.DataFrame`
        this dataframe gets sorted by the column names in the above
            listed hierarhical order.
        :param client: FDSN client
        :type client: string
        :param data: True if you want data False if you want just metadata,
        defaults to True
        :type data: boolean, optional
        :return: An inventory of metadata requested and data
        :rtype: :class:`obspy.Inventory` and :class:`obspy.Stream`

        .. seealso:: https://docs.obspy.org/packages/obspy.clients.fdsn.html#id1

        .. note:: If any of the column values are blank, then any value will
        searched for.  For example if you leave 'station' blank, any station
        within the given start and end time will be returned.

        """
        if client is not None:
            self.client = client

        df = self._validate_dataframe(df)

        # get the metadata from an obspy client
        client = fdsn.Client(self.client, _discover_services=False)

        # Initialize streams and inventory to capture info
        streams = obsread()
        streams.clear()

        inv = Inventory(networks=[], source="MTH5")

        # sort the values to be logically ordered
        df.sort_values(self.column_names[:-1])

        network_inventory = make_network_inventory(df, client)
        station_inventory = make_station_inventory(df, client)
        for row in df.itertuples():
            active_network = network_inventory[row.network][row.start]
            active_station = station_inventory[row.station][row.network][row.start]
            cha_inv = get_channel_from_row(client, row)
            returned_chan = cha_inv.networks[0].stations[0].channels[0]
            active_station.channels.append(returned_chan)

            if data:
                streams = augment_streams(row, streams, client)

            active_network.stations.append(active_station)

        for network_id in network_inventory.keys():
            for start in network_inventory[network_id]:
                inv.networks.append(network_inventory[network_id][start])

        return inv, streams

    def get_df_from_inventory(self, inventory):
        """
        Create an data frame from an inventory object

        :param inventory: inventory object
        :type inventory: :class:`obspy.Inventory`
        :return: dataframe in proper format
        :rtype: :class:`pandas.DataFrame`

        """

        rows = []
        for network in inventory.networks:
            for station in network.stations:
                for channel in station.channels:
                    entry = (
                        network.code,
                        station.code,
                        channel.location_code,
                        channel.code,
                        channel.start_date,
                        channel.end_date,
                    )
                    rows.append(entry)

        return pd.DataFrame(rows, columns=self.column_names)

    def get_unique_networks_and_stations(self, df):
        """
        Get unique lists of networks, stations, locations, and channels from
        a given data frame.

        [{'network': FDSN code, "stations": [list of stations for network]}]

        :param df: request data frame
        :type df: :class:`pandas.DataFrame`
        :return: list of network dictionaries with
        [{'network': FDSN code, "stations": [list of stations for network]}]
        :rtype: list

        """
        unique_list = []
        net_list = df["network"].unique()
        for network in net_list:
            network_dict = {
                "network": network,
                "stations": df[df.network == network].station.unique().tolist(),
            }
            unique_list.append(network_dict)

        return unique_list

    def make_filename(self, df):
        """
        Make a filename from a data frame that is networks and stations

        :param df: request data frame
        :type df: :class:`pandas.DataFrame`
        :return: file name as network_01+stations_network_02+stations.h5
        :rtype: string

        """

        unique_list = self.get_unique_networks_and_stations(df)

        return (
            "_".join([f"{d['network']}_{'_'.join(d['stations'])}" for d in unique_list])
            + ".h5"
        )

    def get_fdsn_channel_map(self):
        FDSN_CHANNEL_MAP = {}

        FDSN_CHANNEL_MAP["BQ2"] = "BQ1"
        FDSN_CHANNEL_MAP["BQ3"] = "BQ2"
        FDSN_CHANNEL_MAP["BQN"] = "BQ1"
        FDSN_CHANNEL_MAP["BQE"] = "BQ2"
        FDSN_CHANNEL_MAP["BQZ"] = "BQ3"
        FDSN_CHANNEL_MAP["BT1"] = "BF1"
        FDSN_CHANNEL_MAP["BT2"] = "BF2"
        FDSN_CHANNEL_MAP["BT3"] = "BF3"
        FDSN_CHANNEL_MAP["LQ2"] = "LQ1"
        FDSN_CHANNEL_MAP["LQ3"] = "LQ2"
        FDSN_CHANNEL_MAP["LT1"] = "LF1"
        FDSN_CHANNEL_MAP["LT2"] = "LF2"
        FDSN_CHANNEL_MAP["LT3"] = "LF3"
        FDSN_CHANNEL_MAP["LFE"] = "LF1"
        FDSN_CHANNEL_MAP["LFN"] = "LF2"
        FDSN_CHANNEL_MAP["LFZ"] = "LF3"
        FDSN_CHANNEL_MAP["LQE"] = "LQ1"
        FDSN_CHANNEL_MAP["LQN"] = "LQ2"
        return FDSN_CHANNEL_MAP
