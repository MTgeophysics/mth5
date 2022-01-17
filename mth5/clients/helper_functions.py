import pandas as pd
from obspy import UTCDateTime

from typing import Dict, Any
import hashlib
import json


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary.
    source:
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    There could be any values, such as lists, floats and other types.
    Note the two following assumptions:
    We will assume any value is serialisable as a string.
    We assume the keys are strings which allows us to order them.
    """

    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_channel_from_row(client, row):
    """
    A helper function that uses obspy client to get a station inventory.  Although
    this could theoretically return muliple stations, as the name of this function
    suggests, it is intended to return only one single channel.

    Parameters
    ----------
    client: obspy.clients.fdsn.client.Clien
    row: pandas.core.frame.Pandas
        A row of a dataframe

    Returns
    -------

    """
    channel_inventory = client.get_stations(
        row.start,
        row.end,
        network=row.network,
        station=row.station,
        loc=row.location,
        channel=row.channel,
        level="response",
    )
    return channel_inventory


def get_network_uuid(network):
    print("this fails because ")
    print("TypeError: Object of type Comment is not JSON serializable")
    try:
        uuid = dict_hash(network.__dict__)
    except TypeError:
        uuid = None
    return uuid


def make_network_inventory(df, client):
    """
    20220109: This can be made more robust.  The issue here is that the same network
    can be returned multiple times, if the dataframe calls data from the same network
    with mutiple start times.  The way to simplify the situationis to return only
    the unique netowrks.  This however is going to require a re-keying of the
    dictionary.  See comments in issue#76

    Parameters
    ----------
    df
    client

    Returns
    -------
    Dictionary keyed first by network_id, and then by starttime of desired streams.
    """
    print("Making Network Inventory")
    df["netork_uuid"] = ""
    networks = {}
    for network_id in df.network.unique():
        networks[network_id] = {}
    for network_id in networks.keys():
        sub_df = df[df.network == network_id]
        for row in sub_df.itertuples():
            print(row)
            if row.start not in networks[network_id].keys():
                print("client.get_stations is super fricken slow!")
                net_inv = client.get_stations(
                    row.start, row.end, network=row.network, level="network"
                )
                networks[network_id][row.start] = net_inv.networks[0]
                # HERE is where you tranlate network into a UUID
                # that UUID is appended to
                # network = net_inv.networks[0]
                # uuid = get_network_uuid(network)
                # if uuid is None:
                #     uuid = f"{network_id}_{row.start}"
                # sub_df["network_uuid"] = uuid
    return networks


def make_station_inventory(df, client):
    """

    Parameters
    ----------
    df
    client

    Returns
    -------
    Dictionary keyed first by station_id, then by network_id, and then by
    starttime of desired streams.
    """
    print("Making Station Inventory")
    stations = {}
    for station_id in df.station.unique():
        # print(f"station_id = {station_id}")
        stations[station_id] = {}

    for row in df.itertuples():
        # make sure that there is a subdict for the active network
        if stations[row.station] == {}:
            stations[row.station][row.network] = {}

    for station_id in stations.keys():
        sub_df = df[df.station == station_id]
        for row in sub_df.itertuples():
            if row.start not in stations[station_id][row.network].keys():
                sta_inv = client.get_stations(
                    row.start,
                    row.end,
                    network=row.network,
                    station=row.station,
                    level="station",
                )
                station = sta_inv.networks[0].stations[0]
                stations[station_id][row.network][row.start] = station
    return stations


def augment_streams(row, streams, client):
    """

    Returns
    -------

    """
    streams = (
        client.get_waveforms(
            row.network,
            row.station,
            row.location,
            row.channel,
            UTCDateTime(row.start),
            UTCDateTime(row.end),
        )
        + streams
    )
    return streams


def channel_summary_to_make_mth5(df, network="ZU"):
    """
    Context is say you have a station_xml that has come from somewhere and you want
    to make an mth5 from it, with all the relevant data.  Then you should use
    make_mth5.  But make_mth5 wants a df with a
    particular schema (which should be written down somewhere!)

    This returns a dataframe with the schema that MakeMTH5() expects.

    TODO: This method could be an option for output format of mth5.channel_summary()

    Parameters
    ----------
    df: the output from mth5_obj.channel_summary

    Returns
    -------

    """
    ch_map = {"ex": "LQN", "ey": "LQE", "hx": "LFN", "hy": "LFE", "hz": "LFZ"}
    number_of_runs = len(df["run"].unique())
    num_rows = 5 * number_of_runs
    networks = num_rows * [network]
    stations = num_rows * [None]
    locations = num_rows * [""]
    channels = num_rows * [None]
    starts = num_rows * [None]
    ends = num_rows * [None]

    i = 0
    for group_id, group_df in df.groupby("run"):
        print(group_id, group_df.start.unique(), group_df.end.unique())
        for index, row in group_df.iterrows():
            stations[i] = row.station
            channels[i] = ch_map[row.component]
            starts[i] = row.start
            ends[i] = row.end
            print("OK")
            i += 1

    out_dict = {
        "network": networks,
        "station": stations,
        "location": locations,
        "channel": channels,
        "start": starts,
        "end": ends,
    }
    out_df = pd.DataFrame(data=out_dict)
    return out_df


def get_trace_start_end_times(msstreams):
    """
    usage: trace_start_times, trace_end_times = get_trace_start_end_times(msstreams)
    Parameters
    ----------
    msstreams

    Returns
    -------

    """
    trace_start_times = sorted(
        list(set([tr.stats.starttime.isoformat() for tr in msstreams]))
    )
    trace_end_times = sorted(
        list(set([tr.stats.endtime.isoformat() for tr in msstreams]))
    )
    if len(trace_start_times) != len(trace_end_times):
        raise ValueError(
            f"Do not have the same number of start {len(trace_start_times)}"
            f" and end times {len(trace_end_times)} from streams"
        )
    return trace_start_times, trace_end_times
