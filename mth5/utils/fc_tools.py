"""
    Work In progress

    This module is concerned with working with Fourier coefficient data

    Tools include prototypes for
    - extacting portions of an FC Run Time Series
    - merging multiple stations runs together into an xarray
    - relabelling channels to avoid namespace clashes for multistation data

"""

from dataclasses import dataclass
from loguru import logger
from typing import Optional, Tuple , Union
import pandas as pd
import xarray


@dataclass
class FCRunChunk():
    """

    This class formalizes the required metadata to specify a chunk of a timeseries of Fourier coefficients.

    This may move to mt_metadata -- for now just use a dataclass as a prototype.
    """
    survey_id: str = "none"
    station_id: str = ""
    run_id: str = ""
    decimation_level_id: str = "0"
    start: str = ""
    end: str = ""
    channels: Tuple[str] = ()

    @property
    def start_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)

    @property
    def duration(self) -> pd.Timestamp:
        return self.end_timestamp - self.start_timestamp


@dataclass
class MultivariateLabelScheme():
    """
    Class to store information about how a multivariate (MV) dataset will be lablelled.

    Has a scheme to handle the how channels will be named.

    This is just a place holder to manage possible future complexity.

    It seemed like a good idea to formalize the fact that we take, by default
    f"{station}_{component}" as the MV channel label.
    It also seemed like a good idea to record what the join character is.
    In the event that we wind up with station names that have underscores in them, then we could,
    for example, set the join character to "__".

     TODO: Consider rename default to ("station", "data_var")

    Parameters
    ----------
    label_elements: tuple
        This is meant to tell what information is being concatenated into an MV channel label.
        Note that concatenation is used is an assumtion (that could change one day).
    join_chan: str
        The string that is used to join the label elements.

    """
    label_elements: tuple = "station", "component",
    join_char: str = "_"

    @property
    def id(self):
       return self.join(self.label_elements)

    def join(self, elements: tuple) -> str:
        """
        Join the label elements to a string

        Parameters
        ----------
        elements: tuple
            expected to be the label elements
            Default (station, component)

        Returns
        -------
        str
            The name of the MV channel.
        """
        return self.join_char.join(elements)

    def split(self, mv_channel_name):
        """
        Splits a MV channel name and returns a dict of strings, keyed by self.label_elements
        - Basically the reverse of Join

        Parameters
        ----------
        mv_channel_name

        Returns
        -------

        """
        splitted = mv_channel_name.split(self.join_char)
        if len(splitted) != len(self.label_elements):
            msg = f"Incompatible map {splitted} and {self.label_elements}"
            logger.error(msg)
            msg = f"cannot map {len(splitted)} to {len(self.label_elements)}"
            raise ValueError(msg)
        output = dict(zip(self.label_elements, splitted))
        return output


class MultivariateDataset():
    """
        Here is a container for a multivariate dataset.
        The xarray is the main underlying item, but it will be useful to have functions that, \
        for example return a list of the associated stations, or that return a list of cahnnels
        that are associated with a station, etc.

    """
    def __init__(
        self,
        xrds: xarray.Dataset,
        label_scheme: Optional[Union[MultivariateLabelScheme, None]] = None,
    ):
        self._xrds = xrds
        self._label_scheme = label_scheme

        self._channels = None
        self._stations = None
        self._station_channels = None

    @property
    def label_scheme(self):
        if self._label_scheme is None:
            msg = f"No label scheme found for {self.__class__} -- setting to default"
            logger.warning(msg)
            self._label_scheme = MultivariateLabelScheme()
        return self._label_scheme

    @property
    def dataset(self):
        return self._xrds

    @property
    def dataarray(self):
        return self._xrds.to_array()

    @property
    def channels(self) -> list:
        """
        returns a list of channels in the dataarray
        """
        if self._channels is None:
            self._channels = list(self.dataarray.coords["variable"].values)
        return self._channels

    @property
    def num_channels(self) -> int:
        """returns a count of the total number of channels in the dataset"""
        return len(self.channels)

    @property
    def stations(self):
        """
        Parses the channel names, extracts the station names

        return a unique list of stations preserving order.
        """
        if self._stations is None:
            if self.label_scheme.id == "station_component":
                tmp = [self.label_scheme.split(x)["station"] for x in self.channels]
                # tmp = [x.split("_")[0] for x in self.channels]
                stations = list(dict.fromkeys(tmp))  # order preserving unique values
                self._stations = stations
            else:
                msg = f"No rule for parsting station names from label scheme {self.label_scheme.id}"
                raise NotImplementedError(msg)

        return self._stations

    def station_channels(self, station):
        """returns a dict with the noise model per station"""
        if self._station_channels is None:
            station_channels = {}
            for station_id in self.stations:
                station_channels[station_id] = [
                    x for x in self.channels if station_id == x.split("_")[0]
                ]
            self._station_channels = station_channels

        return self._station_channels[station]



def make_multistation_spectrogram(
    m,
    fc_run_chunks,
    label_scheme = MultivariateLabelScheme(),
    rtype: Optional[Union[str, None]] = None
) -> Union[xarray.Dataset, MultivariateDataset]:
    """
    see notes in mth5 issue #209.  Takes a list of FCRunChunks and returns the largest contiguous
    block of multichannel FC data available.

    |----------Station 1 ------------|
            |----------Station 2 ------------|
    |--------------------Station 3 ----------------------|


            |-------RETURNED------|

    Handle additional runs in a separate call to this function and then concatenate time series afterwards.

    Input must specify N (station-run-start-end-channel_list) tuples.
    If channel_list is not provided, get all channels.
    If start-end are not provided, read the whole run -- warn if runs are not all synchronous, and
    truncate all to max(starts), min(ends) after the start and end times are sorted out.

    Station IDs must be unique.


    Parameters
    ----------
    m: mth5.mth5.MTH5
        The mth5 object to get the FCs from.
    fc_run_chunks: iterable
        Each element of this describes the run chunk to load.

    Returns
    -------
    output: xarray.core.dataset.Dataset'
    """
    for i_fcrc, fcrc in enumerate(fc_run_chunks):
        station_obj = m.get_station(fcrc.station_id, fcrc.survey_id)
        station_fc_group = station_obj.fourier_coefficients_group
        logger.info(f"Available FC Groups for station {fcrc.station_id}: {station_fc_group.groups_list}")
        run_fc_group = station_obj.fourier_coefficients_group.get_fc_group(fcrc.run_id)
        # print(run_fc_group)
        fc_dec_level = run_fc_group.get_decimation_level(fcrc.decimation_level_id)
        if fcrc.channels:
            channels = list(fcrc.channels)
        else:
            channels = None

        fc_dec_level_xrds = fc_dec_level.to_xarray(channels=channels)
        # could create name mapper dict from run_fc_group.channel_summary here if we wanted to.

        if fcrc.start:
            # TODO: Push slicing into the to_xarray() command so we only access what we need
            cond = fc_dec_level_xrds.time >= fcrc.start_timestamp
            msg = (
                f"trimming  {sum(~cond.data)} samples to {fcrc.start} "
            )
            logger.info(msg)
            fc_dec_level_xrds = fc_dec_level_xrds.where(cond)
            fc_dec_level_xrds = fc_dec_level_xrds.dropna(dim="time")

        if fcrc.end:
            # TODO: Push slicing into the to_xarray() command so we only access what we need
            cond = fc_dec_level_xrds.time <= fcrc.end_timestamp
            msg = (
                f"trimming  {sum(~cond.data)} samples to {fcrc.end} "
            )
            logger.info(msg)
            fc_dec_level_xrds = fc_dec_level_xrds.where(cond)
            fc_dec_level_xrds = fc_dec_level_xrds.dropna(dim="time")

        if label_scheme.id == 'station_component':
            name_dict = {f"{x}": label_scheme.join((fcrc.station_id, x)) for x in fc_dec_level_xrds.data_vars}
        else:
            msg = f"Label Scheme elements {label_scheme.id} not implemented"
            raise NotImplementedError(msg)

        # qq = label_scheme.split(name_dict["ex"])  # test during dev -- To be deleted.
        if i_fcrc == 0:
            xrds = fc_dec_level_xrds.rename_vars(name_dict=name_dict)
        else:
            fc_dec_level_xrds = fc_dec_level_xrds.rename_vars(name_dict=name_dict)
            xrds = xrds.merge(fc_dec_level_xrds)

    # Check that no nan came about as a result of the merge
    if bool(xrds.to_array().isnull().any()):
        msg = "Nan detected in multistation spectrogram"
        logger.warning(msg)

    if rtype == "xrds":
        output = xrds
    else:
        output = MultivariateDataset(xrds=xrds, label_scheme=label_scheme)

    return output
