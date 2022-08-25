#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z3DCollection
=================

An object to hold Z3D file information to make processing easier.


Created on Sat Apr  4 12:40:40 2020

@author: peacock
"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd
from pathlib import Path

from mth5.io.collection import Collection
from mth5.io.zen import Z3D

# =============================================================================
# Collection of Z3D Files
# =============================================================================
class Z3DCollection(Collection):
    """
    An object to deal with a collection of Z3D files. Metadata and information
    are contained with in Pandas DataFrames for easy searching.

    """

    def __init__(self, file_path=None, **kwargs):

        super().__init__(file_path=file_path, **kwargs)

    def get_calibrations(self, calibration_path):
        """
        get coil calibrations
        """
        if calibration_path is None:
            self.logger.warning("Calibration path is None")
            return {}
        if not isinstance(calibration_path, Path):
            calibration_path = Path(calibration_path)
        if not calibration_path.exists():
            self.logger.warning(
                "WARNING: could not find calibration path: "
                "{0}".format(calibration_path)
            )
            return {}
        calibration_dict = {}
        for cal_fn in calibration_path.glob("*.csv"):
            cal_num = cal_fn.stem
            calibration_dict[cal_num] = cal_fn
        return calibration_dict

    def to_dataframe(
        self, sample_rates=[256, 4096], run_name_zeros=4, calibration_path=None
    ):
        """
        Get general z3d information and put information in a dataframe

        :param z3d_fn_list: List of files Paths to z3d files
        :type z3d_fn_list: list

        :return: Dataframe of z3d information
        :rtype: Pandas.DataFrame

        :Example: ::

            >>> zc_obj = zc.Z3DCollection(r"/home/z3d_files")
            >>> z3d_fn_list = zc.get_z3d_fn_list()
            >>> z3d_df = zc.get_z3d_info(z3d_fn_list)
            >>> # write dataframe to a file to use later
            >>> z3d_df.to_csv(r"/home/z3d_files/z3d_info.csv")

        """

        cal_dict = self.get_calibrations(calibration_path)
        entries = []
        for z3d_fn in self.get_files(["z3d"]):
            z3d_obj = Z3D(z3d_fn)
            z3d_obj.read_all_info()
            if not int(z3d_obj.sample_rate) in sample_rates:
                self.logger.warning(
                    f"{z3d_obj.sample_rate} not in {sample_rates}"
                )
                return

            entry = {}
            entry["survey"] = z3d_obj.metadata.job_name
            entry["station"] = z3d_obj.station
            entry["run"] = None
            entry["start"] = z3d_obj.start.isoformat()
            entry["end"] = z3d_obj.end.isoformat()
            entry["channel_id"] = z3d_obj.channel_number
            entry["component"] = z3d_obj.component
            entry["fn"] = z3d_fn
            entry["sample_rate"] = z3d_obj.sample_rate
            entry["file_size"] = z3d_obj.file_size
            entry["n_samples"] = z3d_obj.n_samples
            entry["sequence_number"] = 0
            entry["instrument_id"] = z3d_obj.header.box_number
            if cal_dict:
                try:
                    entry["calibration_fn"] = cal_dict[z3d_obj.coil_number]
                except KeyError:
                    self.logger.warning(
                        f"Could not find {z3d_obj.coil_number}"
                    )

            entries.append(entry)
        # make pandas dataframe and set data types
        df = self._sort_df(
            self._set_df_dtypes(pd.DataFrame(entries)), run_name_zeros
        )

        return df

    def assign_run_names(self, df, zeros=3):

        # assign run names
        starts = sorted(df.start.unique())
        for block_num, start in enumerate(starts):
            sample_rate = df[df.start == start].sample_rate.unique()[0]

            df.loc[
                (df.start == start), "run"
            ] = f"sr{sample_rate:.0f}_{block_num:0{zeros}}"
            df.loc[(df.start == start), "sequence_number"] = block_num
        return df
