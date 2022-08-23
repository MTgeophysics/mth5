# -*- coding: utf-8 -*-
"""
Phoenix file collection

Created on Thu Aug  4 16:48:47 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5.utils.mth5_logger import setup_logger

# =============================================================================


class Collection:
    """
    A general collection class to keep track of files
    """

    def __init__(self, file_path=None, **kwargs):

        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        self.file_path = file_path

        self._columns = [
            "survey",
            "station",
            "run",
            "start",
            "end",
            "channel_id",
            "component",
            "fn",
            "sample_rate",
            "file_size",
            "n_samples",
            "sequence_number",
            "instrument_id",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def file_path(self):
        """
        Path object to z3d directory
        """
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        """
        :param file_path: path to z3d files
        :type file_path: string or Path object

        sets file_path as a Path object
        """
        if file_path is None:
            self._file_path = None
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        self._file_path = file_path

        if not self._file_path.exists():
            raise IOError()

    def _get_files(self, extension):
        """
        Get files with given extension

        :param extension: DESCRIPTION
        :type extension: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return list(self.file_path.rglob(f"*.{extension}"))

    def to_dataframe(self):
        """
        Get a data frame of the file summary

        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass
