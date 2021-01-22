# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:32:55 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# imports
# =============================================================================

import unittest

import numpy as np
import pandas as pd

from mth5 import timeseries
from mth5.utils.exceptions import MTTSError

from mt_metadata import timeseries as metadata

# =============================================================================
#
# =============================================================================


class TestChannelTS(unittest.TestCase):
    def setUp(self):
        self.ts = timeseries.ChannelTS("auxiliary")
        self.maxDiff = None

    def test_input_type_electric(self):
        self.ts = timeseries.ChannelTS("electric")

        electric_meta = metadata.Electric()
        self.assertDictEqual(
            self.ts.channel_metadata.to_dict(), electric_meta.to_dict()
        )

    def test_input_type_magnetic(self):
        self.ts = timeseries.ChannelTS("magnetic")

        magnetic_meta = metadata.Magnetic()
        self.assertDictEqual(
            self.ts.channel_metadata.to_dict(), magnetic_meta.to_dict()
        )

    def test_input_type_auxiliary(self):
        self.ts = timeseries.ChannelTS("auxiliary")

        auxiliary_meta = metadata.Auxiliary()
        self.assertDictEqual(
            self.ts.channel_metadata.to_dict(), auxiliary_meta.to_dict()
        )

    def test_input_type_fail(self):
        self.assertRaises(ValueError, timeseries.ChannelTS, "temperature")

    def test_intialize_with_metadata(self):
        self.ts = timeseries.ChannelTS(
            "electric", channel_metadata={"electric": {"component": "ex"}}
        )
        self.assertEqual(self.ts.channel_metadata.component, "ex")
        self.assertEqual(self.ts._ts.attrs["component"], "ex")

    def test_numpy_input(self):
        self.ts.channel_metadata.sample_rate = 1.0
        self.ts._update_xarray_metadata()

        self.ts.ts = np.random.rand(4096)
        end = self.ts.channel_metadata.time_period._start_dt + (4096 - 1)

        # check to make sure the times align
        self.assertEqual(
            self.ts._ts.coords.to_index()[0].isoformat(),
            self.ts.channel_metadata.time_period._start_dt.iso_no_tz,
        )

        self.assertEqual(
            self.ts._ts.coords.to_index()[-1].isoformat(), end.iso_no_tz
        )

        self.assertEqual(self.ts.n_samples, 4096)

    def test_df_without_index_input(self):
        self.ts.channel_metadata.sample_rate = 1.0

        self.ts._update_xarray_metadata()

        self.ts.ts = pd.DataFrame({"data": np.random.rand(4096)})
        end = self.ts.channel_metadata.time_period._start_dt + (4096 - 1)

        # check to make sure the times align
        self.assertEqual(
            self.ts._ts.coords.to_index()[0].isoformat(),
            self.ts.channel_metadata.time_period._start_dt.iso_no_tz,
        )

        self.assertEqual(
            self.ts._ts.coords.to_index()[-1].isoformat(), end.iso_no_tz
        )

        self.assertEqual(self.ts.n_samples, 4096)

    def test_df_with_index_input(self):
        n_samples = 4096
        self.ts.ts = pd.DataFrame({"data": np.random.rand(n_samples)},
                                  index=pd.date_range(start="2020-01-02T12:00:00",
                                                      periods=n_samples,
                                                      freq="244140N"))

        # check to make sure the times align
        self.assertEqual(
            self.ts._ts.coords.to_index()[0].isoformat(),
            self.ts.channel_metadata.time_period._start_dt.iso_no_tz,
        )

        self.assertEqual(self.ts.sample_rate, 4096.)

        self.assertEqual(self.ts.n_samples, n_samples)

    def test_set_component(self):
        self.ts = timeseries.ChannelTS(
            "electric", channel_metadata={"electric": {"component": "ex"}}
        )

        def set_comp(comp):
            self.ts.component = comp

        self.assertRaises(MTTSError, set_comp, "hx")
        self.assertRaises(MTTSError, set_comp, "bx")
        self.assertRaises(MTTSError, set_comp, "temperature")

    def test_change_sample_rate(self):
        self.ts.sample_rate = 16
        self.ts.start = "2020-01-01T12:00:00"
        self.ts.ts = np.arange(4096)

        self.assertEqual(self.ts.sample_rate, 16.0)

        self.ts.sample_rate = 8
        self.assertEqual(self.ts.sample_rate, 8.0)
        self.assertEqual(self.ts.n_samples, 4096)


# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
