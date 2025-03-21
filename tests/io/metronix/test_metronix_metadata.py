# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:47:59 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path
import json
from collections import OrderedDict
import numpy as np

from mth5.io.metronix import MetronixFileNameMetadata, MetronixChannelJSON

# =============================================================================


class TestMetronixFileNameMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.obj = MetronixFileNameMetadata()

        self.fn_list = [
            {
                "fn": Path(r"084_ADU-07e_C000_TEx_2048Hz.json"),
                "system_number": "084",
                "system_name": "ADU-07e",
                "channel_number": 0,
                "component": "ex",
                "sample_rate": 2048,
                "file_type": "metadata",
            },
            {
                "fn": Path(r"084_ADU-07e_C001_THx_512Hz.json"),
                "system_number": "084",
                "system_name": "ADU-07e",
                "channel_number": 1,
                "component": "hx",
                "sample_rate": 512,
                "file_type": "metadata",
            },
            {
                "fn": Path(r"084_ADU-07e_C002_TEy_128Hz.json"),
                "system_number": "084",
                "system_name": "ADU-07e",
                "channel_number": 2,
                "component": "ey",
                "sample_rate": 128,
                "file_type": "metadata",
            },
            {
                "fn": Path(r"084_ADU-07e_C003_THy_32Hz.json"),
                "system_number": "084",
                "system_name": "ADU-07e",
                "channel_number": 3,
                "component": "hy",
                "sample_rate": 32,
                "file_type": "metadata",
            },
            {
                "fn": Path(r"084_ADU-07e_C004_THz_8Hz.json"),
                "system_number": "084",
                "system_name": "ADU-07e",
                "channel_number": 4,
                "component": "hz",
                "sample_rate": 8,
                "file_type": "metadata",
            },
            {
                "fn": Path(r"084_ADU-07e_C005_TEx_2Hz.json"),
                "system_number": "084",
                "system_name": "ADU-07e",
                "channel_number": 5,
                "component": "ex",
                "sample_rate": 2,
                "file_type": "metadata",
            },
            {
                "fn": Path(r"084_ADU-07e_C006_TEy_2s.json"),
                "system_number": "084",
                "system_name": "ADU-07e",
                "channel_number": 6,
                "component": "ey",
                "sample_rate": 1 / 2,
                "file_type": "metadata",
            },
            {
                "fn": Path(r"084_ADU-07e_C007_THx_8s.atss"),
                "system_number": "084",
                "system_name": "ADU-07e",
                "channel_number": 7,
                "component": "hx",
                "sample_rate": 1 / 8,
                "file_type": "timeseries",
            },
        ]

    def row_test(self, row):
        self.obj.fn = row["fn"]

    def test_sample_rate(self):
        for row in self.fn_list:
            self.obj.fn = row["fn"]
            for key, value in row.items():
                with self.subTest(f"{row['fn']}_{key}"):
                    self.assertEqual(getattr(self.obj, key), value)


class TestMetronixJSONMagnetic(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.magnetic_dict = {
            "datetime": "2009-08-20T13:22:00",
            "latitude": 39.026196666666664,
            "longitude": 29.123953333333333,
            "elevation": 1088.31,
            "angle": 0.0,
            "tilt": 0.0,
            "resistance": 684052.0,
            "units": "mV",
            "filter": "ADB-LF,LF-RF-4",
            "source": "",
            "sensor_calibration": {
                "sensor": "MFS-06",
                "serial": 26,
                "chopper": 1,
                "units_frequency": "Hz",
                "units_amplitude": "mV/nT",
                "units_phase": "degrees",
                "datetime": "2006-12-01T11:23:02",
                "Operator": "",
                "f": [
                    1e-05,
                    1.2589000000000001e-05,
                    1.5849e-05,
                    1.9952e-05,
                    2.5119e-05,
                    3.1623000000000005e-05,
                    3.981e-05,
                    5.0118000000000004e-05,
                    6.3095e-05,
                    7.943e-05,
                    0.0001,
                    0.00012589,
                    0.00015849,
                    0.00019952000000000001,
                    0.00025119,
                    0.00031623,
                    0.0003981,
                    0.00050118,
                    0.00063095,
                    0.0007943000000000001,
                    0.001,
                    0.0012589,
                    0.0015849000000000002,
                    0.0019952,
                    0.0025119,
                    0.0031623000000000003,
                    0.003981,
                    0.0050118,
                    0.0063095,
                    0.007943,
                    0.01,
                    0.012589000000000001,
                    0.015849000000000002,
                    0.019951999999999998,
                    0.025119000000000002,
                    0.031623000000000005,
                    0.03981,
                    0.050118,
                    0.063095,
                    0.07943,
                    0.1,
                    0.12589,
                    0.15849,
                    0.19953,
                    0.25119,
                    0.31623,
                    0.39811,
                    0.50119,
                    0.63095,
                    0.7943,
                    1.0,
                    1.2589,
                    1.5849,
                    1.9952,
                    2.5119,
                    3.1623,
                    3.981,
                    5.0118,
                    6.3095,
                    7.943,
                    10.0,
                    12.589,
                    15.849,
                    19.952,
                    25.119,
                    31.622,
                    39.81,
                    50.118,
                    63.095,
                    79.43,
                    100.0,
                    125.89,
                    158.49,
                    199.52,
                    251.18,
                    316.22,
                    398.1,
                    501.18,
                    630.94,
                    794.3,
                    1000.0,
                    1258.9,
                    1584.9,
                    1995.2,
                    2511.8,
                    3162.2,
                    3981.0,
                    5011.7,
                    6309.4,
                    7943.0,
                    9999.5,
                    10000.0,
                ],
                "a": [
                    0.0019999999999937507,
                    0.0025177999999875305,
                    0.003169799999975118,
                    0.003990399999950359,
                    0.005023799999900942,
                    0.006324599999802356,
                    0.007961999999605673,
                    0.010023599999213205,
                    0.012618999998430126,
                    0.015885999996867916,
                    0.019999999993749996,
                    0.025177999987530366,
                    0.031697999975117984,
                    0.03990399995035912,
                    0.050237999900942555,
                    0.06324599980235342,
                    0.07961999960567287,
                    0.10023599921320551,
                    0.1261899984301254,
                    0.15885999686791308,
                    0.1999999937499987,
                    0.2517799875303634,
                    0.3169799751179777,
                    0.3990399503591332,
                    0.5023799009425741,
                    0.6324598023534992,
                    0.796199605673159,
                    1.002359213206409,
                    1.2618984301283263,
                    1.588596867922261,
                    1.9999937500276823,
                    2.517787530455084,
                    3.169775118267761,
                    3.990350360050149,
                    5.0237009454743236,
                    6.324402362670752,
                    7.961605702157798,
                    10.022813298108742,
                    12.617430418287983,
                    15.882868838977918,
                    20.010376463273992,
                    25.16799310918104,
                    31.712516917010664,
                    39.8980188,
                    50.2405119,
                    63.1068588,
                    79.1203814,
                    99.68167910000001,
                    124.92179050000001,
                    155.881375,
                    193.79,
                    241.38148599999997,
                    297.517428,
                    360.03384,
                    429.38418599999994,
                    502.8373230000001,
                    575.0156400000001,
                    635.796948,
                    688.87121,
                    734.25092,
                    766.36,
                    785.780202,
                    801.1669499999999,
                    810.35048,
                    816.9452370000001,
                    822.0455119999999,
                    825.89826,
                    770.714604,
                    826.0397399999999,
                    827.81946,
                    825.9499999999999,
                    828.0288859999999,
                    830.17062,
                    828.8659359999999,
                    829.044708,
                    826.75719,
                    822.7532699999999,
                    822.6368520000001,
                    821.862444,
                    820.03532,
                    818.2,
                    815.477653,
                    813.814452,
                    840.637616,
                    784.636084,
                    749.12518,
                    698.46645,
                    596.542651,
                    528.601532,
                    551.911412,
                    517.094144,
                    517.0699999999999,
                ],
                "p": [
                    89.99985667036422,
                    89.99981956232152,
                    89.99977283686026,
                    89.9997140287107,
                    89.9996399702879,
                    89.9995467486928,
                    89.99942940471999,
                    89.99928166053145,
                    89.99909566166313,
                    89.99886153270316,
                    89.99856670364251,
                    89.99819562321578,
                    89.99772836860375,
                    89.99714028710932,
                    89.99639970288361,
                    89.99546748693723,
                    89.9942940472185,
                    89.99281660535163,
                    89.99095661670552,
                    89.98861532717969,
                    89.98566703672059,
                    89.98195623274725,
                    89.97728368721353,
                    89.9714028734397,
                    89.9639970335185,
                    89.95467487871475,
                    89.94294049082438,
                    89.92816609070726,
                    89.90956624126156,
                    89.88615341984706,
                    89.8566706626361,
                    89.81956291689636,
                    89.77283804827267,
                    89.7140310808364,
                    89.63997501742048,
                    89.5467581293517,
                    89.42942354657666,
                    89.28169809452952,
                    89.09573660784051,
                    88.8616822137415,
                    88.56650097293976,
                    88.19160912138915,
                    87.72027775680306,
                    87.177,
                    86.459,
                    85.487,
                    84.447,
                    82.992,
                    81.233,
                    79.055,
                    76.345,
                    72.954,
                    69.03,
                    64.201,
                    58.69,
                    52.571,
                    46.212,
                    39.399,
                    33.074,
                    27.567,
                    22.507,
                    18.198,
                    14.529,
                    11.635,
                    9.0507,
                    7.133,
                    5.5026,
                    3.6917,
                    3.2486,
                    2.2856,
                    1.3903,
                    0.8116,
                    0.21197,
                    -0.5109,
                    -1.217,
                    -2.3402,
                    -2.8624,
                    -3.7097,
                    -4.8118,
                    -6.1635,
                    -7.8127,
                    -9.8044,
                    -12.075,
                    -14.807,
                    -23.519,
                    -27.61,
                    -34.118,
                    -40.321,
                    -36.267,
                    -40.999,
                    -51.017,
                    -51.019,
                ],
            },
        }
        self.fn = Path().cwd().joinpath("084_ADU-07e_C002_THx_128Hz.json")
        with open(self.fn, "w") as fid:
            json.dump(self.magnetic_dict, fid)

        self.magnetic = MetronixChannelJSON(self.fn)

        self.expected_magnetic_metadata = OrderedDict(
            [
                ("channel_number", 2),
                ("component", "hx"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [True, True, True]),
                ("filter.name", ["adb-lf", "lf-rf-4", "mfs-06_chopper_1"]),
                ("location.elevation", 1088.31),
                ("location.latitude", 39.026196666666664),
                ("location.longitude", 29.123953333333333),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 128.0),
                ("sensor.id", "26"),
                ("sensor.manufacturer", "Metronix Geophysics"),
                ("sensor.type", "induction coil"),
                ("sensor.model", "MFS-06"),
                ("time_period.end", "2009-08-20T13:22:04.112305+00:00"),
                ("time_period.start", "2009-08-20T13:22:00+00:00"),
                ("type", "magnetic"),
                ("units", "mV"),
            ]
        )

        self.expected_fap = OrderedDict(
            [
                (
                    "amplitudes",
                    np.array(self.magnetic_dict["sensor_calibration"]["a"]),
                ),
                ("calibration_date", "2006-12-01"),
                (
                    "frequencies",
                    np.array(self.magnetic_dict["sensor_calibration"]["f"]),
                ),
                ("gain", 1.0),
                ("instrument_type", None),
                ("name", "mfs-06_chopper_1"),
                (
                    "phases",
                    np.deg2rad(
                        np.array(self.magnetic_dict["sensor_calibration"]["p"])
                    ),
                ),
                ("type", "frequency response table"),
                ("units_out", "mV"),
                ("units_in", "nT"),
            ]
        )

    def test_fn(self):
        self.assertEqual(self.fn, self.magnetic.fn)

    def test_system_number(self):
        self.assertEqual("084", self.magnetic.system_number)

    def test_system_name(self):
        self.assertEqual("ADU-07e", self.magnetic.system_name)

    def test_channel_number(self):
        self.assertEqual(2, self.magnetic.channel_number)

    def test_component(self):
        self.assertEqual("hx", self.magnetic.component)

    def test_sample_rate(self):
        self.assertEqual(128.0, self.magnetic.sample_rate)

    def test_file_type(self):
        self.assertEqual("metadata", self.magnetic.file_type)

    def test_has_metadata(self):
        self.assertTrue(self.magnetic._has_metadata())

    def test_channel_metadata(self):
        magnetic_metadata = self.magnetic.get_channel_metadata()
        self.assertEqual(self.expected_magnetic_metadata, magnetic_metadata)

    def test_get_sensor_response_filter(self):
        fap = self.magnetic.get_sensor_response_filter()
        self.assertEqual(self.expected_fap, fap)

    @classmethod
    def tearDownClass(self):
        self.fn.unlink()


class TestMetronixJSONElectric(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.electric_dict = {
            "datetime": "2009-08-20T13:22:00",
            "latitude": 39.026196666666664,
            "longitude": 29.123953333333333,
            "elevation": 1088.31,
            "angle": 0.0,
            "tilt": 0.0,
            "resistance": 572.3670043945313,
            "units": "mV/km",
            "filter": "ADB-LF,LF-RF-4",
            "source": "",
            "sensor_calibration": {
                "sensor": "EFP-06",
                "serial": 0,
                "chopper": 1,
                "units_frequency": "Hz",
                "units_amplitude": "mV",
                "units_phase": "degrees",
                "datetime": "1970-01-01T00:00:00",
                "Operator": "",
                "f": [],
                "a": [],
                "p": [],
            },
        }

        self.fn = Path().cwd().joinpath("084_ADU-07e_C000_TEx_128Hz.json")
        with open(self.fn, "w") as fid:
            json.dump(self.electric_dict, fid)

        self.expected_electric_metadata = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "ex"),
                ("contact_resistance.start", 572.3670043945312),
                ("data_quality.rating.value", 0),
                ("dipole_length", None),
                ("filter.applied", [True, True]),
                ("filter.name", ["adb-lf", "lf-rf-4"]),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("negative.elevation", 0.0),
                ("negative.id", None),
                ("negative.latitude", 0.0),
                ("negative.longitude", 0.0),
                ("negative.manufacturer", None),
                ("negative.type", None),
                ("positive.elevation", 1088.31),
                ("positive.id", None),
                ("positive.latitude", 39.026196666666664),
                ("positive.longitude", 29.123953333333333),
                ("positive.manufacturer", None),
                ("positive.type", None),
                ("sample_rate", 128.0),
                ("time_period.end", "2009-08-20T13:22:00.456055+00:00"),
                ("time_period.start", "2009-08-20T13:22:00+00:00"),
                ("type", "electric"),
                ("units", "mV/km"),
            ]
        )

        self.electric = MetronixChannelJSON(self.fn)

    def test_fn(self):
        self.assertEqual(self.fn, self.electric.fn)

    def test_system_number(self):
        self.assertEqual("084", self.electric.system_number)

    def test_system_name(self):
        self.assertEqual("ADU-07e", self.electric.system_name)

    def test_channel_number(self):
        self.assertEqual(0, self.electric.channel_number)

    def test_component(self):
        self.assertEqual("ex", self.electric.component)

    def test_sample_rate(self):
        self.assertEqual(128.0, self.electric.sample_rate)

    def test_file_type(self):
        self.assertEqual("metadata", self.electric.file_type)

    def test_has_metadata(self):
        self.assertTrue(self.electric._has_metadata())

    def test_to_mt_metadata(self):
        electric_metadata = self.electric.get_channel_metadata()
        self.assertEqual(self.expected_electric_metadata, electric_metadata)

    def test_get_sensor_response_filter(self):
        fap = self.electric.get_sensor_response_filter()
        self.assertEqual(None, fap)

    @classmethod
    def tearDownClass(self):
        self.fn.unlink()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
