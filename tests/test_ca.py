"""
test_ca.py

Unit tests for ca imaging processing.

"""

import unittest
import pathlib
from fused_org_ephys import FusedOrgCa


################################


class TestMUASpikes(unittest.TestCase):

    def setUp(self):

        # Calcium class
        fpath = pathlib.PurePath('tests', 'test_data', 'gcamp', 'test_traces.pickle')
        self.ca = FusedOrgCa(traces_fname=fpath, prominence=0.4, rel_height=0.99)

        # Ca peak df
        self.ca_peak_df = self.ca.return_caspikewidth_df()

    def test_capeaks(self):

        # ca peak times
        peaks = self.ca.ca_peaks

        # neuron 0, peak 0
        self.assertEqual(peaks[0][0], 570)

    def test_cawidth(self):

        # ca widths
        widths = self.ca.ca_widths

        # neuron 0, peak 0
        self.assertAlmostEqual(widths[0][0], 88.74956445, 3, 1e-5)

    def test_cadf(self):

        # ca peak df
        ca_peak_df = self.ca_peak_df

        # iloc[0,0]
        val = ca_peak_df.iloc[0, 0]
        self.assertAlmostEqual(val, 5.768722, 3, 1e-5)


if __name__ == '__main__':
    unittest.main()
