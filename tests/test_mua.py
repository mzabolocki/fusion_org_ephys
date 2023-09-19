"""
test_mua.py

Unit tests for mua spike extractions.

"""

import unittest
import pathlib
from fused_org_ephys import FusedOrgSpikes


################################


class TestMUASpikes(unittest.TestCase):

    def setUp(self):

        # default dir to tc metadata
        metadata_path = pathlib.PurePath('tests', 'test_data', 'metadata',
                                         'metadata_test.xlsx')
        base_folder = pathlib.PurePath('tests', 'test_data')

        # FusedOrgSpikes class
        self.muaspikes = FusedOrgSpikes(metadata_path=metadata_path, expID=['test_rawfile'],
                                        base_folder=base_folder)

    def test_muathreshold(self):

        # mua threshold
        mua_thresh_set = self.muaspikes.baseline_threshs
        mua_thresh_set_idx0 = mua_thresh_set[0]
        mua_thresh_set_idx10 = mua_thresh_set[10]

        self.assertAlmostEqual(mua_thresh_set_idx0, 88.955, 3, 1e-5)
        self.assertAlmostEqual(mua_thresh_set_idx10, 81.542, 3, 1e-5)

    def test_muaspiketimes(self):

        # mua spike times
        mua_spiketimes_set = self.muaspikes.mua_spikes
        mua_spiketimes_set_idx10_spike10 = mua_spiketimes_set[10][10]
        mua_spiketimes_set_idx22_spike10 = mua_spiketimes_set[22][10]

        self.assertAlmostEqual(mua_spiketimes_set_idx10_spike10, 13.287, 3, 1e-5)
        self.assertAlmostEqual(mua_spiketimes_set_idx22_spike10, 13.428, 3, 1e-5)


if __name__ == '__main__':
    unittest.main()
