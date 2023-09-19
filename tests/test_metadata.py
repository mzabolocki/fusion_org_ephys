"""
test_metadata.py

Unit tests for metadata.

"""

import unittest
import pathlib
import pandas as pd
from fused_org_ephys import FusedOrgSpikes

################################


class TestMetdata(unittest.TestCase):

    def setUp(self):

        # default dir to tc metadata
        metadata_path = pathlib.PurePath('tests', 'test_data', 'metadata',
                                         'metadata_test.xlsx')

        base_folder = pathlib.PurePath('tests', 'test_data')

        # FusedOrgSpikes class
        self.muaspikes = FusedOrgSpikes(metadata_path=metadata_path, expID=['test_rawfile'], base_folder=base_folder)

    def test_metadata_size(self):

        # number of data
        metadata_data_entry = self.muaspikes.metadata.shape[0]

        # number features
        metadata_feature_number = self.muaspikes.metadata.shape[1]

        self.assertEqual(metadata_data_entry, 1)
        self.assertEqual(metadata_feature_number, 30)

    def test_metadata_orgages(self):

        # age of select organoids
        age = pd.DataFrame(self.muaspikes.metadata).ages.values[0]

        self.assertEqual(age, 154.0)


if __name__ == '__main__':
    unittest.main()
