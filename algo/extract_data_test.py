import unittest
import pandas as pd
from algo.extract_data import segment_data

class TestSegmentData(unittest.TestCase):
    def test_segment_data(self):
        data = pd.DataFrame({'a': range(100)})
        n_segments = 10
        n_elements = 5
        segmented_data = segment_data(data, n_segments, n_elements)
        self.assertEqual(len(segmented_data), n_segments * n_elements)

if __name__ == '__main__':
    unittest.main()