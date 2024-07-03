import unittest

import sys
sys.path.append("../")

from Dataset.data_generation import *

class TestDataGeneration(unittest.TestCase):


    def test_segment_heartbeat(self):
        dummy_data  = np.random.randn(100)
        window_size = 20
        
        in_range = segment_heartbeat(dummy_data, peak=50, window_size=window_size)
        self.assertEqual(len(in_range), 40)

        below_range = segment_heartbeat(dummy_data, peak=10, window_size=window_size)
        self.assertIsNone(below_range)

        above_range = segment_heartbeat(dummy_data, peak=90, window_size=window_size)
        self.assertIsNone(above_range)
        

    def test_process_record(self):
        record_path = "../Data/MIT-BIH-Raw/mit-bih-arrhythmia-database-1.0.0/100"
        window_size = 128
        
        heartbeats, labels = process_record(record_path, window_size)
        self.assertEqual(len(heartbeats), len(labels))
        self.assertEqual(len(heartbeats[0]), window_size)

    
if __name__ == "__main__":
    unittest.main()
