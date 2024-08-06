import sys
sys.path.append("../")

import unittest
from Dataset.data_generation import *

import numpy as np


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


    def test_split_value(self):
        base_number = 100
        
        ordered_1, ordered_2 = split_value(base_number, 0.8, shuffle=False)
        self.assertTrue(len(ordered_1) == 80)
        self.assertTrue(len(ordered_2) == 20)

        shuffle_1, shuffle_2 = split_value(base_number, 0.8, shuffle=True)

        # If these ever fail, that's crazy
        self.assertTrue(list(ordered_1) != list(shuffle_1))
        self.assertTrue(list(ordered_2) != list(shuffle_2))


    def test_resize_image(self):
        dummy_input = np.random.randn(2, 3, 256, 256)

        dummy_output = SamplePreprocessor._resize_images(dummy_input, 128)
        self.assertEqual(dummy_output.shape, (2, 3, 128, 128))


    def test_crop_image(self):
        dummy_input = np.random.randn(2, 3, 256, 256)

        dummy_output = SamplePreprocessor._crop_images(dummy_input, 128)
        self.assertEqual(dummy_output.shape, (2, 3, 128, 128))


    def test_random_subset(self):
        indices = random_subset(1000, 1.0, shuffle=True)
        self.assertEqual(len(indices), 1000)

        indices = random_subset(1000, 0.4, shuffle=True)
        self.assertEqual(len(indices), 400)
    

    def test_resample_signal(self):
        dummy_input = np.random.randn(3, 1000)

        dummy_output = SamplePreprocessor._resample_signal(dummy_input, 500)
        self.assertEqual(dummy_output.shape, (3, 500))


    def test_crop_signal(self):
        dummy_input = np.random.randn(3, 256)

        dummy_output = SamplePreprocessor._crop_signal(dummy_input, 128)
        self.assertEqual(dummy_output.shape, (3, 128))

    
if __name__ == "__main__":
    unittest.main()
