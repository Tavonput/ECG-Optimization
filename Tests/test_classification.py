import sys
sys.path.append("../")

import unittest
from Utils.classification import *

import shutil
import os

class TestStats(unittest.TestCase):


    def test_training_stats_v2(self):
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
        os.makedirs("tmp")
        
        stats = TrainingStats()
        stats.best_accuracy      = 1.0
        stats.total_train_time   = 100.0
        stats.running_accuracy   = [1.0, 2.0]
        stats.running_loss       = [1.0, 2.0, 3.0]
        stats.running_train_time = [1.0, 2.0, 3.0, 4.0]
        stats.running_epoch_time = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats.epochs             = 10
        stats.serialize("tmp/stats.txt")

        new_stats = TrainingStats(from_save="tmp/stats.txt")
        self.assertEqual(new_stats.best_accuracy, stats.best_accuracy)
        self.assertEqual(new_stats.total_train_time, stats.total_train_time)
        self.assertEqual(new_stats.running_accuracy, stats.running_accuracy)
        self.assertEqual(new_stats.running_loss, stats.running_loss)
        self.assertEqual(new_stats.running_train_time, stats.running_train_time)
        self.assertEqual(new_stats.running_epoch_time, stats.running_epoch_time)
        self.assertEqual(new_stats.epochs, stats.epochs)

        shutil.rmtree("tmp")

    
if __name__ == "__main__":
    unittest.main()
