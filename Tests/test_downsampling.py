import sys
sys.path.append("../")

import unittest
from Experiments.Downsampling.downsampling import *

import torch
import torch.nn as nn


class TestDownsampling(unittest.TestCase):


    def test_combine_model_weights(self):
        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)


            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model_1 = TestModel()
        for param in model_1.parameters():
            param.data.fill_(2)

        model_2 = TestModel()
        for param in model_2.parameters():
            param.data.fill_(4)

        model_avg = TestModel()
        average_model_weights([model_1, model_2], model_avg)

        for param in model_avg.parameters():
            test_tensor = torch.zeros_like(param.data)
            test_tensor.data.fill_(3)
            self.assertTrue(torch.equal(param.data, test_tensor))


if __name__ == "__main__":
    unittest.main()
