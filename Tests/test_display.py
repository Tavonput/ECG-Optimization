import sys
sys.path.append("../")

import unittest
from Utils.display import *


class TestGroupedValues(unittest.TestCase):


    def test_add_to_group(self):
        group = GroupedValues()

        group.add_to_group("group_1", 1, 1)
        self.assertTrue(isinstance(group.groups["group_1"][0], GroupedValues.Group))

        group.add_to_group("group_1", 2, 2)
        self.assertTrue(len(group.groups["group_1"]) == 2)

        group.add_to_group("group_2", 3, 3)
        self.assertTrue(isinstance(group.groups["group_2"][0], GroupedValues.Group))


    def test_is_empty(self):
        group = GroupedValues()

        self.assertTrue(group.is_empty())

        group.add_to_group("group", 1, 1)
        self.assertFalse(group.is_empty())


    def test_is_valid(self):
        group = GroupedValues()

        group.add_to_group("group_1", 1, 1)
        group.add_to_group("group_2", 1, 1)
        self.assertTrue(group.is_valid())

        group.add_to_group("group_2", 1, 1)
        self.assertFalse(group.is_valid())

    
    def test_size(self):
        group = GroupedValues()

        group.add_to_group("group_1", 1, 1)
        group.add_to_group("group_2", 1, 1)
        self.assertTrue(group.size() == 2)

    
    def test_group_size(self):
        group = GroupedValues()

        group.add_to_group("group_1", 1, 1)
        group.add_to_group("group_2", 1, 1)
        self.assertTrue(group.group_size() == 1)
    

if __name__ == "__main__":
    unittest.main()
    