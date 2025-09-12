"""
Tests nelegolizer.model module
"""
import unittest
import numpy as np
from torch import nn

from nelegolizer.model import brick, shape_model_map
from nelegolizer.constants import BU_RES

#TODO dodać testy z shape_model_map

#class Test_shape_model_map(unittest.TestCase):
#    def test_not_empty(self):
#        self.assertGreater(len(shape_model_map), 0)

#    def test_instance_nn_module(self):
#        for key in shape_model_map.keys():
#            self.assertIsInstance(shape_model_map[key], nn.Module)

#class Test_brick_test_predict(unittest.TestCase):
#    def setUp(self):
#        first_key = list(shape_model_map.keys())[0]
#        self.model = shape_model_map[first_key]

#    def test_returns_int(self):
#        grid = np.zeros(BU_RES).flatten()
#        self.assertIsInstance(brick.test_predict(self.model, grid), int)

#    def test_incorrect_input_shape(self):
#        with self.assertRaises(TypeError):
#            grid = np.zeros(BU_RES)
#            brick.test_predict(self.model, grid), int

#    def test_incorrect_input_size(self):
#        with self.assertRaises(RuntimeError):
#            grid = np.zeros([7, 7, 7]).flatten()
#            brick.test_predict(self.model, grid), int
