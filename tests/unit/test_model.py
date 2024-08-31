"""
Tests nelegolizer.model module
"""
import unittest
import numpy as np
from torch import nn

from nelegolizer.model import brick_classification_models
from nelegolizer.model import brick


class Test_brick_classification_models(unittest.TestCase):
    def test_not_empty(self):
        self.assertGreater(len(brick_classification_models), 0)

    def test_instance_nn_module(self):
        for key in brick_classification_models.keys():
            self.assertIsInstance(brick_classification_models[key], nn.Module)


class Test_brick_test_predict(unittest.TestCase):
    def setUp(self):
        first_key = list(brick_classification_models.keys())[0]
        self.model = brick_classification_models[first_key]

    def test_returns_int(self):
        grid = np.zeros([6, 6, 6]).flatten()
        self.assertIsInstance(brick.test_predict(self.model, grid), int)

    def test_incorrect_input_shape(self):
        with self.assertRaises(TypeError):
            grid = np.zeros([6, 6, 6])
            brick.test_predict(self.model, grid), int

    def test_incorrect_input_size(self):
        with self.assertRaises(RuntimeError):
            grid = np.zeros([7, 7, 7]).flatten()
            brick.test_predict(self.model, grid), int
