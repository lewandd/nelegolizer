"""
Test creation, saving and loading mpd file using LDrawModel and LDrawFile structures.
"""
import unittest
import numpy as np

from nelegolizer import legolize
from nelegolizer.data import LDrawModel, LDrawFile

class Test_mpd_file_operations(unittest.TestCase):
    def test_create(self):
        lego_bricks = legolize("fixtures/cone.obj")
        ldraw_model = LDrawModel.from_bricks(lego_bricks, "Model Name")
        ldraw_file = LDrawFile()
        ldraw_file.add_model(ldraw_model)

    def test_create_and_save_file(self):
        lego_bricks = legolize("fixtures/cone.obj")
        ldraw_model = LDrawModel.from_bricks(lego_bricks, "Model Name")
        ldraw_file = LDrawFile()
        ldraw_file.add_model(ldraw_model)
        ldraw_file.save("fixtures/legolized_cone.mpd")

    def test_load_file(self):
        ldraw_file2 = LDrawFile.load("fixtures/legolized_cone.mpd")

    def test_create_save_load_and_compare_lines(self):
        lego_bricks = legolize("fixtures/cone.obj")
        ldraw_model = LDrawModel.from_bricks(lego_bricks, "Model Name")
        ldraw_file = LDrawFile()
        ldraw_file.add_model(ldraw_model)
        ldraw_file.save("fixtures/legolized_cone.mpd")
        ldraw_file2 = LDrawFile.load("fixtures/legolized_cone.mpd")
        self.assertTrue(np.all(ldraw_file.lines == ldraw_file2.lines))