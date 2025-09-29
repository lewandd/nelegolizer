"""
Test basic usage of legolize function.
"""
import unittest
import pyvista as pv

from nelegolizer import legolize
from nelegolizer.data import LegoBrick


#class Test_legolize(unittest.TestCase):
    #def test_path_init(self):
    #    legolize("tests/integration/fixtures/cone.obj")

    #def test_mesh_init(self):
    #    reader = pv.get_reader("tests/integration/fixtures/cone.obj")
    #    mesh = reader.read()
    #    legolize(mesh)

    #def test_invalid_init(self):
    #    with self.assertRaises(ValueError):
    #        arg = int(5)
    #        legolize(arg)

    #def test_returns_list(self):
    #    lbs = legolize("tests/integration/fixtures/cone.obj")
    #    self.assertIsInstance(lbs, list)

    #def test_return_list_not_empty(self):
    #    lbs = legolize("tests/integration/fixtures/cone.obj")
    #    self.assertGreater(len(lbs), 0)

    #def test_return_list_of_LegoBricks(self):
    #    lbs = legolize("tests/integration/fixtures/cone.obj")
    #    for el in lbs:
    #        self.assertIsInstance(el, LegoBrick)
