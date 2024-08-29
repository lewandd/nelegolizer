"""
Tests nelegolize.utils.mesh module
"""
import unittest
import numpy as np
import pyvista as pv

from nelegolizer.utils.mesh import get_resolution, translate_to_zero, \
    get_position, scale_to


class Test_get_resolution(unittest.TestCase):
    def setUp(self):
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        self.mesh = reader.read()

    def test_returns_ndarray(self):
        self.assertIsInstance(get_resolution(self.mesh), np.ndarray)

    def test_resolution_positive_values(self):
        res = get_resolution(self.mesh)
        self.assertGreater(res[0], 0)
        self.assertGreater(res[1], 0)
        self.assertGreater(res[1], 0)

    def test_correct_values(self):
        mesh = translate_to_zero(self.mesh)
        _, xmax, _, ymax, _, zmax = mesh.bounds
        self.assertTrue(np.allclose(np.array([xmax, ymax, zmax]),
                                    get_resolution(mesh)))

    def test_correct_values_with_translate_by_111(self):
        translate = np.array([1, 1, 1])
        translated_mesh = self.mesh.translate(translate)
        self.assertTrue(np.allclose(get_resolution(translated_mesh),
                                    get_resolution(self.mesh)))

    def test_correct_values_with_translate_by_m1m1m1(self):
        translate = np.array([-1, -1, -1])
        translated_mesh = self.mesh.translate(translate)
        self.assertTrue(np.allclose(get_resolution(translated_mesh),
                                    get_resolution(self.mesh)))

    def test_correct_values_with_another_translate_1(self):
        translate = np.array([-4, 3, 8])
        translated_mesh = self.mesh.translate(translate)
        self.assertTrue(np.allclose(get_resolution(translated_mesh),
                                    get_resolution(self.mesh)))

    def test_correct_values_with_another_translate_2(self):
        translate = np.array([0.3, -0.4, 0])
        translated_mesh = self.mesh.translate(translate)
        self.assertTrue(np.allclose(get_resolution(translated_mesh),
                                    get_resolution(self.mesh)))

    def test_correct_scaled_values_by_111(self):
        scale = np.array([1, 1, 1])
        scaled_mesh = self.mesh.scale(scale)
        self.assertTrue(np.allclose(get_resolution(scaled_mesh),
                                    get_resolution(self.mesh)))

    def test_correct_scaled_values_by_222(self):
        scale = np.array([2, 2, 2])
        scaled_mesh = self.mesh.scale(scale)
        self.assertTrue(np.allclose(get_resolution(scaled_mesh),
                                    scale * get_resolution(self.mesh)))

    def test_correct_scaled_values_by_01_3_09(self):
        scale = np.array([0.1, 3, 0.9])
        scaled_mesh = self.mesh.scale(scale)
        self.assertTrue(np.allclose(get_resolution(scaled_mesh),
                                    scale * get_resolution(self.mesh)))

    def test_correct_scaled_values_by_01_3_09_and_translation(self):
        scale = np.array([0.1, 3, 0.9])
        translate = np.array([0.4, -0.1, 3.2])
        scaled_mesh = self.mesh.scale(scale)
        res_mesh = scaled_mesh.translate(translate)
        self.assertTrue(np.allclose(get_resolution(res_mesh),
                                    scale * get_resolution(self.mesh)))

    def test_different_scaled_values_by_1_1_099(self):
        scale = np.array([1, 1, 0.99])
        scaled_mesh = self.mesh.scale(scale)
        self.assertFalse(np.allclose(get_resolution(scaled_mesh),
                                     get_resolution(self.mesh)))


class Test_get_position(unittest.TestCase):
    def setUp(self):
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        self.mesh = reader.read()

    def test_returns_ndarray(self):
        self.assertIsInstance(get_position(self.mesh), np.ndarray)

    def test_position_correct_after_translation_111(self):
        translate = np.array([1, 1, 1])
        translated_mesh = self.mesh.translate(translate)
        self.assertTrue(np.allclose(get_position(translated_mesh),
                                    translate+get_position(self.mesh)))

    def test_position_correct_after_another_translation_1(self):
        translate = np.array([0, 3.21, -0.32])
        translated_mesh = self.mesh.translate(translate)
        self.assertTrue(np.allclose(get_position(translated_mesh),
                                    translate+get_position(self.mesh)))

    def test_position_incorrect_after_scaling(self):
        scale = np.array([1, 2, 2])
        translated_mesh = self.mesh.scale(scale)
        self.assertFalse(np.allclose(get_position(translated_mesh),
                                     get_position(self.mesh)))

    def test_position_correct_after_translating_to_zero_and_scaling(self):
        start_position = get_position(self.mesh)
        scale = np.array([3.1, 0.4, 1])
        mesh = translate_to_zero(self.mesh)
        mesh = mesh.scale(scale)
        mesh = mesh.translate(start_position)
        self.assertTrue(np.allclose(start_position, get_position(mesh)))


class Test_translate_to_zero(unittest.TestCase):
    def setUp(self):
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        self.mesh = reader.read()

    def test_correct_positions(self):
        self.assertTrue(np.allclose(get_position(translate_to_zero(self.mesh)),
                                    np.array([0, 0, 0])))

    def test_correct_positions_after_translation(self):
        new_position = np.array([-1.3, 2.1, 0])
        mesh = translate_to_zero(self.mesh)
        mesh = mesh.translate(new_position)
        self.assertTrue(np.allclose(get_position(mesh), new_position))

    def test_correct_positions_after_scaling(self):
        translate = np.array([-1.3, 2.1, 0])
        scale = np.array([-1, -2.4, 1.3])
        mesh = self.mesh.scale(scale)
        mesh = mesh.translate(translate)
        mesh = translate_to_zero(mesh)
        self.assertTrue(np.allclose(get_position(mesh), np.array([0, 0, 0])))


class Test_scale_to(unittest.TestCase):
    def setUp(self):
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        self.mesh = reader.read()

    def test_returns_PolyData(self):
        target_res = np.array([2, 3, 2])
        self.assertIsInstance(scale_to(self.mesh, target_res), pv.PolyData)

    def test_returns_correct_resolution(self):
        target_res = np.array([0.2, 3, 2.4])
        self.assertTrue(np.allclose(
            get_resolution(scale_to(self.mesh, target_res)), target_res))

    # TODO: scaling with keeping ratio
