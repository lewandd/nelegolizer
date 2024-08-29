"""
Tests nelegolizer.data module
"""
import unittest
import pyvista
import numpy as np

from nelegolizer.data import LDrawPart, part_by_label, part_by_filename
from nelegolizer.data import LegoBrick, LDrawReference, LDrawFile, LDrawModel
from nelegolizer.data._LegoBrick import ROT_MATRIX_180


class TestLDrawPart(unittest.TestCase):
    def test_initialization_with_valid_parameters(self):
        LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                  geom_path="tests/unit/fixtures/3004.stl",
                  label=3,
                  size=(2, 1, 1))

    def test_brick_id_is_ok(self):
        ldp = LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                        geom_path="tests/unit/fixtures/3004.stl",
                        label=3,
                        size=(2, 1, 1))
        self.assertEqual(ldp.brick_id, '3004')

    def test_initialization_with_no_existing_geom_path(self):
        with self.assertRaises(FileNotFoundError):
            LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                      geom_path="tests/unit/fixtures/invalid_file.stl",
                      label=3,
                      size=(2, 1, 1))

    def test_initialization_with_no_label(self):
        with self.assertRaises(TypeError):
            LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                      geom_path="tests/unit/fixtures/3004.stl",
                      size=(2, 1, 1))

    def test_initialization_with_no_dat_path(self):
        with self.assertRaises(TypeError):
            LDrawPart(geom_path="tests/unit/fixtures/3004.stl",
                      label=3,
                      size=(2, 1, 1))

    def test_initialization_with_no_geom_path(self):
        with self.assertRaises(TypeError):
            LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                      label=3,
                      size=(2, 1, 1))

    def test_mesh_exist(self):
        ldp = LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                        geom_path="tests/unit/fixtures/3004.stl",
                        label=3,
                        size=(2, 1, 1))
        self.assertIsNotNone(ldp.mesh)

    def test_mesh_is_pv_PolyData(self):
        ldp = LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                        geom_path="tests/unit/fixtures/3004.stl",
                        label=3,
                        size=(2, 1, 1))
        self.assertIsInstance(ldp.mesh, pyvista.PolyData)

    def test_grid_exist(self):
        ldp = LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                        geom_path="tests/unit/fixtures/3004.stl",
                        label=3,
                        size=(2, 1, 1))
        self.assertIsNotNone(ldp.grid)

    def test_grid_is_np_ndarray(self):
        ldp = LDrawPart(dat_path="tests/unit/fixtures/3004.dat",
                        geom_path="tests/unit/fixtures/3004.stl",
                        label=3,
                        size=(2, 1, 1))
        self.assertIsInstance(ldp.grid, np.ndarray)


class TestPartByLabel(unittest.TestCase):
    def test_part_by_label_not_empty(self):
        self.assertGreater(len(part_by_label), 0)

    def test_part_by_label_elements_are_LDrawPart(self):
        for key in part_by_label.keys():
            self.assertIsInstance(part_by_label[key], LDrawPart)


class TestPartByFilename(unittest.TestCase):
    def test_part_by_label_not_empty(self):
        self.assertGreater(len(part_by_filename), 0)

    def test_part_by_label_elements_are_LDrawPart(self):
        for key in part_by_filename.keys():
            self.assertIsInstance(part_by_filename[key], LDrawPart)


class TestLegoBrick(unittest.TestCase):
    def setUp(self):
        self.valid_label = list(part_by_label.keys())[0]

    def test_initialization_with_valid_parameters(self):
        LegoBrick(label=self.valid_label,
                  mesh_position=(2, 5, 7),
                  rotation=180)

    def test_label_out_of_bond(self):
        with self.assertRaises(KeyError):
            LegoBrick(label=-1,
                      mesh_position=(2, 5, 7),
                      rotation=180)

    def test_invalid_rotation(self):
        with self.assertRaises(Exception):
            LegoBrick(label=self.valid_label,
                      mesh_position=(2, 5, 7),
                      rotation=45)

    def test_str_cast(self):
        lb = LegoBrick(label=self.valid_label,
                       mesh_position=(2, 5, 7),
                       rotation=180)
        str(lb)

    def test_part_attribute_exists(self):
        lb = LegoBrick(label=self.valid_label,
                       mesh_position=(2, 5, 7),
                       rotation=180)
        self.assertIsNotNone(lb.part)

    def test_part_is_LDrawPart(self):
        lb = LegoBrick(label=self.valid_label,
                       mesh_position=(2, 5, 7),
                       rotation=180)
        self.assertIsInstance(lb.part, LDrawPart)

    def test_mesh_exist(self):
        lb = LegoBrick(label=self.valid_label,
                       mesh_position=(2, 5, 7),
                       rotation=180)
        self.assertIsNotNone(lb.mesh)

    def test_mesh_is_pv_PolyData(self):
        lb = LegoBrick(label=self.valid_label,
                       mesh_position=(2, 5, 7),
                       rotation=180)
        self.assertIsInstance(lb.mesh, pyvista.PolyData)

    def test_grid_exist(self):
        lb = LegoBrick(label=self.valid_label,
                       mesh_position=(2, 5, 7),
                       rotation=180)
        self.assertIsNotNone(lb.grid)

    def test_grid_is_np_ndarray(self):
        lb = LegoBrick(label=self.valid_label,
                       mesh_position=(2, 5, 7),
                       rotation=180)
        self.assertIsInstance(lb.grid, np.ndarray)

    def test_init_from_reference(self):
        line = "1 7 70 -24 -90 1 0 0 0 1 0 0 0 1 3005.dat\n"
        ldr = LDrawReference.from_line(line)
        self.assertIsInstance(LegoBrick.from_reference(ldr), LegoBrick)

    def test_matrix(self):
        lb = LegoBrick(label=self.valid_label,
                       mesh_position=(2, 5, 7),
                       rotation=180)
        self.assertTrue(np.allclose(lb.matrix[:3, :3], ROT_MATRIX_180))
        self.assertTrue(np.allclose(lb.matrix[3, :3], lb.mesh_position))
        self.assertTrue(np.allclose(lb.matrix[-1, -1], 1))


class Test_LDraw_Reference(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [2, 1, 1, 1]
            ])
        self.line = ("1 7 70.0 -24.0 -90.0 "
                     "1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 3005.dat\n")

    def test_init(self):
        LDrawReference(name="3004.dat", matrix=self.matrix, color=16)

    def test_init_from_line_returns_LDrawReference(self):
        self.assertIsInstance(
            LDrawReference.from_line(self.line), LDrawReference)

    def test_init_from_line_has_correct_values(self):
        ldf = LDrawReference.from_line(self.line)
        self.assertEqual(ldf.color, 7)
        self.assertEqual(ldf.name, "3005.dat")
        self.assertEqual(ldf.matrix[0, 0], 1)
        self.assertEqual(ldf.matrix[1, 0], 0)
        self.assertEqual(ldf.matrix[2, 0], 0)
        self.assertEqual(ldf.matrix[3, 0], 70)
        self.assertEqual(ldf.matrix[0, 1], 0)
        self.assertEqual(ldf.matrix[1, 1], 1)
        self.assertEqual(ldf.matrix[2, 1], 0)
        self.assertEqual(ldf.matrix[3, 1], -24)
        self.assertEqual(ldf.matrix[0, 2], 0)
        self.assertEqual(ldf.matrix[1, 2], 0)
        self.assertEqual(ldf.matrix[2, 2], 1)
        self.assertEqual(ldf.matrix[3, 2], -90)
        self.assertEqual(ldf.matrix[0, 3], 0)
        self.assertEqual(ldf.matrix[1, 3], 0)
        self.assertEqual(ldf.matrix[2, 3], 0)
        self.assertEqual(ldf.matrix[3, 3], 1)

    def test_from_line_to_line(self):
        ldf = LDrawReference.from_line(self.line)
        self.assertEqual(ldf.line, self.line)

    def test_from_line_position(self):
        ldf = LDrawReference.from_line(self.line)
        self.assertEqual(ldf.position[0], 70)
        self.assertEqual(ldf.position[1], -24)
        self.assertEqual(ldf.position[2], -90)

    def test_from_line_rotation(self):
        ldf = LDrawReference.from_line(self.line)
        self.assertEqual(ldf.rotation[0, 0], 1)
        self.assertEqual(ldf.rotation[1, 0], 0)
        self.assertEqual(ldf.rotation[2, 0], 0)
        self.assertEqual(ldf.rotation[0, 1], 0)
        self.assertEqual(ldf.rotation[1, 1], 1)
        self.assertEqual(ldf.rotation[2, 1], 0)
        self.assertEqual(ldf.rotation[0, 2], 0)
        self.assertEqual(ldf.rotation[1, 2], 0)
        self.assertEqual(ldf.rotation[2, 2], 1)


class Test_LDrawFile(unittest.TestCase):
    def test_empty_init(self):
        LDrawFile()

    def test_init_from_file(self):
        LDrawFile.load("tests/unit/fixtures/New Model.dat")

    def test_init_from_file2(self):
        LDrawFile.load("tests/unit/fixtures/5935 - Island Hopper.mpd")

    def test_models_len_5(self):
        ldf = LDrawFile.load("tests/unit/fixtures/5935 - Island Hopper.mpd")
        self.assertEqual(len(ldf.models), 5)

    def test_models_are_LDrawModel(self):
        ldf = LDrawFile.load("tests/unit/fixtures/5935 - Island Hopper.mpd")
        for m in ldf.models:
            self.assertIsInstance(m, LDrawModel)

    def test_path_ok(self):
        ldf = LDrawFile.load("tests/unit/fixtures/5935 - Island Hopper.mpd")
        self.assertEqual(
            ldf.path, "tests/unit/fixtures/5935 - Island Hopper.mpd")


class Test_LDrawModel(unittest.TestCase):
    def test_empty_init(self):
        LDrawModel(name="Empty Model")

    def test_init_from_mlt_models(self):
        ldf = LDrawFile.load("tests/unit/fixtures/5935 - Island Hopper.mpd")
        LDrawModel.merge_multiple_models(ldf.models)

    def test_refs_number_203(self):
        ldf = LDrawFile.load("tests/unit/fixtures/5935 - Island Hopper.mpd")
        ldm = LDrawModel.merge_multiple_models(ldf.models)
        self.assertEqual(len(ldm.references), 203)

    def test_init_from_bricks_all_correct(self):
        lbs = [LegoBrick(label=1, mesh_position=(1, 1, 2), rotation=180),
               LegoBrick(label=0, mesh_position=(1, 5, 1), rotation=90),
               LegoBrick(label=1, mesh_position=(0, 0, 0), rotation=0)]
        ldm = LDrawModel.from_bricks(lbs, "Bricks Model")
        for i in range(len(lbs)):
            brick = lbs[i]
            ref = ldm.references[i]
            dat_filename = ref.name
            self.assertEqual(part_by_filename[dat_filename].label, brick.label)
            self.assertTrue(np.allclose(ref.matrix, brick.matrix))
            self.assertEqual(ref.color, brick.color)

    def test_as_bricks_from_bricks(self):
        lbs = [LegoBrick(label=1, mesh_position=(1, 1, 2), rotation=180),
               LegoBrick(label=0, mesh_position=(1, 5, 1), rotation=90),
               LegoBrick(label=1, mesh_position=(0, 0, 0), rotation=0)]
        ldm = LDrawModel.from_bricks(lbs, "Bricks Model")
        lbs2 = ldm.as_bricks()
        for i in range(len(lbs)):
            self.assertEqual(lbs[i].color, lbs2[i].color)
            self.assertEqual(lbs[i].label, lbs2[i].label)
            self.assertTrue(
                np.allclose(lbs[i].mesh_position, lbs2[i].mesh_position))


if __name__ == '__main__':
    unittest.main()
