"""
Tests nelegolize.utils.grid module
"""
import unittest
import numpy as np
import pyvista as pv

from nelegolizer.utils import voxelization as utils_voxelization
from nelegolizer.utils import mesh as utils_mesh
from nelegolizer.utils import grid as utils_grid
from nelegolizer.utils.conversion import bu_to_vu, vu_to_bu

class Test_rotate_grid(unittest.TestCase):
    def test_grid111_rotate_0(self):
        grid111 = np.array([[[True]]])
        grid111_rot = np.array([[[True]]])
        self.assertTrue(np.all(utils_grid.rotate(grid=grid111, degrees=0) == grid111_rot))

    def test_grid111_rotate_90(self):
        grid111 = np.array([[[False]]])
        grid111_rot = np.array([[[False]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid111, degrees=90) == grid111_rot))

    def test_grid111_rotate_180(self):
        grid111 = np.array([[[False]]])
        grid111_rot = np.array([[[False]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid111, degrees=180) == grid111_rot))

    def test_grid111_rotate_270(self):
        grid111 = np.array([[[True]]])
        grid111_rot = np.array([[[True]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid111, degrees=270) == grid111_rot))

    def test_grid222_rotate_0(self):
        grid222 = np.array([[[False, True], [True, False]],
                            [[True, True], [True, False]]])
        grid222_rot = np.array([[[False, True], [True, False]],
                                [[True, True], [True, False]]])
        self.assertTrue(np.all(utils_grid.rotate(grid=grid222, degrees=0) == grid222_rot))

    def test_grid222_rotate_90(self):
        grid222 = np.array([[[False, True], [True, False]],
                            [[True, True], [True, False]]])
        grid222_rot = np.array([[[True, False], [True, True]],
                                [[True, True], [False, False]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid222, degrees=90) == grid222_rot))

    def test_grid222_rotate_180(self):
        grid222 = np.array([[[False, True], [True, False]],
                            [[True, True], [True, False]]])
        grid222_rot = np.array([[[True, True], [False, True]],
                                [[True, False], [False, True]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid222, degrees=180) == grid222_rot))

    def test_grid222_rotate_270(self):
        grid222 = np.array([[[False, True], [True, False]],
                            [[True, True], [True, False]]])
        grid222_rot = np.array([[[True, True], [False, False]],
                                [[False, True], [True, True]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid222, degrees=270) == grid222_rot))

    def test_grid323_rotate_90(self):
        grid323 = np.array([[[False, True, False], [False, True, False]],
                            [[True, False, False], [False, True, True]],
                            [[True, True, False], [True, True, True]]])
        grid323_rot = np.array([[[True, True, False], [True, False, False]],
                                [[True, False, True], [True, True, True]],
                                [[False, False, False], [True, True, False]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid323, degrees=90) == grid323_rot))

    def test_grid323_rotate_180(self):
        grid323 = np.array([[[False, True, False], [False, True, False]],
                            [[True, False, False], [False, True, True]],
                            [[True, True, False], [True, True, True]]])
        grid323_rot = np.array([[[False, True, True], [True, True, True]],
                                [[False, False, True], [True, True, False]],
                                [[False, True, False], [False, True, False]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid323, degrees=180) == grid323_rot))

    def test_grid323_rotate_270(self):
        grid323 = np.array([[[False, True, False], [False, True, False]],
                            [[True, False, False], [False, True, True]],
                            [[True, True, False], [True, True, True]]])
        grid323_rot = np.array([[[False, False, False], [False, True, True]],
                                [[True, False, True], [True, True, True]],
                                [[False, True, True], [False, False, True]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid323, degrees=270) == grid323_rot))

    def test_grid213_rotate_90(self):
        grid213 = np.array([[[False, True, False]],
                            [[True, False, False]]])
        grid213_rot = np.array([[[True, False]],
                                [[False, True]],
                                [[False, False]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid213, degrees=90) == grid213_rot))

    def test_grid213_rotate_180(self):
        grid213 = np.array([[[False, True, False]],
                            [[True, False, False]]])
        grid213_rot = np.array([[[False, False, True]],
                                [[False, True, False]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid213, degrees=180) == grid213_rot))

    def test_grid213_rotate_270(self):
        grid213 = np.array([[[False, True, False]],
                            [[True, False, False]]])
        grid213_rot = np.array([[[False, False]],
                                [[True, False]],
                                [[False, True]]])
        self.assertTrue(
            np.all(utils_grid.rotate(grid=grid213, degrees=270) == grid213_rot))

    def test_invalid_rotation(self):
        with self.assertRaises(Exception):
            grid111 = np.array([[[True]]])
            utils_grid.rotate(grid=grid111, degrees=45)


class Test_get_subgrid(unittest.TestCase):
    def test_subgrid_111_from_111(self):
        grid111 = np.array([[[True]]])
        position = (0, 0, 0)
        shape = np.array([1, 1, 1])
        expected111 = np.array([[[True]]])
        subgrid111 = utils_grid.get_subgrid(grid111, position, shape)
        self.assertTrue(np.all(subgrid111 == expected111))

    def test_subgrid_111_from_222(self):
        grid222 = np.array([[[True, True], [False, True]],
                            [[True, True], [True, True]]])
        position = (0, 1, 0)
        shape = np.array([1, 1, 1])
        expected111 = np.array([[[False]]])
        subgrid111 = utils_grid.get_subgrid(grid222, position, shape)
        self.assertTrue(np.all(subgrid111 == expected111))

    def test_subgrid_112_from_222(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        position = (1, 0, 0)
        shape = np.array([1, 1, 2])
        expected112 = np.array([[[True, False]]])
        subgrid112 = utils_grid.get_subgrid(grid222, position, shape)
        self.assertTrue(np.all(subgrid112 == expected112))

    def test_subgrid_122_from_222(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        position = (0, 0, 0)
        shape = np.array([1, 2, 2])
        expected122 = np.array([[[True, True], [True, True]]])
        subgrid122 = utils_grid.get_subgrid(grid222, position, shape)
        self.assertTrue(np.all(subgrid122 == expected122))

    def test_subgrid_222_from_222(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        position = (0, 0, 0)
        shape = np.array([2, 2, 2])
        expected222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
        subgrid222 = utils_grid.get_subgrid(grid222, position, shape)
        self.assertTrue(np.all(subgrid222 == expected222))

    def test_invalid_shape_and_position(self):
        with self.assertRaises(IndexError):
            grid222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
            position = (1, 1, 1)
            shape = np.array([1, 1, 2])
            utils_grid.get_subgrid(grid222, position, shape)

    def test_invalid_too_big_shape(self):
        with self.assertRaises(IndexError):
            grid222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
            position = (0, 0, 0)
            shape = np.array([1, 3, 1])
            utils_grid.get_subgrid(grid222, position, shape)

    def test_invalid_too_big_position(self):
        with self.assertRaises(IndexError):
            grid222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
            position = (0, 2, 0)
            shape = np.array([1, 1, 1])
            utils_grid.get_subgrid(grid222, position, shape)

    def test_invalid_too_small_position(self):
        with self.assertRaises(IndexError):
            grid222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
            position = (-1, 1, 0)
            shape = np.array([1, 1, 1])
            utils_grid.get_subgrid(grid222, position, shape)


class Test_extend(unittest.TestCase):
    def test_divisibility_222_by_111(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        divider = np.array([1, 1, 1])
        expected222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
        ext_grid222 = utils_grid.provide_divisibility(grid222, divider)
        self.assertTrue(np.all(ext_grid222 == expected222))

    def test_divisibility_222_by_131(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        divider = np.array([1, 3, 1])
        expected232 = np.array([[[True, True], [True, True], [False, False]],
                                [[True, False], [True, True], [False, False]]])
        ext_grid232 = utils_grid.provide_divisibility(grid222, divider)
        self.assertTrue(np.all(ext_grid232 == expected232))

    def test_divisibility_222_by_342(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        divider = np.array([3, 4, 2])
        expected342 = np.array([
            [[True, True], [True, True], [False, False], [False, False]],
            [[True, False], [True, True], [False, False], [False, False]],
            [[False, False], [False, False], [False, False], [False, False]]])
        ext_grid342 = utils_grid.provide_divisibility(grid222, divider)
        self.assertTrue(np.all(ext_grid342 == expected342))

    def test_divisibility_12_12_12_by_3_4_5(self):
        grid = np.random.choice([True, False], size=(12, 12, 12))
        divider = np.array([3, 4, 5])
        expected_shape = np.array((12, 12, 15))
        ext_grid = utils_grid.provide_divisibility(grid, divider)
        self.assertTrue(np.all(ext_grid.shape == expected_shape))


class Test_get_fill_ratio(unittest.TestCase):
    def test_grid_111_fill_0(self):
        grid = np.array([[[False]]])
        self.assertEqual(utils_grid.get_fill_ratio(grid), 0)

    def test_grid_111_fill_1(self):
        grid = np.array([[[True]]])
        self.assertEqual(utils_grid.get_fill_ratio(grid), 1)

    def test_grid_222_fill_0(self):
        grid = np.array([[[False, False], [False, False]],
                         [[False, False], [False, False]]])
        self.assertEqual(utils_grid.get_fill_ratio(grid), 0)

    def test_grid_222_fill_1(self):
        grid = np.array([[[True, True], [True, True]],
                         [[True, True], [True, True]]])
        self.assertEqual(utils_grid.get_fill_ratio(grid), 1)

    def test_grid_222_fill_05(self):
        grid = np.array([[[True, False], [False, True]],
                         [[False, False], [True, True]]])
        self.assertEqual(utils_grid.get_fill_ratio(grid), 0.5)

    def test_grid_252_fill_06(self):
        grid = np.array([[[True, False],
                          [False, True],
                          [False, True],
                          [False, True],
                          [False, True]],
                         [[False, True],
                          [True, True],
                          [True, True],
                          [True, False],
                          [True, False]]])
        self.assertEqual(utils_grid.get_fill_ratio(grid), 0.6)


class Test_from_pv_voxels(unittest.TestCase):
    def setUp(self) -> None:
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        mesh = reader.read()
        self.pv_voxels1 = pv.voxelize(mesh, density=0.2, check_surface=False)
        self.pv_voxels2 = utils_voxelization.from_mesh(
            mesh, voxel_mesh_shape=np.array([0.2, 0.2, 0.2]))

    def test_result_not_None1(self):
        self.assertIsNotNone(utils_grid.from_pv_voxels(self.pv_voxels1))

    def test_result_not_None2(self):
        self.assertIsNotNone(utils_grid.from_pv_voxels(self.pv_voxels2))

    def test_result_is_ndarray1(self):
        self.assertTrue(
            isinstance(utils_grid.from_pv_voxels(self.pv_voxels1), np.ndarray))

    def test_result_is_ndarray2(self):
        self.assertTrue(
            isinstance(utils_grid.from_pv_voxels(self.pv_voxels2), np.ndarray))

    def test_number_of_cells_is_correct1(self):
        grid = utils_grid.from_pv_voxels(self.pv_voxels1)
        n_voxels = 0
        for v in np.nditer(grid):
            if v:
                n_voxels += 1
        self.assertEqual(self.pv_voxels1.n_cells, n_voxels)

    def test_number_of_cells_is_correct2(self):
        grid = utils_grid.from_pv_voxels(self.pv_voxels2)
        n_voxels = 0
        for v in np.nditer(grid):
            if v:
                n_voxels += 1
        self.assertEqual(self.pv_voxels2.n_cells, n_voxels)

    def test_all_cells_are_same_shape1(self):
        first_cell_shape = utils_mesh.get_resolution(
            self.pv_voxels1.extract_cells(0))
        for ind in range(self.pv_voxels1.n_cells):
            cell_shape = utils_mesh.get_resolution(
                self.pv_voxels1.extract_cells(ind))
            self.assertTrue(np.allclose(cell_shape, first_cell_shape))

    def test_all_cells_are_same_shape2(self):
        first_cell_shape = utils_mesh.get_resolution(
            self.pv_voxels2.extract_cells(0))
        for ind in range(self.pv_voxels2.n_cells):
            cell_shape = utils_mesh.get_resolution(
                self.pv_voxels2.extract_cells(ind))
            self.assertTrue(np.allclose(cell_shape, first_cell_shape))


class Test_from_mesh(unittest.TestCase):
    def setUp(self):
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        self.mesh = reader.read()

    def test_result_not_None(self):
        self.assertIsNotNone(
            utils_grid.from_mesh(self.mesh, voxel_mesh_shape=np.array([1, 1, 1])))

    def test_result_is_ndarray(self):
        self.assertIsInstance(
            utils_grid.from_mesh(self.mesh,
                      voxel_mesh_shape=np.array([1, 1, 1])), np.ndarray)


class Test_conversions(unittest.TestCase):
    def test_ok_bu_to_vu(self):
        bu = np.array([1, 1, 2])
        expected_vu = np.array([5, 2, 10])
        self.assertTrue(np.all(bu_to_vu(bu) == expected_vu))

    def test_ok_vu_to_bu(self):
        vu = np.array([5, 2, 10])
        expected_bu = np.array([1, 1, 2])
        self.assertTrue(np.all(vu_to_bu(vu) == expected_bu))

    def test_error_vu_to_bu(self):
        vu = np.array([0, 16, 7])
        with self.assertRaises(Exception):
            vu_to_bu(vu)


if __name__ == '__main__':
    unittest.main()
