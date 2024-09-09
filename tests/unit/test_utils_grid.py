"""
Tests nelegolize.utils.grid module
"""
import unittest
import numpy as np
import pyvista as pv

from nelegolizer.utils import voxelization
from nelegolizer.utils import mesh as umesh
from nelegolizer.utils.grid import find_best_rotation, rotate, get_subgrid, \
    provide_divisibility, get_fill_ratio, from_pv_voxels, \
    from_mesh, add_padding


class Test_find_best_rotation(unittest.TestCase):
    # grid 212

    def test_grid212_with_mass_center_bottom_left(self):
        grid212 = np.array([[[True, False]],
                            [[False, False]]])
        self.assertEqual(find_best_rotation(grid212), 180)

    def test_grid212_with_mass_center_bottom_right(self):
        grid212 = np.array([[[False, False]],
                            [[True, False]]])
        self.assertEqual(find_best_rotation(grid212), 90)

    def test_grid212_with_mass_center_top_right(self):
        grid212 = np.array([[[False, False]],
                            [[False, True]]])
        self.assertEqual(find_best_rotation(grid212), 0)

    def test_grid212_with_mass_center_top_left(self):
        grid212 = np.array([[[False, True]],
                            [[False, False]]])
        self.assertEqual(find_best_rotation(grid212), 270)

    def test_grid212_with_mass_center_bottom(self):
        grid212 = np.array([[[True, False]],
                            [[True, False]]])
        self.assertEqual(find_best_rotation(grid212), 180)

    def test_grid212_with_mass_center_left(self):
        grid212 = np.array([[[True, True]],
                            [[False, False]]])
        self.assertEqual(find_best_rotation(grid212), 270)

    def test_grid212_with_mass_center_right(self):
        grid212 = np.array([[[False, False]],
                            [[True, True]]])
        self.assertEqual(find_best_rotation(grid212), 90)

    def test_grid212_with_m_center_at_center_denser_bottom_left_corner(self):
        grid212 = np.array([[[True, False]],
                            [[False, True]]])
        self.assertEqual(find_best_rotation(grid212), 180)

    def test_grid212_with_m_center_at_center_denser_bottom_right_corner(self):
        grid212 = np.array([[[False, True]],
                            [[True, False]]])
        self.assertEqual(find_best_rotation(grid212), 90)

    # grid 222

    def test_grid222_with_mass_center_bottom(self):
        grid222 = np.array([[[True, False], [True, False]],
                            [[True, False], [True, False]]])
        self.assertEqual(find_best_rotation(grid222), 180)

    def test_grid222_with_mass_center_bottom_left_dense(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, True], [True, False]]])
        self.assertEqual(find_best_rotation(grid222), 180)

    def test_grid222_with_mass_center_bottom_left_sparse(self):
        grid222 = np.array([[[False, False], [True, False]],
                            [[False, False], [False, False]]])
        self.assertEqual(find_best_rotation(grid222), 180)

    def test_grid222_with_mass_center_bottom_right(self):
        grid222 = np.array([[[False, False], [False, False]],
                            [[True, False], [True, False]]])
        self.assertEqual(find_best_rotation(grid222), 90)

    # grid 223

    def test_grid223_with_mass_center_bottom(self):
        grid223 = np.array([[[True, True, True], [True, False, False]],
                            [[True, True, False], [True, True, True]]])
        self.assertEqual(find_best_rotation(grid223), 180)

    def test_grid223_with_mass_center_right(self):
        grid223 = np.array([[[True, True, True], [False, False, True]],
                            [[True, True, True], [True, True, True]]])
        self.assertEqual(find_best_rotation(grid223), 90)

    def test_grid223_with_mass_center_top(self):
        grid223 = np.array([[[True, True, True], [False, False, True]],
                            [[True, True, True], [False, True, True]]])
        self.assertEqual(find_best_rotation(grid223), 0)

    def test_invalid_grid_with_2_dims(self):
        with self.assertRaises(KeyError):
            grid22 = np.array([[True for _ in range(2)] for _ in range(2)])
            find_best_rotation(grid22)


class Test_rotate_grid(unittest.TestCase):
    def test_grid111_rotate_0(self):
        grid111 = np.array([[[True]]])
        grid111_rot = np.array([[[True]]])
        self.assertTrue(np.all(rotate(grid=grid111, degrees=0) == grid111_rot))

    def test_grid111_rotate_90(self):
        grid111 = np.array([[[False]]])
        grid111_rot = np.array([[[False]]])
        self.assertTrue(
            np.all(rotate(grid=grid111, degrees=90) == grid111_rot))

    def test_grid111_rotate_180(self):
        grid111 = np.array([[[False]]])
        grid111_rot = np.array([[[False]]])
        self.assertTrue(
            np.all(rotate(grid=grid111, degrees=180) == grid111_rot))

    def test_grid111_rotate_270(self):
        grid111 = np.array([[[True]]])
        grid111_rot = np.array([[[True]]])
        self.assertTrue(
            np.all(rotate(grid=grid111, degrees=270) == grid111_rot))

    def test_grid222_rotate_0(self):
        grid222 = np.array([[[False, True], [True, False]],
                            [[True, True], [True, False]]])
        grid222_rot = np.array([[[False, True], [True, False]],
                                [[True, True], [True, False]]])
        self.assertTrue(np.all(rotate(grid=grid222, degrees=0) == grid222_rot))

    def test_grid222_rotate_90(self):
        grid222 = np.array([[[False, True], [True, False]],
                            [[True, True], [True, False]]])
        grid222_rot = np.array([[[True, False], [True, True]],
                                [[True, True], [False, False]]])
        self.assertTrue(
            np.all(rotate(grid=grid222, degrees=90) == grid222_rot))

    def test_grid222_rotate_180(self):
        grid222 = np.array([[[False, True], [True, False]],
                            [[True, True], [True, False]]])
        grid222_rot = np.array([[[True, True], [False, True]],
                                [[True, False], [False, True]]])
        self.assertTrue(
            np.all(rotate(grid=grid222, degrees=180) == grid222_rot))

    def test_grid222_rotate_270(self):
        grid222 = np.array([[[False, True], [True, False]],
                            [[True, True], [True, False]]])
        grid222_rot = np.array([[[True, True], [False, False]],
                                [[False, True], [True, True]]])
        self.assertTrue(
            np.all(rotate(grid=grid222, degrees=270) == grid222_rot))

    def test_grid323_rotate_90(self):
        grid323 = np.array([[[False, True, False], [False, True, False]],
                            [[True, False, False], [False, True, True]],
                            [[True, True, False], [True, True, True]]])
        grid323_rot = np.array([[[True, True, False], [True, False, False]],
                                [[True, False, True], [True, True, True]],
                                [[False, False, False], [True, True, False]]])
        self.assertTrue(
            np.all(rotate(grid=grid323, degrees=90) == grid323_rot))

    def test_grid323_rotate_180(self):
        grid323 = np.array([[[False, True, False], [False, True, False]],
                            [[True, False, False], [False, True, True]],
                            [[True, True, False], [True, True, True]]])
        grid323_rot = np.array([[[False, True, True], [True, True, True]],
                                [[False, False, True], [True, True, False]],
                                [[False, True, False], [False, True, False]]])
        self.assertTrue(
            np.all(rotate(grid=grid323, degrees=180) == grid323_rot))

    def test_grid323_rotate_270(self):
        grid323 = np.array([[[False, True, False], [False, True, False]],
                            [[True, False, False], [False, True, True]],
                            [[True, True, False], [True, True, True]]])
        grid323_rot = np.array([[[False, False, False], [False, True, True]],
                                [[True, False, True], [True, True, True]],
                                [[False, True, True], [False, False, True]]])
        self.assertTrue(
            np.all(rotate(grid=grid323, degrees=270) == grid323_rot))

    def test_grid213_rotate_90(self):
        grid213 = np.array([[[False, True, False]],
                            [[True, False, False]]])
        grid213_rot = np.array([[[True, False]],
                                [[False, True]],
                                [[False, False]]])
        self.assertTrue(
            np.all(rotate(grid=grid213, degrees=90) == grid213_rot))

    def test_grid213_rotate_180(self):
        grid213 = np.array([[[False, True, False]],
                            [[True, False, False]]])
        grid213_rot = np.array([[[False, False, True]],
                                [[False, True, False]]])
        self.assertTrue(
            np.all(rotate(grid=grid213, degrees=180) == grid213_rot))

    def test_grid213_rotate_270(self):
        grid213 = np.array([[[False, True, False]],
                            [[True, False, False]]])
        grid213_rot = np.array([[[False, False]],
                                [[True, False]],
                                [[False, True]]])
        self.assertTrue(
            np.all(rotate(grid=grid213, degrees=270) == grid213_rot))

    def test_invalid_rotation(self):
        with self.assertRaises(Exception):
            grid111 = np.array([[[True]]])
            rotate(grid=grid111, degrees=45)


class Test_get_subgrid(unittest.TestCase):
    def test_subgrid_111_from_111(self):
        grid111 = np.array([[[True]]])
        position = (0, 0, 0)
        shape = np.array([1, 1, 1])
        expected111 = np.array([[[True]]])
        subgrid111 = get_subgrid(grid111, position, shape)
        self.assertTrue(np.all(subgrid111 == expected111))

    def test_subgrid_111_from_222(self):
        grid222 = np.array([[[True, True], [False, True]],
                            [[True, True], [True, True]]])
        position = (0, 1, 0)
        shape = np.array([1, 1, 1])
        expected111 = np.array([[[False]]])
        subgrid111 = get_subgrid(grid222, position, shape)
        self.assertTrue(np.all(subgrid111 == expected111))

    def test_subgrid_112_from_222(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        position = (1, 0, 0)
        shape = np.array([1, 1, 2])
        expected112 = np.array([[[True, False]]])
        subgrid112 = get_subgrid(grid222, position, shape)
        self.assertTrue(np.all(subgrid112 == expected112))

    def test_subgrid_122_from_222(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        position = (0, 0, 0)
        shape = np.array([1, 2, 2])
        expected122 = np.array([[[True, True], [True, True]]])
        subgrid122 = get_subgrid(grid222, position, shape)
        self.assertTrue(np.all(subgrid122 == expected122))

    def test_subgrid_222_from_222(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        position = (0, 0, 0)
        shape = np.array([2, 2, 2])
        expected222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
        subgrid222 = get_subgrid(grid222, position, shape)
        self.assertTrue(np.all(subgrid222 == expected222))

    def test_invalid_shape_and_position(self):
        with self.assertRaises(IndexError):
            grid222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
            position = (1, 1, 1)
            shape = np.array([1, 1, 2])
            get_subgrid(grid222, position, shape)

    def test_invalid_too_big_shape(self):
        with self.assertRaises(IndexError):
            grid222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
            position = (0, 0, 0)
            shape = np.array([1, 3, 1])
            get_subgrid(grid222, position, shape)

    def test_invalid_too_big_position(self):
        with self.assertRaises(IndexError):
            grid222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
            position = (0, 2, 0)
            shape = np.array([1, 1, 1])
            get_subgrid(grid222, position, shape)

    def test_invalid_too_small_position(self):
        with self.assertRaises(IndexError):
            grid222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
            position = (-1, 1, 0)
            shape = np.array([1, 1, 1])
            get_subgrid(grid222, position, shape)


class Test_extend(unittest.TestCase):
    def test_divisibility_222_by_111(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        divider = np.array([1, 1, 1])
        expected222 = np.array([[[True, True], [True, True]],
                                [[True, False], [True, True]]])
        ext_grid222 = provide_divisibility(grid222, divider)
        self.assertTrue(np.all(ext_grid222 == expected222))

    def test_divisibility_222_by_131(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        divider = np.array([1, 3, 1])
        expected232 = np.array([[[True, True], [True, True], [False, False]],
                                [[True, False], [True, True], [False, False]]])
        ext_grid232 = provide_divisibility(grid222, divider)
        self.assertTrue(np.all(ext_grid232 == expected232))

    def test_divisibility_222_by_342(self):
        grid222 = np.array([[[True, True], [True, True]],
                            [[True, False], [True, True]]])
        divider = np.array([3, 4, 2])
        expected342 = np.array([
            [[True, True], [True, True], [False, False], [False, False]],
            [[True, False], [True, True], [False, False], [False, False]],
            [[False, False], [False, False], [False, False], [False, False]]])
        ext_grid342 = provide_divisibility(grid222, divider)
        self.assertTrue(np.all(ext_grid342 == expected342))

    def test_divisibility_12_12_12_by_3_4_5(self):
        grid = np.random.choice([True, False], size=(12, 12, 12))
        divider = np.array([3, 4, 5])
        expected_shape = np.array((12, 12, 15))
        ext_grid = provide_divisibility(grid, divider)
        self.assertTrue(np.all(ext_grid.shape == expected_shape))


class Test_get_fill_ratio(unittest.TestCase):
    def test_grid_111_fill_0(self):
        grid = np.array([[[False]]])
        self.assertEqual(get_fill_ratio(grid), 0)

    def test_grid_111_fill_1(self):
        grid = np.array([[[True]]])
        self.assertEqual(get_fill_ratio(grid), 1)

    def test_grid_222_fill_0(self):
        grid = np.array([[[False, False], [False, False]],
                         [[False, False], [False, False]]])
        self.assertEqual(get_fill_ratio(grid), 0)

    def test_grid_222_fill_1(self):
        grid = np.array([[[True, True], [True, True]],
                         [[True, True], [True, True]]])
        self.assertEqual(get_fill_ratio(grid), 1)

    def test_grid_222_fill_05(self):
        grid = np.array([[[True, False], [False, True]],
                         [[False, False], [True, True]]])
        self.assertEqual(get_fill_ratio(grid), 0.5)

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
        self.assertEqual(get_fill_ratio(grid), 0.6)


class Test_from_pv_voxels(unittest.TestCase):
    def setUp(self) -> None:
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        mesh = reader.read()
        self.pv_voxels1 = pv.voxelize(mesh, density=0.2, check_surface=False)
        self.pv_voxels2 = voxelization.from_mesh(
            mesh, voxel_mesh_shape=np.array([0.2, 0.2, 0.2]))

    def test_result_not_None1(self):
        self.assertIsNotNone(from_pv_voxels(self.pv_voxels1))

    def test_result_not_None2(self):
        self.assertIsNotNone(from_pv_voxels(self.pv_voxels2))

    def test_result_is_ndarray1(self):
        self.assertTrue(
            isinstance(from_pv_voxels(self.pv_voxels1), np.ndarray))

    def test_result_is_ndarray2(self):
        self.assertTrue(
            isinstance(from_pv_voxels(self.pv_voxels2), np.ndarray))

    def test_number_of_cells_is_correct1(self):
        grid = from_pv_voxels(self.pv_voxels1)
        n_voxels = 0
        for v in np.nditer(grid):
            if v:
                n_voxels += 1
        self.assertEqual(self.pv_voxels1.n_cells, n_voxels)

    def test_number_of_cells_is_correct2(self):
        grid = from_pv_voxels(self.pv_voxels2)
        n_voxels = 0
        for v in np.nditer(grid):
            if v:
                n_voxels += 1
        self.assertEqual(self.pv_voxels2.n_cells, n_voxels)

    def test_all_cells_are_same_shape1(self):
        first_cell_shape = umesh.get_resolution(
            self.pv_voxels1.extract_cells(0))
        for ind in range(self.pv_voxels1.n_cells):
            cell_shape = umesh.get_resolution(
                self.pv_voxels1.extract_cells(ind))
            self.assertTrue(np.allclose(cell_shape, first_cell_shape))

    def test_all_cells_are_same_shape2(self):
        first_cell_shape = umesh.get_resolution(
            self.pv_voxels2.extract_cells(0))
        for ind in range(self.pv_voxels2.n_cells):
            cell_shape = umesh.get_resolution(
                self.pv_voxels2.extract_cells(ind))
            self.assertTrue(np.allclose(cell_shape, first_cell_shape))


class Test_from_mesh(unittest.TestCase):
    def setUp(self):
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        self.mesh = reader.read()

    def test_result_not_None(self):
        self.assertIsNotNone(
            from_mesh(self.mesh, voxel_mesh_shape=np.array([1, 1, 1])))

    def test_result_is_ndarray(self):
        self.assertIsInstance(
            from_mesh(self.mesh,
                      voxel_mesh_shape=np.array([1, 1, 1])), np.ndarray)


class Test_add_padding(unittest.TestCase):
    def test_111_grid_padding_1_correct_values(self):
        grid = np.array([[[True]]])
        excepted = np.array(
            [[[False, False, False],
              [False, False, False],
              [False, False, False]],
             [[False, False, False],
              [False, True, False],
              [False, False, False]],
             [[False, False, False],
              [False, False, False],
              [False, False, False]]])
        self.assertTrue(
            np.all(add_padding(grid, np.array([1, 1, 1])) == excepted))

    def test_112_grid_padding_1_correct_values(self):
        grid = np.array([[[True, False]]])
        expected = np.zeros((3, 3, 4), dtype=bool)
        expected[1, 1, 1] = True
        expected[1, 1, 2] = False
        self.assertTrue(
            np.all(add_padding(grid, np.array([1, 1, 1])) == expected))

    def test_231_grid_padding_2_correct_shape(self):
        grid = np.random.randint(2, size=(2, 3, 1)).astype(bool)
        grid_with_padding = add_padding(grid, np.array([2, 2, 2]))
        self.assertTrue(np.all(grid_with_padding.shape == np.array([6, 7, 5])))

    def test_231_grid_padding_2_correct_values(self):
        grid = np.random.randint(2, size=(2, 3, 1)).astype(bool)
        grid_with_padding = add_padding(grid, np.array([2, 2, 2]))

        for i in range(2):
            for j in range(3):
                for k in range(1):
                    self.assertEqual(
                        grid[i, j, k], grid_with_padding[i+2, j+2, k+2])

        for i in range(6):
            for j in range(7):
                self.assertFalse(grid_with_padding[i, j, 0])
                self.assertFalse(grid_with_padding[i, j, 1])
                self.assertFalse(grid_with_padding[i, j, 3])
                self.assertFalse(grid_with_padding[i, j, 4])

        for i in range(6):
            for k in range(5):
                self.assertFalse(grid_with_padding[i, 0, k])
                self.assertFalse(grid_with_padding[i, 1, k])
                self.assertFalse(grid_with_padding[i, 5, k])
                self.assertFalse(grid_with_padding[i, 6, k])

        for j in range(7):
            for k in range(5):
                self.assertFalse(grid_with_padding[0, j, k])
                self.assertFalse(grid_with_padding[1, j, k])
                self.assertFalse(grid_with_padding[4, j, k])
                self.assertFalse(grid_with_padding[5, j, k])


if __name__ == '__main__':
    unittest.main()
