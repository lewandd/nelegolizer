"""
Tests nelegolize.utils.voxelization module
"""
import unittest
import numpy as np
import pyvista as pv

from nelegolizer.utils.voxelization import from_mesh, from_grid
from nelegolizer.utils import mesh as umesh

class Test_from_mesh(unittest.TestCase):
    def setUp(self):
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        self.mesh = reader.read()

    def test_returns_PolData(self):
        self.assertIsInstance(from_mesh(self.mesh, voxel_mesh_shape=np.array([1, 1, 1])), pv.UnstructuredGrid)

class Test_from_grid(unittest.TestCase):
    def setUp(self):
        reader = pv.get_reader("tests/unit/fixtures/cone.obj")
        self.mesh = reader.read()

    def test_returns_ndarray_111(self):
        grid = np.array([[[True]]])
        self.assertIsInstance(from_grid(grid, voxel_mesh_shape=np.array([1, 1, 1])), pv.UnstructuredGrid)

    def test_returns_ndarray_222(self):
        grid = np.array([[[False, False], [False, False]], [[True, False], [False, False]]])
        self.assertIsInstance(from_grid(grid, voxel_mesh_shape=np.array([1, 1, 1])), pv.UnstructuredGrid)

    def test_correct_number_of_cells_1(self):
        grid = np.array([[[False, False], [False, False]], [[True, False], [False, False]]])
        self.assertEqual(from_grid(grid, voxel_mesh_shape=np.array([1, 1, 1])).n_cells, 1)

    def test_correct_number_of_cells_3(self):
        grid = np.array([[[True, True], [False, False]], [[True, False], [False, False]]])
        self.assertEqual(from_grid(grid, voxel_mesh_shape=np.array([1, 1, 1])).n_cells, 3)

    def test_correct_cell_shape(self):
        grid = np.array([[[True, True], [False, False]], [[True, False], [False, False]]])
        shape = np.array([1, 1.12, 1])
        pv_voxels = from_grid(grid, voxel_mesh_shape=shape)
        result_cell_shape = umesh.get_resolution(pv_voxels.extract_cells(0))
        self.assertTrue(np.allclose(result_cell_shape, shape))

    def test_all_cells_same_shape(self):
        grid = np.array([[[True, True], [False, False]], [[True, False], [False, False]]])
        shape = np.array([1, 1.12, 1])
        pv_voxels = from_grid(grid, voxel_mesh_shape=shape)
        for ind in range(pv_voxels.n_cells):
            cell_shape = umesh.get_resolution(pv_voxels.extract_cells(ind))
            self.assertTrue(np.allclose(cell_shape, shape))