import pyvista as pv
import nelegolizer.voxel.voxelizer as vox
from nelegolizer.utils import grid
import nelegolizer.constants as CONST

def legolize(path, target_res):
    res = target_res * CONST.GROUP_RES
    
    # read mesh from file
    reader = pv.get_reader(path)
    mesh = reader.read()

    # voxelize and get grid
    voxels = vox.voxelize_from_mesh(mesh, res, 1)
    raw_grid = grid.get_grid_from_voxels(voxels.cell_centers().points, res)