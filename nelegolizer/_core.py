import pyvista as pv
import nelegolizer.voxel as vox
import nelegolizer.constants as CONST

def legolize(path, target_res):
    res = target_res * CONST.GROUP_RES
    
    # read mesh from file
    reader = pv.get_reader(path)
    mesh = reader.read()

    # voxelize and get grid
    voxels = vox.voxelize.from_mesh(mesh, res, 1)
    raw_grid = vox.into_grid(voxels.cell_centers().points, res)