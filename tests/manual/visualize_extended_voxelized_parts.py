from nelegolizer.data.voxelized_parts import ext_part_grid, ext_part_grid2, ext_stud_grid, part3004_rot_90, part_54200_rot_90, part_54200_rot_180, part_54200_rot_270
import nelegolizer.utils.voxelization as uvox
import pyvista as pv
import numpy as np
from nelegolizer.constants import VU

grids = [ext_part_grid2["3023"][0], ext_part_grid2["3023"][90], ext_part_grid2["3023"][180], ext_part_grid2["3023"][270]]

#def xyz():
plotter = pv.Plotter(shape=(2,2))

plotter.subplot(0,0)
active_mesh = uvox.from_grid(np.array(grids[0]), voxel_mesh_shape=VU)
inactive_mesh = uvox.from_grid(np.array(~grids[0]), voxel_mesh_shape=VU)

plotter.add_mesh(active_mesh, show_edges=True, color="red", opacity=0.5)
plotter.add_mesh(inactive_mesh, show_edges=True, color="white", opacity=0.5)

plotter.subplot(0,1)
active_mesh = uvox.from_grid(np.array(grids[1]), voxel_mesh_shape=VU)
inactive_mesh = uvox.from_grid(np.array(~grids[1]), voxel_mesh_shape=VU)

plotter.add_mesh(active_mesh, show_edges=True, color="red", opacity=0.5)
plotter.add_mesh(inactive_mesh, show_edges=True, color="white", opacity=0.5)

plotter.subplot(1,0)
active_mesh = uvox.from_grid(np.array(grids[2]), voxel_mesh_shape=VU)
inactive_mesh = uvox.from_grid(np.array(~grids[2]), voxel_mesh_shape=VU)

plotter.add_mesh(active_mesh, show_edges=True, color="red", opacity=0.5)
plotter.add_mesh(inactive_mesh, show_edges=True, color="white", opacity=0.5)

plotter.subplot(1,1)
active_mesh = uvox.from_grid(np.array(grids[3]), voxel_mesh_shape=VU)
inactive_mesh = uvox.from_grid(np.array(~grids[3]), voxel_mesh_shape=VU)

plotter.add_mesh(active_mesh, show_edges=True, color="red", opacity=0.5)
plotter.add_mesh(inactive_mesh, show_edges=True, color="white", opacity=0.5)

plotter.show()

def fun2():

    for grid in grids:
        active_mesh = uvox.from_grid(np.array(grid), voxel_mesh_shape=VU)
        inactive_mesh = uvox.from_grid(np.array(~grid), voxel_mesh_shape=VU)
                    
        plotter = pv.Plotter()

        plotter.add_mesh(active_mesh, show_edges=True, color="red", opacity=0.5)
        plotter.add_mesh(inactive_mesh, show_edges=True, color="white", opacity=0.5)
                    
    #voxels_scene_mesh = uvox.from_grid(bo.voxel_grid, voxel_mesh_shape=VU)
    #voxels_scene_mesh.translate(np.array([-0.32, -0.32, -0.32]), inplace=True)
    #plotter.add_mesh(voxels_scene_mesh, show_edges=True, color="blue")

    plotter.show()