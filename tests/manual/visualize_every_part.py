from nelegolizer.data import  initilize_parts, part_by_id
import nelegolizer.utils.voxelization as uvox
import nelegolizer.utils.grid as ugrid
import pyvista as pv
from nelegolizer import const
from nelegolizer.data.voxelized_parts import ext_part_grid2

initilize_parts()

mesh_translate = {
    "3005": [0.4,0.16,0.4],
    "3004": [0.8,0.16,0.4],
    "54200": [0.4,1.12,0.4],
    "3024": [0.4,0.16,0.4],
    }


#part54200_rot0 = part_by_id["54200"].grid
#part54200_rot90 = ugrid.rotate(part54200_rot0, 90)
#part54200_rot180 = ugrid.rotate(part54200_rot0, 180)
#part54200_rot270 = ugrid.rotate(part54200_rot0, 270)

#print("90", part54200_rot90)
#print("180", part54200_rot180)
#print("270", part54200_rot270)
plotter = pv.Plotter()

voxels = uvox.from_grid(ext_part_grid2["3005"][0], voxel_mesh_shape=const.VOXEL_MESH_SHAPE)


#plotter2 = pv.Plotter()
plotter.add_mesh(voxels, show_edges=True, opacity=0.9)
plotter.show()

def temp():

    # sprawdzenie wokselizacji w part
    for id in part_by_id:
        part = part_by_id[id]
        grid = part.grid
        mesh = part.mesh

        print(f"id: {id} grid_shape: {grid.shape}")

        # moving to (0, 0, 0) position
        mesh = mesh.translate(mesh_translate[part.id])

        voxels = uvox.from_grid(grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)

        plotter2 = pv.Plotter()
        plotter2.add_mesh(voxels, show_edges=True, opacity=0.9)
        plotter2.add_mesh(mesh, color="white")
        plotter2.show()