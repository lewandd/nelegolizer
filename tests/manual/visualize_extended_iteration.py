from torch import nn
import os
import torch
import pyvista as pv
import numpy as np
from typing import Tuple, List
from nelegolizer.data import LDrawFile, initilize_parts, ObjectOccupancy, BrickCoverage
from nelegolizer.data._GeometryCoverage import GeometryCoverage
from nelegolizer import const
import nelegolizer.utils.voxelization as uvox
from nelegolizer.model.dataset_generation import make_samples
from nelegolizer.data._BrickCoverage import ext_bu_to_vu

# ----------------------------------------------------------------------------
# load LDraw as LegoBrick list

initilize_parts()

filenames = ["fixtures/church.mpd"]

ldf = LDrawFile.load(filenames[0])
lbm = ldf.models[0]
bricks = lbm.as_bricks()

shapes = [(2,3,2), (2,3,1), (1,3,1), (1,1,1)]
bo = BrickCoverage.from_bricks(bricks)
oo = GeometryCoverage(bo.voxel_grid)

#brick = bo.get_brick_at(bricks, (2, 6, 2))
#brick = bo.get_brick_at(bricks, (2, 8, 2))
#if brick is not None:
#    bo.remove_brick(brick)
#    print("Mamy klocka do usunięcia")

for brick in bricks:
    if brick.part.id == "3024":
        print(bo.get_brick_position(brick))

print("grid shape:", oo.brick_grid.shape)
for shape in shapes:
    print("current shape:", shape)
    
    for y in range(oo.brick_grid.shape[1] - shape[1], -1, -1):
        for x in range(oo.brick_grid.shape[0] - shape[0] + 1):
            for z in range(oo.brick_grid.shape[2] - shape[2] + 1):
                pos = np.array([x, y, z])
                # TODO : do zmiany
                vu_pos = ext_bu_to_vu(pos)#+np.array([0, 2, 0])
                vu_shape = ext_bu_to_vu(shape) + 2*const.PADDING
                patch = oo.brick_grid[x:x+shape[0], y:y+shape[1], z:z+shape[2]]

                print("------------------------------------------------------")
                print(f"possition {pos}")
                #print(f"{patch}")
                
                channel1 = oo.ext_voxel_grid[vu_pos[0]:vu_pos[0]+vu_shape[0],
                                        vu_pos[1]:vu_pos[1]+vu_shape[1],
                                        vu_pos[2]:vu_pos[2]+vu_shape[2]]

                temp_bo = BrickCoverage.from_bricks(bricks)

                # TODO iteracja powinna iść o jedno dalej niż wielkość całości?
                # nie, jeżeli się nie mieści, to... nie no, powinno się powiększyć cały grid
                # o 1 klocek w przestrzeni X Z, podczas właściwej iteracji też
                # chcemy zawsze... ale właśnie trzeba zaimplementiować mechanizm
                # gdzie nie ma miejsca na ten duży shape i trzeba zmienić go na mniejszy
                # dla danego miejsca... a zatem... ale też nie

                # TODO tak czy inaczej na krańcach duża grupa się nie zmieści, a
                # jeżeli zostanie zmniejszona to nadal te krańcowe fragmenty nie zostaną 
                # pokryte

                # TODO zmienione, trzeba zrobić po prostu zmienną o nazwie seleceted location

                brick = temp_bo.get_brick_at(bricks, (pos[0], pos[1]+shape[1]-2, pos[2]))
                label = 0
                if brick is not None:
                    print(f"removed brick {brick.part.id}")
                    label = brick.id
                    temp_bo.remove_brick(brick)

                channel2 = temp_bo.ext_voxel_grid[vu_pos[0]:vu_pos[0]+vu_shape[0],
                                            vu_pos[1]:vu_pos[1]+vu_shape[1],
                                            vu_pos[2]:vu_pos[2]+vu_shape[2]]

                # -------------------------------------------------------------
                # plotting below

                plotter = pv.Plotter(shape=(1, 3))

                # subplot - whole voxels scene with moving kernel
                plotter.subplot(0, 0)
                active_mesh = None
                inactive_mesh = None
                try:
                    active_mesh = uvox.from_grid(np.array(patch), voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
                except Exception:
                    pass
                try:
                    inactive_mesh = uvox.from_grid(np.array(~patch), voxel_mesh_shape=const.BRICK_UNIT_MESH_SHAPE)
                except Exception:
                    pass
                if active_mesh is not None:
                    active_mesh.translate(np.array([x, y, z])*const.BRICK_UNIT_MESH_SHAPE, inplace=True)
                    plotter.add_mesh(active_mesh, show_edges=True, color="red", opacity=0.5)
                if inactive_mesh is not None:
                    inactive_mesh.translate(np.array([x, y, z])*const.BRICK_UNIT_MESH_SHAPE, inplace=True)
                    plotter.add_mesh(inactive_mesh, show_edges=True, color="white", opacity=0.5)
                
                voxels_scene_mesh = uvox.from_grid(bo.voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
                voxels_scene_mesh.translate(np.array([-0.32, -0.32, -0.32]), inplace=True)
                plotter.add_mesh(voxels_scene_mesh, show_edges=True, color="blue")

                # subplot - choosed voxels from oo
                plotter.subplot(0, 1)
                oo_choosed_voxels_mesh = None
                try:
                    oo_choosed_voxels_mesh = uvox.from_grid(channel1, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
                    plotter.add_mesh(oo_choosed_voxels_mesh, show_edges=True, color="red", opacity=0.5)
                except Exception:
                    pass

                # subplot - choosed voxels from bo
                plotter.subplot(0, 2)
                bo_choosed_voxels_mesh = None
                try:
                    bo_choosed_voxels_mesh = uvox.from_grid(channel2, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
                    plotter.add_mesh(bo_choosed_voxels_mesh, show_edges=True, color="red", opacity=0.5)
                except Exception:
                    pass
                
                cpos = [(4, -11.5, 4), 
                        (1.28, 0.64, 1.28), 
                        (-1, -0.17, 0.16)]
                cpos2 = [(-3.15, -14.13, -4.28), 
                        (2.00, 1.36, 2.4), 
                        (0.37, -0.47, 0.8)]
                cpos_moving = [(4+x*const.BRICK_UNIT_MESH_SHAPE[0], -11.5+y*const.BRICK_UNIT_MESH_SHAPE[1], 4+z*const.BRICK_UNIT_MESH_SHAPE[2]), 
                        (1.28+x*const.BRICK_UNIT_MESH_SHAPE[0], 0.64+y*const.BRICK_UNIT_MESH_SHAPE[1], 1.28+z*const.BRICK_UNIT_MESH_SHAPE[2]), 
                        (-1, -0.17, 0.16)]
                plotter.subplot(0, 0)
                plotter.camera_position = cpos2#_moving
                plotter.subplot(0, 1)
                plotter.camera_position = cpos2
                plotter.subplot(0, 2)
                plotter.camera_position = cpos2
                plotter.subplot(0, 0)
                plotter.show(window_size=(1800, 800))