from torch import nn
import os
import torch
import pyvista as pv
import numpy as np
from typing import Tuple, List
from nelegolizer.data import LDrawFile, initilize_parts, ObjectOccupancy, BrickCoverage, LegoBrick
from nelegolizer.data._GeometryCoverage import GeometryCoverage
from nelegolizer import const
import nelegolizer.utils.voxelization as uvox
from nelegolizer.model.dataset_generation import make_samples
from nelegolizer.data._BrickCoverage import ext_bu_to_vu
from nelegolizer.utils.grid import get_fill, bu_to_mesh
import copy

# ----------------------------------------------------------------------------
# load LDraw as LegoBrick list

initilize_parts()

filenames = ["fixtures/church.mpd"]

ldf = LDrawFile.load(filenames[0])
lbm = ldf.models[0]
bricks = lbm.as_bricks()

def find_next_to_cover(cover_grid, covered_grid):
    to_cover_mask = np.logical_and(cover_grid, np.logical_not(covered_grid))
    indices = np.argwhere(to_cover_mask)
    sorted_indices = indices[np.lexsort((indices[:,0], indices[:,2], -indices[:,1]))]
    return sorted_indices[0] if len(sorted_indices) > 0 else None

shapes3 = []

filled_bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, top_extension=4, side_extension=1)
gc = GeometryCoverage(filled_bc.voxel_grid, bottom_extension=3, top_extension=4, side_extension=1)
training_bc = BrickCoverage(gc.interior_shape, bottom_extension=3, top_extension=4, side_extension=1)
training_bc.pos_min = 0#filled_bc.pos_min
training_bc.pos_max = 0#filled_bc.pos_max

analyzed_grid = np.zeros_like(training_bc.brick_grid)
next_to_cover = find_next_to_cover(gc.brick_grid, analyzed_grid)
num_found = 0
num_all = 0

shapes = [(3, 3, 3), (1, 3, 1), (1, 1, 1)]

while next_to_cover is not None:
    shape = np.array([3, 3, 3])
    shape_side_ext = int((shape[0]-1)/2)
    shape_top_ext = int(shape[1]-1)

    x, y, z = next_to_cover
    ext_vu_pos = ext_bu_to_vu(np.array([x, y, z]))

    # TODO SIECI ZORGANIZOWANE POD WZGLĘDEM KLOCKÓW KTÓRE NIE SĄ SWOIMI PODKLOCKAMI

    # TODO trzeba zrobić funkcję która sprawdza czy dany klocek posiada prawidłową
    # pozycję w gridzie, sprawdza się to poprzez wykonani funkcji analogicznej do
    # wklejenia tylko że zamiast paste_grid robi się get_subgrid i sprawdza się
    # czy cała tablica jest pusta
    # UWAGA : nie jest sprawdzana poprawność względem geometrii, gdyż jeżeli klocek
    # będzie wystawał, to będzie nieładnie ale nadal poprawnie
    # przy wprowadzaniu szumów do sieci może się zdarzyć, że klocek będzie wystawał
    # poza granicę geometrii, a nadal będzie pasował

    # TODO następnie tą funkcję trzeba wprowadzić i w procesie uczenia żeby przypadkiem
    # nie uczyć na niemożliwych sytuacjach w praktyce

    # TODO jeżeli żaden klocek się nie mieści, to wybierana jest mniejsza sieć

    # TODO jeżeli został zwrócony błędny klocek, który się nie mieści, to
    # wybierana jest mniejsza sieć

    # TODO jeżeli nic nie zostanie sklasyfikowane to znaczy, że geometria
    # nie odpowiada dużym klockom i wybierana jest mniejsza sieć

    selected_location = (x, y+shape[1]-2-shape_top_ext, z)
    print(f"selected location: {selected_location}")
    brick = filled_bc.get_brick_at(bricks, selected_location)
    label = 0
    if brick is not None:
        label = brick.id # wybierany z listy
        print(f"selected {brick.part.id} at position {selected_location}")
        placement_pos = np.array(selected_location) - np.array([training_bc.SIDE_EXT, training_bc.TOP_EXT-1 + brick.rotated_shape[1], training_bc.SIDE_EXT])
        if brick.id == "54200":
            placement_pos = placement_pos + np.array([0, 2, 0])
        placement_mesh_pos = bu_to_mesh(placement_pos)
        
        new_lb = LegoBrick(id=brick.id, mesh_position=placement_mesh_pos, rotation=brick.rotation)
        training_bc.place_brick(new_lb)
        num_found += 1
    else:
        label = 0
        analyzed_grid[x, y, z] = True
    num_all += 1

    analyzed_grid = np.logical_or(analyzed_grid, training_bc.brick_grid)
    next_to_cover = find_next_to_cover(gc.brick_grid, analyzed_grid)

    # grid kernel and channels

    patch = np.ones(shape=(3,3,3))

    
    ext_vu_pos = ext_bu_to_vu(np.array([x-shape_side_ext, y-shape_top_ext-1, z-shape_side_ext]))
    ext_vu_shape = ext_bu_to_vu(shape+np.array([0, 2, 0])) + 2*const.PADDING

    channel1 = gc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                 ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                 ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

    channel2 = training_bc.ext_voxel_grid[ext_vu_pos[0]:ext_vu_pos[0]+ext_vu_shape[0],
                                          ext_vu_pos[1]:ext_vu_pos[1]+ext_vu_shape[1],
                                          ext_vu_pos[2]:ext_vu_pos[2]+ext_vu_shape[2]]

    print(f"channel size: {channel1.shape}")

    plotter = pv.Plotter(shape=(2, 2))

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
        active_mesh.translate(np.array([x-1, y, z-1])*const.BRICK_UNIT_MESH_SHAPE, inplace=True)
        #plotter.add_mesh(active_mesh, show_edges=True, color="red", opacity=0.5)
    if inactive_mesh is not None:
        inactive_mesh.translate(np.array([x-1, y, z-1])*const.BRICK_UNIT_MESH_SHAPE, inplace=True)
        #plotter.add_mesh(inactive_mesh, show_edges=True, color="white", opacity=0.5)
    

    selection_mesh = uvox.from_grid(np.array([[[1]]]), voxel_mesh_shape=(np.array([6, 8, 6]))*const.VOXEL_MESH_SHAPE)
    selection_mesh.translate(np.array([6, 3, 6])*np.array([x, y-2, z])*const.VOXEL_MESH_SHAPE, inplace=True)
    plotter.add_mesh(selection_mesh, show_edges=True, color="blue", opacity=0.2)

    #voxels_scene_mesh = uvox.from_grid(bo.voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
    voxels_scene_mesh = uvox.from_grid(training_bc.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
    voxels_scene_mesh.translate(np.array([0, -0.16, 0]), inplace=True)
    plotter.add_mesh(voxels_scene_mesh, show_edges=True, color="white")

    # subplot - choosed voxels from oo
    plotter.subplot(0, 1)
    plotter.add_title("training_bc.brick_grid")
    mesh = uvox.from_grid(training_bc.brick_grid, voxel_mesh_shape=np.array([5, 3, 5]))
    plotter.add_mesh(mesh, show_edges=True, color="white")


    plotter.subplot(1, 0)
    plotter.add_title("channel1", 8)
    oo_choosed_voxels_mesh = None
    try:
        oo_choosed_voxels_mesh = uvox.from_grid(channel1, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
        plotter.add_mesh(oo_choosed_voxels_mesh, show_edges=True, color="red", opacity=0.5)
    except Exception:
        pass
    

    # subplot - choosed voxels from bo
    plotter.subplot(1, 1)
    plotter.add_title("channel2", 8)
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
    plotter.subplot(1, 0)
    plotter.camera_position = cpos2
    plotter.subplot(1, 1)
    plotter.camera_position = cpos2
    plotter.show(window_size=(1800, 800))


print(f"{get_fill(filled_bc.brick_grid)}")
print(f"Found {num_found} bricks blocks. All locations looked: {num_all}.")
