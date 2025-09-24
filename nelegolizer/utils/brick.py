import numpy as np
from .conversion import mesh_to_bu, bu_to_mesh
import copy
from typing import List
from ..data import LegoBrick

def compute_bounds(lb_list):
        mins = []
        maxs = []
        
        for b in lb_list:
            b_min = np.array(b.position)
            b_max = b_min + np.array(b.rotated_shape)
            mins.append(b_min)
            maxs.append(b_max)
        
        mins = np.min(mins, axis=0)
        maxs = np.max(maxs, axis=0)
        
        return mins, maxs

def normalize_positions(bricks: List[LegoBrick], offset: np.ndarray):
    mins, _ = compute_bounds(bricks)
    for brick in bricks:
        brick.mesh_position = bu_to_mesh((np.round(brick.position - mins)+offset).astype(int))

def rotate_bricks_y(bricks, k=1):
    """
    Obraca listę bricków o k*90° wokół osi Y.
    """
    angle = np.deg2rad(90 * k)
    c, s = np.cos(angle), np.sin(angle)
    rot_y = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ], dtype=int)

    rotated = []
    for brick in bricks:
        b = copy.deepcopy(brick)

        # obrót pozycji
        b.mesh_position = rot_y @ brick.mesh_position
        
        if b.part.size[0] > 1:
            if k == 1 and b.rotation == 0:
                bu_pos = mesh_to_bu(b.mesh_position)
                b.mesh_position = bu_to_mesh(bu_pos - np.array([0, 0, 1]))
            if k == 1 and b.rotation == 180:
                bu_pos = mesh_to_bu(b.mesh_position)
                b.mesh_position = bu_to_mesh(bu_pos - np.array([0, 0, 1]))
            if k == 2 and b.rotation == 0:
                bu_pos = mesh_to_bu(b.mesh_position)
                b.mesh_position = bu_to_mesh(bu_pos - np.array([1, 0, 0]))
            if k == 2 and b.rotation == 180:
                bu_pos = mesh_to_bu(b.mesh_position)
                b.mesh_position = bu_to_mesh(bu_pos - np.array([1, 0, 0]))
            if k == 2 and b.rotation == 270:
                bu_pos = mesh_to_bu(b.mesh_position)
                b.mesh_position = bu_to_mesh(bu_pos - np.array([0, 0, 1]))
            if k == 2 and b.rotation == 90:
                bu_pos = mesh_to_bu(b.mesh_position)
                b.mesh_position = bu_to_mesh(bu_pos - np.array([0, 0, 1]))
            if k == 3 and b.rotation == 90:
                bu_pos = mesh_to_bu(b.mesh_position)
                b.mesh_position = bu_to_mesh(bu_pos - np.array([1, 0, 0]))
            if k == 3 and b.rotation == 270:
                bu_pos = mesh_to_bu(b.mesh_position)
                b.mesh_position = bu_to_mesh(bu_pos - np.array([1, 0, 0]))

        if hasattr(brick, "rotation"):  
            b.rotation = (brick.rotation + 90 * k) % 360

        rotated.append(b)

    return rotated