"""
Create lego bricks objects by hand and visualize.

Plot 2 lego bricks with custom parameters.
"""
import numpy as np
import pyvista as pv

from nelegolizer import const
from nelegolizer.data import LegoBrick

lego_brick1 = LegoBrick(
    label=0,
    mesh_position=np.array((0, 0, 0)) * const.BRICK_UNIT_MESH_SHAPE,
    rotation=0)
lego_brick2 = LegoBrick(
    label=1,
    mesh_position=np.array((2, 0, 0)) * const.BRICK_UNIT_MESH_SHAPE,
    rotation=90)

print("Brick size:")
print(f"{lego_brick1.part.brick_id} brick size: {lego_brick1.part.size}")
print(f"{lego_brick2.part.brick_id} brick size: {lego_brick2.part.size}")

print("Mesh size:")
print(f"{lego_brick1.part.brick_id} "
      "mesh size: {umesh.get_resolution(lego_brick1.mesh).round(2)}")
print(f"{lego_brick2.part.brick_id} "
      "mesh size: {umesh.get_resolution(lego_brick2.mesh).round(2)}")

print("Mesh position:")
print(f"{lego_brick1.part.brick_id} "
      "mesh position: {umesh.get_position(lego_brick1.mesh).round(2)}")
print(f"{lego_brick2.part.brick_id} "
      "mesh position: {umesh.get_position(lego_brick2.mesh).round(2)}")

pl = pv.Plotter()
pl.add_mesh(lego_brick1.mesh)
pl.add_mesh(lego_brick2.mesh)
pl.show_bounds()
pl.show()
