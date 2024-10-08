"""
Visualize basic legolize usage.

Plot input mesh and output list of LegoBricks.
"""
import pyvista as pv

from nelegolizer import legolize

reader = pv.get_reader("fixtures/cone.obj")
mesh = reader.read()
mesh = mesh.scale(4)

plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_title("cone.obj", 8)
plotter.add_mesh(mesh)

lego_bricks = legolize(mesh)
plotter.subplot(0, 1)
plotter.add_title("legolized cone.obj", 8)
for lb in lego_bricks:
    plotter.add_mesh(lb.mesh)

plotter.show()
