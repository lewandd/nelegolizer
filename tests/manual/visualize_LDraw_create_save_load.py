"""
Visualize LDrawFile operations - creation from list of LegoBrick,
saving and loading.

Plot original legolize output - lego bricks, and lego bricks after
saving and loading from .mpd file.
"""
import pyvista as pv

from nelegolizer import legolize
from nelegolizer.data import LDrawFile, LDrawModel

# load mesh and scale by 4
reader = pv.get_reader("fixtures/cone.obj")
mesh = reader.read()
mesh = mesh.scale(4)

# legolize cone.obj
lego_bricks = legolize(mesh)

# save results as legolized_cone.mpd
ldraw_model = LDrawModel.from_bricks(lego_bricks, "Model Name")
ldraw_file = LDrawFile()
ldraw_file.add_model(ldraw_model)
ldraw_file.save("fixtures/legolized_cone.mpd")

# load legolized_cone.mpd and convert into lego bricks
loaded_ldraw_file = LDrawFile.load("fixtures/legolized_cone.mpd")
loaded_ldraw_model = LDrawModel.merge_multiple_models(loaded_ldraw_file.models)
loaded_lego_bricks = loaded_ldraw_model.as_bricks()

# plot
plotter = pv.Plotter(shape=(1, 3))
plotter.subplot(0, 0)
plotter.add_title("cone.obj", 8)
plotter.add_mesh(mesh)

plotter.subplot(0, 1)
plotter.add_title("legolized cone.obj", 8)
for lb in lego_bricks:
    plotter.add_mesh(lb.mesh)

plotter.subplot(0, 2)
plotter.add_title("loaded from mpd", 8)
for lb in loaded_lego_bricks:
    plotter.add_mesh(lb.mesh)
plotter.show()
