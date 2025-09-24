"""
Visualize basic legolize usage.

Plot input mesh and output list of LegoBricks.
"""
import pyvista as pv
from pathlib import Path

from nelegolizer.legolizer._core import legolize_from_mpd
from nelegolizer.paths import MODEL555, MODEL555CONFIG

reader = pv.get_reader("../../data/raw/obj_examples/cone.obj")
mesh = reader.read()
mesh = mesh.scale(4)

#plotter = pv.Plotter(shape=(1, 2))
#plotter.subplot(0, 0)
#plotter.add_title("cone.obj", 8)
#plotter.add_mesh(mesh)


#lego_bricks = legolize_from_mpd(mesh, )
#plotter.subplot(0, 1)
#plotter.add_title("legolized cone.obj", 8)
#for lb in lego_bricks:
#    plotter.add_mesh(lb.mesh)

#plotter.show()
