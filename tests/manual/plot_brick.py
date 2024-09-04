import pyvista as pv

reader = pv.get_reader("fixtures/3005.stl")
mesh = reader.read()

plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show_bounds()
plotter.show()
