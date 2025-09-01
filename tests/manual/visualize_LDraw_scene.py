from nelegolizer.data import LDrawFile, initilize_parts
import pyvista as pv

initilize_parts()

filenames = ["fixtures/church.mpd"]

ldf = LDrawFile.load(filenames[0])
lbm = ldf.models[0]
lb_list = lbm.as_bricks()


plotter = pv.Plotter()
for brick in lb_list:
    if brick.id == "3005":
        plotter.add_mesh(brick.mesh, color="white")
    if brick.id == "3004":
        plotter.add_mesh(brick.mesh, color="#c0c0c0")
    if brick.id == "54200":
        plotter.add_mesh(brick.mesh, color="#808069")
    if brick.id == "3024":
        plotter.add_mesh(brick.mesh, color="#778899")
plotter.show()