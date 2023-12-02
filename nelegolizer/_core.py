import pyvista as pv

def legolize(path, res):
    # read mesh from file
    reader = pv.get_reader(path)
    mesh = reader.read()

