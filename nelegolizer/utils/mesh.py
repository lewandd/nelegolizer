import pyvista as pv
import numpy as np

def scale_to_resolution(mesh, res):
    """Scale mesh in place to target resolution 
    
    Mesh proportions can be changed, by scaling different dimensions with different factors.

    Args:
        mesh (pyvista.PolyData) : mesh to scale
        res (list) : target resolution, (3) shape list of ints
    """

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    xlen, ylen, zlen = xmax-xmin, ymax-ymin, zmax-zmin
    xres, yres, zres = res
    mesh.scale([xres/xlen, yres/ylen, zres/zlen], inplace=True)
