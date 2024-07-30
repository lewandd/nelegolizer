import pyvista as pv
import numpy as np

def get_resolution(mesh: pv.PolyData) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    return np.array([xmax-xmin, ymax-ymin, zmax-zmin])

def get_position(mesh: pv.PolyData) -> np.ndarray:
    xmin, _, ymin, _, zmin, _ = mesh.bounds
    return np.array([xmin, ymin, zmin])

def translate_to_zero(mesh: pv.PolyData) -> pv.PolyData:
    translate = (-1) * get_position(mesh)
    return mesh.translate(translate)

def scale_to(mesh: pv.PolyData, 
             target_res: np.ndarray,
             keep_ratio: bool = False) -> pv.PolyData:
    mesh_res = get_resolution(mesh)
    if keep_ratio:
        ratio = min(target_res/mesh_res)
    else:
        ratio = target_res/mesh_res
    return mesh.scale(ratio)