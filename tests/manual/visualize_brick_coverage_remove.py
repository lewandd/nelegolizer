from nelegolizer.data import LDrawFile, initilize_parts
import pyvista as pv
import nelegolizer.utils.voxelization as uvox
from nelegolizer import const
from nelegolizer.data import BrickCoverage
import random
import numpy as np

initilize_parts()

filenames = ["fixtures/church.mpd"]

ldf = LDrawFile.load(filenames[0])
ldm = ldf.models[0]
bricks = ldm.as_bricks()

bo = BrickCoverage.from_bricks(bricks)

BRICK_COLORS = {
    "3005": "white",
    "3004": "#c0c0c0",
    "54200": "#808069",
    "3024": "#778899"
}

def choose_brick_to_remove():
    return random.choice(bricks)

def get_occupancy_mesh(brick_to_remove):
    nbo = BrickCoverage.from_bricks(bricks)
    nbo.remove_brick(brick_to_remove)
    nbo.ext_voxel_grid[15, 0:2, 15] = True
    nbo.ext_voxel_grid[15, -1, 15] = True
    nbo.ext_voxel_grid[15, -2, 15] = True
    nbo.voxel_grid[15, 0:2, 15] = True
    nbo.voxel_grid[15, -1, 15] = True
    nbo.voxel_grid[15, -2, 15] = True
    
    voxels = uvox.from_grid(nbo.ext_voxel_grid, voxel_mesh_shape=const.VOXEL_MESH_SHAPE)
    return voxels

def add_all_bricks_to_plotter(plotter, brick_to_highlight=None):
    """Dodaje wszystkie klocki z kolorami do sceny."""
    for brick in bricks:
        col = BRICK_COLORS.get(brick.id, "white")  # domyślnie biały
        if brick_to_highlight is not None and np.allclose(brick.position, brick_to_highlight.position):
            col = "red"  # wyróżnienie klocka
        plotter.add_mesh(brick.mesh, color=col)

def update_scene(plotter):
    """Aktualizuje obiekt occupancy i rysuje całą scenę od nowa."""
    global actor, brick_to_remove
    # Usuń stary occupancy mesh
    plotter.remove_actor(actor)
    # Wybierz klocek i narysuj occupancy
    brick_to_remove = choose_brick_to_remove()
    occupancy_mesh = get_occupancy_mesh(brick_to_remove)
    actor = plotter.add_mesh(occupancy_mesh, show_edges=True, color="white")
    # Dodaj wszystkie klocki
    add_all_bricks_to_plotter(plotter, brick_to_highlight=brick_to_remove)
    plotter.render()

def close_plotter():
    plotter.close()

# --- Inicjalizacja sceny ---
plotter = pv.Plotter()
brick_to_remove = choose_brick_to_remove()
occupancy_mesh = get_occupancy_mesh(brick_to_remove)

actor = plotter.add_mesh(occupancy_mesh, show_edges=True, color="white")
add_all_bricks_to_plotter(plotter, brick_to_highlight=brick_to_remove)

# Obsługa klawiszy
plotter.add_key_event("space", lambda: update_scene(plotter))  # spacja -> nowy obiekt
plotter.add_key_event("escape", close_plotter)                # esc -> wyjście

plotter.show()