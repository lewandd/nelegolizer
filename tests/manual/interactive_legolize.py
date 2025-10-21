import pyvista as pv
import numpy as np
import yaml

from nelegolizer.data import LDrawFile, initilize_parts, BrickCoverage
from nelegolizer.utils.brick import compute_bounds
from nelegolizer.data._GeometryCoverage import GeometryCoverage
import nelegolizer.utils.voxelization as uvox
from nelegolizer.utils.conversion import bu_to_mesh, ext_bu_to_vu
from nelegolizer.constants import VU, BU
from nelegolizer.legolizer.iterator import find_next_pos_to_cover, place_brick, make_brick_variants
from nelegolizer.model.dataset_generation import get_brick_id_rotation
import copy

# ---- CONFIG / FILENAME ----
initilize_parts()
filename = "fixtures/church.mpd"
config_path = "../../configs/datasets/church.dataset.yaml"

# ---- LOAD DATA / PREPARE GRID ----
ldf = LDrawFile.load(filename)
lbm = ldf.models[0]
bricks = lbm.as_bricks()

mins, maxs = compute_bounds(bricks)
for brick in bricks:
    brick.mesh_position = bu_to_mesh((np.round(brick.position - mins) + np.array([2, 4, 2])).astype(int))
    #print(f"{brick.id} position {brick.position}")

with open(config_path) as f:
    config = yaml.safe_load(f)

filled_bc = BrickCoverage.from_bricks(bricks, bottom_extension=3, top_extension=4, side_extension=2)
interior_voxel_grid = filled_bc.voxel_grid[10:-10, 8:-6, 10:-10]

gc = GeometryCoverage(interior_voxel_grid, bottom_extension=3, top_extension=4, side_extension=2)
training_bc = BrickCoverage(gc.interior_shape, bottom_extension=3, top_extension=4, side_extension=2)

height_map = {net: conf['iteration']['height'] for net, conf in config['dataset']['subsets'].items()}
analyzed = {net: np.zeros_like(training_bc.brick_grid) for net in config['dataset']['subsets']}
ntcs = find_next_pos_to_cover(gc, training_bc, analyzed, height_map)

# ---- STATE ----
state = {
    "gc": gc,
    "filled_bc": filled_bc,
    "training_bc": training_bc,
    "config": config,
    "bricks": bricks,
    "height_map": height_map,
    "analyzed": analyzed,
    "ntcs": ntcs,
    "num_found": 0,
    "num_all": 0,
}

# ---- PLOTTING HELPERS ----
def make_selection_mesh(subset_used, x, y, z, shape):
    """Create selection box mesh matching your original logic."""
    if subset_used == 'subset1':
        shape_of_selection = np.array([5, 9, 5]) * VU
    elif subset_used == 'subset2':
        shape_of_selection = np.array([5, 3, 5]) * VU
    else:
        # fallback size
        shape_of_selection = np.array([5, max(1, shape[1]), 5]) * VU

    # compute selection position using same formula as original
    selection_pos = (np.array([6, 3, 6]) * np.array([x, y - shape[1] + 3, z]) - np.array([0, 1, 0])) * VU
    selection_mesh = uvox.from_grid(np.array([[[1]]]), voxel_mesh_shape=shape_of_selection)
    selection_mesh.translate(selection_pos, inplace=True)
    return selection_mesh

# ---- ITERATION (ONE STEP) ----
def run_one_iteration(state):
    """Run a single iteration of the algorithm. Returns True if did something, False if finished."""
    gc = state["gc"]
    filled_bc = state["filled_bc"]
    training_bc = state["training_bc"]
    config = state["config"]
    analyzed = state["analyzed"]
    ntcs = state["ntcs"]

    if not any(x is not None for x in ntcs.values()):
        # finished
        return False

    # pick subset and position (same logic as original)
    not_none_ntcs = {k: v for k, v in ntcs.items() if v is not None}
    subset_used = list(not_none_ntcs.keys())[0]
    pos = not_none_ntcs[subset_used]
    for subs, indices in not_none_ntcs.items():
        if pos[1] < indices[1]:
            subset_used = subs
            pos = indices
    x, y, z = pos

    subset_cfg = config['dataset']['subsets'][subset_used]
    shape = np.array(subset_cfg['iteration']['group_shape'])
    shape_top_ext = subset_cfg['iteration']['shape_top_ext']
    placement_pos = np.array([x, y - 1 - shape_top_ext, z])
    looking_pos = np.array([x, y - 1, z])
    shape_side_ext = int(np.round((shape[0] - 1) / 2))

    # prepare channel slices for visualization (return them)
    ext_vu_pos = ext_bu_to_vu(np.array([x - shape_side_ext, y - shape_top_ext - 1, z - shape_side_ext]))
    ext_vu_shape = ext_bu_to_vu(shape)

    channel1 = copy.deepcopy(gc.ext_voxel_grid[
        ext_vu_pos[0]:ext_vu_pos[0] + ext_vu_shape[0],
        ext_vu_pos[1]:ext_vu_pos[1] + ext_vu_shape[1],
        ext_vu_pos[2]:ext_vu_pos[2] + ext_vu_shape[2],
    ])

    channel2 = copy.deepcopy(training_bc.ext_voxel_grid[
        ext_vu_pos[0]:ext_vu_pos[0] + ext_vu_shape[0],
        ext_vu_pos[1]:ext_vu_pos[1] + ext_vu_shape[1],
        ext_vu_pos[2]:ext_vu_pos[2] + ext_vu_shape[2],
    ])

    # try place brick if any variant available
    brick_variants = make_brick_variants(placement_pos, subset_cfg['bricks'])
    placed = False
    if any(training_bc.is_placement_available(b) for b in brick_variants):
        brick_id, rotation = get_brick_id_rotation(filled_bc, looking_pos, placement_pos, config, subset_used, state["bricks"])
        if brick_id != "None":
            placed = place_brick(brick_id, rotation, placement_pos, training_bc)
            if placed:
                state["num_found"] += 1

    state["num_all"] += 1
    analyzed[subset_used][x, y, z] = True
    state["analyzed"] = analyzed
    state["ntcs"] = find_next_pos_to_cover(gc, training_bc, analyzed, state["height_map"])



    # return info needed to update visualization
    return {
        "subset_used": subset_used,
        "pos": (x, y, z),
        "shape": shape,
        "placement_pos": placement_pos,
        "channel1": channel1,
        "channel2": channel2,
    }

# ---- PLOTTER INITIALIZATION ----
plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 800))
actors = {
    "left_selection": None,
    "left_voxels": None,
    "right_channel1": None,
    "right_channel2": None,
    "right_grid": None
}

# initial left scene: training_bc ext grid
left_voxels_mesh = uvox.from_grid(training_bc.ext_voxel_grid, voxel_mesh_shape=VU)
left_voxels_mesh.translate(np.array([0, -0.16, 0]), inplace=True)
plotter.subplot(0, 0)
actors["left_voxels"] = plotter.add_mesh(left_voxels_mesh, show_edges=False, color="white")

# initial right scene: empty (we'll fill on iteration)
plotter.subplot(0, 1)
grid = uvox.from_grid(np.ones((1, 1, 1)), voxel_mesh_shape=VU*np.array([30, 15, 30]))
actors["right_grid"] = plotter.add_mesh(grid, show_edges=True, color="white", opacity=0.1)

# configure camera positions (reuse your preset)
cpos2 = [(-3.15, -14.13, -4.28),
         (2.00, 1.36, 2.4),
         (0.37, -0.47, 0.8)]
plotter.subplot(0, 0)
plotter.camera_position = cpos2
plotter.subplot(0, 1)
plotter.camera_position = cpos2

# ---- KEY HANDLER ----
def on_space():
    """Callback executed when SPACE is pressed - runs one iteration and updates the scene."""
    result = run_one_iteration(state)
    if not result:
        # finished
        print(f"Found {state['num_found']} bricks blocks. All locations looked: {state['num_all']}.")
        # optional: compute and print metrics if you want
        plotter.close()  # close window and end program
        return

    # update visualization after this iteration
    info = result  # dict with channel1/channel2 etc.
    subset_used = info["subset_used"]
    x, y, z = info["pos"]
    shape = info["shape"]
    channel1 = info["channel1"]
    channel2 = info["channel2"]

    # LEFT: update selection (remove previous selection actor if present, then add new)
    plotter.subplot(0, 0)
    if actors["left_selection"] is not None:
        try:
            plotter.remove_actor(actors["left_selection"])
        except Exception:
            # fallback: ignore if already removed
            actors["left_selection"] = None



    # LEFT: update voxels mesh (remove old and add new)
    new_left_voxels_mesh = uvox.from_grid(state["training_bc"].ext_voxel_grid, voxel_mesh_shape=VU)
    new_left_voxels_mesh.translate(np.array([0, -0.16, 0]), inplace=True)
    selection_mesh = make_selection_mesh(subset_used, x, y, z, shape)

    if actors["left_voxels"] is not None:
        try:
            plotter.remove_actor(actors["left_voxels"])
        except Exception:
            actors["left_voxels"] = None

    actors["left_voxels"] = plotter.add_mesh(new_left_voxels_mesh, show_edges=False, color="white")

    
    if subset_used == 'subset2':
        actors["left_selection"] = plotter.add_mesh(selection_mesh, show_edges=True, color="orange", opacity=0.2)
    else:
        actors["left_selection"] = plotter.add_mesh(selection_mesh, show_edges=True, color="purple", opacity=0.2)

    # RIGHT: update channel1 and channel2
    plotter.subplot(0, 1)
    # remove old channel actors if any
    for key in ("right_channel1", "right_channel2", "right_grid"):
        if actors[key] is not None:
            try:
                plotter.remove_actor(actors[key])
            except Exception:
                actors[key] = None

    # add new ones (try/except as in original)
    try:
        channel1_mesh = uvox.from_grid(channel1, voxel_mesh_shape=VU)
        actors["right_channel1"] = plotter.add_mesh(channel1_mesh, show_edges=True, color="blue", opacity=0.4)
        channel2_mesh = uvox.from_grid(channel2, voxel_mesh_shape=VU)
        actors["right_channel2"] = plotter.add_mesh(channel2_mesh, show_edges=True, color="white", opacity=1)
        if subset_used == 'subset2':
            grid = uvox.from_grid(np.ones((1, 1, 1)), voxel_mesh_shape=VU*np.array([18, 9, 18]))
        else:
            grid = uvox.from_grid(np.ones((1, 1, 1)), voxel_mesh_shape=VU*np.array([30, 15, 30]))
        actors["right_grid"] = plotter.add_mesh(grid, show_edges=True, color="white", opacity=0.1)

    except Exception:
        # keep silent as original code did
        pass

    # adjust cameras (reuse same presets)
    plotter.subplot(0, 0)
    plotter.camera_position = cpos2
    plotter.subplot(0, 1)
    plotter.camera_position = cpos2

    # render updated scene
    plotter.render()

# register key
plotter.add_key_event("space", on_space)

# show and start interactive loop
print("Interactive legolization. Press SPACE to perform one iteration, ESC to close early.")
try:
    plotter.show()
except AttributeError:
    pass
# when plotter.show() returns, the window was closed (either by finishing or manually)
# If finished normally, the on_space callback will have printed the summary and closed the plotter.
