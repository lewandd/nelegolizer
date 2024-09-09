import numpy as np

from nelegolizer.utils import grid as gridtools
from nelegolizer import const


def crop(grid: np.ndarray,
         target_resolution: np.ndarray) -> np.ndarray:
    return grid[-target_resolution[0]:,
                -target_resolution[1]:,
                -target_resolution[2]:]


def paste_grid(copy_grid: np.ndarray,
               target_grid: np.ndarray,
               center_xz=False,
               up_y=False) -> np.ndarray:
    if np.any(target_grid.shape < copy_grid.shape):
        raise Exception("target_grid cannot be smaller than copy_grid")
    start = np.array([0, 0, 0])
    end = np.array(copy_grid.shape)
    if center_xz:
        start[0] = (target_grid.shape[0] - copy_grid.shape[0]) / 2
        start[2] = (target_grid.shape[2] - copy_grid.shape[2]) / 2
        end[0] = start[0] + copy_grid.shape[0]
        end[2] = start[2] + copy_grid.shape[2]
    if up_y:
        start[1] = target_grid.shape[1] - copy_grid.shape[1]
        end[1] = start[1] + copy_grid.shape[1]

    target_grid[start[0]:end[0],
                start[1]:end[1],
                start[2]:end[2]] = copy_grid
    return target_grid


def flood_fill(x: int,
               y: int,
               z: int,
               grid: np.ndarray,
               visited: np.ndarray) -> None:
    if x >= 0 and x < grid.shape[0] and z >= 0 and z < grid.shape[2]:
        if not visited[x, y, z] and not grid[x, y, z]:
            visited[x, y, z] = True
            flood_fill(x+1, y, z, grid, visited)
            flood_fill(x-1, y, z, grid, visited)
            flood_fill(x, y, z-1, grid, visited)
            flood_fill(x, y, z-1, grid, visited)


def grid_xz_hull(grid: np.ndarray) -> np.ndarray:
    visited = np.zeros_like(grid)
    height = grid.shape[1]
    for y in range(height):
        for x in range(grid.shape[0]):
            flood_fill(x, y, 0, grid, visited)
        for x in range(grid.shape[0]):
            flood_fill(x, y, grid.shape[2]-1, grid, visited)
        for z in range(grid.shape[2]):
            flood_fill(0, y, z, grid, visited)
        for z in range(grid.shape[2]):
            flood_fill(grid.shape[0]-1, y, z, grid, visited)
    return ~visited


def prepare_part_grid(grid: np.ndarray,
                      brick_resolution: np.ndarray) -> np.ndarray:
    grid = crop(grid, brick_resolution)
    grid = paste_grid(copy_grid=grid,
                      target_grid=np.zeros(brick_resolution),
                      center_xz=True,
                      up_y=True)
    grid = gridtools.add_padding(grid, const.PADDING)
    grid = grid_xz_hull(grid)
    best_rotation = gridtools.find_best_rotation(grid)
    grid = gridtools.rotate(grid, best_rotation)
    return grid


__all__ = [prepare_part_grid]
