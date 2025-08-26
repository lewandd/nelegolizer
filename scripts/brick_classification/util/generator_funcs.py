import numpy as np

from nelegolizer.data import part_by_filename
from nelegolizer import const
from .geom_preparation import prepare_part_grid


prepared_part_grids = {}
for filename in part_by_filename.keys():
    part = part_by_filename[filename]
    prepared_part_grids[part.id] = prepare_part_grid(
        part.grid, part.size * const.BRICK_UNIT_RESOLUTION)


def add_noise(grid, treshold):
    for (x, y, z), val in np.ndenumerate(grid):
        if np.random.rand(1) < treshold:
            if val == 0:
                grid[x][y][z] = 1
            else:
                grid[x][y][z] = 0
    return grid


def with_001_noise(id: str):
    grid = prepared_part_grids[id].copy()
    grid = add_noise(grid, 0.01)
    return grid


__all__ = [with_001_noise]
