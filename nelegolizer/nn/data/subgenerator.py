import numpy as np
import nelegolizer.constants as CONST

_111shape = (CONST.GROUP_RES, CONST.GROUP_RES, CONST.GROUP_RES)

def _add_noise(grid, treshold):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            for k in range(len(grid[i][j])):
                if np.random.rand(1) < treshold:
                    grid[i][j][k] = 1 if grid[i][j][k] == 0 else 0

def _111_full():
    grid = np.ones(_111shape, dtype=int)
    _add_noise(grid, 0.1)
    return grid.flatten()

def _111_lower_half():
    grid = np.zeros(_111shape)
    for i in range(_111shape[0]):
        for j in range(_111shape[1]):
            for k in range(_111shape[2]):
                if j < CONST.GROUP_RES/2:
                    grid[i][j][k] = 1
    _add_noise(grid, 0.1)
    return grid.flatten()

def _111_upper_half():
    grid = np.zeros(_111shape)
    for i in range(_111shape[0]):
        for j in range(_111shape[1]):
            for k in range(_111shape[2]):
                if j >= CONST.GROUP_RES/2:
                    grid[i][j][k] = 1
    _add_noise(grid, 0.1)
    return grid.flatten()