import numpy as np

GROUP_RES = 4

n111shape = (GROUP_RES, GROUP_RES, GROUP_RES)

def add_noise(grid, treshold):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            for k in range(len(grid[i][j])):
                if np.random.rand(1) < treshold:
                    grid[i][j][k] = 1 if grid[i][j][k] == 0 else 0

def n111_full():
    grid = np.ones(n111shape, dtype=int)
    add_noise(grid, 0.1)
    return grid.flatten()

def n111_lower_half():
    grid = np.zeros(n111shape)
    for i in range(n111shape[0]):
        for j in range(n111shape[1]):
            for k in range(n111shape[2]):
                if j < GROUP_RES/2:
                    grid[i][j][k] = 1
    add_noise(grid, 0.1)
    return grid.flatten()

def n111_upper_half():
    grid = np.zeros(n111shape)
    for i in range(n111shape[0]):
        for j in range(n111shape[1]):
            for k in range(n111shape[2]):
                if j >= GROUP_RES/2:
                    grid[i][j][k] = 1
    add_noise(grid, 0.1)
    return grid.flatten()