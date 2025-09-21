import numpy as np

# constants

BU = np.array([0.8, 0.32, 0.8])
LDU = np.array([0.04, 0.04, 0.04])
VU = np.array([0.16, 0.16, 0.16])
BU_RES = np.array([5, 2, 5], dtype=int)
EXT_BU_RES = BU_RES + np.array([1, 1, 1])