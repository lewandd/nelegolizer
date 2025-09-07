import numpy as np
from nelegolizer import const
from typing import Union, Tuple

UNIT_EXT = np.array([1, 1, 1])
EXTBU = const.BRICK_UNIT_RESOLUTION + UNIT_EXT

def ext_bu_to_vu(bu: Union[np.ndarray, Tuple[int, int, int]]) -> np.ndarray:
    arr = np.array(bu, dtype=int)

    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape} from {bu!r}")

    result = arr * (const.BRICK_UNIT_RESOLUTION + UNIT_EXT)

    if isinstance(bu, tuple):
        return tuple(result.tolist())
    return result

def vu_to_bu(vu: np.ndarray) -> np.ndarray:
    if np.any((vu / const.BRICK_UNIT_RESOLUTION) != (vu // const.BRICK_UNIT_RESOLUTION)):
        raise Exception(
            "{vu} is not divisible by {const.BRICK_UNIT_RESOLUTION}."
            "Cannot convert VoxelUnit to BrickUnit.")
    else:
        return (vu / const.BRICK_UNIT_RESOLUTION).astype(int)

def bu_to_vu(bu: np.ndarray) -> np.ndarray:
    return bu * const.BRICK_UNIT_RESOLUTION

def bu_to_mesh(bu: np.ndarray) -> np.ndarray:
    return bu * const.BRICK_UNIT_MESH_SHAPE

def mesh_to_bu(mesh: np.ndarray) -> np.ndarray:
    return mesh / const.BRICK_UNIT_MESH_SHAPE

def ldu_to_mesh(ldu_position: np.ndarray) -> np.ndarray:
    return (ldu_position/const.BRICK_UNIT_LDU_SHAPE) * const.BRICK_UNIT_MESH_SHAPE

def mesh_to_ldu(mesh_position: np.ndarray) -> np.ndarray:
    return (mesh_position/const.BRICK_UNIT_MESH_SHAPE) * const.BRICK_UNIT_LDU_SHAPE 