import numpy as np
import constants as const

def ldu_to_mesh(ldu_position: np.ndarray, brick_id: int) -> np.ndarray:
    return ((ldu_position-const.LDRAW_PART_OFFSET[brick_id])/
           const.BRICK_UNIT_LDU_SHAPE) * const.BRICK_UNIT_MESH_SHAPE

def mesh_to_ldu(mesh_position: np.ndarray, brick_id: int) -> np.ndarray:
    return ((mesh_position/const.BRICK_UNIT_MESH_SHAPE)*
            const.BRICK_UNIT_LDU_SHAPE)+const.LDRAW_PART_OFFSET[brick_id]

from ._LDrawPart import LDrawPart, part_by_id, part_by_filename
from ._LDrawReference import LDrawReference
from ._LegoBrick import LegoBrick
from ._LDrawModel import LDrawModel
from ._LDrawFile import LDrawFile
from ._ClassificationResult import ClassificationResult

__all__ = [LDrawPart,
           part_by_filename,
           part_by_id,
           LDrawReference,
           LegoBrick,
           LDrawModel,
           LDrawFile,
           ClassificationResult,
           ldu_to_mesh,
           mesh_to_ldu]
