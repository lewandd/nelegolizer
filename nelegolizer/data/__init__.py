import numpy as np
import constants as const

from ._LDrawPart import LDrawPart, part_by_id, part_by_filename, initilize_parts

def ldu_to_mesh(ldu_position: np.ndarray) -> np.ndarray:
    return (ldu_position/const.BRICK_UNIT_LDU_SHAPE) * const.BRICK_UNIT_MESH_SHAPE

def mesh_to_ldu(mesh_position: np.ndarray) -> np.ndarray:
    return (mesh_position/const.BRICK_UNIT_MESH_SHAPE) * const.BRICK_UNIT_LDU_SHAPE 

from ._LDrawReference import LDrawReference
from ._LegoBrick import LegoBrick
from ._LDrawModel import LDrawModel
from ._LDrawFile import LDrawFile
from ._ClassificationResult import ClassificationResult, ClassificationResult2
from ._BrickOccupancy import BrickOccupancy
from ._BrickCoverage import BrickCoverage
from ._GeometryCoverage import GeometryCoverage
from ._ObjectOccupancy import ObjectOccupancy

__all__ = [LDrawPart,
           part_by_filename,
           part_by_id,
           initilize_parts,
           LDrawReference,
           LegoBrick,
           LDrawModel,
           LDrawFile,
           ClassificationResult,
           ClassificationResult2,
           BrickOccupancy,
           BrickCoverage,
           ObjectOccupancy,
           ldu_to_mesh,
           mesh_to_ldu]
