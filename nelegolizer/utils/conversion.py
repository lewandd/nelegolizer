from ..constants import BU_RES, BU, LDU, EXT_BU_RES
import numpy as np
from typing import Union, Tuple, Type


#UNIT_EXT = np.array([1, 1, 1])


def _cast(input_arr: Union[np.ndarray, Tuple],
          output_type: Type = np.ndarray) -> Union[np.ndarray, Tuple]:
    arr = np.asarray(input_arr)#.squeeze()
    if arr.shape != (3,):
        raise ValueError(f"Expected shape (3,), got {arr.shape} from {input_arr!r}")
    if output_type is tuple:
        return tuple(arr.tolist())
    elif output_type is np.ndarray:
        return arr
    else:
        raise TypeError(f"Unsupported output_type: {output_type}")

def ext_bu_to_vu(bu: Union[np.ndarray, Tuple]) -> Union[np.ndarray, Tuple]:
    npbu = np.array(bu)
    return _cast((npbu*EXT_BU_RES).astype(int), type(bu)) 

def vu_to_bu(vu: Union[np.ndarray, Tuple]) -> Union[np.ndarray, Tuple]:
    npvu = np.array(vu)
    if np.any((npvu / BU_RES) != (npvu // BU_RES)):
        raise Exception(f"{vu} is not divisible by {BU_RES}."
                         "Cannot convert VoxelUnit to BrickUnit.")
    else:
        return _cast((npvu / BU_RES).astype(int), type(vu))

def bu_to_vu(bu: Union[np.ndarray, Tuple]) -> Union[np.ndarray, Tuple]:
    npbu = np.array(bu)
    return _cast((npbu * BU_RES).astype(int), type(bu))

def bu_to_mesh(bu: Union[np.ndarray, Tuple]) -> Union[np.ndarray, Tuple]:
    npbu = np.array(bu)
    return _cast(npbu * BU, type(bu))

def mesh_to_bu(mesh: Union[np.ndarray, Tuple]) -> Union[np.ndarray, Tuple]:
    npmesh = np.array(mesh)
    return _cast(npmesh/BU, type(mesh))

def ldu_to_mesh(ldu: Union[np.ndarray, Tuple]) -> Union[np.ndarray, Tuple]:
    npldu = np.array(ldu)
    return _cast(npldu*LDU, type(ldu))

def mesh_to_ldu(mesh: Union[np.ndarray, Tuple]) -> Union[np.ndarray, Tuple]:
    npmesh = np.array(mesh)
    return _cast(npmesh/LDU, type(mesh))