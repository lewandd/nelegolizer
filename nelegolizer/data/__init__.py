import pandas as pd
import nelegolizer.constants as CONST

from ._LegoBrick import LegoBrick
from ._LDrawPart import LDrawPart

_PART_LABEL_PATH = CONST.PATH + "/LDraw/part/part_label.csv"
_PART_LABEL_DF = pd.read_csv(_PART_LABEL_PATH)

_PART_DETAILS_PATH = CONST.PATH + "/LDraw/part/part_details.csv"
_PART_DETAILS_DF = pd.read_csv(_PART_DETAILS_PATH)