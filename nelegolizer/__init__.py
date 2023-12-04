import os

# path to library main folder
PATH = os.path.join(os.path.dirname(__file__), '..')
# group resolution
GROUP_RES = 4

from ._core import legolize
from ._LegoBrick import LegoBrick