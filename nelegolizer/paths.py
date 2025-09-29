from pathlib import Path

PACKAGE = Path(__file__).parent
REPO = PACKAGE.parent

PARTS_CSV = REPO / "data" / "raw" / "metadata" / "parts.csv"
PART_DAT_DIR = REPO / "data" / "raw" / "ldraw_parts" / "dat"
PART_GEOM_DIR = REPO / "data" / "raw" / "ldraw_parts" / "geom"

DEFAULT_CONFIG = REPO / "configs" / "legolization" / "stable" / "st_conenet1v1.yaml"