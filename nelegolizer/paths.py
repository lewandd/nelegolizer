from pathlib import Path

PACKAGE = Path(__file__).parent
REPO = PACKAGE.parent

PARTS_CSV = REPO / "data" / "raw" / "metadata" / "parts.csv"
PART_DAT_DIR = REPO / "data" / "raw" / "ldraw_parts" / "dat"
PART_GEOM_DIR = REPO / "data" / "raw" / "ldraw_parts" / "geom"