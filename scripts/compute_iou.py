from nelegolizer.model.evaluation import compute_iou
import argparse
from nelegolizer.data import initilize_parts, LDrawFile, BrickCoverage, GeometryCoverage, LegoBrick, LDrawModel
from typing import List
import numpy as np
import nelegolizer.utils.brick as utils_brick

def load_bricks(path: str) -> List[LegoBrick]:
    ldf = LDrawFile.load(path)
    lbm = LDrawModel.merge_multiple_models(ldf.models)
    return lbm.as_bricks()


def main(model1_path: str, model2_path: str):
    initilize_parts()
    
    bottom_ext, top_ext, side_ext = 3, 4, 2
    offset = (side_ext, top_ext, side_ext)

    
    bricks1 = load_bricks(model1_path)
    bricks2 = load_bricks(model2_path)
    utils_brick.normalize_positions(bricks1+bricks2, offset)
    pos_min1, pos_max1 = utils_brick.compute_bounds(bricks1)
    pos_min2, pos_max2 = utils_brick.compute_bounds(bricks2)
    pos_min = np.minimum(pos_min1, pos_min2)
    pos_max = np.maximum(pos_max1, pos_max2)
    shape = (pos_max-pos_min).astype(int)
    grids = []
    for bricks in [bricks1, bricks2]:
        bc = BrickCoverage.from_bricks(bricks, bottom_extension=bottom_ext, top_extension=top_ext, side_extension=side_ext, shape=shape)
        grids.append(bc.voxel_grid)

    print(f"IoU: {compute_iou(grids[0], grids[1])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", required=True, help="Path to MPD LDraw model")
    parser.add_argument("--model2", required=True, help="Path to MPD LDraw model")
    args = parser.parse_args()

    main(args.model1, args.model2)