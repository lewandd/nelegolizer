from nelegolizer.data import LDrawFile, initilize_parts, BrickOccupancy, ObjectOccupancy
from nelegolizer.model.dataset_generation import make_samples, save_dataset
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Script that takes an mpd " \
                                    "input file and an output txt file path.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input MPD file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output TXT file"
    )
    parser.add_argument(
        "shape",
        type=str,
        help="Shape"
    )
    
    args = parser.parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    shape = tuple(int(ch) for ch in args.shape)

    if not input_path.exists():
        print(f"File doesn't exist: {input_path}")
        return
    
    initilize_parts()

    ldf = LDrawFile.load(input_path)
    lbm = ldf.models[0]
    bricks = lbm.as_bricks()

    #shapes = [(2,3,2), (2,3,1), (1,3,1), (1,1,1)]
    bo = BrickOccupancy.from_bricks(bricks)
    oo = ObjectOccupancy(bo.voxel_grid)

    samples = make_samples(oo, shape, bricks, debug=True)
    print("samples generated:", len(samples))
    #save_dataset(samples, output_path)

if __name__ == "__main__":
    main()

