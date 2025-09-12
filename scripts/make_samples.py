from nelegolizer.data import initilize_parts
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
        "net_type",
        type=str,
        help="Network type"
    )
    
    args = parser.parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    #shape = tuple(int(ch) for ch in args.shape)
    net_type = args.net_type

    if not input_path.exists():
        print(f"File doesn't exist: {input_path}")
        return
    
    if net_type == "type3":
        network_type = 3
    elif net_type == "type1":
        network_type = 1
    initilize_parts()
    samples = make_samples(input_path, network_type)
    print("samples generated:", len(samples))
    save_dataset(samples, output_path)
    #ldf = LDrawFile.load(input_path)
    #lbm = ldf.models[0]
    #bricks = lbm.as_bricks()

    
    #bo = BrickOccupancy.from_bricks(bricks)
    #oo = ObjectOccupancy(bo.voxel_grid)

    #samples = make_samples(oo, shape, bricks, debug=True)
    #print("samples generated:", len(samples))
    #save_dataset(samples, output_path)

if __name__ == "__main__":
    main()

