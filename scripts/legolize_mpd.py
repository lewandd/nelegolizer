from nelegolizer.data import initilize_parts, LDrawFile, LDrawModel
from nelegolizer.model.dataset_generation import make_samples, save_dataset
import argparse
from nelegolizer.legolizer._core import legolize_from_mpd
from pathlib import Path
import yaml
import random

def main(input_path: str, config_path: str, output_path: str = None):
    initilize_parts()
    bricks = legolize_from_mpd(input_path, config_path)

    if output_path:
        ldraw_model = LDrawModel.from_bricks(bricks, "main.ldr")
        ldraw_file = LDrawFile()
        ldraw_file.add_model(ldraw_model)
        ldraw_file.save(output_path)
        print(f"Model zapisany do: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPD file legolization.")
    parser.add_argument("--input", required=True, help="Input MPD file path.")
    parser.add_argument("--config", required=True, help="Legolization configuration path.")
    parser.add_argument("--output", required=False, default=None, help="Optional: generated MPD file path.")
    
    args = parser.parse_args()
    main(args.input, args.config, args.output)

