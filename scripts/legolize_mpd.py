from nelegolizer.data import initilize_parts
from nelegolizer.model.dataset_generation import make_samples, save_dataset
import argparse
from nelegolizer.legolizer._core import legolize_from_mpd
from pathlib import Path
import yaml
import random

def train_test_split(data, train_ratio=0.8, seed=42):
    """Split data list into train/test parts."""
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def main(input_path: str, config_path: str):
    initilize_parts()
    legolize_from_mpd(input_path, config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to config YAML")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    #parser.add_argument("--output", required=True, help="Path to config YAML")
    args = parser.parse_args()

    main(args.input, args.config)

