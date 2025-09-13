from nelegolizer.data import initilize_parts
from nelegolizer.model.dataset_generation import make_samples, save_dataset
import argparse
from pathlib import Path
import yaml
import random

def train_test_split(data, train_ratio=0.8, seed=42):
    """Split data list into train/test parts."""
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def main(config_path: str):
    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    initilize_parts()
    samples = make_samples(config)

    print("Samples generated:", sum(len(v) for v in samples.values()))

    for subset_name, subset_cfg in config["dataset"]["subsets"].items():
        subset_samples = samples[subset_name]

        # split into train/test
        train_samples, test_samples = train_test_split(
            subset_samples, 
            train_ratio=subset_cfg["data"]["split"]["train"],
            seed=config['dataset']['misc']['seed']
        )

        # resolve paths
        train_path = Path(subset_cfg["data"]["train_path"])
        test_path = Path(subset_cfg["data"]["test_path"])

        # save datasets
        save_dataset(train_samples, train_path)
        save_dataset(test_samples, test_path)

        print(f"Subset '{subset_name}' samples: {len(train_samples)} train, {len(test_samples)} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    main(args.config)

