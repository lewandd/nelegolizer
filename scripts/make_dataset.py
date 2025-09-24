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
    train_samples = make_samples(config, type="train")
    test_samples = make_samples(config, type="test")

    print("Train samples generated:", sum(len(v) for v in train_samples.values()))
    print("Test samples generated:", sum(len(v) for v in test_samples.values()))

    for subset_name, subset_cfg in config["dataset"]["subsets"].items():
        subset_train_samples = train_samples[subset_name]
        subset_test_samples = test_samples[subset_name]
        train_path = Path(subset_cfg["data"]["train_path"])
        save_dataset(subset_train_samples, train_path)
        print(f"Subset '{subset_name}' train samples: {len(subset_train_samples)}")

        test_path = Path(subset_cfg["data"]["test_path"])
        save_dataset(subset_test_samples, test_path)
        print(f"Subset '{subset_name}' test samples: {len(subset_test_samples)}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    main(args.config)

