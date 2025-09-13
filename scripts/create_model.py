import argparse
import torch
import yaml
from nelegolizer.model.registry import get_model
from nelegolizer.model.cnn import *
from pathlib import Path

def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def main(config_path: str, output_path: str):
    # load config
    config = load_config(config_path)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config).to(device)
    # save fresh state dict
    torch.save(model.state_dict(), Path(output_path))
    print(f"Saved untrained model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to config output model")
    args = parser.parse_args()

    main(args.config, args.output)