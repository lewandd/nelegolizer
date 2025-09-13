import argparse
import torch
from nelegolizer.model.dataset import VoxelDataset
from nelegolizer.model.training import train_model
from nelegolizer.model.registry import get_model
from nelegolizer.model.cnn import *
from nelegolizer.model.label_encoder import build_label_encoder
import yaml

def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def main(train_config_path: str):
    # load configs
    train_config = load_config(train_config_path)
    model_config_path = train_config['model']['config_path']
    model_config = load_config(model_config_path)

    # datasets
    label_encoder = build_label_encoder(model_config)
    train_dataset = VoxelDataset(train_config["data"]["train_path"], label_encoder)
    val_dataset = VoxelDataset(train_config["data"]["test_path"], label_encoder)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_config)
    model.load_state_dict(torch.load(train_config["model"]["checkpoint"], map_location=device))

    # train
    train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=train_config["training"]["epochs"],
        batch_size=train_config["training"]["batch_size"],
        lr=train_config["training"]["lr"],
        seed=train_config["misc"]["seed"]
    )

    # save model
    torch.save(model.state_dict(), train_config['training']['save_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    main(args.config)