import argparse
import torch
from nelegolizer.model.dataset import VoxelDataset
from nelegolizer.model.training import train_model
from torch.utils.data import random_split
from nelegolizer.model.registry import get_model
from nelegolizer.model.cnn import *
import yaml

def main(config_path: str, output_path: str):
    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # dataset
    dataset = VoxelDataset(config["data"]["train_path"])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config)
    model.load_state_dict(torch.load(config["model"]["checkpoint"], map_location=device))

    train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["lr"],
        seed=config["misc"]["seed"]
    )

    torch.save(model.state_dict(), output_path)

    #args = parser.parse_args()
    #model_path = Path(args.model)
    #dataset_path = Path(args.dataset)
    #shape = tuple(int(ch) for ch in args.shape)
    #shape = (shape[0]*5, shape[1]*2, shape[2]*5)

    #net_type = args.net_type
    #model = net_types[net_type](input_shape=(30, 15, 30), num_classes=10)

    #if not model_path.exists():
    #    print(f"File doesn't exist: {model_path}")
    #    return
    #if not dataset_path.exists():
    #    print(f"File doesn't exist: {dataset_path}")
    #    return
    
    #initilize_parts()

    #dataset = VoxelDataset(dataset_path)
    
    #train_size = int(0.8 * len(dataset))
    #val_size = len(dataset) - train_size
    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #config = yaml.safe_load(open("configs/brick_classification.yaml"))
    #model = get_model(config)
    #print(model)

    #model = net_types[net_type](input_shape=(30, 15, 30), num_classes=10)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    #model = load_model_str(model_path, shape)
    #create_model(shape)#Voxel3DCNN(num_classes=NUM_CLASSES)


    #train_model(model, train_dataset, val_dataset, epochs=30, batch_size=8, lr=1.3e-3)

    

    # Trening
    # dla 232 najlepsze lr=1.3e-3
    #iter = 1
    #lr_1_3_acc = []
    #for i in range(iter):
    #    model = create_model(shape)#Voxel3DCNN(num_classes=NUM_CLASSES)
    #    train_model(model, train_dataset, val_dataset, epochs=30, batch_size=8, lr=1.3e-3)
    #    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #    _, val_acc = evaluate_model(model, val_loader)
    #    lr_1_3_acc.append(val_acc)

    #lr_1_4_acc = []
    #for i in range(iter):
    #    model = create_model(shape)#Voxel3DCNN(num_classes=NUM_CLASSES)
    #    train_model(model, train_dataset, val_dataset, epochs=30, batch_size=8, lr=1.4e-3)
    #    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #    _, val_acc = evaluate_model(model, val_loader)
    #    lr_1_4_acc.append(val_acc)

    #print(f"1.3e-3: {lr_1_3_acc} (mean: {sum(lr_1_3_acc)/len(lr_1_3_acc)}), 1.4e-3: {lr_1_4_acc} (mean: {sum(lr_1_4_acc)/len(lr_1_4_acc)})")
    
    #print(f"Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to output pth")
    args = parser.parse_args()

    main(args.config, args.output)