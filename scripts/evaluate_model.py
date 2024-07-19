"""
Evaluate accuracy of models from models/

Usage:
python3 evaluate_model.py model_name_1 model_name2 ...
or
python3 evaluate_model.py all 
"""

import sys
import os
import torch
from torch import nn
from util.dataset import CustomVoxelsDataset
from torch.utils.data import DataLoader
from scripts.util.modules import model_class

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PACKAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)))
MODELS_PATH = os.path.join(PACKAGE_PATH, "models/")
GENERATED_DATA_PATH = os.path.join(PACKAGE_PATH, "data/generated/")

model_hyperparameters = {
    "model_n111": {"train_labels_filename": "labels_111.csv",
                    "test_labels_filename": "labels_111.csv",
                    "loss_fn": nn.CrossEntropyLoss(),
                    "lr": 0.1}
}

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    argc = len(sys.argv) - 1
    if argc == 0 :
        raise Exception("Usage: python3 create_model.py [model name] [model2 name] ... or python3 create_model.py all")
    elif "all" in sys.argv:
        args = list(model_class.keys())
    else:
        args = sys.argv[1:]
    if "debug=true" in [s.lower() for s in sys.argv]:
        debug = True
    elif "debug=false" in [s.lower() for s in sys.argv]:
        debug = False

    for arg in args:
        if arg.lower() == "debug=true" or arg.lower() == "debug=false":
            continue

        # Create model
        try:
            model = model_class[arg]().to(device)
        except KeyError:
            print(f"No model like {arg}. Available models: {list(model_class.keys())}")
            continue

        # Load model
        MODEL_NAME = arg + ".pth"
        MODEL_SAVE_PATH = MODELS_PATH + MODEL_NAME
        try:
            loaded = torch.load(f=MODEL_SAVE_PATH)
            model.load_state_dict(loaded)    
        except FileNotFoundError as e:
            print(f"No file {MODEL_SAVE_PATH}")
            continue
        else:
            print(f"Model succesfully loaded from: {MODEL_SAVE_PATH}")
        
        # Prepare datasets and dataloaders
        training_data = CustomVoxelsDataset(GENERATED_DATA_PATH + model_hyperparameters[arg]["train_labels_filename"], '.')
        test_data = CustomVoxelsDataset(GENERATED_DATA_PATH + model_hyperparameters[arg]["test_labels_filename"], '.')

        test_dataloader = DataLoader(test_data, batch_size=30, shuffle=True)

        # Test
        loss_fn = model_hyperparameters[arg]["loss_fn"]
        test(test_dataloader, model, loss_fn)            