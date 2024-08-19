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
from torch.utils.data import DataLoader
from util.dataset import CustomVoxelsDataset
from importlib.machinery import SourceFileLoader
import importlib.util

__PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
__PATHS_FILE = os.path.join(__PACKAGE_DIR, "paths.py")

path_loader = SourceFileLoader("paths", __PATHS_FILE)
path_spec = importlib.util.spec_from_loader(path_loader.name, path_loader)
path = importlib.util.module_from_spec(path_spec)
path_loader.create_module(path_spec)
path_loader.exec_module(path)

bc_models_loader = SourceFileLoader("bc_models", path.BRICK_MODULES_FILE)
bc_models_spec = importlib.util.spec_from_loader(bc_models_loader.name, bc_models_loader)
bc_models_module = importlib.util.module_from_spec(bc_models_spec)
bc_models_loader.create_module(bc_models_spec)
bc_models_loader.exec_module(bc_models_module)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        args = bc_models_module.get_model_names()
    else:
        args = sys.argv[1:]
    if "debug=true" in [s.lower() for s in sys.argv]:
        debug = True
    elif "debug=false" in [s.lower() for s in sys.argv]:
        debug = False

    for arg in args:
        if arg.lower() == "debug=true" or arg.lower() == "debug=false":
            continue

        model = bc_models_module.create_model(arg)
        model = bc_models_module.load_model(model, arg)
        
        # Prepare datasets and dataloaders
        MODEL_DATA_DIR = os.path.join(path.BRICK_CLASSFICATION_DATA_DIR, arg[6:])
        TRAIN_LABEL_FILE_PATH = os.path.join(MODEL_DATA_DIR, "train_data_labels.csv")
        TEST_LABEL_FILE_PATH = os.path.join(MODEL_DATA_DIR, "test_data_labels.csv")
        TRAIN_DATA_DIR = os.path.join(MODEL_DATA_DIR, "train_data")
        TEST_DATA_DIR = os.path.join(MODEL_DATA_DIR, "test_data")
        
        training_data = CustomVoxelsDataset(TRAIN_LABEL_FILE_PATH, TRAIN_DATA_DIR)
        test_data = CustomVoxelsDataset(TEST_LABEL_FILE_PATH, TEST_DATA_DIR)

        test_dataloader = DataLoader(test_data, batch_size=30, shuffle=True)

        # Test
        loss_fn = model_hyperparameters[arg]["loss_fn"]
        test(test_dataloader, model, loss_fn)            