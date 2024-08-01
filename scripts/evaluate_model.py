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
from util.modules import model_class
from util import path

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
        MODEL_FILENAME = arg + ".pth"
        MODEL_PTH_FILE_PATH = os.path.join(path.BRICK_MODELS_DIR, MODEL_FILENAME)
        try:
            loaded = torch.load(f=MODEL_PTH_FILE_PATH)
            model.load_state_dict(loaded)    
        except FileNotFoundError as e:
            print(f"No file {MODEL_PTH_FILE_PATH}")
            continue
        else:
            print(f"Model succesfully loaded from: {MODEL_PTH_FILE_PATH}")
        
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