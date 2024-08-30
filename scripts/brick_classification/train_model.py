"""
Train models from models/

Usage:
python3 train_models.py [debug=True|False] model_name_1 model_name2 ...
or
python3 train_models.py [debug=True|False] all
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
bc_models_spec = importlib.util.spec_from_loader(bc_models_loader.name,
                                                 bc_models_loader)
bc_models_module = importlib.util.module_from_spec(bc_models_spec)
bc_models_loader.create_module(bc_models_spec)
bc_models_loader.exec_module(bc_models_module)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug = False

model_hyperparameters = {
    "model_n111": {"loss_fn": nn.CrossEntropyLoss(),
                   "lr": 0.1}
}


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if debug:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")


def set_args(argv):
    argc = len(argv) - 1
    if argc == 0:
        print("Usage: python3 create_model.py [model name] [model2 name] ... "
              "or python3 create_model.py all")
        sys.exit()
    elif "all" in argv:
        return bc_models_module.get_model_names()
    else:
        return argv[1:]


def print_debug(text):
    if debug:
        print(text)


if __name__ == '__main__':
    args = set_args(sys.argv)
    if "debug=true" in [s.lower() for s in sys.argv]:
        debug = True
    elif "debug=false" in [s.lower() for s in sys.argv]:
        debug = False

    for arg in args:
        if arg.lower() == "debug=true" or arg.lower() == "debug=false":
            continue

        model = bc_models_module.create_model(arg)
        model = bc_models_module.load_model(model, arg, debug)

        # Prepare datasets and dataloaders
        MODEL_DATA_DIR = os.path.join(
            path.BRICK_CLASSFICATION_DATA_DIR, arg[6:])
        TRAIN_LABEL_FILE_PATH = os.path.join(
            MODEL_DATA_DIR, "train_data_labels.csv")
        TEST_LABEL_FILE_PATH = os.path.join(
            MODEL_DATA_DIR, "test_data_labels.csv")
        TRAIN_DATA_DIR = os.path.join(MODEL_DATA_DIR, "train_data")
        TEST_DATA_DIR = os.path.join(MODEL_DATA_DIR, "test_data")

        train_data = CustomVoxelsDataset(
            annotations_file=TRAIN_LABEL_FILE_PATH,
            data_dir=TRAIN_DATA_DIR)
        test_data = CustomVoxelsDataset(
            annotations_file=TEST_LABEL_FILE_PATH,
            data_dir=TEST_DATA_DIR)

        train_dataloader = DataLoader(train_data,
                                      batch_size=60,
                                      shuffle=True)
        test_dataloader = DataLoader(test_data,
                                     batch_size=30,
                                     shuffle=True)

        # Train
        loss_fn = model_hyperparameters[arg]["loss_fn"]
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=model_hyperparameters[arg]["lr"])

        print_debug(f"Training {arg}...")
        epochs = 6
        for t in range(epochs):
            print_debug(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            if debug:
                test(test_dataloader, model, loss_fn)

        print_debug("Done!")

        bc_models_module.save_model(model, arg, debug)
