"""
Train models from models/

Usage:
python3 train_models.py [debug=True|False] model_name_1 model_name2 ...
or
python3 train_models.py [debug=True|False] all 
"""

import sys
import os
from importlib.machinery import SourceFileLoader
import torch
from torch import nn
from torch.utils.data import DataLoader
from util.dataset import CustomVoxelsDataset
from util import path

bc_modules = SourceFileLoader("modules", path.BRICK_MODULES_FILE).load_module()
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    argc = len(sys.argv) - 1
    if argc == 0 :
        print("Usage: python3 create_model.py [model name] [model2 name] ... or python3 create_model.py all")
        sys.exit()
    elif "all" in sys.argv:
        args = bc_modules.get_model_names()
    else:
        args = sys.argv[1:]
    if "debug=true" in [s.lower() for s in sys.argv]:
        debug = True
    elif "debug=false" in [s.lower() for s in sys.argv]:
        debug = False

    for arg in args:
        if arg.lower() == "debug=true" or arg.lower() == "debug=false":
            continue

        model = bc_modules.create_model(arg)
        model = bc_modules.load_model(model, arg, debug)

        # Prepare datasets and dataloaders
        MODEL_DATA_DIR = os.path.join(path.BRICK_CLASSFICATION_DATA_DIR, arg[6:])
        TRAIN_LABEL_FILE_PATH = os.path.join(MODEL_DATA_DIR, "train_data_labels.csv")
        TEST_LABEL_FILE_PATH = os.path.join(MODEL_DATA_DIR, "test_data_labels.csv")
        TRAIN_DATA_DIR = os.path.join(MODEL_DATA_DIR, "train_data")
        TEST_DATA_DIR = os.path.join(MODEL_DATA_DIR, "test_data")
        
        training_data = CustomVoxelsDataset(TRAIN_LABEL_FILE_PATH, TRAIN_DATA_DIR)
        test_data = CustomVoxelsDataset(TEST_LABEL_FILE_PATH, TEST_DATA_DIR)

        train_dataloader = DataLoader(training_data, batch_size=60, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=30, shuffle=True)

        # Train
        loss_fn = model_hyperparameters[arg]["loss_fn"]
        optimizer = torch.optim.SGD(model.parameters(), lr=model_hyperparameters[arg]["lr"])

        if debug:
            print(f"Training {arg}...")
        epochs = 6
        for t in range(epochs):
            if debug:
                print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            if debug:
                test(test_dataloader, model, loss_fn)

        if debug:
            print("Done!")

        bc_modules.save_model(model, arg, debug)