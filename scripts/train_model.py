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
from util.modules import nn_modules
from util import path

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
        args = list(nn_modules.keys())
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
            model = nn_modules[arg]().to(device)
        except KeyError:
            print(f"No model like {arg}. Available models: {list(nn_modules.keys())}")
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

        train_dataloader = DataLoader(training_data, batch_size=60, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=30, shuffle=True)

        # Train
        loss_fn = model_hyperparameters[arg]["loss_fn"]
        optimizer = torch.optim.SGD(model.parameters(), lr=model_hyperparameters[arg]["lr"])

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

        # Save model
        MODEL_FILENAME = arg + ".pth"
        MODEL_PTH_FILE_PATH = os.path.join(path.BRICK_MODELS_DIR, MODEL_FILENAME)
        try:
            torch.save(obj=model.state_dict(), f=MODEL_PTH_FILE_PATH)
        except Exception as e:
            print(f"Exception occured while saving model to {MODEL_PTH_FILE_PATH}: {e}")
            continue
        else:
            print(f"Model succesfully saved to: {MODEL_PTH_FILE_PATH}")