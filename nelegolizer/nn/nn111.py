import torch
from nelegolizer.nn.dataset import CustomVoxelsDataset
import nelegolizer.nn.common as comm
from torch.utils.data import DataLoader
from torch import nn
from nelegolizer.data import LegoBrick, LegoBrickList
from nelegolizer import constants as CONST

# load data
train_labels_filename = 'labels_111.csv'
test_labels_filename = 'labels_111.csv'

training_data = CustomVoxelsDataset(CONST.PATH + CONST.DIR_LABELS + train_labels_filename, '.')
test_data = CustomVoxelsDataset(CONST.PATH + CONST.DIR_LABELS + test_labels_filename, '.')

train_dataloader = DataLoader(training_data, batch_size=60, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=30, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
if comm.debug:
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

# create model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Conv1d(60, 32, 8, stride=8),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(comm.device)

# train
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def train():
    epochs = 6
    for t in range(epochs):
        if comm.debug:
            print(f"Epoch {t+1}\n-------------------------------")
        comm.train(train_dataloader, model, loss_fn, optimizer)
        comm.test(test_dataloader, model, loss_fn)
if comm.debug:
    print("Done!")