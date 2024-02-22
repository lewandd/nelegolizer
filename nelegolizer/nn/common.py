import torch
import nelegolizer.nn.utils as ut
from nelegolizer.data import LegoBrick, LegoBrickList

debug = False

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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
    if debug:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_predict(model, group):
    group_float = list(map(float, group))
    X = torch.tensor([group_float]).to(device)
    pred = model(X)
    return pred.argmax(1).item()

fill_treshold = 0.1

def get_brick(model, group, gres, position):
  """Choose brick most matching the given voxel group

  Args:
    model (NeuralNetwork) : neural network model
    group (list) : list of bools with shape (gres, gres, gres)
    gres (int) : used to determine shape

  Returns:
    LegoBrickList : list of LegoBrick containing single chosen brick
  """
  best_rotation = ut.find_best_rotation(group, gres)
  group = ut.rotate_group(group, gres, best_rotation)

  fill = 0
  for i in range(gres):
     for j in range(gres):
        for k in range(gres):
           if (group[i][j][k]):
              fill += 1
  #fill = list(filter(lambda a: a != 0, group))
  count = fill/(gres*gres*gres)
  if count > fill_treshold:
    label = test_predict(model, group.flatten())
    lego_brick = LegoBrick(label, position, best_rotation)
    #333print(lego_brick)
    return LegoBrickList([lego_brick])
  else:
     return None
