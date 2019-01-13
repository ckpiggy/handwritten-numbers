import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, 5)
    self.conv2 = nn.Conv2d(10, 20, 5)
    self.fc1 = nn.Linear(20 * 4 * 4, 10)
    self.pool = nn.MaxPool2d(2)

  def forward(self, x):
    
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 20 * 4 * 4)
    x = F.relu(self.fc1(x))
    return x


