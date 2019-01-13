import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from model import Net

def load_data(train_batch_size, test_batch_size):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
  ])
  trainset = torchvision.datasets.MNIST(root='./data', transform=transform, download=True)
  trainloader = torch.utils.data.DataLoader(trainset, train_batch_size, shuffle=True)

  testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
  testloader = torch.utils.data.DataLoader(testset, test_batch_size, shuffle=True)

  return (trainloader, testloader)

def train(model, optimizer, epoch, trainloader, log_interval):
  model.train()
  cost_fn = nn.CrossEntropyLoss()
  for i, (inputs, labels) in enumerate(trainloader):
    optimizer.zero_grad()
    output = model(inputs)
    cost = cost_fn(output, labels)
    cost.backward()
    optimizer.step()

    if i % log_interval == 0:
      print('Train epoch: {0} [{1} / {2}]\tcost:{3}'.format(
        epoch, 
        i * len(inputs), 
        len(trainloader.dataset), 
        cost.data.item()))

def evaluate (model, trainloader, testloader):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, labels in trainloader:
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the train images: %d %%' % (
        100 * correct / total))
    correct = 0
    total = 0
    for inputs, labels in testloader:
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='PyTorch MNIST Demo')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                      help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                      help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                      help='learning rate (default: 0.01)')
  parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                      help='how many batches to wait before logging training status')
  args = parser.parse_args()

  model = Net()

  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
  # optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

  trainloader, testloader = load_data(args.batch_size, args.test_batch_size)

  for epoch in range(1, args.epochs + 1):
    train(model, optimizer, epoch, trainloader, args.log_interval)
  
  evaluate(model, trainloader, testloader)

  package_dir = os.path.dirname(os.path.abspath(__file__))
  model_path = os.path.join(package_dir, 'model')
  torch.save(model.state_dict(), model_path)

