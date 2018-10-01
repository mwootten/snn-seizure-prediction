import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys

#   Title: Neural Networks Tutorial
#   Author: Chintala, S
#   Date: 9/14/2017
#   Code version: 1.0
#   Source: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

class Net(nn.Module):

    def __init__(self, sizes):
        super(Net, self).__init__()
        # 2x2 square convolution
        self.conv1 = nn.Conv2d(sizes[0], sizes[1], 1)
        self.conv2 = nn.Conv2d(sizes[1], sizes[2], 1)
        self.conv3 = nn.Conv2d(sizes[2], sizes[3], 1)
        self.conv4 = nn.Conv2d(sizes[3], sizes[4], 1)
        # Convolutional to linear neuron
        self.fc1 = nn.Linear(sizes[5], sizes[6])
        self.fc2 = nn.Linear(sizes[6], 1)

    def forward(self, x):
        # Max pooling over a (1, 1) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 1)
        x = F.max_pool2d(F.relu(self.conv2(x)), 1)
        x = F.max_pool2d(F.relu(self.conv3(x)), 1)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#   Title: PyTorch with Examples
#   Author: Johnson, J
#   Date: 2017
#   Code version: 1.0
#   Availability: http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#nn-module

model = Net([1, 68, 23, 10, 5, 2700, 64])
# sample test/train data, telling if the inputs are from identical types of function either tan or cos
xtest = []
ytest = []
x = []
y = []
for a in range(4):
  func1 = []
  func2 = []
  for b in range(135):
    func1.append(math.cos(b+a))
  for b in range(135):
    func2.append(math.cos(b-a))
  xtest.append([[func1,func2]])
  ytest.append([1])
for a in range(4):
  func1 = []
  func2 = []
  for b in range(135):
    func1.append(math.tan(b+a))
  for b in range(135):
    func2.append(math.tan(b-a))
  xtest.append([[func1,func2]])
  ytest.append([1])
for a in range(4):
  func1 = []
  func2 = []
  for b in range(135):
    func1.append(math.tan(b+a))
  for b in range(135):
    func2.append(math.cos(b-a))
  xtest.append([[func1,func2]])
  ytest.append([0])
for a in range(4):
  func1 = []
  func2 = []
  for b in range(135):
    func1.append(math.cos(b+a))
  for b in range(135):
    func2.append(math.tan(b-a))
  xtest.append([[func1,func2]])
  ytest.append([0])
for a in range(16):
  func1 = []
  func2 = []
  for b in range(135):
    func1.append(math.cos(b+a))
  for b in range(135):
    func2.append(math.cos(b-a))
  x.append([[func1,func2]])
  y.append([1])
for a in range(16):
  func1 = []
  func2 = []
  for b in range(135):
    func1.append(math.tan(b+a))
  for b in range(135):
    func2.append(math.tan(b-a))
  x.append([[func1,func2]])
  y.append([1])
for a in range(16):
  func1 = []
  func2 = []
  for b in range(135):
    func1.append(math.tan(b+a))
  for b in range(135):
    func2.append(math.cos(b-a))
  x.append([[func1,func2]])
  y.append([0])
for a in range(16):
  func1 = []
  func2 = []
  for b in range(135):
    func1.append(math.cos(b+a))
  for b in range(135):
    func2.append(math.tan(b-a))
  x.append([[func1,func2]])
  y.append([0])
xtest = np.array(xtest)
ytest = np.array(ytest)
x = np.array(x)
y = np.array(y)


x = Variable(torch.from_numpy(x).float())
y = Variable(torch.from_numpy(y).float(), requires_grad=False)
xtest = Variable(torch.from_numpy(xtest).float(), requires_grad=False)
ytest = Variable(torch.from_numpy(ytest).float(), requires_grad=False)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
errorTime = []
testErrorTime = []

for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(t, loss.data[0])
    errorTime.append(loss.data[0])
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_pred = model(xtest)
    testLoss = criterion(test_pred, ytest)
    testErrorTime.append(testLoss.data[0])

import matplotlib.pyplot as plt
xs = range(len(errorTime))
ys = errorTime
plt.plot(xs, ys)
xsb = range(len(testErrorTime))
ysb = testErrorTime
plt.plot(xsb, ysb)
plt.show()
