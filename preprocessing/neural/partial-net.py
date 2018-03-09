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
#   Availability: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3756 input neurons, 1x1 square convolution
        # 3 Temporal Convolutions 4256 -> 1418 -> 472 -> 157
        self.conv1 = nn.Conv2d(187, 45, 1)
        self.conv2 = nn.Conv2d(45, 15, 1)
        self.conv3 = nn.Conv2d(15, 5, 1)
        # Convolutional to output neuron
        self.fc1 = nn.Linear(5, 1)

    def forward(self, x):
        # Max pooling over a (1, 1) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 1)
        x = F.max_pool2d(F.relu(self.conv2(x)), 1)
        x = F.max_pool2d(F.relu(self.conv3(x)), 1)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
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
model = Net()
N, D_in, D_out = 45, 187, 1
# batch size. input dimensions, output dimensions
xtest = []
ytest = []
x = []
y = []
for filename in sys.argv:
    if "test" in filename:
        xtest.append(np.fromfile(filename, dtype = np.dtype("i4")) / 10000)
        if "positive" in filename:
            ytest.append(1)
        else:
            ytest.append(0)
    if "training" in filename:
        x.append(np.fromfile(filename, dtype = np.dtype("i4")) / 10000)
        if "positive" in filename:
            y.append(1)
        else:
            y.append(0)
xtest = np.array(xtest)
ytest = np.array(ytest)
x = np.array(x)
y = np.array(y)
x = Variable(torch.from_numpy(x).float())
y = Variable(torch.from_numpy(y).float(), requires_grad=False)
xtest = Variable(torch.from_numpy(xtest).float(), requires_grad=False)
ytest = Variable(torch.from_numpy(ytest).float(), requires_grad=False)
xtest = xtest.unsqueeze(-1).unsqueeze(-1)
x = x.unsqueeze(-1).unsqueeze(-1)
# adding fake dimensions to x for compatibility with convolutions
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
errorTime = []
testErrorTime = []
t = 0

while t < 500:
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
    t += 1

x0 = xtest
x1 = F.max_pool2d(F.relu(model.conv1(x0)), 1)
x2 = F.max_pool2d(F.relu(model.conv2(x1)), 1)
x3 = F.max_pool2d(F.relu(model.conv3(x2)), 1)
output = np.array(x3.data)[:,:,0,0]

output.tofile('ltd4-test.matrix')
