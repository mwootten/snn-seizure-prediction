import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
from optparse import OptionParser

parser = OptionParser("usage: %prog [options] inputs")

parser.add_option("-f", "--full",
                  action="store_const", dest="deleteLayers", const=0,
                  help="Run the full neural network")
parser.add_option("-w", "--without",
                  action="store", dest="deleteLayers", type="int", metavar="n",
                  help="Delete n layers off the end of the network")

(options, args) = parser.parse_args()

#   Title: Neural Networks Tutorial
#   Author: Chintala, S
#   Date: 9/14/2017
#   Code version: 1.0
#   Source: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class Net(nn.Module):

    def __init__(self, sizes):
        super(Net, self).__init__()
        # 1x1 square convolution
        self.conv1 = nn.Conv2d(sizes[0], sizes[1], 1)
        self.conv2 = nn.Conv2d(sizes[1], sizes[2], 1)
        self.conv3 = nn.Conv2d(sizes[2], sizes[3], 1)
        # Convolutional to output neuron
        self.fc1 = nn.Linear(sizes[3], 1)

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

model = Net([187, 45, 15, 5])
xtest = []
ytest = []
x = []
y = []
for filename in args:
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

# adding fake dimensions to x for compatibility with convolutions
x = x.unsqueeze(-1).unsqueeze(-1)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
errorTime = []
testErrorTime = []

for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(t, loss.data[0], file=sys.stderr)
    errorTime.append(loss.data[0])
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_pred = model(xtest)
    testLoss = criterion(test_pred, ytest)
    testErrorTime.append(testLoss.data[0])

if options.deleteLayers == 0:
	rawPredictions = model(xtest).data
	predictions = list(np.array(rawPredictions)[:, 0])
	print('\n'.join(map(str, predictions)))
else:
	layers = [model.conv1, model.conv2, model.conv3]
	xs = xtest
	for n in range(4 - options.deleteLayers):
		xs = F.max_pool2d(F.relu((layers[n])(xs)), 1)
	output = np.array(xs.data)[:,:,0,0]
	print(list(map(list, output)))
