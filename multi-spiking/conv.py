import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt

#   Title: Neural Networks Tutorial
#   Author: Chintala, S
#   Date: 9/14/2017
#   Code version: 1.0
#   Availability: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3756 input neurons, 1x1 square convolution
        # 3 Temporal Convolutions 4696 -> 1565 -> 521 -> 173
        self.conv1 = nn.Conv2d(4696, 1565, 1)
        self.conv2 = nn.Conv2d(1565, 521, 1)
        self.conv3 = nn.Conv2d(521, 173, 1)
        # Convolutional to output neuron
        self.fc1 = nn.Linear(173, 1)

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
N, D_in, D_out = 192, 4696, 1
# batch size. input dimensions, output dimensions
x = []
y = []
for a in range (0, N):
  xbatch = []
  ybatch = []
  if (a < N/2):
    ybatch.append(1)
  else:
    ybatch.append(0)
  for b in range (0, D_in):
    if (a < N/2):
      xbatch.append(math.cos(a+b))
    else:
      xbatch.append(random.uniform(-1.0,1.0))
  x.append(xbatch)
  y.append(ybatch)
x = np.array(x)
y = np.array(y)
x = Variable(torch.from_numpy(x).float())
# x = Variable(torch.randn(N, D_in))
y = Variable(torch.from_numpy(y).float(), requires_grad=False)
# y = Variable(torch.randn(N, D_out), requires_grad=False)
x = x.unsqueeze(-1)
x = x.unsqueeze(-1)
# test data
xtest = []
ytest = []
for a in range (0, int(N/4)):
  xbatch = []
  ybatch = []
  if (a < N/8):
    ybatch.append(1)
  else:
    ybatch.append(0)
  for b in range (0, D_in):
    if (a < N/8):
      xbatch.append(math.sin(a+b))
    else:
      xbatch.append(random.uniform(-1.0,1.0))
  xtest.append(xbatch)
  ytest.append(ybatch)
xtest = np.array(xtest)
ytest = np.array(ytest)
xtest = Variable(torch.from_numpy(xtest).float(), requires_grad=False)
# x = Variable(torch.randn(N, D_in))
ytest = Variable(torch.from_numpy(ytest).float(), requires_grad=False)
# y = Variable(torch.randn(N, D_out), requires_grad=False)
xtest = xtest.unsqueeze(-1)
xtest = xtest.unsqueeze(-1)
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
    ## if t > 2:
    ##    if abs(errorTime[-1] - testErrorTime[-1]) > abs(errorTime[-2] - testErrorTime[-2]):
    ##        t += 500

xs = range(len(errorTime))
ys = errorTime
plt.plot(xs, ys)
xstest = range(len(testErrorTime))
ystest = testErrorTime
plt.plot(xstest,ystest)
plt.show()
