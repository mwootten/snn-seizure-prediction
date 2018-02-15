import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import json

neuronThreshold = 1

if True:  #  input("Use defaults? ") == "yes":
    random.seed(1)
    useConstantInput = False
    synapseNumber = 4
    network = [3, 5, 1]
    simulationTime = 25
    encodingInterval = 6
    refractorinessDecay = 80
    latestOutputSpike = 16
else:
    random.seed(int(input("Random seed? ")))
    useConstantInput = (input("Train on same input (yes/no)? ") == "yes")
    if useConstantInput:
        [x_in, y_in] = json.loads(input("Which input (format [a, b])? "))
        constantInput = [[x_in], [y_in], [0]]
    synapseNumber = int(input("Number of Synapses?"))
    network = []
    network.append(3)
    hiddenLayer = int(input("Number of Hidden Layers?"))
    for x in range(1, hiddenLayer + 1):
        network.append(int(input("Number of Neurons in Hidden Layer " + str(x) + "?")))
    network.append(1)
    # describing network structure as neurons per layer
    simulationTime = float(input("Simulation Time?"))
    encodingInterval = float(input("Encoding Interval?"))
    refractorinessDecay = float(input("Refractoriness Decay?"))
    latestOutputSpike = int(input("Latest Output Spike?"))

weightMatrices = []
for w in range(synapseNumber):
    weightMatrix = []
    for x in range(sum(network)):
        matrixRow = []
        for y in range(sum(network)):
            if (x < network[0]):
                matrixRow.append(0)
            else:
                if x < (sum(network[0:2])):
                    matrixRow.append(y < network[0])
                else: 
                    matrixRow.append((y >= network[0])*(y != (sum(network)-1)))
        weightMatrix.append(matrixRow)
    initialWeightMatrix = torch.FloatTensor(weightMatrix)
    weightMatrix = Variable(torch.FloatTensor(weightMatrix))
    weightMatrices.append(weightMatrix)


neuronInput = []
for x in range(0, network[0]):
    singleInput = [1]
    for y in range(0, simulationTime):
        singleInput.append(0)
    neuronInput.append(singleInput)
synapseDelayDelta = int(latestOutputSpike / synapseNumber)
synapseDelay = [1 + (n * synapseDelayDelta) for n in range(synapseNumber)]
# calculating synapse delays so that they cover the simulation time
timeDecay = encodingInterval + 1
# setting time decay based on encoding interval
inputMatrices = []
for w in range(0, simulationTime + 1):
    inputMatrix = []
    for x in range(0, network[0]):
        inputMatrix.append(neuronInput[x][w])
    for x in range(0, sum(network[1:len(network)])):
        inputMatrix.append(0)
    inputMatrix = Variable(torch.FloatTensor(inputMatrix), requires_grad = False)
    inputMatrices.append(inputMatrix)
# initial inputs to network at time 0 
timeMatrix = []
for x in range(0, simulationTime + 1):
    for y in range(0, sum(network)):
        timeMatrix.append(x)
timeMatrix = Variable(torch.FloatTensor(timeMatrix), requires_grad = False)

def alpha(t):
    return (t / timeDecay) * torch.exp(1 - (t/timeDecay)) * ((t + 0.0000001)/torch.abs(t + 0.0000001) + 1)/2

def refractoriness(time):
    return -2 * neuronThreshold * torch.exp(-1*time/refractorinessDecay) * ((time + 0.0000001)/torch.abs(time + 0.0000001) + 1)/2

def runNetwork(neuronInput):
    for x in range(1, simulationTime + 1):
        internalState = Variable(torch.zeros(sum(network)).unsqueeze(-1),requires_grad = False)
        refractorinessValue = Variable(torch.zeros(sum(network)),requires_grad = False)
        for y in range(0, synapseNumber):
            for z in range(0, len(outputMatrices)):
                internalState = internalState + torch.mm(weightMatrices[y],(alpha(x - outputMatrices[z] * timeMatrix[z*sum(network):(z+1)*sum(network)] - synapseDelay[y])*outputMatrices[z]).unsqueeze(-1))
        for y in range(0, len(outputMatrices)):
            refractorinessValue = refractorinessValue + refractoriness((x - outputMatrices[y]*y))*outputMatrices[y] - refractorinessValue*outputMatrices[y]
        internalState = internalState + refractorinessValue.unsqueeze(-1)       
        output = ((internalState - 1)/(torch.abs(internalState - 1)) + 1)/2
        outputMatrix = inputMatrices[x] + output.squeeze(1)
        outputMatrices.append(outputMatrix)

outputMatrices = [inputMatrices[0]]
runNetwork(neuronInput)
weightMatrices = []
for w in range(synapseNumber):
    weightMatrix = []
    for x in range(sum(network)):
        matrixRow = []
        for y in range(sum(network)):
            if (x < network[0]):
                matrixRow.append(0)
            else:
                if x < (sum(network[0:2])):
                    matrixRow.append((y < network[0])*(random.uniform(1.0,10.0)))
                else: 
                    matrixRow.append((y >= network[0])*(y != (sum(network)-1))*(random.uniform(1.0,10.0)))
        weightMatrix.append(matrixRow)
    weightMatrix = torch.FloatTensor(weightMatrix)
    weightMatrices.append(weightMatrix)
weightSum = 0
for x in range(len(weightMatrices)):
    weightSum = weightSum + sum(sum(weightMatrices[x]))
weightNumber = (network[0]+network[2])*network[1]*synapseNumber
meanWeight = weightSum/weightNumber
networkOutput = 0
for x in range(simulationTime+1):
    networkOutput = networkOutput + outputMatrices[-x][-1]*(simulationTime-x+1) - networkOutput*outputMatrices[-x][-1]
for x in range(len(weightMatrices)):
    weightMatrices[x] = Variable(weightMatrices[x]/(meanWeight * networkOutput).data, requires_grad = True)

inputData = [[0, 0, 0], [0, 6, 0], [0, 0, 6], [0, 6, 6]]
outputData = [[10], [16], [16], [10]]
learningRate = 0.005
epochErrorTime = []
for a in range(500):
    epochError = []
    for b in range(len(inputData)):
        neuronInput = []
        for x in range(0, network[0]):
            singleInput = []
            for y in range(0, simulationTime+1):
                singleInput.append(y == inputData[b][x])
            neuronInput.append(singleInput)
        inputMatrices = []
        for w in range(0, simulationTime + 1):
            inputMatrix = []
            for x in range(0, network[0]):
                inputMatrix.append(neuronInput[x][w])
            for x in range(0, sum(network[1:len(network)])):
                inputMatrix.append(0)
            inputMatrix = Variable(torch.FloatTensor(inputMatrix), requires_grad = False)
            inputMatrices.append(inputMatrix) 
        outputMatrices = [inputMatrices[0]]
        runNetwork(neuronInput)      
        networkOutput = 0
        for x in range(simulationTime+1):
            networkOutput = networkOutput + outputMatrices[-x][-1]*(simulationTime-x+1) - networkOutput*outputMatrices[-x][-1]
        error = 0.5*(networkOutput - outputData[b][0])**2
        epochError.append(error)
    weightShifts = []
    for b in range(synapseNumber):
        weightShifts.append(torch.zeros(9, 9))
    for b in range(len(epochError)):
        epochError[b].backward()
        for c in range(synapseNumber):
            weightShifts[c] = (weightShifts[c] - learningRate * weightMatrices[c].grad.data)*initialWeightMatrix
    for b in range(synapseNumber):
        weightMatrices[b] = Variable(weightMatrices[b].data + weightShifts[b], requires_grad = True)
    epochErrorTime.append(sum(epochError)/len(epochError))
    print(sum(epochError)/len(epochError))
