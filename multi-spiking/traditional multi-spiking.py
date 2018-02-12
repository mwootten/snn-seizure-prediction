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

synapseWeight = []
for w in range(1, len(network)):
    outputNeuronWeight = []
    for x in range(0, network[w]):  # w: 1--len
        inputNeuronWeight = []
        for y in range(0, network[w - 1]):  # w-1: 0--len-1
            synapsesPerConnection = []
            for z in range(0, synapseNumber):
                synapsesPerConnection.append(1)
            inputNeuronWeight.append(Variable(torch.FloatTensor(synapsesPerConnection)))
        outputNeuronWeight.append(inputNeuronWeight)
    synapseWeight.append(outputNeuronWeight)
# Organization of synapse weights:
# > layer
#   > number of neuron in layer
#     > number of layer inputted from
#       > synapse number
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
    for x in range(0, simulationTime):
        internalState = Variable(torch.zeros(sum(network)).unsqueeze(-1),requires_grad = False)
        refractorinessValue = Variable(torch.zeros(sum(network)),requires_grad = False)
        for y in range(0, synapseNumber):
            for z in range(0, len(outputMatrices)):
                internalState = internalState + torch.mm(weightMatrices[y],(alpha(x - outputMatrices[z] * timeMatrix[z*sum(network):(z+1)*sum(network)] - synapseDelay[y])*outputMatrices[z]).unsqueeze(-1))
        for y in range(0, len(outputMatrices)):
            refractorinessValue = refractorinessValue + refractoriness((x - outputMatrices[y]*y))*outputMatrices[y] - refractorinessValue*outputMatrices[y]
        internalState = internalState + refractorinessValue.unsqueeze(-1)       
        output = ((internalState - 1)/(torch.abs(internalState - 1)) + 1)/2
        outputMatrix = inputMatrices[x + 1] + output.squeeze(1)
        outputMatrices.append(outputMatrix)

outputMatrices = [inputMatrices[0]]
runNetwork(neuronInput)
