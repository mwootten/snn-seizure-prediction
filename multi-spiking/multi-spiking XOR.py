import math
import random
from copy import deepcopy
import json
import time

start = time.time()

random.seed(1)

# Constants
neuronThreshold = 1
synapseNumber = 4
network = [3, 5, 1]
simulationTime = 25
encodingInterval = 6
refractorinessDecay = 80
latestOutputSpike = 16
timeDecay = encodingInterval + 1

# calculating synapse delays so that they cover the simulation time
synapseDelayDelta = int(latestOutputSpike / synapseNumber)
synapseDelay = [1 + (n * synapseDelayDelta) for n in range(synapseNumber)]

# Organization of synapse weights:
# > layer
#   > number of neuron in layer
#     > number of layer inputted from
#       > synapse number
synapseWeight = []
# Initialize all weights to 1
for w in range(1, len(network)):
    outputNeuronWeight = []
    for x in range(0, network[w]):  # w: 1--len
        inputNeuronWeight = []
        for y in range(0, network[w - 1]):  # w-1: 0--len-1
            synapsesPerConnection = []
            for z in range(0, synapseNumber):
                synapsesPerConnection.append(1)
            inputNeuronWeight.append(synapsesPerConnection)
        outputNeuronWeight.append(inputNeuronWeight)
    synapseWeight.append(outputNeuronWeight)

neuronInput = []
for x in range(0, network[0]):
    neuronInput.append([0])


def alpha(t):
    if t > 0:
        return (t / timeDecay) * math.exp(1 - (t/timeDecay))
    else:
        return 0


def refractoriness(time):
    if time > 0:
        return -2 * neuronThreshold * math.exp(-time/refractorinessDecay)
    else:
        return 0


# Modifies the global layerOutput
def runNetwork(neuronInput, synapseWeight):
    inp = deepcopy(neuronInput)
    networkOutput = [deepcopy(neuronInput)]
    networkInternalState = []

    for a in range(0, len(network) - 1):
        outputNeuronNetworkOutput = []
        layerInternalState = []
        for b in range(0, network[a + 1]):
            # placeholder for outputs
            time = 0
            # counter for time
            output = []
            neuronInternalState = []
            while time <= simulationTime:
                activationFunction = (time > min(min(inp)))
                internalState = 0
                if len(output) > 0:
                    activationFunction = ((time - output[-1]) > 2)
                    # heuristic rule: prevent spikes that are <= 2 ms apart
                if a == (len(network) - 2):
                    activationFunction *= (1 - len(output))
                if activationFunction == 1:
                    for x in range(0, network[a]):
                        for y in range(0, synapseNumber):
                            for z in range(0, len(inp[x])):
                                adjustedTime = -inp[x][z] - synapseDelay[y] + time
                                internalState += synapseWeight[a][b][x][y] * alpha(adjustedTime)
            # summing alpha function values for all received inputs to a neuron
            # an input is recieved from the previous layer when the sum of the
            # input time and the delay is equal to the time
                    if (len(output) > 0)*(internalState > neuronThreshold):
                        internalState += refractoriness(time - output[-1])
                # adding the refractoriness term for the most recent output
                    if internalState > neuronThreshold:
                        output.append(time)
                # storing output time if the neuron outputs
                neuronInternalState.append(internalState)
                if time == simulationTime:
                    if len(output) == 0:
                        output.append(neuronInternalState.index(max(neuronInternalState)))
                        # heuristic rule
                    layerOutput.append(output)
                time = time + 1
            layerInternalState.append(neuronInternalState)
            outputNeuronNetworkOutput.append(output)
        networkInternalState.append(layerInternalState)
        inp = deepcopy(layerOutput)
        networkOutput.append(outputNeuronNetworkOutput)
    return networkOutput

layerOutput = []
runNetwork(neuronInput, synapseWeight)
sumSynapseWeight = 0
weightNumber = 0
for w in range (1, len(network)):
  for x in range (0, network[w]):
    for y in range (0, network[w-1]):
      for z in range (0, synapseNumber):
        synapseWeight[w-1][x][y][z] = random.uniform(1.0,10.0)
        sumSynapseWeight = sumSynapseWeight + synapseWeight[w-1][x][y][z]
        weightNumber = weightNumber + 1
meanSynapseWeight = sumSynapseWeight / weightNumber
# randomizing synapse weights
# calculating average synapse weight
for w in range (1, len(network)):
  for x in range (0, network[w]):
    for y in range (0, network[w-1]):
      for z in range (0, synapseNumber):
        synapseWeight[w-1][x][y][z] /= (meanSynapseWeight * layerOutput[-1][0])
# reducing the synapse weights based on the average and network output
# reduces the number of outputs from each neuron
# print("Network Outputs: " + str(networkOutput))
learningRate = 0.005  # float(input("Learning Rate?"))
maxEpoch = 500  # int(input("Max Epochs?"))
iteration = 1
examplesPerEpoch = 4
errorTime = []
trainingData = [
    ([[0], [0], [0]], [10]),
    ([[0], [6], [0]], [16]),
    ([[6], [0], [0]], [16]),
    ([[6], [6], [0]], [10])
]
examples = []
for epoch in range(maxEpoch):
    examples.extend(deepcopy(trainingData))
for (neuronInput, expectedOutput) in examples:
  layerOutput = []
  networkOutput = runNetwork(neuronInput, synapseWeight)

  print("COMPARE: {} vs {}".format(networkOutput[-1][0], expectedOutput))

  error = 0.5*sum([
    (networkOutput[-1][x][0] - expectedOutput[x])**2
    for x in range(len(expectedOutput))
  ])

  for (neurons, zorblax) in trainingData:
      name = '-'.join(list(map(lambda x: str(x[0]), neurons)))
      actualOutput = runNetwork(neurons, synapseWeight)
      print(actualOutput)
      if actualOutput[-1][0] == zorblax:
          print("Correct for " + name + "   => " + str(zorblax))
      else:
          print("Incorrect for " + name + " => " + str(zorblax))
  print("")

  errorTime.append(error)
  previousSynapseWeight = deepcopy(synapseWeight)

  # Backpropagate through the final level of synapses: from output to the last
  # hidden layer
  for x in range (0, network[-1]):
    for y in range (0, network[-2]):
      for z in range (0, synapseNumber):
        errorOutput = networkOutput[-1][x][0] - expectedOutput[x]
        internalStateWeight = 0
        for a in range (0, len(networkOutput[-2][y])):
          adjustedTimeOutput = networkOutput[-1][x][0] - networkOutput[-2][y][a] - synapseDelay[z]
          internalStateWeight += alpha(adjustedTimeOutput)
        denominatorOutputInternalState = 0
        for a in range (0, network[-2]):
          for b in range (0, len(networkOutput[-2][a])):
            for c in range (0, synapseNumber):
              adjustedTimeOutput = networkOutput[-1][x][0] - networkOutput[-2][a][b] - synapseDelay[c]
              if adjustedTimeOutput > 0:
                alphaFunctionOutput = previousSynapseWeight[-1][x][a][c] * alpha(adjustedTimeOutput)
                denominatorOutputInternalState += alphaFunctionOutput*(1/adjustedTimeOutput - 1/timeDecay)
        if abs(denominatorOutputInternalState) < 0.1:
          denominatorOutputInternalState = 0.1 - 0.2*(denominatorOutputInternalState < 0)
        outputInternalState = -1 / (denominatorOutputInternalState)
        errorGradient = errorOutput * outputInternalState * internalStateWeight
        synapseWeight[-1][x][y][z] = previousSynapseWeight[-1][x][y][z] - learningRate*errorGradient

  # Backpropagate through all the remaining layers
  for w in range (1, len(network)-1):
    for x in range (0, network[w]):
      for y in range (0, network[w-1]):
        for z in range (0, synapseNumber):
          errorGradient = 0
          for a in range (0, network[-1]):
            errorOutput = networkOutput[-1][a][0] - expectedOutput[a]
            denominatorOutputInternalState = 0
            for b in range (0, network[-2]):
              for c in range (0, len(networkOutput[-2][b])):
                for d in range (0, synapseNumber):
                  adjustedTimeOutput = networkOutput[-1][a][0] - networkOutput[-2][b][c] - synapseDelay[d]
                  if adjustedTimeOutput > 0:
                    alphaFunctionOutput = previousSynapseWeight[-1][a][b][d] * alpha(adjustedTimeOutput)
                    denominatorOutputInternalState += alphaFunctionOutput*(1/adjustedTimeOutput - 1/timeDecay)
            if abs(denominatorOutputInternalState) < 0.1:
              denominatorOutputInternalState = 0.1 - 0.2*(denominatorOutputInternalState < 0)
            outputInternalState = -1 / (denominatorOutputInternalState)
            internalStateInputSum = 0
            if errorOutput != 0:
                for b in range (0, len(networkOutput[w][x])):
                    for c in range (0, synapseNumber):
                        adjustedTimeInput = networkOutput[-1][a][0] - networkOutput[w][x][b] - synapseDelay[c]
                        alphaFunctionInput = previousSynapseWeight[w-1][x][y][c] * alpha(adjustedTimeInput)
                        if adjustedTimeInput > 0:
                            internalStateInputSum = internalStateInputSum + alphaFunctionInput*(1/adjustedTimeInput - 1/timeDecay)
                internalStateInput = -1*internalStateInputSum
            errorInput = errorOutput*outputInternalState*internalStateInput
            inputWeight = 0
            for b in range (0, len(networkOutput[w][x])):
              if b == 0:
                inputInternalStateDenominator = 0
                for c in range (0, network[w-1]):
                  for d in range (0, len(networkOutput[w-1][c])):
                    for e in range (0, synapseNumber):
                      adjustedTimeInput = networkOutput[w][x][b] - networkOutput[w-1][c][d] - synapseDelay[e]
                      alphaFunctionInput = previousSynapseWeight[w-1][x][c][e] * alpha(adjustedTimeInput)
                      if adjustedTimeInput > 0:
                        inputInternalStateDenominator += alphaFunctionInput*(1/adjustedTimeInput - 1/timeDecay)
                if abs(inputInternalStateDenominator) < 0.1:
                    inputInternalStateDenominator = 0.1 - 0.2*(inputInternalStateDenominator < 0)
                inputInternalState = -1/inputInternalStateDenominator
                internalStateWeight = 0
                for c in range (0, len(networkOutput[w-1][y])):
                  adjustedTimeInput = networkOutput[w][x][b] - networkOutput[w-1][y][c] - synapseDelay[z]
                  if adjustedTimeInput > 0:
                    internalStateWeight += alpha(adjustedTimeInput)
                inputWeight = inputInternalState*internalStateWeight
              if b > 0:
                inputInternalStateDenominator = 0
                for c in range (0, network[w-1]):
                  for d in range (0, len(networkOutput[w-1][c])):
                    for e in range (0, synapseNumber):
                      adjustedTimeInput = networkOutput[w][x][b] - networkOutput[w-1][c][d] - synapseDelay[e]
                      adjustedTimeRefractoriness = networkOutput[w][x][b] - networkOutput[w][x][b-1]
                      refractorinessInput = refractoriness(adjustedTimeRefractoriness)
                      if adjustedTimeInput > 0:
                        alphaFunctionInput = previousSynapseWeight[w-1][x][c][e] * alpha(adjustedTimeInput)
                        inputInternalStateDenominator += alphaFunctionInput*(1/adjustedTimeInput - 1/timeDecay) + (2*neuronThreshold/refractorinessDecay)*refractorinessInput
                      else:
                        inputInternalStateDenominator += (2*neuronThreshold/refractorinessDecay)*refractorinessInput
                if abs(inputInternalStateDenominator) < 0.1:
                    inputInternalStateDenominator = 0.1 - 0.2*(inputInternalStateDenominator < 0)
                inputInternalState = -1/inputInternalStateDenominator
                internalStateWeight = 0
                for c in range (0, len(networkOutput[w-1][y])):
                  adjustedTimeInput = networkOutput[w][x][b] - networkOutput[w-1][y][c] - synapseDelay[z]
                  internalStateWeight += alpha(adjustedTimeInput)
                adjustedTimeRefractoriness = networkOutput[w][x][b] - networkOutput[w][x][b-1]
                refractorinessInput = refractoriness(adjustedTimeRefractoriness)
                internalStateInput = (2*neuronThreshold/refractorinessDecay)*refractorinessInput
                inputWeight = inputInternalState*(internalStateWeight+internalStateInput*inputWeight)
            errorGradient = errorGradient + errorInput*inputWeight
          synapseWeight[w-1][x][y][z] = previousSynapseWeight[w-1][x][y][z] - learningRate*errorGradient
end = time.time()
print("Time = " + str(end - start))
print("")
print("")

print("Network output:")
for (neurons, expectedOutput) in trainingData:
    name = '-'.join(list(map(lambda x: str(x[0]), neurons)))
    actualOutput = runNetwork(neurons, synapseWeight)
    print(actualOutput)
    if actualOutput[-1][0] == expectedOutput:
        print("Correct for " + name + "   => " + str(expectedOutput))
    else:
        print("Incorrect for " + name + " => " + str(expectedOutput))

print()
print(errorTime)

"""
import matplotlib.pyplot as plt
xs = range(len(errorTime))
ys = errorTime
plt.plot(xs, ys)
plt.show()
"""
