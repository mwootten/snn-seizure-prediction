import math
from random import randint

neuronThreshold = 1
synapseNumber = int(input("Number of Synapses?"))
network = []
network.append(int(input("Number of Input Neurons?")))
hiddenLayer = int(input("Number of Hidden Layers?"))
for x in range (1, hiddenLayer + 1):
  network.append(int(input("Number of Neurons in Hidden Layer " + str(x) + "?")))
network.append(int(input("Number of Output Neurons?")))
synapseWeight = []
for w in range (1, len(network)):
  outputNeuronWeight = []
  for x in range (0, network[w]):
    inputNeuronWeight = []
    for y in range (0, network[w - 1]):
      synapsesPerConnection = []
      for z in range (0, synapseNumber):
        synapsesPerConnection.append(1)  
      inputNeuronWeight.append(synapsesPerConnection)
    outputNeuronWeight.append(inputNeuronWeight)
  synapseWeight.append(outputNeuronWeight)
simulationTime = float(input("Simulation Time?"))
neuronInput = []
for x in range (0, network[0]):
    neuronInput.append([0])
encodingInterval = float(input("Encoding Interval?"))
refractorinessDecay = float(input("Refractoriness Decay?"))
# setting constants
synapseDelay = [1]
for x in range (0, synapseNumber-1): 
  synapseDelay.append(synapseDelay[x] + simulationTime/synapseNumber - (simulationTime%synapseNumber)/synapseNumber)
# calculating synapse delays so that they cover the simulation time
timeDecay = encodingInterval + 1
# setting time decay based on encoding interval
networkOutput = []
for a in range (0, len(network)-1):
  output = []
  for b in range (0, network[a + 1]):
    # placeholder for outputs
    time = 0
    # counter for time
    output = []
    while time <= simulationTime:
      internalState = 0
      for x in range (0, network[a]):
        for z in range (0, len(neuronInput[x])):
          for y in range (0, synapseNumber):
            adjustedTime = -neuronInput[x][z] - synapseDelay[y] + time
            if adjustedTime > 0:
              internalState = internalState + synapseWeight[a][b][x][y] * (adjustedTime/timeDecay) * math.exp(1 - (adjustedTime/timeDecay))
    # summing alpha function values for received inputs
    # an input is recieved when the sum of the input time and the delay is equal to the time
      if len(output) > 0:
            internalState = internalState - 2 * neuronThreshold * math.exp(-1*(time - output[-1])/refractorinessDecay)
    # adding the refractoriness term for the most recent output
      if internalState > neuronThreshold:
          output.append(time)
  # storing output time if the neuron outputs
      if time == simulationTime:
          print("Outputs: " + str(output))
          networkOutput.append(output)
  # final printout of outputs
      time = time + 1
      # counter
  neuronInput = networkOutput
synapseWeight = []
for w in range (1, len(network)):
  outputNeuronWeight = []
  for x in range (0, network[w]):
    inputNeuronWeight = []
    for y in range (0, network[w - 1]):
      synapsesPerConnection = []
      for z in range (0, synapseNumber):
        synapsesPerConnection.append(randint(1,10))  
      inputNeuronWeight.append(synapsesPerConnection)
    outputNeuronWeight.append(inputNeuronWeight)
  synapseWeight.append(outputNeuronWeight)
sumSynapseWeight = 0
for w in range (1, len(network)):
  for x in range (0, network[w]):
    for y in range (0, network[w-1]):
      for z in range (0, synapseNumber):
        sumSynapseWeight = sumSynapseWeight + synapseWeight[w-1][x][y][z]
weightNumber = 0
for w in range (1, len(network)):
  for x in range (0, network[w]):
    for y in range (0, network[w-1]):
      for z in range (0, synapseNumber):
        weightNumber = weightNumber + 1
meanSynapseWeight = sumSynapseWeight / weightNumber
for w in range (1, len(network)):
  for x in range (0, network[w]):
    for y in range (0, network[w-1]):
      for z in range (0, synapseNumber):
        synapseWeight[w-1][x][y][z] = synapseWeight[w-1][x][y][z]/(meanSynapseWeight * networkOutput[-1][0])
