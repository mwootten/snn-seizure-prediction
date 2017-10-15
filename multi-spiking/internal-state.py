neuronThreshold = float(input("Neuron Threshold?"))
synapseWeightA = float(input("Synapse Weight 1?"))
synapseWeightB = float(input("Synapse Weight 2?"))
synapseWeightC = float(input("Synapse Weight 3?"))
synapseWeightD = float(input("Synapse Weight 4?"))
simulationTime = float(input("Simulation Time?"))
neuronInput = float(input("Neuron Input?"))
encodingInterval = float(input("Encoding Interval?"))
refractorinessDecay = float(input("Refractoriness Decay?"))
# setting constants
synapseDelayA = 1
synapseDelayB = 1 + simulationTime/4 - (simulationTime%4)/4
synapseDelayC = synapseDelayB + simulationTime/4 - (simulationTime%4)/4
synapseDelayD = synapseDelayC + simulationTime/4 - (simulationTime%4)/4
# calculating synapse delays so that they cover the simulation time
timeDecay = encodingInterval + 1
# setting time decay based on encoding interval
outputA = 0
outputB = 0
outputC = 0
outputD = 0
outputE = 0
# placeholders for outputs
time = 0
# counter for time
while time <= simulationTime:
  internalState = 0
  if(neuronInput - synapseDelayA + time) > 0:
    internalState = synapseWeightA * ((-neuronInput - synapseDelayA + time) / timeDecay) * 2.71828 ** (1 - ((-neuronInput - synapseDelayA + time) / timeDecay))
  if(neuronInput - synapseDelayB + time) > 0:
    internalState = internalState + synapseWeightB * ((-neuronInput - synapseDelayB + time) / timeDecay) * 2.71828 ** (1 - ((-neuronInput - synapseDelayB + time) / timeDecay))
  if(neuronInput - synapseDelayC + time) > 0:
    internalState = internalState + synapseWeightC * ((-neuronInput - synapseDelayC + time) / timeDecay) * 2.71828 ** (1 - ((-neuronInput - synapseDelayC + time) / timeDecay))
  if(neuronInput - synapseDelayD + time) > 0:
    internalState = internalState + synapseWeightD * ((-neuronInput - synapseDelayB + time) / timeDecay) * 2.71828 ** (1 - ((-neuronInput - synapseDelayD + time) / timeDecay))
  # summing alpha function values for received inputs
  # an input is recieved when the sum of the input time and the delay is equal to the time
  if outputE > 0:
    internalState = internalState - 2 * 2.71828 ** (-1*(time - outputE)/refractorinessDecay)
  else:
    if outputD > 0:
      internalState = internalState - 2 * 2.71828 ** (-1*(time - outputD)/refractorinessDecay)
    else:
      if outputC > 0:
        internalState = internalState - 2 * 2.71828 ** (-1*(time - outputC)/refractorinessDecay)
      else:
        if outputB > 0:
          internalState = internalState - 2 * 2.71828 ** (-1*(time - outputB)/refractorinessDecay)
        else:
          if outputA > 0:
            internalState = internalState - 2 * 2.71828 ** (-1*(time - outputA)/refractorinessDecay)
  # adding the refractoriness term for the most recent output
  if internalState > neuronThreshold:
    if outputA == 0:
      outputA = time
    else:
      if outputB == 0:
        outputB = time
      else:
        if outputC == 0:
          outputC = time
        else:
          if outputD == 0:
            outputD = time
          else:
            if outputE == 0:
              outputE = time
  # storing output time if the neuron outputs
  if time == simulationTime:
    print("Output A is " + str(outputA))
    print("Output B is " + str(outputB))
    print("Output C is " + str(outputC))
    print("Output D is " + str(outputD))
    print("Output E is " + str(outputE))
  # final printout of outputs
  time = time + 1
  # counter
