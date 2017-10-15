neuronThreshold = float(input("Neuron Threshold?"))
synapseNumber = int(input("Synapse Number?"))
synapseWeight = []
for x in range (1, (synapseNumber + 1)):
  synapseWeight.append(float(input("Synapse Weight " + str(x) + "?")))
simulationTime = float(input("Simulation Time?"))
neuronInput = float(input("Neuron Input?"))
encodingInterval = float(input("Encoding Interval?"))
refractorinessDecay = float(input("Refractoriness Decay?"))
# setting constants
synapseDelay = [1]
for x in range (0, synapseNumber-1): 
  synapseDelay.append(synapseDelay[x] + simulationTime/synapseNumber - (simulationTime%synapseNumber)/synapseNumber)
# calculating synapse delays so that they cover the simulation time
timeDecay = encodingInterval + 1
# setting time decay based on encoding interval
output = []
# placeholder for outputs
time = 0
# counter for time
while time <= simulationTime:
  internalState = 0
  for x in range (0, synapseNumber):
    if(-neuronInput - synapseDelay[x] + time) > 0:
      internalState = internalState + synapseWeight[x] * ((-neuronInput - synapseDelay[x] + time)/timeDecay) * 2.71828 ** (1 - ((-neuronInput - synapseDelay[x] + time)/timeDecay))
  # summing alpha function values for received inputs
  # an input is recieved when the sum of the input time and the delay is equal to the time
  if len(output[:]) > 0:
         internalState = internalState - 2 * neuronThreshold * 2.71828 ** (-1*(time - output[-1])/refractorinessDecay)
  # adding the refractoriness term for the most recent output
  if internalState > neuronThreshold:
    output.append(time)
  # storing output time if the neuron outputs
  if time == simulationTime:
    print("Outputs: " + str(output[:]))
  # final printout of outputs
  time = time + 1
  # counter
