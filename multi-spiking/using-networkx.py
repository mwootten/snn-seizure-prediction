import networkx as nx
from enum import Enum
import math

layerSizes = [3, 5, 1]
synapsesPerConnection = 4
lastOutput = 16
simulationTime = 25
refractorinessDecay = 80
encodingInterval = 6
timeDecay = 7
neuronThreshold = 1
xorInput = [True, True]


class NeuronType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

    def __str__(self):
        lookup = {
            NeuronType.INPUT: "input",
            NeuronType.HIDDEN: "hidden",
            NeuronType.OUTPUT: "output"
        }
        return lookup[self]


class Neuron:
    """docstring for Neuron."""
    def __init__(self, spikeTimes, internalState, type):
        self.spikeTimes = spikeTimes
        self.internalState = internalState
        self.type = type

    def __repr__(self):
        return str(self.type)

    def timeSinceLastSpike(self, time):
        if len(self.spikeTimes) == 0:
            return float('inf')
        else:
            return time - self.spikeTimes[-1]

    def update(self, internalState, time):
        print(" -- ".join(map(repr, [self, internalState, time])))
        self.internalState = internalState
        if internalState > neuronThreshold:
            # heuristic to prevent rapid repeated spikes
            if postneuron.timeSinceLastSpike(time) > 2:
                self.spikeTimes.append(time)


def createLayer(layerSize, type):
    return [Neuron([], 0, type) for x in range(layerSize)]


def synapseDelay(synapseNumber):
    return 1 + synapseNumber * int(lastOutput / synapsesPerConnection)


def spikeResponse(time):
    if time > 0:
        return (time/timeDecay) * math.exp(1 - (time/timeDecay))
    else:
        return 0


def refractoriness(time):
    if time > 0:
        return -2 * neuronThreshold * math.exp(-time)
    else:
        return 0


def boolToSpikes(b):
    if b:
        return [0]
    else:
        return [encodingInterval]


def createXorInputNeurons(lhs, rhs):
    lhsNeuron = Neuron(boolToSpikes(lhs), 0, NeuronType.INPUT)
    rhsNeuron = Neuron(boolToSpikes(rhs), 0, NeuronType.INPUT)
    biasNeuron = Neuron(boolToSpikes(True), 0, NeuronType.INPUT)
    return [lhsNeuron, rhsNeuron, biasNeuron]


def perNeuronAdjustment(time, preneuron, postneuron):
    internalState = 0
    synapses = G.edge[preneuron][postneuron]
    for (num, data) in synapses.items():
        for spikeTime in preneuron.spikeTimes:
            adjustedTime = time - (spikeTime + data['delay'])
            internalState += data['weight'] * spikeResponse(adjustedTime)
    return internalState


G = nx.MultiDiGraph()

previousLayer = []
for layerSize in layerSizes:
    if previousLayer == []:
        # Setting up input neurons is a little more complicated.
        # layer = createLayer(layerSize, NeuronType.INPUT)
        layer = createXorInputNeurons(xorInput[0], xorInput[1])
        G.add_nodes_from(layer)
    else:
        layer = createLayer(layerSize, NeuronType.HIDDEN)
        G.add_nodes_from(layer)
        for presynapticNeuron in previousLayer:
            for postsynapticNeuron in layer:
                for synapseNumber in range(synapsesPerConnection):
                    G.add_edge(
                        presynapticNeuron,
                        postsynapticNeuron,
                        delay=synapseDelay(synapseNumber),
                        weight=1
                    )
    previousLayer = layer

for outputNeuron in previousLayer:
    outputNeuron.type = NeuronType.OUTPUT

# some relevent methods:
# G.predecessors(neuron) - get the presynaptic neurons
# G.successors(neuron) - get the postsynaptic neurons
# G.edge[presynaptic][postsynaptic] - get a dict like:
# {
#     0: {'delay': 1, 'weight': 1},
#     1: {'delay': 5, 'weight': 1},
#     2: {'delay': 9, 'weight': 1},
#     3: {'delay': 13, 'weight': 1}
# }

all_presynaptic = [
    n for n in G.nodes()
    if (n.type == NeuronType.INPUT)
    or (n.type == NeuronType.HIDDEN)
]
all_postsynaptic = [
    n for n in G.nodes()
    if (n.type == NeuronType.HIDDEN)
    or (n.type == NeuronType.OUTPUT)
]

for time in range(simulationTime):
    for postneuron in all_postsynaptic:
        internalState = 0
        for preneuron in G.predecessors(postneuron):
            internalState += perNeuronAdjustment(time, preneuron, postneuron)
        internalState += refractoriness(postneuron.timeSinceLastSpike(time))
        postneuron.update(internalState, time)
    input("")
outN = G.nodes()[-1]
print(outN.spikeTimes)
