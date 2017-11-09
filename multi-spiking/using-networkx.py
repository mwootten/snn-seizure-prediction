import networkx as nx
from enum import Enum
import math
from copy import deepcopy

layerSizes = [3, 5, 1]
synapsesPerConnection = 4
lastOutput = 16
simulationTime = 25
refractorinessDecay = 80
encodingInterval = 6
timeDecay = 7
neuronThreshold = 1
xorInput = [True, True]


class Neuron:
    """Represent a single neuron, and encapsulate its internal state"""
    def __init__(self, spikeTimes=[], internalState=0):
        # The deepcopy is not redundant.
        # The default parameter value is initialized exactly once.
        # If the deepcopy was removed, every default (every hidden/output
        # neuron) will have identical spike times, causing great confusion.
        self.spikeTimes = deepcopy(spikeTimes)
        self.internalState = internalState

    def timeSinceLastSpike(self, time):
        if len(self.spikeTimes) == 0:
            return float('inf')
        else:
            return time - self.spikeTimes[-1]

    def update(self, internalState, time):
        self.internalState = internalState
        if internalState > neuronThreshold:
            # heuristic to prevent rapid repeated spikes
            if self.timeSinceLastSpike(time) > 2:
                self.spikeTimes.append(time)


class InputNeuron(Neuron):
    """docstring for InputNeuron."""
    def __repr__(self):
        return "InputNeuron({}, {})".format(
            self.spikeTimes, self.internalState)


class HiddenNeuron(Neuron):
    """docstring for HiddenNeuron."""
    def __repr__(self):
        return "HiddenNeuron({}, {})".format(
            self.spikeTimes, self.internalState)


class OutputNeuron(Neuron):
    """docstring for OutputNeuron."""
    def __repr__(self):
        return "OutputNeuron({}, {})".format(
            self.spikeTimes, self.internalState)

    def update(self, internalState, time):
        # Output neurons should spike only once
        if len(self.spikeTimes) == 0:
            super().update(internalState, time)


def synapseDelay(synapseNumber):
    return 1 + synapseNumber * int(lastOutput / synapsesPerConnection)


def spikeResponse(time):
    if time > 0:
        return (time/timeDecay) * math.exp(1 - (time/timeDecay))
    else:
        return 0


def refractoriness(time):
    if time > 0:
        return -2 * neuronThreshold * math.exp(-time/refractorinessDecay)
    else:
        return 0


def boolToSpikes(b):
    if b:
        return [0]
    else:
        return [encodingInterval]


def createXorInputNeurons(lhs, rhs):
    lhsNeuron = InputNeuron(boolToSpikes(lhs))
    rhsNeuron = InputNeuron(boolToSpikes(rhs))
    biasNeuron = InputNeuron(boolToSpikes(True))
    return [lhsNeuron, rhsNeuron, biasNeuron]


def perNeuronAdjustment(G, time, preneuron, postneuron):
    internalState = 0
    synapses = G.edge[preneuron][postneuron]
    for (num, data) in synapses.items():
        for spikeTime in preneuron.spikeTimes:
            adjustedTime = time - (spikeTime + data['delay'])
            internalState += data['weight'] * spikeResponse(adjustedTime)
    return internalState


def connectLayers(G, prelayer, postlayer, weightFunction):
            for presynapticNeuron in prelayer:
                for postsynapticNeuron in postlayer:
                    for synapseNumber in range(synapsesPerConnection):
                        G.add_edge(
                            presynapticNeuron,
                            postsynapticNeuron,
                            delay=synapseDelay(synapseNumber),
                            weight=weightFunction(
                                presynapticNeuron,
                                postsynapticNeuron,
                                synapseNumber
                            )
                        )


def flatten(list):
    return sum(list, [])


class Network:
    """docstring for Network."""
    def __init__(self, inputLayer, layerSizes, weightFunction):
        self.G = nx.MultiDiGraph()
        self.G.add_nodes_from(inputLayer)
        self.inputLayer = inputLayer

        self.hiddenLayers = []
        previousLayer = inputLayer
        for layerSize in layerSizes[1:-1]:
            layer = [HiddenNeuron() for x in range(layerSize)]
            self.G.add_nodes_from(layer)
            self.hiddenLayers.append(layer)
            connectLayers(self.G, previousLayer, layer, weightFunction)
            previousLayer = layer

        outputLayer = [OutputNeuron() for x in range(layerSizes[-1])]
        self.G.add_nodes_from(outputLayer)
        connectLayers(self.G, previousLayer, outputLayer, weightFunction)
        self.outputLayer = outputLayer

        self.all_presynaptic = flatten([inputLayer] + self.hiddenLayers)
        self.all_postsynaptic = flatten(self.hiddenLayers + [outputLayer])


def constantOneWeightFunction(preneuron, postneuron, synapseNumber):
    return 1.0


def randomWeightFunction(preneuron, postneuron, synapseNumber):
    return random.uniform(1.0, 10.0)


# some relevant methods:
# G.predecessors(neuron) - get the presynaptic neurons
# G.successors(neuron) - get the postsynaptic neurons
# G.edge[presynaptic][postsynaptic] - get a dict like:
# {
#     0: {'delay': 1, 'weight': 1},
#     1: {'delay': 5, 'weight': 1},
#     2: {'delay': 9, 'weight': 1},
#     3: {'delay': 13, 'weight': 1}
# }


def simulateNetwork(net, simulationTime):
    for time in range(simulationTime):
        for post in net.all_postsynaptic:
            internalState = 0
            for pre in net.G.predecessors(post):
                internalState += perNeuronAdjustment(net.G, time, pre, post)
            internalState += refractoriness(post.timeSinceLastSpike(time))
            post.update(internalState, time)


# input layer starts out here as all zeros
initialInputLayer = [InputNeuron([0]) for x in range(layerSizes[0])]
onesNetwork = Network(initialInputLayer, layerSizes, constantOneWeightFunction)

simulateNetwork(onesNetwork, simulationTime)
for node in onesNetwork.G.nodes():
    print(node)
