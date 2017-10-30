import networkx as nx
from enum import Enum


class NeuronType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3  # currently not set correctly - fix with low priority


class Neuron:
    """docstring for Neuron."""
    def __init__(self, spikeTimes, internalState, neuronType):
        self.spikeTimes = spikeTimes
        self.internalState = internalState
        self.neuronType = neuronType


layerSizes = [3, 5, 1]
synapsesPerConnection = 4
lastOutput = 16


def createLayer(layerSize, neuronType):
    return [Neuron([], 0, neuronType) for x in range(layerSize)]


def synapseDelay(synapseNumber):
    return 1 + synapseNumber * int(lastOutput / synapsesPerConnection)


G = nx.MultiDiGraph()

previousLayer = []
for layerSize in layerSizes:
    if previousLayer == []:
        layer = createLayer(layerSize, NeuronType.INPUT)
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
