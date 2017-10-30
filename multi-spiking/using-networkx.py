import networkx as nx
from enum import Enum


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

    def update(self):
        pass


layerSizes = [3, 5, 1]
synapsesPerConnection = 4
lastOutput = 16


def createLayer(layerSize, type):
    return [Neuron([], 0, type) for x in range(layerSize)]


def synapseDelay(synapseNumber):
    return 1 + synapseNumber * int(lastOutput / synapsesPerConnection)


def spikeResponse(time, timeDecay):
    if time > 0:
        return (time/timeDecay) * math.exp(1 - (time/timeDecay))
    else:
        return 0


def refractoriness(time, refractorinessDecay):
    if time > 0:
        return -2 * NEURON_THRESHOLD * math.exp(-time / refractorinessDecay)
    else:
        return 0


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

for postneuron in all_postsynaptic:
    preneurons = G.predecessors(postneuron)
    for preneuron in preneurons:
        for (synapseNum, associatedData) in G.edge[preneuron][postneuron].items():
            print(" -- ".join(map(repr, [preneuron, postneuron, synapseNum, associatedData])))
