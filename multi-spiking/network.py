import copy
INITIAL_WEIGHT = 1
NEURON_THRESHOLD = 1


def initWeights(layerNeuronCounts, synapsesPerConnection):
    # Weights contains an entry for every synapse, organized by four
    # parameters:
    # * w - the layer of the network of the source neuron
    # * x - the neuron index of the source neuron within its layer
    # * y - the neuron index of the destination neuron within its layer
    # * z - the index among the multiple synapses connecting those two

    weights = []

    # sourceLayerNumber uses zero-based indexing
    # It stops one short of the end because output neurons have no
    # further connections
    for sourceLayerNumber in range(len(layerNeuronCounts) - 1):
        xs = range(layerNeuronCounts[sourceLayerNumber])      # sources
        ys = range(layerNeuronCounts[sourceLayerNumber + 1])  # destinations
        # Since synapses/connection is constant throughout the network,
        # we can reuse this across the network. However, we need to make
        # copies so that different weights don't influence each other.
        eachConnection = [INITIAL_WEIGHT] * synapsesPerConnection
        # Create the connections for each destination neuron, and then each
        # source neuron. Since we're building these from the bottom up,
        # they appear to go backwards.
        layerConnections = [
            [copy.deepcopy(eachConnection) for y in ys]
            for x in xs
        ]
        weights.append(layerConnections)
    return weights


# This produces an array that functions as a map from a synapse index (z) to
# its respective delay.
def synapseDelays(synapseCount, lastOutput):
    synapseDelay = [1]
    increment = int(lastOutput / synapseCount)
    for x in range(0, synapseCount - 1):
        synapseDelay.append(synapseDelay[x] + increment)
    return synapseDelay


class MultiSpikingNetwork(object):
    """
    """
    def __init__(self, layerNeuronCounts, synapsesPerConnection,
                 encodingInterval, refractorinessDecay):
        super(MultiSpikingNetwork, self).__init__()
        self.layerNeuronCounts = layerNeuronCounts
        self.weights = initWeights(layerNeuronCounts, synapsesPerConnection)
        self.encodingInterval = encodingInterval
        self.refractorinessDecay = refractorinessDecay  # $\tau_r$ in the paper
        self.timeDecay = encodingInterval + 1  # plain $\tau$ in the paper

    def makeXorNetwork():
        # Explanation of inputs:
        # 3: two inputs + 1 bias neuron
        # 5: Arbitary, from what I can tell
        # 1: one output (of the XOR function)

        # 4: smallest number needed to model the problem
        # 6: Conformity with Bohte et al. (2002)
        # 80: Chosen by trial-and-error
        return MultiSpikingNetwork([3, 5, 1], 4, 6, 80)

    def simulate(self, duration, lastOutput, inputs):
        pass


if __name__ == "__main__":  # run this code only if executed, not when imported
    print(MultiSpikingNetwork.makeXorNetwork().weights)
