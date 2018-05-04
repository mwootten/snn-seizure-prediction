import json
import random
import sys

positiveOutputTime = 15
negativeOutputTime = 20
iterationsPerExample = 500

rawInputs = json.load(open(sys.argv[1], 'r'))
inputsPerClass = int(len(rawInputs) / 2)
rawOutputs = [negativeOutputTime] * inputsPerClass + [positiveOutputTime] * inputsPerClass

assert len(rawInputs) == len(rawOutputs)

inputs = [[[x] for x in neurons] for neurons in rawInputs]
outputs = [[x] for x in rawOutputs]

possibilities = list(zip(inputs, outputs))

sequence = possibilities * iterationsPerExample
random.shuffle(sequence)

print(sequence)
