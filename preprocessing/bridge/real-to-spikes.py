import math
import json
import sys

M = 4
I_min = 0
I_max = 0.086
γ = 1.5

gaussianSpread = (1 / γ) * (I_max - I_min) / (M - 2)

def gaussianCenter(i):
    return I_min + (((2 * i - 3) / 2) * (I_max - I_min) / (M - 2))

def gaussianEval(μ, σ, x):
    return math.exp((-(x - μ) ** 2) / (2 * σ**2))

def gauss(I_a, i):
    return gaussianEval(gaussianCenter(i), gaussianSpread, I_a)

def spikeTime(I_a, i):
    return 10 - round(10 * gauss(I_a, i))

def encode(I_a):
    return list([spikeTime(I_a, i) for i in range(1, M + 1)])


in_filename = sys.argv[1]
in_file = open(in_filename, 'r')
in_matrix = json.load(in_file)
in_file.close()

out_matrix = []
for sample in in_matrix:
	sample_vec = []
	for val in sample:
		sample_vec.extend(encode(val))
	sample_vec.append(0)
	out_matrix.append(sample_vec)

print(json.dumps(out_matrix))
