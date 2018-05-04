import numpy as np
import sys
import matplotlib.pyplot as plt
from optparse import OptionParser

usage = "usage: %prog [options] *.raw32"
parser = OptionParser(usage)
parser.add_option("-x", "--max-freq", dest="max_hz", type="float", default=30,
                  help="the maximum frequency (in Hz) of the spectrum")
parser.add_option("-n", "--min-freq", dest="min_hz", type="float", default=8,
                  help="the minimum frequency (in Hz) of the spectrum")
(options, args) = parser.parse_args()

lofreq = options.min_hz / 500
hifreq = options.max_hz / 500

train_pre = []
test_pre = []

for filename in args:
	word = ''
	if "test" in filename:
		word = 'test'
	if "training" in filename:
		word = 'training'
	afterwards_index = filename.index(word)
	afterwards = filename[afterwards_index:].replace('raw32', 'freq32')
	newName = 'out/' + afterwards

	initial_signal = np.fromfile(filename, dtype = np.dtype("i4")) / 10000
	signal = np.array([0] * 100 + list(initial_signal) + [0] * 100)

	frequencies = np.fft.rfftfreq(len(signal))
	fft1 = np.abs(np.fft.rfft(signal))

	importantValues = []
	for (i, val) in enumerate(fft1):
		if lofreq <= frequencies[i] <= hifreq:
			importantValues.append(val)

	importantArr = np.asarray((np.array(importantValues) * 100000).round(), dtype=np.int32)
	importantArr.tofile(newName)
