import numpy as np
import math

def e_ix(x):
    return math.cos(x) + 1j * math.sin(x)

def spectral_power(f, delta_t, x):
    return (delta_t ** 2) * (abs(sum([e_ix(-2 * math.pi * f * n * delta_t) for (n, x) in enumerate(x)])) ** 2)

signal1 = [math.sin(x) for x in range(500)]
signal2 = [math.sin(2 * x) for x in range(500)]
signal3 = [signal1[n] + signal2[n] for n in range(500)]

frequencies = np.fft.rfftfreq(len(signal1))

fft1 = np.fft.rfft(signal1)
fft2 = np.fft.rfft(signal2)
fft3 = np.fft.rfft(signal3)

import matplotlib.pyplot as plt
# plt.plot(frequencies, np.abs(fft1))
# plt.plot(frequencies, np.abs(fft2))
plt.plot(frequencies, np.abs(fft3))

plt.show()
