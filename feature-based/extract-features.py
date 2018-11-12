import numpy as np
import scipy.stats
import pywt

sample_rate = 1

def mean(signal):
    return np.mean(signal)

def variance(signal):
    return np.var(signal)

def skewness(signal):
    return scipy.stats.skew(signal)

def kurtosis(signal):
    return scipy.stats.kurtosis(signal)

def stdev(signal):
    return np.std(signal)

def zero_crossings(signal):
    return len(np.where(np.diff(np.signbit(signal)))[0])

def peak_to_peak(signal):
    return np.max(signal) - np.min(signal)

def total_signal_area(signal):
    c = signal[:-1]
    d = signal[1:]
    small = np.finfo(signal.dtype).tiny
    sc = np.sign(c + small)
    sd = np.sign(d)
    dx = sample_rate
    parts = dx * sc * (c ** 2 - sc * sd * d ** 2) / (2 * (c - d))
    return np.sum(parts)

def total_signal_energy(signal):
    return np.sum(signal ** 2)

def energy_percentages(signal):
    # delta (<= 3 Hz)
    # theta (4-7 Hz)
    # alpha (8-13 Hz)
    # beta (14-30 Hz)
    # gamma1 (30-55 Hz)
    # gamma2 (65-110 Hz)
    padded = np.array([0] * 100 + list(signal) + [0] * 100)
    freqs = np.fft.rfftfreq(len(padded)) / sample_rate
    fft = np.abs(np.fft.rfft(signal))
    overall = np.trapz(fft, freqs)
    delta = np.trapz(fft[freqs <= 3], dx=sample_rate)
    theta = np.trapz(fft[freqs > 4 && freqs < 7], freqs[freqs > 4 && freqs < 7])
    alpha = np.trapz(fft[freqs > 8 && freqs < 13], freqs[freqs > 8 && freqs < 13])
    beta = np.trapz(fft[freqs > 14 && freqs < 30], freqs[freqs > 14 && freqs < 30])
    gamma1 = np.trapz(fft[freqs > 30 && freqs < 55], freqs[freqs > 30 && freqs < 55])
    gamma2 = np.trapz(fft[freqs > 65 && freqs < 110], freqs[freqs > 65 && freqs < 110])
    return np.array([delta, theta, alpha, beta, gamma1, gamma2]) / overall

def discrete_wavelet(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=7)
    return coeffs

def max_absolute_cross_correlations(signal1, signal2):
    assert len(signal1) == len(signal2)
    """
    correlate x1[:1] x2[-1:]
    correlate x1[:2] x2[-2:]
    correlate x1[:3] x2[-3:]
    correlate x1 x2
    correlate x1[:-3] x2[3:]
    correlate x1[:-2] x2[2:]
    correlate x1[:-1] x2[1:]
    """
    for tau in range(len(signal1 - 1)):
        pass

def decorrelation_times(signal):
    pass

# Graph theoretic
def make_graph(signal):
    pass

def clustering_coefficient(signal):
    pass
