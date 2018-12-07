import numpy as np
import scipy.stats
import pywt
import networkx as nx

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
and
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
    theta = np.trapz(fft[freqs > 4 and freqs < 7], freqs[freqs > 4 and freqs < 7])
    alpha = np.trapz(fft[freqs > 8 and freqs < 13], freqs[freqs > 8 and freqs < 13])
    beta = np.trapz(fft[freqs > 14 and freqs < 30], freqs[freqs > 14 and freqs < 30])
    gamma1 = np.trapz(fft[freqs > 30 and freqs < 55], freqs[freqs > 30 and freqs < 55])
    gamma2 = np.trapz(fft[freqs > 65 and freqs < 110], freqs[freqs > 65 and freqs < 110])
    return np.array([delta, theta, alpha, beta, gamma1, gamma2]) / overall

def discrete_wavelet(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=7)
    return coeffs

def max_absolute_cross_correlations(signal1, signal2):
    assert len(signal1) == len(signal2)
    correlations = []
    for i in range(1, len(signal1)):
        correlations.append(np.correlate(signal1[:i], signal2[-i:]))
    correlations.append(np.correlate(signal1, signal2))
    for i in range(len(signal2), 0, -1):
        correlations.append(np.correlate(signal1[:-i], signal2[i:]))
    return max(correlations)

# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
def autocorr(x, t=1):
    return numpy.corrcoef(numpy.array([x[0:len(x)-t], x[t:len(x)]]))

def decorrelation_times(signal):
    t = 0
    while autocorr(signal, t) > np.exp(-1):
        t += 1
    return t

# Graph theoretic
def make_cross_correlation_graph(signals):
    G = nx.Graph()
    for (c1, v1) in signals.items():
        for (c2, v2) in signals.items():
            G.add_edge(c1, c2, weight=max_absolute_cross_correlations(v1, v2))

def parameters(G):
    return [
        nx.algorithms.cluster.clustering(G),
        nx.algorithms.efficiency.local_efficiency(G),
        nx.algorithms.centrality.betweenness_centrality(G),
        nx.algorithms.distance_measures.eccentricity(G),
        nx.algorithms.efficiency.global_efficiency(G),
        nx.algorithms.distance_measures.diameter(G),
        nx.algorithms.distance_measures.radius(G),
        nx.algorithms.shortest_paths.generic.average_shortest_path_length(G, weight='weight')
    ]
