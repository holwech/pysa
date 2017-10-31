import numpy as np
import scipy.signal as signal
from . import utils


def nhht(imfs, sample_frequency):
    n_imfs = len(imfs)
    max_freq = sample_frequency / 2.0
    amplitudes = np.zeros(imfs.shape, np.float32)
    scaled_imfs = np.zeros(imfs.shape, np.float32)
    frequencies = np.zeros(imfs.shape, np.float32)

    for i in range(n_imfs):
        scaled_imf, am = utils.scale_amplitudes(imfs[i])
        scaled_imfs[i] = scaled_imf
        h = signal.hilbert(scaled_imf)
        amplitudes[i] = am
        frequencies[i] = np.r_[
            0.0,
            0.5*(np.angle(-h[2:]*np.conj(h[0:-2]))+np.pi)/(2.0*np.pi) * np.float32(sample_frequency),
            0.0
        ]

        frequencies[i, 0] = frequencies[i, 1]
        frequencies[i, -1] = frequencies[i, -2]

        frequencies[i] = utils.check_rapid_changes_in_frequency(frequencies[i], max_freq)

    return frequencies, amplitudes


def get_instantaneous_frequency(imfs, sample_frequency=500.0):
    sample_frequency = float(sample_frequency)
    max_freq = sample_frequency / 2.0
    freq = np.zeros(imfs.shape, np.float)

    for i in range(len(imfs)):
        # Do Hilbert Transform - NB! Must be normalized (scaled amplitudes)
        hi = signal.hilbert(imfs[i])
        freq[i, :] = np.r_[
            0.0,
            0.5*(np.angle(-hi[2:]*np.conj(hi[0:-2]))+np.pi)/(2.0*np.pi) * sample_frequency,
            0.0
        ]

        freq[i, 0] = freq[i, 1]
        freq[i, -1] = freq[i, -2]

        for k in range(len(freq[i])):

            if freq[i, k] > max_freq:
                if k > 0:
                    freq[i, k] = freq[i, k-1]
                else:
                    freq[i, k] = max_freq

            # Check if change in frequency is unrealistic (too rapid change):
            if k > 0:
                if np.fabs(freq[i, k] - freq[i, k-1]) > 50.0:
                    if freq[i, k] > freq[i, k-1]:
                        freq[i, k] = freq[i, k-1]
                    else:
                        freq[i, k-1] = freq[i, k]
    return freq




