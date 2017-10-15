import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate


def normalize_data(data, min_data, max_data, n_samples):
    diff = max_data - min_data
    data_norm = np.zeros(n_samples)

    for i in range(n_samples):
        data_norm[i] = (data[i] - min_data) / diff

    return data_norm


def reverse_normalization(data_norm, min_data, max_data, n_samples):
    diff = float(max_data - min_data)
    data_original = np.zeros(n_samples)

    for i in range(n_samples):
        data_original[i] = data_norm[i]*diff + min_data

    return data_original


def normalize_data_one_to_minus_one(data, min_data, max_data, n_samples):
    diff = max_data - min_data
    data_norm = np.zeros(n_samples)

    for i in range(n_samples):
        data_norm[i] = (data[i] - min_data) / diff

    data_norm = 2*data_norm - 1
    return data_norm


def reverse_normalization_one_to_minus_one(data_norm, min_data, max_data, n_samples):
    diff = float(max_data - min_data)
    data_original = np.zeros(n_samples)

    for i in range(n_samples):
        data_original[i] = ((data_norm[i] + 1)/2)*diff + min_data

    return data_original


def scale_imfs_amplitude(imfs, n_scalings=4, threshold=0.001):
    n_imfs = len(imfs)
    n = len(imfs[0])
    scaled_imfs = np.ndarray(imfs.shape, np.float)
    inst_amp = np.ndarray(imfs.shape, np.float)

    for i in range(n_imfs):
        correction_curves = np.ndarray((n_scalings, n), np.float)
        scaled_imfs[i] = imfs[i]

        for k in range(n_scalings):
            absolute_value = np.fabs(scaled_imfs[i])
            x = signal.argrelextrema(absolute_value, np.greater)[0]
            y = absolute_value[x]
            correction_curve = interpolate.PchipInterpolator(x, y)(np.arange(n))

            for j in range(n):
                if correction_curve[j] == threshold:
                    if j > 0:
                        correction_curve[j] = correction_curve[j-1]
                    else:
                        correction_curve[j] = scaled_imfs[i][j]
                scaled_imfs[i][j] = scaled_imfs[i][j] / correction_curve[j]

            correction_curves[k] = correction_curve

        if n_scalings == 1:
            inst_amp[i] = correction_curves[0]
        else:
            inst_amp[i] = get_instantaneous_amplitude(correction_curves)

    return scaled_imfs, inst_amp


def reverse_scaling_imfs(imfs_scaled, correction_curves):
    n_imfs = len(imfs_scaled)
    imfs_reconstructed = np.ndarray(imfs_scaled.shape, np.float)

    for i in range(n_imfs):
        imfs_reconstructed[i] = np.multiply(imfs_scaled[i], correction_curves[i])

    return imfs_reconstructed


def scale_amplitudes(data, n_scalings=4, threshold=0.001):
    n = len(data)
    scaled_data = data
    correction_curves = np.zeros((n_scalings, n))

    for k in range(n_scalings):
        absolute_value = np.fabs(scaled_data)
        x = signal.argrelextrema(absolute_value, np.greater)[0]
        y = absolute_value[x]
        correction_curve = interpolate.PchipInterpolator(x, y)(np.arange(n))

        for i in range(n):
            if correction_curve[i] == threshold:
                if i > 0:
                    correction_curve[i] = correction_curve[i-1]
                else:
                    correction_curve[i] = threshold
            scaled_data[i] = data[i] / correction_curve[i]

        correction_curves[k] = correction_curve

    if n_scalings == 1:
        am = correction_curves[0]
    else:
        am = get_instantaneous_amplitude(correction_curves)

    return scaled_data, am


def get_instantaneous_amplitude(correction_curves):
    n_correction_curves = len(correction_curves)
    am = correction_curves[0]

    for i in range(1, n_correction_curves):
        am = np.multiply(am, correction_curves[i])

    return am


def check_rapid_changes_in_frequency(frequency, max_freq, rapid_change_threshold=50.0):
    n_samples = len(frequency)

    for k in range(n_samples):
        if frequency[k] > max_freq:
            if k > 0:
                frequency[k] = frequency[k-1]
            else:
                frequency[k] = max_freq
        # Check if change in frequency is unrealistic (too rapid change):
        if k > 0:
            if np.fabs(frequency[k] - frequency[k-1]) > rapid_change_threshold:
                if frequency[k] > frequency[k-1]:
                    frequency[k] = frequency[k-1]
                else:
                    frequency[k-1] = frequency[k]
    return frequency


def reverse_scaling(data_scaled, correction_curve):
    return np.multiply(data_scaled, correction_curve)


def five_point_stencil_differentiate(data, fs=500):
    differentiated = np.zeros(len(data), np.float)

    for k in range(2, len(data)-2, 1):
        stencil = -data[k+2] + 8.0*data[k+1] - 8.0*data[k-1] + data[k-2]
        differentiated[k] = (float(fs) * float(stencil)) / (24.0*np.pi)

    differentiated[:2] = differentiated[2]
    differentiated[-2:] = differentiated[-3]
    return differentiated


def forward_euler_differentiate(data, fs=500.0):
    differentiated = np.zeros(len(data), np.float)
    for k in range(len(data)-1):

        differentiated[k] = (data[k+1] - data[k]) * (fs/(2.0*np.pi))

    differentiated[-1] = differentiated[-2]
    return differentiated



