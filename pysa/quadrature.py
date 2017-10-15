import numpy as np
import scipy.interpolate as interpolate
import scipy.signal as signal
import utils


def gaussian_quadrature(imfs, fs):
    amplitudes = np.zeros(imfs.shape, np.float32)
    frequencies = np.zeros(imfs.shape, np.float32)
    scaled_imfs = np.zeros(imfs.shape, np.float32)
    n_imfs = len(imfs)
    n_points = len(imfs[0])

    for i in range(n_imfs):
        x, am = utils.scale_amplitudes(imfs[i])
        dx_dt = utils.five_point_stencil_differentiate(x, fs)
        mask = ((dx_dt > 0).astype(int))*(-2) + 1
        y = np.real(np.sqrt(1-x**2))
        q = y * mask
        quadrature = x + q*1j
        amplitudes[i] = am
        frequency = np.r_[
            0.0,
            0.5*(np.angle(-quadrature[2:]*np.conj(quadrature[0:-2]))+np.pi)/(2.0*np.pi) * np.float32(fs),
            0.0
        ]
        frequency = fill_in_invalid_points_using_spline_interpolation(frequency, n_points)
        frequencies[i] = signal.medfilt(frequency, kernel_size=5)
        frequencies[i][-1] = frequencies[i][-2]
        scaled_imfs[i] = x

    return frequencies, amplitudes, scaled_imfs


def direct_quadrature_arccosine(imfs, fs):
    amplitudes = np.zeros(imfs.shape, np.float32)
    frequencies = np.zeros(imfs.shape, np.float32)
    scaled_imfs = np.zeros(imfs.shape, np.float32)
    n_imfs = len(imfs)
    n_points = len(imfs[0])
    dt = 1/fs

    for i in range(n_imfs):
        x, am = (utils.scale_amplitudes(imfs[i]))
        x[np.where(np.fabs(x) > 1)] = float('NaN')
        amplitudes[i] = am
        phase = np.real(np.arccos(x))
        phase = set_nan_where_slope_of_phase_changes(phase, n_points)
        frequency = np.fabs(utils.five_point_stencil_differentiate(phase, fs))
        frequencies[i] = fill_in_invalid_points_using_spline_interpolation(frequency, n_points)
        scaled_imfs[i] = x

    return frequencies, amplitudes, scaled_imfs


def direct_quadrature_arccosine_with_wus_approach(imfs, fs):
    amplitudes = np.zeros(imfs.shape, np.float32)
    frequencies = np.zeros(imfs.shape, np.float32)
    scaled_imfs = np.zeros(imfs.shape, np.float32)
    n_imfs = len(imfs)
    n_points = len(imfs[0])
    dt = 1/fs

    for i in range(n_imfs):
        x, am = (utils.scale_amplitudes(imfs[i]))
        amplitudes[i] = am
        phase = np.real(np.arccos(x))
        phase = set_nan_where_slope_of_phase_changes(phase, n_points)
        frequency = np.fabs(get_instantaneous_frequency_using_wus_method_arcos_style(phase, fs))/(2.0*np.pi*dt)
        frequencies[i] = fill_in_invalid_points_using_spline_interpolation(frequency, n_points)
        scaled_imfs[i] = x

    return frequencies, amplitudes, scaled_imfs


def direct_quadrature_arctan(imfs, fs):
    amplitudes = np.zeros(imfs.shape, np.float32)
    phase = np.zeros(imfs.shape, np.float32)
    frequencies = np.zeros(imfs.shape, np.float32)
    scaled_imfs = np.zeros(imfs.shape, np.float32)
    n_imfs = len(imfs)
    n_points = len(imfs[0])
    dt = 1/fs

    for i in range(n_imfs):
        x, am = utils.scale_amplitudes(imfs[i])
        y = np.real(np.sqrt(1-x**2))
        amplitudes[i] = am
        phase[i] = np.real(np.arctan(x/y))
        phase[i] = set_nan_where_slope_of_phase_changes(phase[i], n_points)
        frequency = np.fabs(utils.five_point_stencil_differentiate(phase[i], fs))/(2.0*np.pi*dt)
        frequencies[i] = fill_in_invalid_points_using_spline_interpolation(frequency, n_points)
        scaled_imfs[i] = x

    return frequencies, amplitudes, scaled_imfs


def wus_approach(imfs, fs):
    amplitudes = np.zeros(imfs.shape, np.float32)
    frequencies = np.zeros(imfs.shape, np.float32)
    scaled_imfs = np.zeros(imfs.shape, np.float32)
    n_imfs = len(imfs)
    n_points = len(imfs[0])

    for i in range(n_imfs):
        x, am = utils.scale_amplitudes(imfs[i])
        y = np.real(np.sqrt(1-x**2))
        scaled_imfs[i] = x
        amplitudes[i] = am
        frequency = np.fabs(get_instantaneous_frequency_using_wus_method(x, y, fs))
        frequencies[i] = fill_in_invalid_points_using_spline_interpolation(frequency, n_points)

    return frequencies, amplitudes, scaled_imfs


def hous_approach(imfs, fs):
    amplitudes = np.zeros(imfs.shape, np.float32)
    frequencies = np.zeros(imfs.shape, np.float32)
    scaled_imfs = np.zeros(imfs.shape, np.float32)
    n_imfs = len(imfs)
    n_points = len(imfs[0])

    for i in range(n_imfs):
        imf, am = utils.scale_amplitudes(imfs[i])
        scaled_imfs[i] = imf
        amplitudes[i] = am
        frequency = np.fabs(get_instantaneous_frequency_using_hous_method(imf, fs))
        frequencies[i] = fill_in_invalid_points_using_spline_interpolation(frequency, n_points)

    return frequencies, amplitudes, scaled_imfs


def set_nan_where_slope_of_phase_changes(phase, n_points):
    for k in range(1, n_points-1):
        prev = phase[k-1]
        cur = phase[k]
        next_p = phase[k+1]
        if (prev < cur and cur > next_p) or (prev > cur and cur < next_p):
            phase[k] = float('NaN')

    return phase


def fill_in_invalid_points_using_spline_interpolation(data, n_points):
    t = np.arange(0, len(data))
    invalid_indexes = np.isnan(data)

    for j in range(n_points):
        if np.isnan(data[j]):
            counter = 0
            while invalid_indexes[j+counter]:
                counter += 1
                if j+counter >= n_points-1:
                    break
            data[j - 1] = data[j - 2]
            j += counter

    finite_indexes = np.isfinite(data)
    finite_data = data[finite_indexes]
    tck = interpolate.splrep(t[finite_indexes], finite_data)
    interpolated = interpolate.splev(t, tck)

    return interpolated


def get_instantaneous_frequency_using_wus_method(x, y, fs=500):
    length_of_data = len(x)
    instantanenous_frequency = np.zeros(length_of_data, np.float32)
    delta_t = 1.0/float(fs)

    for i in range(2, length_of_data-2, 1):
        if y[i] == 0:
            instantanenous_frequency[i] = float('NaN')
        else:
            instantanenous_frequency[i] = (1.0/delta_t)*((x[i + 1] - x[i - 1])/(2.0 * y[i]))

    instantanenous_frequency[:2] = instantanenous_frequency[2]
    instantanenous_frequency[-2:] = instantanenous_frequency[-3]
    return instantanenous_frequency


def get_instantaneous_frequency_using_wus_method_arcos_style(phase, fs=500):
    length_of_data = len(phase)
    instantanenous_frequency = np.zeros(length_of_data, np.float32)
    delta_t = 1.0/float(fs)

    for i in range(2, length_of_data-2, 1):
        if phase[i] == 0:
            instantanenous_frequency[i] = float('NaN')
        else:
            instantanenous_frequency[i] = (1.0/delta_t)*((np.cos(phase[i + 1]) - np.cos(phase[i - 1]))/(2.0 * np.sin(phase[i])))

    instantanenous_frequency[:2] = instantanenous_frequency[2]
    instantanenous_frequency[-2:] = instantanenous_frequency[-3]
    return instantanenous_frequency


def get_instantaneous_frequency_using_hous_method(imf, fs):
    instantanenous_frequency = np.zeros(len(imf), np.float32)
    delta_t = 1.0/float(fs)

    for i in range(2, len(imf)-2, 1):
        if imf[i] == 0:
            imf[i] = imf[i-1]
        instantanenous_frequency[i] = (1.0/delta_t)*np.divide(imf[i + 1] - imf[i - 1], 2.0 * imf[i])

    instantanenous_frequency[:2] = instantanenous_frequency[2]
    instantanenous_frequency[-2:] = instantanenous_frequency[-3]
    return instantanenous_frequency
