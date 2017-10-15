import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
from . import utils


def emd(x, min_data, max_data, max_modes=10, max_siftings=200):
    '''
    Adaptive Signal Analysis algorithm, Empirical Mode Decomposition, to get Intrinsic Mode Functions (IMF)

    :param x: Signal to decompose
    :param max_modes: Maximimum IMFs to decompose the signal into.
    :param max_siftings: Maximum number of siftings per each decomposed IMF
    :return: Array (n, m) of decomposed IMFs. n = max_modes, m = length of signal x
    '''
    n_samples = len(x)
    imfs = np.ndarray(([]))
    norm_x = utils.normalize_data(x, min_data, max_data, n_samples)
    residue = norm_x
    first_imf = True
    n = 0

    while n < max_modes:
        imf = sift_process(residue, max_siftings)

        if first_imf:
            first_imf = False
            imfs = imf
        else:
            imfs = np.vstack((imfs, imf))

        residue -= imf
        n += 1

        extrema = get_number_of_extrema(residue)
        if extrema <= 2:
            break

    imfs = np.vstack((imfs, residue))
    return imfs


def sift_process(residue, max_siftings):
    '''
    Sift process for current mode.

    :param residue: current mode/residue of data x
    :param max_siftings: Maximum number of siftings
    :return: array (1, n) Mode that satisfies conditions for being an IMF
    '''
    mode = residue
    n_siftings = 0
    n_extrema = 0
    n_zero_crossings = 0
    extrema_counter = 0
    zero_crossings_counter = 0

    while n_siftings < max_siftings:
        mode = sift_one(mode)
        extrema, zero_crossings, mean = analyze_mode(mode)
        n_siftings += 1

        if abs(extrema - zero_crossings) <= 1 and -0.001 <= mean <= 0.001:
            if n_extrema == extrema and n_zero_crossings == zero_crossings:
                extrema_counter += 1
                zero_crossings_counter += 1

            n_extrema = extrema
            n_zero_crossings = zero_crossings

            if extrema_counter >= 5 and zero_crossings_counter >= 5:
                break

    extrema = get_number_of_extrema(mode)
    if extrema > 1:
        n_extra_siftings = 5
        for i in range(n_extra_siftings):
            mode = sift_one(mode)

    return mode


def sift_one(mode):
    '''
    Process for sifting one time.

    :param mode: array(n) Current mode
    :return: array (1, n) mean subtracted from current mode.
    '''
    maxima = find_maxima(mode)
    minima = find_minima(mode)
    upper_signal = interpolate_maxima(mode, maxima)
    lower_signal = interpolate_minima(mode, minima)
    mean = find_local_mean(lower_signal, upper_signal)
    return mode - mean


def find_minima(x):
    return signal.argrelextrema(x, np.less)[0]


def find_maxima(x):
    return signal.argrelextrema(x, np.greater)[0]


def analyze_mode(mode):
    '''
    This method analyze the current mode, providing the number of extremas, zero crossings and the mean.
    It is required so that the conditions for being and IMF can be confirmed.

    :param mode: array(n) Current mode
    :return: Integer, Integer, Float
    '''
    maxima = signal.argrelextrema(mode, np.greater)
    minima = signal.argrelextrema(mode, np.less)
    extrema = np.size(maxima) + np.size(minima)
    zero_crossings = find_number_of_zero_crossings(mode)
    return extrema, zero_crossings, np.mean(mode)


def get_number_of_extrema(mode):
    maxima = signal.argrelextrema(mode, np.greater)
    minima = signal.argrelextrema(mode, np.less)
    extrema = np.size(maxima) + np.size(minima)
    return extrema


def find_local_mean(lower_signal, upper_signal):
    return (lower_signal + upper_signal)/2.0


def find_number_of_zero_crossings(x):
    crossings = len(np.where(np.diff(np.signbit(x)))[0])
    return crossings


def get_sum_of_the_differences(imf_old, imf_new):
    '''
    Stopping criterion introduced by Dr. Norden Huang to stop the EMD procedure if all oscillatory modes er extracted.

    :param imf_old: array(n)
    :param imf_new: array(n)
    :return: (Float) sum of difference
    '''
    imf_length = len(imf_old)
    sd = 0

    for i in range(imf_length):
        sd += ((abs(imf_old[i]-imf_new[i]))**2)/(imf_old[i]**2)

    return sd


def interpolate_maxima(x, maxima):
    '''
    Cubic interpolation of maxima points.

    :param x: original data
    :param maxima: indexes for maxima points of original data
    :return: array(n) upper envelope
    '''
    t = np.arange(0, len(x))
    size = np.size(maxima)

    if size == 0:
        return np.ones(len(x)) * 0

    if size == 1:
        return np.ones(len(x)) * x[maxima]

    points, maxima = correct_end_effects(x, maxima, True)
    tck = interpolate.splrep(maxima, points)
    return interpolate.splev(t, tck)


def interpolate_minima(x, minima):
    '''
    Cubic interpolation of minima points.

    :param x: original data
    :param minima: indexes for minima points of original data
    :return: array(n) lower envelope
    '''
    t = np.arange(0, len(x))
    size = np.size(minima)

    if size == 0:
        return np.ones(len(x)) * 0

    if size == 1:
        return np.ones(len(x)) * x[minima]

    points, minima = correct_end_effects(x, minima, False)
    tck = interpolate.splrep(minima, points)
    return interpolate.splev(t, tck)


def correct_end_effects(x, extrema, is_maxima):
    '''
    Correction of end effect by near boundary spline interpolation. This method will append end point and
    insert start point to the interpolation samples.

    :param x: array(n): original data
    :param extrema: array: indexes for extreme points of original data
    :param is_maxima: Boolean reflecting whether the extrema is maxima or not.

    :return: array (n), array(n)
    '''
    interpolation_points = x[extrema]
    start_point = get_start_point_from_linear_spline_of_two_extrema_near_boundary(x, extrema)
    end_point = get_end_point_from_linear_spline_of_two_extrema_near_boundary(x, extrema)
    interpolation_points, extrema = add_end_point_to_interpolation(x, interpolation_points, extrema, end_point, is_maxima)
    interpolation_points, extrema = add_start_point_to_interpolation(x, interpolation_points, extrema, start_point, is_maxima)
    return interpolation_points, extrema


def get_end_point_from_linear_spline_of_two_extrema_near_boundary(x, extrema):
    '''
    This will calculate the end point on the line connected by the two last extrema.

    :param x: array(n) current mode
    :param extrema: array of current extrema
    :return:
    '''
    slope = (x[extrema[-1]] - x[extrema[-2]]) / (extrema[-1] - extrema[-2])
    dt = len(x) - extrema[-1]
    return slope * dt + x[extrema[-1]]


def get_start_point_from_linear_spline_of_two_extrema_near_boundary(x, extrema):
    '''
    This will calculate the start point on the line connected by the two first extrema.

    :param x: array(n)
    :param extrema: array of current extrema
    :return:
    '''
    slope = (x[extrema[0]] - x[extrema[1]]) / (extrema[0] - extrema[1])
    dt = extrema[0]
    return slope * dt + x[extrema[0]]


def add_end_point_to_interpolation(x, interpolation_points, extrema, end_point, is_maxima):
    '''
    Adding the last end point to the interpolation samples. Checks whether if it is an maxima or not.

    :param x: array(n)
    :param interpolation_points: Current interpolation points of the envelope
    :param extrema: array current extrema
    :param end_point: calculated endpoint on line connecting the two last extrema of the envelope
    :param is_maxima: boolean
    :return: interpolation points, extrema
    '''
    if is_maxima:
        return add_end_point_to_upper_envelope(x, interpolation_points, extrema, end_point)
    else:
        return add_end_point_to_lower_envelope(x, interpolation_points, extrema, end_point)


def add_start_point_to_interpolation(x, interpolation_points, extrema, start_point, is_maxima):
    '''
    Adding the start point to the interpolation samples. Checks whether if it is an maxima or not.

    :param x: array(n)
    :param interpolation_points: Current interpolation points of the envelope
    :param extrema: array current extrema
    :param start_point: calculated start point on line connecting the two first extrema of the envelope
    :param is_maxima: boolean
    :return: interpolation points, extrema
    '''
    if is_maxima:
        return add_start_point_to_upper_envelope(x, interpolation_points, extrema, start_point)
    else:
        return add_start_point_to_lower_envelope(x, interpolation_points, extrema, start_point)


def add_end_point_to_upper_envelope(x, interpolation_points, extrema, end_point):
    length_of_data = len(x)-1

    if end_point < x[length_of_data]:
        interpolation_points = np.append(interpolation_points, [x[length_of_data]])
        extrema = np.append(extrema, [length_of_data])
    else:
        interpolation_points = np.append(interpolation_points, [end_point])
        extrema = np.append(extrema, [length_of_data])

    return interpolation_points, extrema


def add_end_point_to_lower_envelope(x, interpolation_points, extrema, end_point):
    length_of_data = len(x)-1

    if end_point < x[length_of_data]:
        interpolation_points = np.append(interpolation_points, [end_point])
        extrema = np.append(extrema, [length_of_data])
    else:
        interpolation_points = np.append(interpolation_points, [x[length_of_data]])
        extrema = np.append(extrema, [length_of_data])

    return interpolation_points, extrema


def add_start_point_to_upper_envelope(x, interpolation_points, extrema, start_point):
    if start_point < x[0]:
        interpolation_points = np.insert(interpolation_points, [0], [x[0]])
        extrema = np.insert(extrema, [0], [0])
    else:
        interpolation_points = np.insert(interpolation_points, [0], [start_point])
        extrema = np.insert(extrema, [0], [0])

    return interpolation_points, extrema


def add_start_point_to_lower_envelope(x, interpolation_points, extrema, start_point):
    if start_point < x[0]:
        interpolation_points = np.insert(interpolation_points, [0], [start_point])
        extrema = np.insert(extrema, [0], [0])
    else:
        interpolation_points = np.insert(interpolation_points, [0], [x[0]])
        extrema = np.insert(extrema, [0], [0])

    return interpolation_points, extrema
