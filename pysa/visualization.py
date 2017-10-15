import utils
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def plot_single_channel(sample_frequency, channel_data, title=str, plotter=plt):
    plotter.ion()
    f, ax = plotter.subplots(1, 1)
    f.suptitle(title)
    plotter.subplots_adjust(left=0.08, right=0.99, bottom=0.04, top=0.95, wspace=0.1, hspace=0.38)
    time_axis = scipy.linspace(start=0, stop=len(channel_data) / sample_frequency, num=len(channel_data))
    ylabel_str = "%s [%sV]" % ('EEG', u'\u00b5')
    ax.set_ylabel(ylabel_str)
    ax.set_xlabel('Time [s]')
    ax.plot(time_axis, channel_data)
    plotter.draw()


def plot_multiple_channels(sample_frequency, channels, plotter=plt, filename=str):
    plotter.ion()
    f, ax = plotter.subplots(1, 1)
    f.suptitle(filename)
    n_points = len(channels[0])
    plotter.subplots_adjust(left=0.08, right=0.99, bottom=0.04, top=0.95, wspace=0.1, hspace=0.38)
    time_axis = scipy.linspace(start=0, stop=n_points / sample_frequency, num=n_points)
    ylabel_str = "%s [%sV]" % ('EEG', u'\u00b5')
    ax.set_ylabel(ylabel_str)
    ax.set_xlabel('Time [s]')

    n_channels = len(channels)
    for i in range(n_channels):
        ax.plot(time_axis, channels[i])

    plotter.draw()


def plot_all_electrodes_for_one_trial(sample_frequency, trial_data, offset_y=50.0, event=str, plotter=plt):
    n_channels = len(trial_data)
    data_length = len(trial_data[0])
    f, axis = plotter.subplots(1, 1, sharex=True, sharey=False)
    f.suptitle("Stimuli " + event)
    time_axis = scipy.linspace(start=0, stop=data_length / sample_frequency, num=data_length)

    offset = offset_y
    for i in range(n_channels):
        data = trial_data[i]
        axis.plot(time_axis, data - np.mean(data) + offset)
        offset += 50.0

    ylabel_str = "%s [%sV]" % ('Amplitude', u'\u00b5')
    axis.set_ylabel(ylabel_str)
    axis.set_xlabel('Time [s]')


def plot_intrinsic_mode_functions(sample_frequency, imfs, channel=str, plotter=plt):
    n_rows = len(imfs)
    data_length = len(imfs[0])
    f, axis = plotter.subplots(n_rows, 1, sharex=True, sharey=False)
    time_axis = scipy.linspace(start=0, stop=data_length / sample_frequency, num=data_length)
    sup_title = "Channel " + channel
    f.suptitle(sup_title, fontsize=18)

    for i in range(n_rows):
        title = 'IMF: ' + str(i+1)
        axis[i].plot(time_axis, imfs[i])
        axis[i].set_title(title)
        axis[i].grid()

    axis[n_rows-1].set_xlabel('Time [s]')
    f.subplots_adjust(hspace=.5)


def plot_intrinsic_mode_functions_with_frequency_and_amplitude(sample_frequency, imfs, frequencies, amplitudes, channel=str, plotter=plt):
    '''

    :param sample_frequency:
    :param imfs:
    :param frequencies:
    :param amplitudes:
    :param channel:
    :param plotter:
    :return:
    '''
    n_rows = len(imfs)
    data_length = len(imfs[0])
    sup_title = "Channel " + channel
    f, axis = plotter.subplots(n_rows, 2, sharex=False, sharey=False)
    time_axis = scipy.linspace(start=0, stop=data_length / sample_frequency, num=data_length)
    f.suptitle(sup_title, fontsize=18)

    for i in range(0, n_rows):
        title = 'IMF: ' + str(i+1)
        axis[i][0].plot(time_axis, imfs[i])
        axis[i][0].set_title(title)
        axis[i][0].grid()
        axis[i][1].plot(frequencies[i], amplitudes[i])
        axis[i][1].set_ylabel('Amplitude')
        axis[i][1].grid()

    axis[n_rows - 1][0].set_xlabel('Time [s]')
    axis[n_rows - 1][1].set_xlabel('Frequency [Hz]')
    f.subplots_adjust(hspace=.5)


def plot_intrinsic_mode_functions_with_time_frequency_series(sample_frequency, imfs, frequencies, channel=str, plotter=plt):
    '''

    :param sample_frequency:
    :param imfs:
    :param frequencies:
    :param channel:
    :param plotter:
    :return:
    '''
    n_rows = len(imfs)
    data_length = len(imfs[0])
    sup_title = "Channel " + channel
    f, axis = plotter.subplots(n_rows, 2, sharex=False, sharey=False)
    time_axis = scipy.linspace(start=0, stop=data_length / sample_frequency, num=data_length)
    f.suptitle(sup_title, fontsize=18)

    for i in range(0, n_rows):
        title = 'IMF: ' + str(i+1)
        axis[i][0].plot(time_axis, imfs[i])
        axis[i][0].set_title(title)
        axis[i][0].grid()
        axis[i][1].plot(time_axis, frequencies[i])
        axis[i][1].set_ylabel('Frequency [Hz]')
        axis[i][1].grid()

    axis[n_rows - 1][0].set_xlabel('Time [s]')
    axis[n_rows - 1][1].set_xlabel('Time [s]')
    f.subplots_adjust(hspace=.5)


def plot_time_frequency_series(sample_frequency, frequencies, channel=str, plotter=plt):
    '''

    :param sample_frequency:
    :param frequencies:
    :param channel:
    :param plotter:
    :return:
    '''
    n_rows = len(frequencies)
    data_length = len(frequencies[0])
    sup_title = "Channel " + channel
    f, axis = plotter.subplots(n_rows, 1, sharex=False, sharey=False)
    time_axis = scipy.linspace(start=0, stop=data_length / sample_frequency, num=data_length)
    f.suptitle(sup_title, fontsize=18)

    for i in range(0, n_rows):
        title = 'IMF: ' + str(i+1)
        axis[i].plot(time_axis, frequencies[i])
        axis[i].set_title(title)
        axis[i].grid()

    axis[n_rows - 1].set_xlabel('Frequency [Hz]')
    axis[n_rows - 1].set_xlabel('Time [s]')
    f.subplots_adjust(hspace=.5)


def plot_original_signal_from_intrinsic_mode_functions(sample_frequency, imfs, residue, channel, plotter=plt):
    '''

    :param sample_frequency:
    :param imfs:
    :param residue:
    :param channel:
    :param plotter:
    :return:
    '''
    n_rows = len(imfs)
    data_length = len(imfs[0])
    final_signal = scipy.zeros(len(imfs[1]))
    time_axis = scipy.linspace(start=0, stop=data_length / sample_frequency, num=data_length)

    for i in range(n_rows):
        final_signal = scipy.add(final_signal, imfs[i + 1])

    final_signal = scipy.add(final_signal, residue)
    f, axis = plotter.subplots(1, 1)
    sup_title = "Channel " + channel
    f.suptitle(sup_title, fontsize=18)
    axis.plot(time_axis, final_signal)
    axis.grid()


def plot_power_spectral_density(frequency, amplitude, plotter=plt, title=str):
    '''

    :param frequency:
    :param amplitude:
    :param plotter:
    :param title:
    :return:
    '''
    fig = plotter.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(title)
    power = np.multiply(amplitude, amplitude)
    ax.plot(frequency, power)
    ax.set_xlabel('Frequency [Hz]')
    ylabel = 'Power Spectral Density [%sV^2]' % u'\u00b5'
    ax.set_ylabel(ylabel)
    plotter.draw()


# -------------------------------------------------------------------------------------------------------------------- #
#                                                HHT PLOTTING                                                          #
# -------------------------------------------------------------------------------------------------------------------- #

# FUNCTION FOR PLOTTING THE HILBERT SPECTRA FOR IMFs
def plot_hilbert_spectra_log(time, frequency, amplitude, title, plotter=plt, fs=500, n_levels=180, events=[0],
                         stimulus_onset_in_ms=200, correct_baseline=False):

    # Scale factor (to plot frequency with decimal precision)
    scale_freq = 5.0
    # Defining minimum number of samples needed to define the frequency accurately, n:
    n = 5.0
    # Max scaled frequency
    max_freq = int(scale_freq*fs/n)

    # Creating time axis
    time_ax = np.linspace(0, len(time)-1, len(time))
    # Allocating memory for the rounded frequency
    freq_rounded_array = np.zeros(np.shape(frequency), np.int)

    # Create GRID based on time axis and maximum frequency
    yi = np.linspace(0, max_freq, max_freq + 1)

    absolute_min_log_power = 0.1
    if isinstance(frequency[0], np.ndarray):
        for k in range(len(frequency)):
            min_data = amplitude[k]
            max_data = amplitude[k]
            n_samples = len(amplitude[k])
            amplitude[k] = utils.normalize_data(amplitude[k], min_data, max_data, n_samples)
            min_log_power = min(amplitude[k]*amplitude[k])
            if min_log_power < absolute_min_log_power:
                print min_log_power
                absolute_min_log_power = min_log_power

    absolute_min_log_power -= 0.30*abs(absolute_min_log_power)
    base_value = -absolute_min_log_power

    Z = np.ones((max_freq + 1, len(time_ax)))*absolute_min_log_power
    X, Y = np.meshgrid(time_ax, yi)
    log_array = np.zeros(np.shape(time))

    # Enter loop if more than one IMF exists
    if isinstance(frequency[0], np.ndarray):
        print 'Plotting multiple IMFs in Hilbert Spectra.'
        for i in range(len(frequency)):

            # Round the frequency to the nearest (results in OK resolution if scale_freq > 1, eg scale_freq=10)
            freq_rounded_array[i] = np.ceil(frequency[i]*scale_freq)

            # Compute the logarithmic power, and add it to the previous if the same inst. frequency exists.
            for k in range(len(time_ax)):
                # if amplitude[i, k] <= 0.0:
                #         amplitude[i, k] = 0.001
                if freq_rounded_array[i, k] >= max_freq:
                    freq_rounded_array[i, k] = max_freq - 1
                    # amplitude[i, k] = 0.001
                    amplitude[i, k] = 0.0

                # logarithmic_power = 20.0*np.log10(amplitude[i, k])
                logarithmic_power = amplitude[i, k]*amplitude[i, k]

                current_power = Z[int(freq_rounded_array[i, k]), int(time_ax[k])]
                if np.fabs(current_power) > base_value:
                    # print "Crossing"

                    Z[int(freq_rounded_array[i, k]), int(time_ax[k])] = current_power + logarithmic_power
                else:
                    Z[int(freq_rounded_array[i, k]), int(time_ax[k])] = logarithmic_power
                    # print logarithmic_power

                log_array[k] = Z[int(freq_rounded_array[i, k]), int(time_ax[k])]
    else:

        # Round the frequency to the nearest (results in OK resolution if scale_freq > 1, eg scale_freq=10)
        freq_rounded_array = np.ceil(frequency*scale_freq)
        # Compute the logarithmic power, and add it to the previous if the same inst. frequency exists.
        for k in range(len(time_ax)):
            if amplitude[k] == 0.0:
                amplitude[k] = 0.001
            if freq_rounded_array[k] >= max_freq:
                freq_rounded_array[k] = max_freq - 1
                amplitude[k] = 0.001
            Z[int(freq_rounded_array[k]), int(time_ax[k])] = 20.0*np.log10(amplitude[k])

    # Create figure and subplot.
    # Set titles and labels.
    fig = plotter.figure()
    suptitle = title
    fig.suptitle(suptitle)
    ax = plotter.subplot(111)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Frequency [Hz]')

    # Create contour plot og time, frequency and logarithmic power. Scale frequencies back to original values.
    cax = ax.contourf(X*2, Y/scale_freq, Z, n_levels)
    # Assign color bar to the contour plot
    cb = fig.colorbar(cax)
    # Set label and draw plot
    cb.set_label('Logarithmic power [dB]')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Event triggers')

    if events[0] != 0:
        labels = []
        for k in range(len(events)):
            ax2.set_xticks([k+stimulus_onset_in_ms])
            labels.append(events[k])
            ax2.axvline(k+stimulus_onset_in_ms, color='w', linestyle='--')
        ax2.set_xticklabels(labels)
    else:
        ax2.set_xticks([stimulus_onset_in_ms])
        ax2.set_xticklabels(['stm'])
        ax2.axvline(stimulus_onset_in_ms, color='w', linestyle='--')
    plotter.draw()


# FUNCTION FOR PLOTTING THE HILBERT SPECTRA FOR IMFs
def plot_hilbert_spectra_power(time, frequency, amplitude, title, plotter=plt, fs=500, n_levels=180, events=[0],
                         stimulus_onset_in_ms=500, baseline_correct=True):

    # Scale factor (to plot frequency with decimal precision)
    scale_freq = 10.0
    # Defining minimum number of samples needed to define the frequency accurately, n:
    n = 10.0
    # Max scaled frequency
    max_freq = int(scale_freq*fs/n)

    # Creating time axis
    time_ax = np.linspace(0, len(time)-1, len(time))
    # Allocating memory for the rounded frequency
    freq_rounded_array = np.zeros(np.shape(frequency), np.int)

    # Create GRID based on time axis and maximum frequency
    yi = np.linspace(0, max_freq, max_freq + 1)

    if baseline_correct:
        if isinstance(frequency[0], np.ndarray):
            for k in range(len(amplitude)):
                amplitude[k] /= np.mean(amplitude[k, :int(fs*stimulus_onset_in_ms/1000)])
                # amplitude[k] = [0 if i < 0 else i for i in amplitude[k]]

    absolute_min_power = 100.0
    if isinstance(frequency[0], np.ndarray):
        for k in range(len(frequency)):
            min_data = amplitude[k]
            max_data = amplitude[k]
            n_samples = len(amplitude[k])
            amplitude[k] = utils.normalize_data(amplitude[k], min_data, max_data, n_samples)
            min_power = min(amplitude[k]*amplitude[k])

            if min_power < absolute_min_power:
                absolute_min_power = min_power

    Z = np.ones((max_freq + 1, len(time_ax)))*absolute_min_power
    X, Y = np.meshgrid(time_ax, yi)
    power_array = np.zeros(np.shape(time))

    # Enter loop if more than one IMF exists
    if isinstance(frequency[0], np.ndarray):
        print 'Plotting multiple IMFs in Hilbert Spectra.'
        for i in range(len(frequency)):

            # Round the frequency to the nearest (results in OK resolution if scale_freq > 1, eg scale_freq=10)
            freq_rounded_array[i] = np.ceil(frequency[i]*scale_freq)
            # Compute the power, and add it to the previous if the same inst. frequency exists.
            for k in range(len(time_ax)):
                if freq_rounded_array[i, k] >= max_freq:
                    freq_rounded_array[i, k] = max_freq - 1
                    amplitude[i, k] = 0.0

                power = amplitude[i, k]*amplitude[i, k]

                current_power = Z[int(freq_rounded_array[i, k]), int(time_ax[k])]
                if current_power > absolute_min_power:  # + 0.15:
                    # print "Crossing"
                    Z[int(freq_rounded_array[i, k]), int(time_ax[k])] = current_power + power
                else:
                    Z[int(freq_rounded_array[i, k]), int(time_ax[k])] = power

                power_array[k] = Z[int(freq_rounded_array[i, k]), int(time_ax[k])]
    else:

        # Round the frequency to the nearest (results in OK resolution if scale_freq > 1, eg scale_freq=10)
        freq_rounded_array = np.ceil(frequency*scale_freq)
        # Compute the logarithmic power, and add it to the previous if the same inst. frequency exists.
        for k in range(len(time_ax)):

            if freq_rounded_array[k] >= max_freq:
                freq_rounded_array[k] = max_freq - 1
                amplitude[k] = 0.0
            Z[int(freq_rounded_array[k]), int(time_ax[k])] = amplitude[k]*amplitude[k]

    # Create figure and subplot.
    # Set titles and labels.
    fig = plotter.figure()
    suptitle = title
    fig.suptitle(suptitle)
    ax = plotter.subplot(111)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Frequency [Hz]')

    # Create contour plot og time, frequency and logarithmic power. Scale frequencies back to original values.
    cax = ax.contourf(X*2, Y/scale_freq, Z, n_levels)
    # Assign color bar to the contour plot
    cb = fig.colorbar(cax)
    # Set label and draw plot
    lab = "%s [%s$V^2$]" % ('Normalized Power', u'\u00b5')
    cb.set_label(lab)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # ax2.set_xlabel('Event triggers')

    if events[0] != 0:
        labels = []
        for k in range(len(events)):
            ax2.set_xticks([k+stimulus_onset_in_ms])
            labels.append(events[k])
            ax2.axvline(k+stimulus_onset_in_ms, color='w', linestyle='--')
        ax2.set_xticklabels(labels)
    else:
        ax2.set_xticks([stimulus_onset_in_ms])
        ax2.set_xticklabels(['stm'])
        ax2.axvline(stimulus_onset_in_ms, color='w', linestyle='--')
    plotter.draw()


def plot_single_data_sliding(data_array, events, fs=500.0, label='', subtract_mean=True,  n_sec_per_trial=3, n_trial_per_frame=3):
    '''

    :param data_array:
    :param events:
    :param fs:
    :param label:
    :param subtract_mean:
    :param n_sec_per_trial:
    :param n_trial_per_frame:
    :return:
    '''
    n_sec_per_frame = n_trial_per_frame * n_sec_per_trial
    n_frames = 3*len(events)/n_sec_per_frame

    xdata = np.linspace(0, n_sec_per_frame, n_sec_per_frame*fs)
    # set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='lightgoldenrodyellow')
    ax.autoscale(True)
    plt.subplots_adjust(left=0.03, bottom=0.09, right=0.99, top=0.96)

    # plot first data set
    frame = 0
    data = np.array(data_array[int(frame*n_sec_per_frame*fs):int((frame+1)*n_sec_per_frame*fs)])
    if subtract_mean:
        mean_data = np.mean(data)
    else:
        mean_data = 0.0
    ln, = ax.plot(xdata, data - mean_data, label=label)
    legend = ax.legend(loc='upper center', shadow=True)
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    ax.plot(0.05, 0.9*ax.get_ylim()[0], 'r+', markersize=10)
    ax.axvline(1, color='k', linestyle='--', linewidth=3)
    if n_sec_per_frame == 6:
        ax.plot(3, 0.9*ax.get_ylim()[0], 'r+', markersize=10)
        ax.axvline(4, color='k', linestyle='--', linewidth=3)
    elif n_sec_per_frame == 9:
        ax.plot(3, 0.9*ax.get_ylim()[0], 'r+', markersize=10)
        ax.axvline(4, color='k', linestyle='--', linewidth=3)
        ax.plot(6, 0.9*ax.get_ylim()[0], 'r+', markersize=10)
        ax.axvline(7, color='k', linestyle='--', linewidth=3)
    trial_num = int(frame)*n_trial_per_frame+1
    title = '(Trial ' + str(trial_num) + ', Stm = ' + str(events[trial_num-1]) + \
            ')            &           (Trial ' + str(trial_num+1) + ', Stm = ' + str(events[trial_num]) + \
            ')            &           (Trial ' + str(trial_num+2) + ', Stm = ' + str(events[trial_num+1]) + ')'
    ax.set_title(title)

    # make the slider
    axframe = plt.axes([0.13, 0.02, 0.75, 0.03], axisbg='lightgoldenrodyellow')
    sframe = Slider(axframe, 'Frame', 0, n_frames-1, valinit=0, valfmt='%d')

    # call back function
    def update(val):
        frame = np.floor(sframe.val)
        data2 = data_array[int(frame*n_sec_per_frame*fs):int((frame+1)*n_sec_per_frame*fs)]
        ln.set_xdata(xdata)
        if subtract_mean:
            mean_data2 = np.mean(data2)
        else:
            mean_data2 = 0.0
        ln.set_ydata(data2 - mean_data2)
        trial_num2 = int(frame)*n_trial_per_frame+1
        title2 = '(Trial ' + str(trial_num2) + ', Stm = ' + str(events[trial_num2-1]) + \
                 ')            &           (Trial ' + str(trial_num2+1) + ', Stm = ' + str(events[trial_num2]) + \
                 ')            &           (Trial ' + str(trial_num2+2) + ', Stm = ' + str(events[trial_num2+1]) + ')'
        ax.set_title(title2)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
    # connect callback to slider
    sframe.on_changed(update)


def plot_data_sliding(data_matrix, events, fs=500.0, offset=50.0, n_sec_per_trial=6, n_trial_per_frame=2, plot=plt):
    '''

    :param data_matrix:
    :param events:
    :param fs:
    :param offset:
    :param n_sec_per_trial:
    :param n_trial_per_frame:
    :param plot:
    :return:
    '''
    n_sec_per_frame = n_trial_per_frame * n_sec_per_trial
    n_frames = len(events)/n_trial_per_frame
    n_channels = len(data_matrix)

    xdata = np.linspace(0, n_sec_per_frame, n_sec_per_frame*fs)

    # set up figure
    fig = plot.figure()
    plot.subplots_adjust(left=0.03, bottom=0.09, right=0.99, top=0.96)
    ax = fig.add_subplot(111, axisbg='lightgoldenrodyellow')
    ax.autoscale(True)

    # plot first data set
    frame = 0
    lns = []

    for i in range(n_channels):
        data = np.array(data_matrix[i, int(frame*n_sec_per_frame*fs):int((frame+1)*n_sec_per_frame*fs)])

        if n_channels == 2:
            if i == 0:
                ln, = ax.plot(xdata, data - np.mean(data) + (i*offset), label='C3')
            elif i == 1:
                ln, = ax.plot(xdata, data - np.mean(data) + (i*offset), label='C4')
        elif n_channels == 3:
            if i == 0:
                ln, = ax.plot(xdata, data - np.mean(data) + (i*offset), label='C3')
            elif i == 1:
                ln, = ax.plot(xdata, data - np.mean(data) + (i*offset), label='C4')
            elif i == 2:
                ln, = ax.plot(xdata, data - np.mean(data) + (i*offset), label='Cz')
        else:
            ln, = ax.plot(xdata, data - np.mean(data) + (i*offset))
        lns.append(ln)

    if n_channels == 2 or n_channels == 3:
        legend = ax.legend(loc='upper center', shadow=True)
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        legend_frame = legend.get_frame()
        legend_frame.set_facecolor('0.90')

        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')

        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width

    # ax.plot(0.05, 0.9*ax.get_ylim()[0], 'r+', markersize=10)
    if n_sec_per_trial == 8:
        ax.axvline(2, color='k', linestyle='--', linewidth=3)

        if n_trial_per_frame == 2:
            # ax.plot(8, 0.9*ax.get_ylim()[0], 'r+', markersize=10)
            ax.axvline(10, color='k', linestyle='--', linewidth=3)

    elif n_sec_per_trial == 6:
        ax.axvline(1, color='k', linestyle='--', linewidth=3)

        if n_sec_per_frame == 12:
            ax.axvline(7, color='k', linestyle='--', linewidth=3)
        elif n_sec_per_frame == 18:
            ax.axvline(7, color='k', linestyle='--', linewidth=3)
            ax.axvline(13, color='k', linestyle='--', linewidth=3)

    else:
        ax.axvline(1, color='k', linestyle='--', linewidth=3)

        if n_sec_per_frame == 6:
            ax.axvline(4, color='k', linestyle='--', linewidth=3)
        elif n_sec_per_frame == 9:
            ax.axvline(4, color='k', linestyle='--', linewidth=3)
            ax.axvline(7, color='k', linestyle='--', linewidth=3)

    trial_num = int(frame)*n_trial_per_frame+1
    if n_trial_per_frame == 1:
        title = '(Trial ' + str(trial_num) + ', Stm = ' + str(events[trial_num-1]) + ')'

    elif n_trial_per_frame == 2:
        title = '(Trial ' + str(trial_num) + ', Stm = ' + str(events[trial_num-1]) + \
                 ')            &           (Trial ' + str(trial_num+1) + ', Stm = ' + str(events[trial_num]) + ')'

    elif n_trial_per_frame == 3:
        title = '(Trial ' + str(trial_num) + ', Stm = ' + str(events[trial_num-1]) + \
                 ')            &           (Trial ' + str(trial_num+1) + ', Stm = ' + str(events[trial_num]) + \
                 ')            &           (Trial ' + str(trial_num+2) + ', Stm = ' + str(events[trial_num+1]) + ')'
    ax.set_title(title)

    # make the slider
    axframe = plot.axes([0.13, 0.02, 0.75, 0.03], axisbg='lightgoldenrodyellow')
    sframe = Slider(axframe, 'Frame', 0, n_frames-1, valinit=0, valfmt='%d')

    # call back function
    def update(val):
        frame = np.floor(sframe.val)
        for k in range(n_channels):

            data2 = data_matrix[k, int(frame*n_sec_per_frame*fs):int((frame+1)*n_sec_per_frame*fs)]
            lns[k].set_xdata(xdata)
            lns[k].set_ydata(data2 - np.mean(data2) + (k*offset))

        trial_num2 = int(frame)*n_trial_per_frame+1
        if n_trial_per_frame == 1:
            title2 = '(Trial ' + str(trial_num2) + ', Stm = ' + str(events[trial_num2-1]) + ')'

        elif n_trial_per_frame == 2:
            title2 = '(Trial ' + str(trial_num2) + ', Stm = ' + str(events[trial_num2-1]) + \
                     ')            &           (Trial ' + str(trial_num2+1) + ', Stm = ' + str(events[trial_num2]) + ')'

        elif n_trial_per_frame == 3:
            title2 = '(Trial ' + str(trial_num2) + ', Stm = ' + str(events[trial_num2-1]) + \
                     ')            &           (Trial ' + str(trial_num2+1) + ', Stm = ' + str(events[trial_num2]) + \
                     ')            &           (Trial ' + str(trial_num2+2) + ', Stm = ' + str(events[trial_num2+1]) + ')'
        ax.set_title(title2)
        ax.relim()
        ax.autoscale_view()
        # plot.draw()
    # connect callback to slider
    sframe.on_changed(update)


# -------------------------------------------------------------------------------------------------------------------- #
#                                                ERP-related PLOTTING                                                  #
# -------------------------------------------------------------------------------------------------------------------- #

def plot_avg_erp(erp_data, my_plot=plt, filename=str, event='L'):
    my_plot.ion()
    f, ax = my_plot.subplots(1, 1)
    f.suptitle(filename)
    my_plot.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.92, wspace=0.1, hspace=0.38)
    time_axis = scipy.linspace(start=-200, stop=800, num=len(erp_data))
    ylabel_str = "%s [%sV]" % ('ERP', u'\u00b5')
    ax.set_ylabel(ylabel_str)
    ax.set_xlabel('Time [ms]')
    ax.plot(time_axis, erp_data)
    ax.grid()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Stimulus onset')
    ax2.set_xticks([0])
    ax2.set_xticklabels([event])
    ax.axvline(1, color='r', linestyle='--', linewidth=3)
    my_plot.draw()

