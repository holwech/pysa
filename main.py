import numpy as np
import matplotlib.pylab as plt
import pysa.emd as emddev
import pysa.eemd as eemdev
import pysa.visualization as plotter
import pysa.utils as utils


if __name__ == '__main__':
    max_modes = 15
    ensembles = 100
    ensembles_per_process = 10
    max_siftings = 200
    end_time = 1
    sample_freq = 1000
    sample_freq_trial = 500
    noise_std = 10
    noise_std2 = 1500

    a_0 = 10
    a_1 = 2
    a_2 = 3000
    a_3 = 2000
    a_4 = -1.0/256.0
    a_5 = 1.0/512.0
    alpha = 5.0
    beta = 50.0
    gamma = 0.04
    omega_0 = alpha * 2.0 * np.pi
    omega_1 = beta * 2.0 * np.pi
    omega_2 = 10.0 * 2.0 * np.pi
    omega_3 = 5.0 * 2.0 * np.pi
    time_ax = np.linspace(0, end_time, end_time * sample_freq)

    sine_wave = a_0 * np.sin(omega_0 * time_ax) + \
                a_1 * np.sin(omega_1 * time_ax) + \
                a_2 * np.sin(omega_2 * time_ax) + \
                a_3 * np.sin(omega_3 * time_ax)

    sine_wave[500:] *= 5.0
    simulated_signal = a_0 * np.sin(omega_0 * time_ax) + a_1 * np.sin(omega_1 * time_ax) + np.exp(2.0*time_ax)

    sinusoidal_wave = a_0 * np.sin(omega_0 * time_ax)
    sinusoidal_wave_2 = 100 * np.sin(omega_1 * time_ax)
    noise = np.multiply(np.random.randn(len(sinusoidal_wave)), noise_std)
    sinus_with_noise = sinusoidal_wave + sinusoidal_wave_2 + noise

    max_data = max(sinus_with_noise)
    min_data = min(sinus_with_noise)
    n_samples = end_time * sample_freq

    plotter.plot_single_channel(sample_freq, sinus_with_noise, plt)

    imfs = emddev.emd(sinus_with_noise, min_data, max_data, max_modes, max_siftings)

    component_of_interest = np.zeros(len(sinus_with_noise), dtype=np.float64)
    for i in range(len(imfs)):
        imfs[i] = utils.reverse_normalization(imfs[i], min_data, max_data, len(sinusoidal_wave))
        print np.std(imfs[i])
        if i >= 2:
            component_of_interest += imfs[i]

    plotter.plot_intrinsic_mode_functions(sample_freq, imfs, "Normalized", plt)

    plotter.plot_single_channel(sample_freq, component_of_interest, plt)

    plt.show(block=True)
