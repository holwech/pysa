from . import emd
import numpy as np
from multiprocessing import Process, Queue


def eemd(x, noise_std, max_modes, max_siftings, ensembles, ensembles_per_process):
    '''
    Threaded Ensemble empirical mode decomposition

    :param x: array(n) of the signal to be analyzed
    :param noise_std: Standard deviation of the noise to be added to the original signal
    :param max_modes: (Integer) Maximimum IMFs to decompose the signal into.
    :param max_siftings: (Integer) Maximum number of siftings per each decomposed IMF
    :param ensembles: (Integer) Number of ensembles
    :param ensembles_per_process: Number of ensebles per process
    :return:
    '''
    n_processes = int(ensembles // ensembles_per_process)
    data_length = len(x)
    noise_std *= np.std(x)
    output = Queue(n_processes)

    processes = [
        Process(
            target=ensemble_process,
            args=(
                x,
                data_length,
                max_modes,
                max_siftings,
                noise_std,
                ensembles_per_process,
                output
            )
        )
        for p in range(n_processes)
    ]

    for p in processes:
        p.start()

    results = [output.get() for p in processes]

    imfs = ensemble_all_processes(data_length, results, n_processes, ensembles, max_modes)

    return imfs


def ensemble_all_processes(data_length, results, n_processes, ensembles, max_modes):
    '''
    Ensemble all processes gathered from the multiprocess

    :param data_length:
    :param results:
    :param n_processes:
    :param ensembles:
    :param max_modes:
    :return:
    '''
    imfs = np.zeros((max_modes + 1, data_length))

    for j in range(n_processes):
        imfs = np.add(imfs, results[j])

    imfs = np.multiply(imfs, 1.0/float(ensembles))

    return imfs


def ensemble_process(x, data_length, max_modes, max_siftings, noise_std, ensembles_per_process, output):
    '''

    :param x:
    :param data_length:
    :param max_modes:
    :param max_siftings:
    :param noise_std:
    :param ensembles_per_process:
    :param output:
    :return:
    '''
    imfs = np.zeros((max_modes + 1, data_length))

    for i in range(ensembles_per_process):
        noise = np.multiply(np.random.randn(data_length), noise_std)
        noise_assisted_data = np.add(x, noise)
        ensemble = emd.emd(noise_assisted_data, max_modes, max_siftings)
        imfs = np.add(imfs, ensemble)

    output.put(imfs)
