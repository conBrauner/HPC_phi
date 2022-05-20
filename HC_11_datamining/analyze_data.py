from datetime import datetime
from decimal import Decimal
from fractions import Fraction
import math
import os, sys
import warnings
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from HC_11_datamining.manipulate_data import *

def compute_psd(signal, fs, band: list, relative=False, window_length_seconds=1, suppress=True):

    band = np.asarray(band)
    low_freq, high_freq = band

    welch_window_length = window_length_seconds*fs

    # Compute Welch periodogram
    freq_axis, psd = welch(signal, fs, nperseg=welch_window_length)
    df = freq_axis[1] - freq_axis[0]

    # True for indices which fall in the frequency band of interest
    band_mask = np.logical_and(freq_axis >= low_freq, freq_axis <=high_freq)

    # Integral approximation using Simpson's quadrature over the frequency band 
    band_power = sum(psd[band_mask]*df)

    # Normalize by the total power, if requested
    if relative == True:
        band_power /= sum(psd*df)

    if suppress != True:
        
        plt.close()
        fig, ax = plt.subplots()

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.plot(freq_axis, psd, linewidth=0.8, color='k')

        ax.set_xlabel('Frequency ($Hz$)')
        ax.set_ylabel('Power spectral density ($A^2/Hz)$')

        plt.show()

    return band_power

def windowed_bandpower(window_duration, signal, sf, band: list, overlap=False, relative=False, suppress=True):

    window_width_n = math.floor(window_duration*sf)

    if overlap == True:
        num_windows = len(signal) - window_width_n + 1
    else:
        num_windows = len(signal)//window_width_n

    window_bandpowers = np.empty((num_windows,))
    bandpower_sf = num_windows/(len(signal)/sf)

    with tqdm(total=num_windows, desc='Computing window psd:') as pbar:
        for i in range(num_windows):

            if overlap:
                window = signal[i: window_width_n + i]
            else:
                window = signal[i*window_width_n:(i+1)*window_width_n]

            window_bandpowers[i] = compute_psd(
                window, 
                sf, 
                band, 
                relative=relative, 
                suppress=suppress
        )
            pbar.update(1)
    
    return window_bandpowers, bandpower_sf

def windowed_linear_regression(predictors, responses, window_width, window_increment):

    x = np.asarray(predictors).reshape((-1, 1))
    y = np.asarray(responses).reshape((1, -1))

    try:
        num_windows = int((np.amax(predictors) - np.amin(predictors) - window_width)//window_increment)
    except ValueError:
        return [np.NAN], [np.NAN], [np.NAN]
    window_markers = np.empty((num_windows,))

    beta_1 = []
    R_sq = []

    for i in range(num_windows):

        window = ((x >= (window_increment*i)) & (x <= (window_width + window_increment*i)))
        x_window = x[window].reshape((-1, 1))
        y_window = y[window.T].reshape((-1,))
        
        if x_window.shape[0] == y_window.shape[0] and x_window.shape[0] != 0:
            model = LinearRegression().fit(x_window, y_window)
            window_markers[i] = np.mean([np.amin(x_window), np.amax(x_window)])
            beta_1.append(model.coef_[0])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                R_sq.append(model.score(x_window, y_window))

        else:
            window_markers[i] = np.NAN
            beta_1.append(np.NAN)
            R_sq.append(np.NAN)
            continue

    
    return beta_1, R_sq, window_markers

def classical_phase_prec_quant(session_data_dict, theta_phase_time_series, analytic_signal, epoch_of_interest, neuron_ID_index, predefined_spike_data=False, predefined_spike_phase=None, predefined_spike_times=None, verbose=True, suppress=False, save_dir=False):

    position_array = session_data_dict['position_array']
    position_time_stamps = session_data_dict['position_time_stamps']

    all_spike_times = session_data_dict['all_spike_times']
    all_spike_IDs = session_data_dict['all_spike_IDs']
    pyramidal_IDs = session_data_dict['pyramidal_IDs']

    selected_neuron_ID = pyramidal_IDs[neuron_ID_index]

    spike_times = extract_spike_times(all_spike_times, all_spike_IDs, selected_neuron_ID, epoch_of_interest)

    if verbose:
        print(f'\nSelected pyramidal ID: {selected_neuron_ID}')
        print(f'Number of spikes fired by neuron {selected_neuron_ID} over entire lfp: {spike_times.shape[0]}')

    if predefined_spike_data == False:
        spike_phase = extract_spike_phase(theta_phase_time_series, analytic_signal, spike_times, epoch_of_interest, suppress=True)
        spike_times = spike_times[(spike_times >= epoch_of_interest[0]) & (spike_times <= epoch_of_interest[1])]
    else:
        spike_phase = predefined_spike_phase
        spike_times = predefined_spike_times
    if verbose:
        print(f'Number of spikes fired by neuron {selected_neuron_ID} over EOI: {spike_times.shape[0]}')
        print(f'Proportion of spikes fired by neuron {selected_neuron_ID} over EOI: {spike_times.shape[0]/all_spike_times[(all_spike_times >= epoch_of_interest[0]) & (all_spike_times <= epoch_of_interest[1])].shape[0]}')
    
    defined_spike_phase = []
    defined_position = []

    nan_counter = 0
    for i, j in enumerate(spike_times): 
        nearest_position_index = np.abs(position_time_stamps - j).argmin()
        if np.isnan(position_array[nearest_position_index]):
            nan_counter += 1
            continue
        else:
            defined_spike_phase.append(spike_phase[i])
            defined_position.append(position_array[nearest_position_index])

    window_width = 0.1
    window_increment = 0.01
    #reg_coeff, coeff_of_deter, window_markers = windowed_linear_regression(defined_position, defined_spike_phase, window_width, window_increment)

    if not suppress and nan_counter < spike_times.shape[0]:
        print(f'{spike_times.shape[0]} spikes found')
        print(f'{nan_counter} spikes with undefined position')
    
        plt.close()

        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.scatter(defined_position, defined_spike_phase, s=10, color='k')
        #ax.plot(window_markers, reg_coeff, linewidth=0.8, color='k')
        #ax.plot(window_markers, coeff_of_deter, linewidth=0.8, color='r')

        ax.set_xlabel('Position ($m$)')
        ax.set_ylabel('Spike Phase')
        ax.set_ylim(-np.pi/2, 5*np.pi/2)
        ax.set_yticks([-np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2])
        ax.set_yticklabels(['$-\\frac{\pi}{2}$', '$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$', '$\\frac{5\pi}{2}$'])
        #ax.legend(['Action potentials', 'Regression coefficient', 'Coefficient of determination'])
        plt.show()

        #if save_dir != False:
        #    figure_object = plt.gcf()
        #    figure_object.savefig(save_dir + f'/{neuron_ID_index}_{int(datetime.now().hour)}_{int(datetime.now().day)}_{int(datetime.month().day)}', transparent=True, format='svg', bbox_inches='tight')

def compute_PRQ(cycle_boundaries, spike_times, spike_phase, fs, neuron_ID, animal_name, epoch_type, save_dir=False, return_map_suppress=False, p_val_suppress=False):

    # Interpret inputs as arrays
    cycle_boundaries = np.asarray(cycle_boundaries)/fs
    spike_times = np.asarray(spike_times)
    spike_phase = np.asarray(spike_phase)

    num_cycles = cycle_boundaries.shape[0] - 1
    nan_counter = 0

    # Loop over each cycle
    phi_array = np.array([])
    for cycle_index, right_boundary in enumerate(cycle_boundaries[1:]):

        left_boundary = cycle_boundaries[cycle_index]
        cycle_phase = spike_phase[(spike_times >= left_boundary) & (spike_times <= right_boundary)]
        
        if np.any(cycle_phase):
            cycle_phi = np.mean(cycle_phase)
            phi_array = np.concatenate((phi_array, np.array([cycle_phi])))
        else:
            phi_array = np.concatenate((phi_array, np.array([np.nan])))
            nan_counter += 1
    
    difference_array = np.subtract(np.delete(phi_array, -1), np.delete(phi_array, 0))
    
    RM_x = np.delete(phi_array, -1).reshape((1, -1))
    RM_y = np.delete(phi_array, 0).reshape((1, -1))
    RM_coords = np.concatenate((RM_x, RM_y), axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        PRQ = np.nanmean(difference_array)
    
    difference_array = difference_array[~np.isnan(difference_array)]
    print(f'{nan_counter} of {num_cycles} cycles had no spikes')
    print(f'PRQ: {PRQ}')

    if not p_val_suppress:
        p_val = compute_PRQ_p(difference_array)
    else:
        p_val = None

    line_of_identity = np.linspace(0, 2*np.pi, num=3)
    plt.close()
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.plot(line_of_identity, line_of_identity, linewidth=1.0, color='k', linestyle='dashed')
    ax.scatter(np.delete(phi_array, -1), np.delete(phi_array, 0), s=10, color='k')

    ax.set_xlabel('$\phi_{k-1}$')
    ax.set_ylabel('$\phi_k$')
    ax.set_ylim(0, 2*np.pi)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$'])
    ax.set_xlim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$'])

    if save_dir != False: 
        plt.savefig(save_dir + '/'.join(['/modelFigures', 'datamining_PRQ', animal_name, epoch_type, f'{neuron_ID}_{datetime.now().month}_{datetime.now().day}_{datetime.now().hour}.svg']), format='svg', bbox_inches='tight')
    if not return_map_suppress:
        plt.show()


    return PRQ, p_val, RM_coords

def compute_PRQ_p(difference_array):

    N = difference_array.shape[0]
    num_successes = 0
    num_failures = 0
    prob_success = 1/2

    for difference in difference_array:
        if difference > 0:
            num_successes += 1
        else:
            num_failures += 1

    p_val = 0
    for x in range(num_successes, N + 1):
        p_val += Decimal(math.comb(N, x))*Decimal(prob_success**x)*Decimal((1 - prob_success)**(N - x))
    print(f'p value: {float(p_val)}')
    return float(p_val)



