from itertools import cycle
import math
import time
import warnings
from matplotlib import colors

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from numpy.lib.type_check import real
from tqdm import tqdm
from scipy.signal import hilbert, find_peaks, chirp

def vanilla_HPC_phi():

    # Time parameters
    simulation_duration = 3000 # ms
    dt = 0.01 # ms
    
    # Neuron biophysical parameters
    neuron_threshold = -35 # mV
    neuron_time_constant = 10 # ms
    rest_V = -75 # mV
    spike_V = 50 # mV
    refractory_period_duration = 2 # ms
    
    # Spike frequency adaptation biophysical parameters
    adaptation_response_constant = 40 # mV
    adaptation_decay_constant = 5 # mV
    adaptation_time_constant = 10 # ms
    
    # Ornstein-Uhlenbeck process parameters
    OU_sigma = 100
    OU_mu = 0.7
    OU_time_constant = 20

    # Dual oscillator input parameters
    theta_amplitude = 30 # mV
    interference_amplitude = 30 # mV
    constant_input = False
    constant_input_amplitude = 40.0000001 # mV

    theta_frequency = 10 # Hz
    interference_frequency = 11 # Hz
    # =============================================================
    num_timesteps = int(simulation_duration/dt)

    if constant_input == False:
        theta_vector = 2*np.pi*theta_frequency/1000*dt*np.arange(num_timesteps)
        interference_vector = 2*np.pi*interference_frequency/1000*dt*np.arange(num_timesteps)

        theta_rhythm = theta_amplitude*np.sin(theta_vector)
        interference_rhythm = interference_amplitude*np.sin(interference_vector)
        forcing = np.sum((theta_rhythm, interference_rhythm), axis=0)
    else:
        forcing = np.ones((num_timesteps,))*constant_input_amplitude

    refractory_period_length = int(refractory_period_duration/dt)

    V = np.ones((num_timesteps,))
    V = V*rest_V
    xi = 0
    W_list = [0]
    W = 0#np.zeros((num_timesteps,))

    spike_state = False
    refractory_period = 0
    spike_times = np.zeros((num_timesteps,))

    random_numbers = np.random.normal(loc=OU_mu, scale=1.0, size=(num_timesteps,))

    adaptation_decay_rate = adaptation_decay_constant/adaptation_time_constant
    weiner_coefficient = OU_sigma*math.sqrt(dt*OU_time_constant)

    for step in tqdm(range(1, num_timesteps), desc='Simulation'):

        # Update adaptation variable
        if spike_state == True:
            W += dt*adaptation_response_constant/adaptation_time_constant
            #W[step] = W[step - 1] + dt*adaptation_response_constant
        W += -dt*adaptation_decay_rate*W
        #W[step] = W[step - 1] - dt*adaptation_decay_rate

        # Update Ornstein-Uhlenbeck process
        xi = (OU_mu - xi)*OU_time_constant*dt + weiner_coefficient*random_numbers[step] 

        # Conditionally update voltage based on spike and refractory period state
        if spike_state == False:
            V[step] = V[step - 1] + dt*((-(V[step - 1] - rest_V) - W + xi + forcing[step])/neuron_time_constant)
        elif spike_state == True and refractory_period <= refractory_period_length:
            V[step] = spike_V
        elif spike_state == True and refractory_period > refractory_period_length:
            V[step] = rest_V

        # If the absolute refractory period is over
        if refractory_period > refractory_period_length:
            refractory_period = 0

        # If the voltage is subthreshold, set the spike state to be false
        if V[step] < neuron_threshold:
            spike_state = False
        # If the voltage is threshold or suprathreshold then set spike state to be true
        else:
            spike_state = True

        # If the neuron is spiking and the refractory period is zero, take the current step to be a spike time
        if spike_state == True and refractory_period == 0:
            spike_times[step] = 1
        
        # If the neuron is spiking and the absolute refractory period has not elapsed, then increment it
        if spike_state == True and refractory_period <= refractory_period_length:
            refractory_period += 1

        if type(W) == float:
            W_list.append(W)
    if type(W) == float:
        W = W_list
    # =============================================================
    plt.close()
    fig, axes = plt.subplots(3,1)
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.spines['right'].set_visible(False) 
        ax.spines['top'].set_visible(False) 

    time_axis = np.linspace(0, simulation_duration, num=int(simulation_duration/dt))

    axes[0].plot(time_axis, V, linewidth=0.8, color='k')
    spike_times[spike_times == 0] = np.nan
    axes[0].scatter(time_axis, spike_times*spike_V, color='red', s=10)

    axes[1].plot(time_axis, forcing, linewidth=0.8, color='deeppink')
    axes[1].scatter(time_axis, spike_times*forcing, color='red', s=10)
    axes[2].plot(time_axis, theta_rhythm, linewidth=0.8, color='slateblue')
    axes[2].scatter(time_axis, spike_times*theta_rhythm, color='red', s=10)

    axes[-1].set_xlabel('Time ($ms$)')
    axes[0].set_ylabel('$V(t)$')
    axes[1].set_ylabel('$I(t)$')
    axes[2].set_ylabel('$\Theta$')

    plt.show()
def disc_regimeSearch(i_freq_min=1, i_freq_max=20, i_freq_num=20, plot_all=False, signal_index_to_plot=15, verbose=True):
    
    # Time parameters
    simulation_duration = 2000 # ms
    dt = 0.01 # ms
    num_timesteps = int(simulation_duration/dt)

    # Dual oscillator input parameters
    theta_amplitude = 1
    interference_amplitude = 1
    theta_frequency = 10
    interference_frequencies = np.linspace(i_freq_min, i_freq_max, i_freq_num)

    solution_array = np.empty((i_freq_num, num_timesteps), dtype=float)
    theta_analytic_signal = np.empty((num_timesteps,), dtype=complex)
    theta_phase = np.empty((num_timesteps,), dtype=float)
    # ==============================================================================================
    t0 = time.time()
    with tqdm(total=i_freq_num, desc='Generating interference patterns and analytic theta...') as pbar1:
        for i in range(i_freq_num):
            solution_array[i, :] = interference_amplitude*np.sin(2*np.pi*interference_frequencies[i]/1000*dt*np.arange(num_timesteps))
            solution_array[i, :] += theta_amplitude*np.sin(2*np.pi*theta_frequency/1000*dt*np.arange(num_timesteps))
            pbar1.update(1)
   
    theta_analytic_signal = hilbert(theta_amplitude*np.sin(2*np.pi*theta_frequency/1000*dt*np.arange(num_timesteps)))
    theta_phase = np.arctan2(theta_analytic_signal.imag, theta_analytic_signal.real) + np.pi
    inst_freq = np.diff(theta_phase)
    #theta_phase = np.unwrap(theta_phase)
    last_index = len(inst_freq)
    cycle_boundaries = np.where(inst_freq < 0)[0].astype(int)
    t1 = time.time()
    print(f'Time to simulate: {(t1 - t0):.2f} s\n')
    # ==============================================================================================
    with tqdm(total=i_freq_num*(len(cycle_boundaries) + 1), desc='Computing local maxima theta phase...') as pbar2:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in range(i_freq_num):
                peak_indices = find_peaks(solution_array[i, :], height=1.0)[0]
                if i == signal_index_to_plot:
                    plotable_peak_indices = peak_indices.astype(int)

                for j, cycle_index in enumerate(cycle_boundaries, start=0):
                    if j == 0:
                        cycle_peak_indices = peak_indices[peak_indices < cycle_index]
                        cycle_phi = np.mean(theta_phase[cycle_peak_indices]).reshape((1,))
                        cycle_phi_vector = cycle_phi
                        pbar2.update(1)
                    else:
                        cycle_peak_indices = peak_indices[(peak_indices < cycle_index) & (peak_indices >= cycle_boundaries[j - 1])]
                        cycle_phi = np.mean(theta_phase[cycle_peak_indices]).reshape((1,))
                        cycle_phi_vector = np.concatenate((cycle_phi_vector, cycle_phi), axis=0)
                        pbar2.update(1)
                cycle_peak_indices = peak_indices[peak_indices > cycle_index]
                cycle_phi = np.mean(theta_phase[cycle_peak_indices]).reshape((1,))
                cycle_phi_vector = np.concatenate((cycle_phi_vector, cycle_phi), axis=0).reshape((-1, 1))

                if i == 0:
                    cycle_phi_array = cycle_phi_vector
                else:
                    cycle_phi_array = np.concatenate((cycle_phi_array, cycle_phi_vector), axis=1)
                pbar2.update(1)

    cycle_phi_array = np.transpose(cycle_phi_array)
    t2 = time.time()
    print(f'cycle_phi_array shape: {cycle_phi_array.shape}')
    print(f'Time to project spikes onto theta phase: {(t2 - t1):.2f} s\n')
    # ==============================================================================================
    with tqdm(total=i_freq_num, desc='Computing RMQ...') as pbar3:
        RMQ_array = np.empty((i_freq_num,))
        for i in range(i_freq_num):
            series = cycle_phi_array[i, :]
            RMQ = np.mean(np.diff(np.flip(series[~np.isnan(series)])))
            if verbose == True:
                print(f'\nReturn map coordinates for signal {i} with frequency {interference_frequencies[i]} Hz\n{np.diff(np.flip(series[~np.isnan(series)]))}')
                print(f'Corresponding RMQ: {RMQ}')
            RMQ_array[i] = RMQ
            pbar3.update(1)
    print(f'\nRMQ_array: {RMQ_array}')
    t3 = time.time()
    print(f'Time to compute RMQs: {(t3 - t2):.2f} s\n')
    # ==============================================================================================
    fig, axes = plt.subplots(3, 1, sharex=False)
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
        ax.spines['right'].set_visible(False) 
        ax.spines['top'].set_visible(False) 
    
    if plot_all == True:
        for signal_index in range(i_freq_num):
            axes[0].plot(dt*np.arange(num_timesteps), solution_array[signal_index, :], linewidth=0.4)
    else:
        print(f'Interference pattern frequencies\ntheta: {theta_frequency} Hz\ninterference: {interference_frequencies[signal_index_to_plot]} Hz')
        axes[0].plot(dt*np.arange(num_timesteps), solution_array[signal_index_to_plot, :], linewidth=0.8, color='deeppink')
        axes[0].scatter(dt*np.arange(num_timesteps)[plotable_peak_indices], solution_array[signal_index_to_plot, plotable_peak_indices], s=10, color='deeppink')

    axes[1].plot(dt*np.arange(num_timesteps), theta_phase, linewidth=0.8, color='slateblue')
    axes[1].scatter(dt*np.arange(num_timesteps)[np.append(cycle_boundaries, last_index)], cycle_phi_array[signal_index_to_plot, :], s=10, color='slateblue')

    axes[2].plot(interference_frequencies, RMQ_array, linewidth=0.8, color='k')

    axes[0].set_xlabel('Time ($ms$)')
    axes[0].set_ylabel('Amplitude ($agnostic$)')
    axes[1].set_xlabel('Time ($ms$)')
    axes[1].set_ylabel('Theta phase ($radians$)')
    axes[2].set_xlabel('Interference_frequency ($Hz$)')
    axes[2].set_ylabel('RMQ ($radians$)')

    plt.show()
def cont_regimeSearch(num_s, dt, theta_freq, interference_min_freq, interference_max_freq, prettyplot=True):
    # =============================================================
    time_axis = np.linspace(0, num_s, num=int(num_s/dt))
    X = np.cos(2*np.pi*theta_freq*time_axis)
    Y = chirp(time_axis, interference_min_freq, np.max(time_axis), interference_max_freq)

    Z = X + Y

    F = interference_min_freq + (interference_max_freq - interference_min_freq)*time_axis/max(time_axis)

    analytic_signal = hilbert(X)
    P = np.arctan2(analytic_signal.imag, analytic_signal.real) + np.pi

    peak_indices = find_peaks(Z, height=1.0)[0]
    peak_magnitudes = Z[peak_indices]
    phase_magnitudes = P[peak_indices]
    theta_magnitudes = X[peak_indices]
    # =============================================================
    inst_freq = np.diff(P)
    cycle_boundaries = np.where(inst_freq < 0)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, cycle_index in enumerate(cycle_boundaries, start=0):
            if i == 0:
                cycle_peak_indices = peak_indices[peak_indices < cycle_index]
                cycle_phi = np.mean(P[cycle_peak_indices]).reshape((1,))
                cycle_phi_array = cycle_phi
            else:
                cycle_peak_indices = peak_indices[(peak_indices < cycle_index) & (peak_indices >= cycle_boundaries[i - 1])]
                cycle_phi = np.mean(P[cycle_peak_indices]).reshape((1,))
                cycle_phi_array = np.concatenate((cycle_phi_array, cycle_phi), axis=0)
        
        cycle_peak_indices = peak_indices[peak_indices > cycle_index]
        cycle_phi = np.mean(P[cycle_peak_indices]).reshape((1,))
        cycle_phi_array = np.concatenate((cycle_phi_array, cycle_phi), axis=0)
    # =============================================================
    coarseness = 10

    defined_cycle_indices = np.argwhere(~np.isnan(cycle_phi_array)).flatten().astype(int)
    F_defined = F[cycle_boundaries[defined_cycle_indices][10:].astype(int)]

    RMQ_coords = np.flip(np.diff(np.flip(cycle_phi_array[~np.isnan(cycle_phi_array)])))

    num_RMQs = RMQ_coords.shape[0] - coarseness + 1
    print(f'Number of defined cycles: {defined_cycle_indices.shape[0]}\nNumber of defined cycle differences: {RMQ_coords.shape[0]}\nNumber of RMQs to compute: {num_RMQs}')

    RMQs= []
    for i in range(num_RMQs):
        RMQs.append(np.mean(RMQ_coords[i:i + coarseness]))
    RMQs = np.array(RMQs)
    # =============================================================
    plt.close()
    if not prettyplot:
        fig, axes = plt.subplots(3, 1, sharex=True)

        fig.patch.set_alpha(0.0)

        for ax in axes:
            ax.patch.set_alpha(0.0)
            ax.spines['right'].set_visible(False) 
            ax.spines['top'].set_visible(False) 
        
        
        axes[0].plot(F, Z, linewidth=0.8, color='k', zorder=1)
        axes[0].scatter(F[peak_indices], peak_magnitudes, s=5, color='r', zorder=2)

        axes[1].plot(F, P, linewidth=0.8, color='darkgrey', linestyle='dashed', zorder=1)
        axes[1].scatter(F[peak_indices], phase_magnitudes, s=5, color='k', zorder=2)
        axes[1].vlines(F[cycle_boundaries], 0, 2*np.pi, linewidth=0.6, color='crimson', zorder=0)

        axes[2].scatter(F_defined, RMQs, s=5, color='deeppink')
        axes[2].hlines(0, 0, 20, colors='k', linestyle='dashed')
        plt.show()
    if prettyplot:
        fig, ax = plt.subplots()

        fig.patch.set_facecolor('k')

        ax.patch.set_color('k')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.scatter(F[peak_indices], phase_magnitudes, s=.1, color='w')

        axis_object = plt.gca()
        axis_object.axes.get_xaxis().set_visible(False)
        axis_object.axes.get_yaxis().set_visible(False)

        plt.tight_layout()


        plt.show()
    # =============================================================
    
def main():

    test_sim = True
    regimeSearch_continuous = False
    regimeSearch_discrete = False

    if test_sim:
        vanilla_HPC_phi()
    if regimeSearch_continuous:
        cont_regimeSearch(8000, 0.0001, 10, 1, 20, prettyplot=True)
    if regimeSearch_discrete:
        disc_regimeSearch(i_freq_min=1, i_freq_max=20, i_freq_num=20, plot_all=False, signal_index_to_plot=9, verbose=False)

if __name__ == '__main__':
    main()