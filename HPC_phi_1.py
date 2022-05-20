from itertools import cycle
import math
import time
import warnings
from matplotlib import colors

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from numpy import random
from numpy.lib.type_check import real
from tqdm import tqdm
from scipy.signal import hilbert, find_peaks, chirp

def vanilla_HPC_phi(
    simulation_duration,
    dt,
    neuron_threshold,
    neuron_time_constant,
    rest_V,
    spike_V,
    refractory_period_duration,
    adaptation_response_constant,
    adaptation_decay_constant,
    adaptation_time_constant,
    OU_sigma,
    OU_mu,
    OU_time_constant,
    theta_amplitude,
    interference_amplitude,
    theta_frequency,
    interference_frequency,
    burst_analogue_threshold,
    constant_input=False,
    constant_input_amplitude=40,
    fig_suppress = False
    ):
    # Initialize simulation 
    num_timesteps = int(simulation_duration/dt)
    refractory_period_length = int(refractory_period_duration/dt)
    rheobase = burst_analogue_threshold 

    V = np.ones((num_timesteps,))
    V = V*rest_V
    xi = 0
    W = 0
    spike_state = False
    refractory_period = 0
    spike_times = np.array([]).astype(int)

    adaptation_decay_rate = adaptation_decay_constant/adaptation_time_constant
    weiner_coefficient = OU_sigma*math.sqrt(dt*OU_time_constant)

    # Initialize forcing
    if constant_input == False:
        theta_rhythm = theta_amplitude*np.sin(2*np.pi*theta_frequency/1000*dt*np.arange(num_timesteps))
        interference_rhythm = interference_amplitude*np.sin(2*np.pi*interference_frequency/1000*dt*np.arange(num_timesteps))
        forcing = np.sum((theta_rhythm, interference_rhythm), axis=0)
    else:
        forcing = np.ones((num_timesteps,))*constant_amplitude
    
    for step in tqdm(range(1, num_timesteps), desc='Simulation progress:'):

        # Update adaptation
        if spike_state == True:
            W += dt*adaptation_response_constant/adaptation_time_constant
        W += -dt*adaptation_decay_rate*W

        # Update Ornstein-Uhlenbeck process
        xi = (OU_mu - xi)*OU_time_constant*dt + weiner_coefficient*np.random.normal(loc=OU_mu, scale =1.0)

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
            spike_times = np.append(spike_times, step)

        # If the neuron is spiking and the absolute refractory period has not elapsed, then increment it
        if spike_state == True and refractory_period <= refractory_period_length:
            refractory_period += 1

    # Phase analytics
    theta_analytic = hilbert(theta_rhythm)

    theta_phase = np.arctan2(theta_analytic.imag, theta_analytic.real)
    theta_phase[theta_phase < 0] += 2*np.pi
    theta_phase += -3*np.pi/2
    theta_phase[theta_phase < 0] += 2*np.pi
    
    theta_omega = np.diff(theta_phase) 
    cycle_boundary_indices = np.where(theta_omega < 0)[0]

    forcing_peak_indices = find_peaks(forcing, height=rheobase)[0]

    cycle_spikes_per_burst = np.array([])
    bad_thresholding_count = 0
    good_thresholding_count = 0
    spike_causing_peaks = np.array([]).astype(int)
    cycle_spike_mean_indices = np.array([]).astype(int)

    for i in tqdm(range(len(cycle_boundary_indices)), desc='Analysis progress:'):
        if i == 0:
            continue

        
        cycle_spike_indices = np.where((spike_times > cycle_boundary_indices[i - 1]) & (spike_times < cycle_boundary_indices[i]))[0]
        num_spikes_i = cycle_spike_indices.shape[0]
        cycle_spike_mean_index = int(np.mean(spike_times[cycle_spike_indices]))
        cycle_peak_indices = np.where((forcing_peak_indices > cycle_boundary_indices[i - 1]) & (forcing_peak_indices < cycle_boundary_indices[i]))[0]
        num_peaks_i = cycle_peak_indices.shape[0]

        if num_peaks_i != 0 and num_spikes_i != 0:
            cycle_spikes_per_burst = np.append(cycle_spikes_per_burst, num_spikes_i/num_peaks_i)
            spike_causing_peaks = np.append(spike_causing_peaks, forcing_peak_indices[cycle_peak_indices])
            cycle_spike_mean_indices = np.append(cycle_spike_mean_indices, cycle_spike_mean_index)
            good_thresholding_count += 1
        elif num_peaks_i == 0 and num_spikes_i == 0:
            good_thresholding_count += 1
        else:
            bad_thresholding_count += 1

    average_burst_number = np.mean(cycle_spikes_per_burst, axis=0)

    if interference_frequency < theta_frequency:
        print('\nGround truth: recession')
    elif interference_frequency == theta_frequency:
        print('\nGround truth: locking')
    else:
        print('\nGround truth: precession')

    print(f'Average burst number: {average_burst_number}')
    print(f'Minimum membrane potential: {np.amin(V)}')
    print(f'Bad thresholding proportion: {bad_thresholding_count}/{good_thresholding_count + bad_thresholding_count}\n')

    print(spike_causing_peaks)
    if fig_suppress == False:
        plt.close()
        fig, axes = plt.subplots(2, 1, sharex=True)
        fig.patch.set_alpha(0.0)

        for ax in axes:
            ax.patch.set_alpha(0.0)
            ax.spines['right'].set_visible(False) 
            ax.spines['top'].set_visible(False) 
            #ax.spines['bottom'].set_color('#CCCCCC') 
            #ax.spines['left'].set_color('#CCCCCC') 

        time_axis = range(num_timesteps) 

        axes[0].plot(time_axis, V, linewidth=0.8, color='k')
        axes[1].plot(time_axis, theta_rhythm, linewidth=0.8, color='k')
        #axes[0].scatter(forcing_peak_indices, np.ones(forcing_peak_indices.shape[0])*spike_V, color='red', s=10)
        #axes[1].scatter(spike_causing_peaks, theta_rhythm[spike_causing_peaks], color='r', s=10)
        axes[1].scatter(cycle_spike_mean_indices, theta_rhythm[cycle_spike_mean_indices], color='r', s=10)
        axes[1].vlines(cycle_boundary_indices, -theta_amplitude, theta_amplitude, color='r', linewidth=0.8)

        axes[1].set_xlabel('$t$')
        axes[1].set_ylabel('$V$')
        #ax.tick_params(axis='x', colors='#CCCCCC')
        #ax.tick_params(axis='y', colors='#CCCCCC')
        #ax.yaxis.label.set_color('#CCCCCC')
        #ax.xaxis.label.set_color('#CCCCCC')
        plt.show()
def HPC_phi_numerical_GT_probe(
    simulation_duration,
    dt,
    mesh_side_length,
    theta_frequency,
    interference_frequency,
    max_amplitude_threshold_proportion,
    theta_mesh_amp_params: list,
    interference_mesh_amp_params: list,
    suppress=True 
    ):

    num_timesteps = int(simulation_duration/dt)
    theta_rhythm = np.sin(2*np.pi*theta_frequency/1000*dt*np.arange(num_timesteps))
    interference_rhythm = np.sin(2*np.pi*interference_frequency/1000*dt*np.arange(num_timesteps))

    tiled_theta = np.empty((mesh_side_length, mesh_side_length, num_timesteps))
    tiled_interference = np.empty((mesh_side_length, mesh_side_length, num_timesteps)) 

    for i, v in enumerate(theta_rhythm):
        tiled_theta[:, :, i] = v
        tiled_interference[:, :, i] = interference_rhythm[i]
    
    forcing_amplitude_mesh = np.transpose(np.array(np.meshgrid(
        np.linspace(
            theta_mesh_amp_params[0],
            theta_mesh_amp_params[1],
            num=mesh_side_length
        ),
        np.linspace(
            interference_mesh_amp_params[0],
            interference_mesh_amp_params[1],
            num=mesh_side_length
        ),
    )))
    
    forcing_array = np.empty((mesh_side_length, mesh_side_length, num_timesteps))
    for i in range(num_timesteps):
        tiled_theta[:, :, i] = tiled_theta[:, :, i]*forcing_amplitude_mesh[:, :, 0]
        tiled_interference[:, :, i] = tiled_interference[:, :, i]*forcing_amplitude_mesh[:, :, 1]
        forcing_array[:, :, i] = tiled_theta[:, :, i] + tiled_interference[:, :, i]
    
    frequency_array = np.empty((mesh_side_length, mesh_side_length))
    theta_frequency_array = np.ones((mesh_side_length, mesh_side_length))*theta_frequency
    for i in range(mesh_side_length): 
        for j in range(mesh_side_length): 
            frequency_array[i, j] = find_peaks(forcing_array[i, j, :], height=max_amplitude_threshold_proportion*np.amax(forcing_array[i, j, :]))[0].shape[0]/simulation_duration*1000
    
    theta_detuning_array = np.flipud(np.transpose(frequency_array - theta_frequency_array))
    if interference_frequency > theta_frequency:
        theta_detuning_array = theta_detuning_array*(-1)
    theta_detuning_array = theta_detuning_array.astype(int)
    print(f'{theta_detuning_array}\n')

    if not suppress:
        plt.close()
        fig, ax = plt.subplots()

        fig.patch.set_alpha(0.0)
        ax.spines['right'].set_visible(False) 
        ax.spines['top'].set_visible(False) 

        time_axis = np.linspace(0, num_timesteps, num=num_timesteps)
        ax.plot(time_axis, forcing_array[0, 0, :])
        # ax.scatter(peaks, np.ones((peaks.shape[0],))*(theta_mesh_amp_params[1] + interference_mesh_amp_params[1]), s=10, color='r')

        plt.show()

    return theta_detuning_array
def GUI_doi(duration, dt, interference_frequency):

    plt.close()
    
    def doi_function(t, A_1, A_2, f_1, f_2):
        function = A_1*np.sin(2*np.pi*f_1*t) + A_2*np.sin(2*np.pi*f_2*t) 
        return function/(np.amax(function))

    num_timesteps = int(duration/dt)
    t = np.linspace(0, duration, num=num_timesteps)

    init_A_1 = 35
    init_A_2 = 35
    init_f_1 = 10
    init_f_2 = interference_frequency

    theta_rhythm = np.sin(2*np.pi*init_f_1*t)
    interference_rhythm = np.sin(2*np.pi*init_f_2*t)

    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    theta_line, = plt.plot(t, theta_rhythm, lw=2, color='lightgrey')
    theta_line.set_visible(False)
    interference_line, = plt.plot(t, interference_rhythm, lw=2, color='lightgrey')
    interference_line.set_visible(False)
    line, = plt.plot(t, doi_function(t, init_A_1, init_A_2, init_f_1, init_f_2), lw=2, color='k')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, duration)
    #ax.hlines(0.2, -10, 10, color='r', linestyle='--')

    plt.subplots_adjust(left=0.25, bottom=0.25)

    theta_amp = plt.axes([0.25, 0.1, 0.65, 0.03])
    theta_amp_ticks = [20, 50, 100]
    
    theta_amp_slider = Slider(
        ax=theta_amp,
        label='Theta amplitude',
        valmin=20,
        valmax=100,
        valinit=init_A_1,
        color='k'
    )
    theta_amp.add_artist(theta_amp.xaxis)
    theta_amp.set_xticks(theta_amp_ticks)

    interference_amp = plt.axes([0.1, 0.25, 0.0225, 0.63])
    interference_amp_ticks = [20, 50, 100]

    interference_amp_slider = Slider(
        ax=interference_amp,
        label='Interference amplitude',
        valmin=20,
        valmax=100,
        valinit=init_A_2,
        orientation='vertical',
        color='k'
    )
    interference_amp.add_artist(interference_amp.yaxis)
    interference_amp.set_yticks(interference_amp_ticks)

    def update_slider(val):
        line.set_ydata(doi_function(t, theta_amp_slider.val, interference_amp_slider.val, init_f_1, init_f_2))
        fig.canvas.draw_idle()

    theta_amp_slider.on_changed(update_slider)
    interference_amp_slider.on_changed(update_slider)

    show_theta_button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    show_theta_button = Button(
        show_theta_button_ax,
        'Show theta',
        hovercolor='0.975'
    )
    def theta_button_press(event):
        if theta_line.get_visible():
            theta_line.set_visible(False)
        else:
            theta_line.set_visible(True)
    show_theta_button.on_clicked(theta_button_press)

    show_interference_button_ax = plt.axes([0.6, 0.025, 0.15, 0.04])
    show_interference_button = Button(
        show_interference_button_ax,
        'Show interference',
        hovercolor='0.975'
    )
    def interference_button_press(event):
        if interference_line.get_visible():
            interference_line.set_visible(False)
        else:
            interference_line.set_visible(True)
    show_interference_button.on_clicked(interference_button_press)

    plt.show()

def main():

    # Time parameters
    simulation_duration = 3000 # ms
    dt = 0.001 # ms

    # Neuron biophysical parameters
    neuron_threshold = -40 # mV
    neuron_time_constant = 10 # ms
    rest_V = -75 # mV
    spike_V = 50 # mV
    refractory_period_duration = 2 # ms

    # Spike frequency adaptation biophysical parameters
    adaptation_response_constant = 10 # mV
    adaptation_decay_constant = 1 # mV
    adaptation_time_constant = 20 # ms

    # Ornstein-Uhlenbeck process parameters
    OU_sigma = 100
    OU_mu = 0.6
    OU_time_constant = 50

    # Dual oscillator input parameters
    theta_amplitude = 20 # mV
    interference_amplitude = 50  # mV
    theta_frequency = 10 # Hz
    interference_frequency = 11 # Hz
    constant_input = False
    constant_input_amplitude = 40

    # Miscellaneous parameters
    fig_suppress = False
    burst_analogue_threshold = 1 
    GT_confirm = False

    visualize_forcing_functions = False
    visualization_duration = 2

    if GT_confirm == False and visualize_forcing_functions == False:
        for interference_frequency in [11]:
            vanilla_HPC_phi(
                simulation_duration,
                dt,
                neuron_threshold,
                neuron_time_constant,
                rest_V,
                spike_V,
                refractory_period_duration,
                adaptation_response_constant,
                adaptation_decay_constant,
                adaptation_time_constant,
                OU_sigma,
                OU_mu,
                OU_time_constant,
                theta_amplitude,
                interference_amplitude,
                theta_frequency,
                interference_frequency,
                burst_analogue_threshold,
                constant_input=constant_input,
                constant_input_amplitude=constant_input_amplitude,
                fig_suppress = fig_suppress
            )
    elif GT_confirm == True and visualize_forcing_functions == False:
        subset_marker_increment = (50 - 20)/11
        detuning_SD = []
        for i, threshold_prop in enumerate(np.linspace(0.2, 0.2, num=1)):
            print(f'{i + 1} -- Threshold: {threshold_prop}')
            theta_detuning_array = HPC_phi_numerical_GT_probe(
                simulation_duration,
                dt,
                10,
                theta_frequency,
                interference_frequency,
                threshold_prop,
                [20 + subset_marker_increment, 50 - subset_marker_increment, 10],
                [20 + subset_marker_increment, 50 - subset_marker_increment, 10],
                suppress=True
            )
            detuning_SD.append(np.std(theta_detuning_array))    
        detuning_SD = np.asarray(detuning_SD)
        print(f'Maximum detuning SD: {np.amax(detuning_SD)}\nAchieved under threshold: {np.linspace(0, 1, num=100)[detuning_SD.argmax()]}')
        precession_GT_array = np.array([-1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 
                                        -1, -1, -1, 0, 0, 0, 1, 1, 1, 1,
                                        -1, -1, -1, 0, 0, 1, 1, 1, 1, 1,
                                        -1, -1, 0, 0, 1, 1, 1, 1, 1, 1,
                                        -1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                        -1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                        0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                                        0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                                        1, 1, 1, 1, 0, 0, 0, 0, 0, 0]).reshape((10, 10))
        recession_GT_array = np.array([-1, -1, -1, -1, -1, -2, -2, -2, -2, -2,
                                       -1, -1, -1, -1, -2, -2, -2, -2, -2, -2,
                                       -1, -1, -1, -2, -2, -2, -2, -2, -2, -2,
                                       -1, -1, -1, -2, -2, -2, -2, -2, -2, -1,
                                       -1, -1, -2, -2, -2, -2, -2, -1, -1, -1,
                                       -1, -2, -2, -2, -2, -2, -1, -1, -1, -1,
                                       -2, -2, -2, -2, -2, -1, -1, -1, -1, 0,
                                       -2, -2, -2, -2, -1, -1, -1, 0, 0, 0,
                                       -2, -2, -2, -1, -1, -1, 0, 0, 0, 0,
                                       -2, -2, -1, -1, 0, 0, 0, 0, 0, 0]).reshape((10, 10))
        print(recession_GT_array == theta_detuning_array)
    if visualize_forcing_functions == True:
        GUI_doi(visualization_duration, dt, interference_frequency)

if __name__ == '__main__':
    main()
