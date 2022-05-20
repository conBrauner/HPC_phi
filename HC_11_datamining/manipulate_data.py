import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def extract_spike_times(all_spike_times, spike_IDs, neuron_ID, epoch):

    neuron_spec_spike_indices = np.where(np.abs(spike_IDs - neuron_ID) == np.abs(spike_IDs - neuron_ID).min())[0] 
    spike_times = all_spike_times[neuron_spec_spike_indices]
    spike_times = np.asarray(spike_times[(spike_times >= epoch[0]) & (spike_times <= epoch[1])]) 

    return spike_times

def extract_spike_phase(theta_phase_time_series, analytic_signal, spike_times, epoch, lfp_sf=1250, spike_sf=20000, suppress=False):

    spike_times = np.asarray(spike_times[(spike_times >= epoch[0]) & (spike_times <= epoch[1])])

    spike_to_lfp_indices = (spike_times*lfp_sf).astype(int)
    spike_phase = theta_phase_time_series[spike_to_lfp_indices]

    if not suppress:

        plt.close()
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.plot(analytic_signal[:200000].real, analytic_signal[:200000].imag, linewidth=0.8, color='k')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (unkown)')

        plt.show()

    return spike_phase

def extract_signal_phase(periodic_signal):

    analytic_signal = hilbert(periodic_signal)
    theta_t = np.arctan2(analytic_signal.imag, analytic_signal.real) # Extract phase
    theta_t[theta_t < 0] += 2*np.pi # Shift from interval [-pi, pi) to [0, 2pi)

    omega_t = np.diff(theta_t)
    boundaries = []
    for i, w in enumerate(omega_t):
        if w < -np.pi:
            boundaries.append(i)

    return boundaries, theta_t, analytic_signal

def extract_high_psd_epochs(psd_signal, epoch_start_time, fs, inclusion_dur, exclusion_dur, inclusion_threshold):

    # Convert durations in seconds to N
    inclusion_n = inclusion_dur*fs
    exclusion_n = exclusion_dur*fs

    # Initialize control parameters
    inclusion_epoch_init_index = -1
    inclusion_epoch_end_index = -1
    suprathresh_count = 0
    subthresh_count = 0

    # Initialize inclusion/exclusion epoch arrays
    inclusion_epochs = np.array([], dtype=int).reshape((-1, 2)) # Two columns and n rows
        
    for i, prop in enumerate(psd_signal.flatten()):
        
        # If this is a new putative inclusion epoch
        if inclusion_epoch_init_index < 0 and prop >= inclusion_threshold:
            inclusion_epoch_init_index = i # Note the index where it started
            suprathresh_count += 1 # Count it towards the total number of indices needed to make it a valid duration
            subthresh_count = 0 # Reset the subthresh count (i.e. how long we have been 'out of bounds')

        # If we're already in a putative inclusion epoch
        elif inclusion_epoch_init_index >= 0 and prop >= inclusion_threshold:
            suprathresh_count += 1 # Simply increment the count...
            subthresh_count = 0 # ...and reset 'out of bounds' counter

            # If we thought that the epoch had ended
            if inclusion_epoch_end_index >= 0:
                inclusion_epoch_end_index = -1 # Reset it

        # If we're already in a putative inclusion epoch but we're 'out of bounds'
        elif inclusion_epoch_init_index >= 0 and prop < inclusion_threshold:

            # If we don't have an active potential endpoint already
            if inclusion_epoch_end_index < 0:
                inclusion_epoch_end_index = i # Note where the inclusion epoch appears to have ended

            subthresh_count += 1  # Start counting how long we're out of bounds

        else:
            continue
    
        if inclusion_epoch_init_index >= 0: # If we're currently in a putative inclusion epoch...
            if subthresh_count >= exclusion_n: # ...but we've been out of bounds for too long...
                if suprathresh_count >= inclusion_n: # ...but the inclusion epoch is long enough 

                    # Save that sucker
                    new_inclusion_epoch = np.array([(inclusion_epoch_init_index/fs) + epoch_start_time, (inclusion_epoch_end_index/fs) + epoch_start_time]).reshape((1, 2))
                    inclusion_epochs = np.concatenate((inclusion_epochs, new_inclusion_epoch), axis=0) # Add this epoch to the running list of exclusions

                    # Reset control parameters
                    inclusion_epoch_init_index = -1
                    inclusion_epoch_end_index = -1
                    suprathresh_count = 0
                    subthresh_count = 0

                else: # ... but the inclusion epoch is not long enough
                
                    # Just reset control parameters, don't save the epoch
                    inclusion_epoch_init_index = -1
                    inclusion_epoch_end_index = -1
                    suprathresh_count = 0
                    subthresh_count = 0

            elif i == psd_signal.size - 1: # Otherwise if we're at the end of the signal and in an inclusion epoch...

                    # Save that sucker
                    new_inclusion_epoch = np.array([(inclusion_epoch_init_index/fs) + epoch_start_time, (i/fs) + epoch_start_time]).reshape((1, 2))
                    inclusion_epochs = np.concatenate((inclusion_epochs, new_inclusion_epoch), axis=0) # Add this epoch to the running list of exclusions



    return inclusion_epochs