import math
import os
import time
import warnings
from datetime import datetime, date
from pathlib import Path
import operator as op

import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, inferno
import numba as nb
import numpy as np
from scipy.signal import hilbert, find_peaks
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from numba_ufuncs import (scalar_add, condit_add,
                                  condit_assign_bool, condit_assign_num,
                                  hadamard, ornstein_uhlenbeck,
                                  subthresh_SASLIF_integrate, integer_add,
                                  stack_topography_flatten, heterogenous_normal)
from HPC_phi_ARC_instructions import *

# Pipeline manager
class PipelineManager():

    def __init__(self, simulation_specifications: dict, analysis_specifications: dict, ARC=False):

        self.__dict__.update(simulation_specifications)
        self.__dict__.update(analysis_specifications)

        self.sim_spec_dict = simulation_specifications
        self.num_timesteps = int(self.simulation_duration/self.dt)
        self.num_subsets = 0
        self.ARC = ARC
        self.probe_TS = False
        self.probe_RM = False
        self.date_of_sim = datetime.now()
        self.no_V = False
        self.cmap = 'cool'

        if self.contour_popup_suppress == False or self.figure_path != False:
            self._set_cmap()

        if self.theta_frequency < self.interference_frequency:
            self.ground_truth = 'precession'
        elif self.theta_frequency > self.interference_frequency:
            self.ground_truth = 'recession'
        else:
            self.ground_truth = 'locking'

        if self.ARC == True:
            matplotlib.use('Agg')
            self.contour_popup_suppress = True
            self.dbscan_fig_suppress = True

    def _init_forcing(self):

        # Infer subspace sizes from class attribute dictionary
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            theta_subset_key = '_'.join(list(self.__dict__.keys())[list(self.__dict__.values()).index('theta_amplitude')].split('_')[:2])
            interference_subset_key = '_'.join(list(self.__dict__.keys())[list(self.__dict__.values()).index('interference_amplitude')].split('_')[:2])

        # Synthetic theta rhythm for phase extraction later
        theta_rhythm = np.sin(2*np.pi*self.theta_frequency/1000*self.dt*np.arange(self.num_timesteps))

        # Create parameter mesh so simulation can generate forcing current internally
        forcing_amplitude_mesh = np.transpose(np.array(np.meshgrid(self.__dict__[theta_subset_key], self.__dict__[interference_subset_key])))

        return forcing_amplitude_mesh, theta_rhythm, theta_subset_key, interference_subset_key

    def _set_cmap(self, RGB_start=[(221)/256, (33)/256, (255)/256], RGB_end=[(4)/256, (251)/256, (255)/256], blunt_end=False, blunt_end_length=25, preset_name=False):
        #[(119 + -93)/256, (176 + -93)/256, (93 + -93)/256], RGB_end=[(41 + 50)/256, (171 + 50)/256, (202 + 50)/256]
        if preset_name != False:
            self.cmap = preset_name
        else:
            N = 256
            value_array = np.ones((N, 4))

            value_array[:, 0] = np.linspace(RGB_start[0], RGB_end[0], N)
            value_array[:, 1] = np.linspace(RGB_start[1], RGB_end[1], N)
            value_array[:, 2] = np.linspace(RGB_start[2], RGB_end[2], N)

            if blunt_end != False:
                value_array[:blunt_end_length, :] = np.array(blunt_end.append(1))

            self.cmap = ListedColormap(value_array)

    def add_parameter_subset(self, start: float, stop: float, num: int, name: str):

        self.__dict__.update({f'subset_{self.num_subsets + 1}': np.linspace(start, stop, num)})
        self.__dict__.update({f'subset_{self.num_subsets + 1}_name': name})
        self.num_subsets += 1

    def execute_pipeline(self, mesh_type: str):

        # Add run number to encoded variables
        self.sim_spec_dict.update({
            'run_number': self.run_number
        })

        # Initialize simulation file name and encode path
        if mesh_type == 'forcing':
            self.sim_name = f"{self.ground_truth}_basemodel_---_{self.date_of_sim.day}-{self.date_of_sim.month}-{self.date_of_sim.year}_{self.run_number}"
            self.mesh_param_1 = '-'
            self.mesh_param_2 = '-'
            self.sim_spec_dict.update({
                'mesh_param_1': '-',
                'mesh_param_2': '-'
                })
        elif mesh_type == 'neurodynamics':
            self.sim_name = f"{self.ground_truth}_neurodynamics_{self.adaptation_response_constant}-{self.adaptation_time_constant}_{self.date_of_sim.day}-{self.date_of_sim.month}-{self.date_of_sim.year}_{self.run_number}"
            self.mesh_param_1 = self.adaptation_response_constant
            self.mesh_param_2 = self.adaptation_time_constant
            self.sim_spec_dict.update({
                'mesh_param_1': self.mesh_param_1,
                'mesh_param_2': self.mesh_param_2
            })
        elif mesh_type == 'stochastic':
            self.sim_name = f"{self.ground_truth}_stochastic_{self.OU_mu}-{self.OU_sigma}_{self.date_of_sim.day}-{self.date_of_sim.month}-{self.date_of_sim.year}_{self.run_number}"
            self.mesh_param_1 = self.OU_mu
            self.mesh_param_2 = self.OU_sigma
            self.sim_spec_dict.update({
                'mesh_param_1': self.mesh_param_1,
                'mesh_param_2': self.mesh_param_2
            })
        

        if self.encode_path != None:
           self.encode_path = '/'.join([self.encode_path, 'GT_' + self.ground_truth, self.sim_name])

        # Ensure a 2d mesh has been specified
        if self.num_subsets != 2:
            raise Exception(f'Must pass 2 parameter subsets, but {self.num_subsets} were specified.')

        # Determine meshfig dimensions in advance
        self.mesh_fig_dimensions =(min(4, len(self.subset_1)), min(4, len(self.subset_2)))

        # Simulating new data
        if self.decode == False:

            forcing_amplitude_mesh, theta_rhythm, theta_subset_key, interference_subset_key = self._init_forcing()

            t0 = time.time()
            spike_stack = HPC_phi_6_sim_forcing_subspace_ARC(
                forcing_amplitude_mesh,
                self.theta_frequency,
                self.interference_frequency,
                self.__dict__[theta_subset_key].size,
                self.__dict__[interference_subset_key].size,
                self.simulation_duration,
                self.dt,
                self.neuron_threshold,
                self.neuron_time_constant,
                self.rest_V,
                self.spike_V,
                self.refractory_period_duration,
                self.adaptation_response_constant,
                self.adaptation_decay_constant,
                self.adaptation_time_constant,
                self.OU_sigma,
                self.OU_mu,
                self.OU_time_constant
            )
            t1 = time.time()

            # Summarize simulation output
            print(f"Simulation output construct sizes:\n     - Spike times matrix: {spike_stack.nbytes/1000000:.2f} MB")
            print(f"Time to simulate: {(t1 - t0):.2f} s\n")
            print(f'Spike times matrix shape: {spike_stack.shape}\n')

            # Encode simulation if requested
            if self.encode == True:

                handle_sim_encoding(self.sim_spec_dict, self.encode_path, spike_stack=spike_stack, encode=True, decode=False)

        # Decoding previously simulated data
        elif self.decode == True:

            # Get simulation data and update the class attributes to reflect state at time of original simulation
            spike_stack, decoded_sim_specifications = handle_sim_encoding(self.sim_spec_dict, self.decode_path, encode=False, decode=True)
            self.__dict__.update(decoded_sim_specifications)

            # Reconstruct theta rhythm using parameters at original time of simulation
            theta_rhythm = self.theta_amplitude*np.sin(2*np.pi*self.theta_frequency/1000*self.dt*np.arange(self.num_timesteps))

        # Construct the block of mean phase on each cycle (0 for cycles with no spikes)
        cycle_phi_block, cycle_boundary_indices, cycle_index_block = HPC_phi_phase_analysis(
            spike_stack,
            hilbert(theta_rhythm),
            central_tendency_technique=self.central_tendency_technique
        )

        # Compute PRQ
        PRQ_array = HPC_phi_compute_PRQ(
            self.__dict__,
            cycle_phi_block,
            dbscan_scrub_data=self.dbscan_scrub_data,
            dbscan_epsilon=self.dbscan_epsilon,
            dbscan_min_samples=self.dbscan_min_samples,
            dbscan_fig_suppress=self.dbscan_fig_suppress,
            dbscan_accumulate_data_fig=self.dbscan_accumulate_data_fig
        )

        # Flip and transpose for ascending parameters along x and y axis
        PRQ_array = np.flipud(np.transpose(PRQ_array))

        print(f"PRQ array :\n{np.around(PRQ_array, decimals=3)}")

        # Compute and print the error over the mesh
        if self.compute_PRQ_error == True:
            PRQ_error = HPC_phi_compute_PRQ_error(
                PRQ_array,
                self.theta_frequency,
                self.interference_frequency,
                self.subset_1[0],
                self.subset_1[-1],
                self.subset_2[0],
                self.subset_2[-1]
                )
            print(f'Mesh error: {PRQ_error}')

        # Generate and save contour plot
        PRQ_contour(
            PRQ_array,
            self.subset_1,
            self.subset_2,
            self.subset_1_name,
            self.subset_2_name,
            self.cmap,
            self.ground_truth,
            mesh_type,
            self.mesh_param_1,
            self.mesh_param_2,
            self.run_number,
            figure_path=self.figure_path,
            dbscan_scrub_data=self.dbscan_scrub_data,
            numeric_GT_label=self.numeric_GT_label,
            suppress=self.contour_popup_suppress
            )

@nb.njit
def HPC_phi_6_sim_forcing_subspace_ARC(
    forcing_amplitude_mesh,
    theta_frequency,
    interference_frequency,
    subspace_1_size,
    subspace_2_size,
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
    track_progress=True
    ):
    """
    Simulates a mesh of uncoupled SASLIF neurons receiving different forcing.

    The ARC variant internally generates forcing from a parameter mesh and does not save voltage time series.
    """

    # Initialize integer constants
    num_timesteps = int(simulation_duration/dt)
    refractory_period_length = int(refractory_period_duration/dt)

    # Initialize system variable arrays
    V = np.ones((subspace_1_size, subspace_2_size), dtype=nb.float32)
    V = hadamard(V, rest_V)
    xi = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)
    W = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)

    # Initialize neuron state arrays
    spike_state_matrix = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.bool_)
    refractory_period_matrix = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)
    subthreshold_matrix = V[:, :] < neuron_threshold
    suprathreshold_matrix = ~subthreshold_matrix

    # Initialize output arrays
    stack_height = 10000 # Maximum number of spikes supported per neuron
    spike_stack = np.zeros((subspace_1_size, subspace_2_size, stack_height), dtype=nb.int32) # To contain indices of each spike along axis 2
    stack_topography = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.int64) # 2d array indicating 'height' of each column when visualized from above

    # Initialize equation constants
    adaptation_decay_rate = adaptation_decay_constant/adaptation_time_constant
    weiner_coefficient = nb.float32(OU_sigma*math.sqrt(dt*OU_time_constant))

    # Initialize mesh state
    spike_state_matrix = condit_assign_bool(spike_state_matrix, suprathreshold_matrix, True)
    spike_state_matrix = condit_assign_bool(spike_state_matrix, subthreshold_matrix, False)
    refractory_period_matrix = condit_add(refractory_period_matrix, spike_state_matrix & (refractory_period_matrix <= refractory_period_length), 1)
    refractory_period_matrix = condit_assign_num(refractory_period_matrix, refractory_period_matrix > refractory_period_length, 0)

    forcing_amplitude_mesh = forcing_amplitude_mesh.astype(nb.float32) # For compatibility with vectorized functions
    V_dummy = V # Redundancy to avoid access/update conflicts

    print('\n')
    for step in range(num_timesteps):

        # Generate forcing mesh from parameters
        theta = hadamard(forcing_amplitude_mesh[:, :, 0], nb.float32(np.sin(2*np.pi*theta_frequency/1000*dt*step)))
        interference = hadamard(forcing_amplitude_mesh[:, :, 1], nb.float32(np.sin(2*np.pi*interference_frequency/1000*dt*step)))
        forcing = scalar_add(theta, interference)

        # Generate stochastic element
        random_numbers = np.random.normal(loc=OU_mu, scale=1.0, size=(subspace_1_size, subspace_2_size)).astype(nb.float32)

        # Update adaptation depending on neuron state (subthreshold/suprathreshold)
        W = condit_add(W, spike_state_matrix, nb.float32(dt*adaptation_response_constant))
        W = condit_add(W, True, nb.float32(-dt*adaptation_decay_rate)*W)

        # Update Ornstein-Uhlenbeck noise contribution
        xi = ornstein_uhlenbeck(xi, OU_mu, OU_time_constant, dt, weiner_coefficient, random_numbers[:, :])

        # Conditionally integrate voltage
        V[:, :] = subthresh_SASLIF_integrate(V_dummy[:, :], ~spike_state_matrix, dt, rest_V, forcing[:, :], W[:, :], neuron_time_constant, xi[:, :])
        V[:, :] = condit_assign_num(V_dummy[: , :], spike_state_matrix & (refractory_period_matrix <= refractory_period_length), spike_V)
        V[:, :] = condit_assign_num(V_dummy[: , :], spike_state_matrix & (refractory_period_matrix > refractory_period_length), rest_V)

        # Reset refractory period wherever it exceeds the absolute duration
        refractory_period_matrix = condit_assign_num(refractory_period_matrix, (refractory_period_matrix > refractory_period_length), 0)

        # Update neuron state arrays
        subthreshold_matrix = V[:, :] < neuron_threshold
        suprathreshold_matrix = ~subthreshold_matrix
        spike_state_matrix = condit_assign_bool(spike_state_matrix, suprathreshold_matrix, True)
        spike_state_matrix = condit_assign_bool(spike_state_matrix, subthreshold_matrix, False)

        # Log all spikes for this index
        x, y = np.where((spike_state_matrix == True) & (refractory_period_matrix == 0)) # Coordinates where neurons were newly assigned to the spike state
        z = np.ones((x.shape[0],), dtype=nb.int64) # Initialize a vector for the index of the top of each column
        for i in range(x.shape[0]):
            z[i] = stack_topography_flatten(z[i], stack_topography[x[i], y[i]]) # Take heights wherever a spike was recorded
            spike_stack[x[i], y[i], z[i]] = step # Assign the current index to each coordinate in the stack
            stack_topography[x[i], y[i]] = integer_add(stack_topography[x[i], y[i]], 1) # Increment the topography wherever a spike was added

        # Increment refractory period state for neurons which recently fired
        refractory_period_matrix = condit_add(refractory_period_matrix, spike_state_matrix & (refractory_period_matrix <= refractory_period_length), 1)
        V_dummy = V # Update redundancy

        # Numba nopython JIT friendly progress bar
        if track_progress == True:
            if step % 100000 == 0:
                print ("\033[A                             \033[A")
                percent = str(int(step/num_timesteps*100)) + '.' + str(int((step/num_timesteps*100) % 1))
                print('Simulation progress: ', percent, '%')
            elif step == num_timesteps - 1:
                print ("\033[A                             \033[A")
                print('Simulation progress: 100.0% - COMPLETE')

    return spike_stack

# Non-compiled pipeline functions
def HPC_phi_phase_analysis(spike_stack, analytic_signal, central_tendency_technique='mean'):
    """
    Return a block of mean phase on each cycle for each neuron.

    spike_stack -- 3d array where axes 0 and 1 give coordinates for a neuron and axis 2 contains int
                   indices of spikes on time series.
    analytic_signal -- complex 1d array of Hilbert-transformed theta signal.
    """

    cycle_index_block = False

    # Preprocess the analytic signal
    theta_phase_time_series = np.arctan2(analytic_signal.imag, analytic_signal.real) # Extract phase
    theta_phase_time_series[theta_phase_time_series < 0] += 2*np.pi # Shift from interval [-pi, pi) to [0, 2pi)
    theta_phase_time_series += -3*np.pi/2 # Offset of 3pi/2 shifts phase of 0 to origin for sin function
    theta_phase_time_series[theta_phase_time_series < 0] += 2*np.pi # readjust interval to [0, 2pi)
    theta_omega_time_series = np.diff(theta_phase_time_series) # Monotonically increasing for linear oscillators except where remainder mod 2pi returns to zero
    cycle_boundary_indices = np.where(theta_omega_time_series < 0)[0] # Hence, negative frequencies readily delineate cycles

    # Identify index on axis 2 where the data ends
    for slice_index in range(spike_stack.shape[2]):
        if np.all(spike_stack[:, :, slice_index] == 0):
            truncation_index = slice_index
            break
        elif slice_index == (spike_stack.shape[2] - 1):
            truncation_index = -1

    # Mask zeros to exclude from calculations, truncate the stack where the data ends
    spike_stack = np.ma.masked_equal(spike_stack[:, :, :truncation_index], 0)

    # Cycle phi block generation algorithm; spikeless cycles have mean phase of zero
    t0 = time.time()
    num_cycles = len(cycle_boundary_indices) + 1
    for i in range(num_cycles):
        cycle_phi_slice = np.zeros((spike_stack.shape[0], spike_stack.shape[1], 1)) # Initialize a slice to populate with means
        for row_index in range(spike_stack.shape[0]):
            for column_index in range(spike_stack.shape[1]):
                stack_column = spike_stack[row_index, column_index, :] # The current stack column (spike indices for neuron at coorindate (row_index, column_index)).
                if i == 0:
                    cycle_phi = theta_phase_time_series[stack_column[(stack_column < cycle_boundary_indices[i]) & (stack_column.mask == False)]]
                    if cycle_phi.size != 0:
                        cycle_phi_slice[row_index, column_index, 0] = central_tendency_selector(cycle_phi, central_tendency_technique)
                elif i < len(cycle_boundary_indices):
                    cycle_phi = theta_phase_time_series[stack_column[(stack_column >= cycle_boundary_indices[i - 1]) & (stack_column < cycle_boundary_indices[i]) & (stack_column.mask == False)]]
                    if cycle_phi.size != 0:
                        cycle_phi_slice[row_index, column_index, 0] = central_tendency_selector(cycle_phi, central_tendency_technique)
                else:
                    cycle_phi = theta_phase_time_series[stack_column[(stack_column >= cycle_boundary_indices[i - 1]) & (stack_column.mask == False)]]
                    if cycle_phi.size != 0:
                        cycle_phi_slice[row_index, column_index, 0] = central_tendency_selector(cycle_phi, central_tendency_technique)
        if i == 0:
            cycle_phi_block = cycle_phi_slice
        else:
            cycle_phi_block = np.concatenate((cycle_phi_block, cycle_phi_slice), axis=2)

        # Numba nopython JIT friendly progress bar
        # if i % 10 == 0:
        #     #print ("\033[A                             \033[A")
        #     percent = str(int(i/num_cycles*100)) + '.' + str(int((i/num_cycles*100) % 1))
        #     print(f'Central tendency computation progress: {percent} %', end="\r")
        # elif i == num_cycles - 1:
        #     #print ("\033[A                             \033[A")
        #     print('Central tendency computation progress: 100.0% - COMPLETE')
    t1 = time.time()
    print(f"Time to compute cycle phase central tendencies: {(t1 - t0):.2f} s\n")

    return cycle_phi_block, cycle_boundary_indices, cycle_index_block
def HPC_phi_compute_PRQ(post_sim_specifications, cycle_phi_block, dbscan_scrub_data=True, dbscan_epsilon=0.1, dbscan_min_samples=10, dbscan_fig_suppress=False, dbscan_accumulate_data_fig=False):
    """
    Return an array of PRQ assessments over a mesh of finite neuron time series.

    cycle_phi_block -- 3d array where axes 0 and 1 give coordinates for a neuron and axis 2 contains some
                       measure of spike phase central tendency over each cycle of theta.
    """
    # Set the number of cycles over which to compute PRQ
    num_cycles = post_sim_specifications['num_cycles']
    if num_cycles <= 0 or num_cycles == 'all':
        num_cycles = cycle_phi_block.shape[2] # For negative values or str literal 'all', compute PRQ over all cycles

    # In case number of cycles is > the number available, set number of cycles to number available
    num_cycles = min(num_cycles, cycle_phi_block.shape[2])
    
    # Compute PRQ using only contiguously defined sequences of phi central tendency 
    cycle_phi_block_slice = cycle_phi_block[:, :, -num_cycles:] # Consider only last n cycles, where n is determined above

    masked_cycle_phi_block_slice = np.ma.masked_equal(cycle_phi_block_slice, 0) # Do not consider zeros in computation (default value for cycles with no spike activity)
    first_order_differences = np.diff(np.flip(masked_cycle_phi_block_slice, axis=2), axis=2) # The backward differences are samples over the time series
    
    PRQ_array = np.empty((cycle_phi_block_slice.shape[0], cycle_phi_block_slice.shape[1])) # Initialize PRQ array

    # Loop over all entries in the PRQ_array
    accumulating_RM_coordinates = np.array([]).reshape((-1, 2))
    accumulating_RM_cluster_colors = np.array([], dtype=object)
    for i in range(PRQ_array.shape[0]): 
        for j in range(PRQ_array.shape[1]):

            defined_differences = first_order_differences[i, j, :].compressed() # remove masked elements
            
            # Use dbscan algorithm to remove outliers and/or plot return maps that indicate outliers
            if dbscan_fig_suppress == True or dbscan_scrub_data == False: # If no outlier removal or corresponding plots
                RM_coordinates = False # Skip computing RMQ coordinates
            else:
                phase_sequence_slices = np.ma.flatnotmasked_contiguous(masked_cycle_phi_block_slice[i, j, :]) # Sequences of slices for contiguously defined sequences of phi central tendency

                for k, slice in enumerate(phase_sequence_slices): 
                    phase_sequence = masked_cycle_phi_block_slice[i, j, :][slice]
                    if k == 0:
                        RM_coordinates = np.concatenate((np.reshape(np.delete(phase_sequence, -1), (1, -1)), np.reshape(np.delete(phase_sequence, 0), (1, -1))), axis=0)
                    else:
                        next_segment = np.concatenate((np.reshape(np.delete(phase_sequence, -1), (1, -1)), np.reshape(np.delete(phase_sequence, 0), (1, -1))), axis=0)
                        RM_coordinates = np.concatenate((RM_coordinates, next_segment), axis=1)
            
            # Scrub data if requested; removes cluster with the highest absolute mean
            if dbscan_scrub_data == True:
                if dbscan_accumulate_data_fig == True:
                    if i == (PRQ_array.shape[0] - 1) and (j == PRQ_array.shape[1] - 1):
                        scrubbed_data, accumulating_RM_coordinates, accumulating_RM_cluster_colors = RM_dbscan_scrub(defined_differences, RM_coordinates, epsilon=0.1, min_samples=10, suppress=dbscan_fig_suppress, dbscan_accumulate_data_fig=True, accumulating_RM_coordinates=accumulating_RM_coordinates, accumulating_RM_cluster_colors=accumulating_RM_cluster_colors, done_accumulating=True)
                    else:
                        scrubbed_data, accumulating_RM_coordinates, accumulating_RM_cluster_colors = RM_dbscan_scrub(defined_differences, RM_coordinates, epsilon=0.1, min_samples=10, suppress=dbscan_fig_suppress, dbscan_accumulate_data_fig=True, accumulating_RM_coordinates=accumulating_RM_coordinates, accumulating_RM_cluster_colors=accumulating_RM_cluster_colors, done_accumulating=False)
                else:
                    scrubbed_data = RM_dbscan_scrub(defined_differences, RM_coordinates, epsilon=0.1, min_samples=10, suppress=dbscan_fig_suppress)
            else:
                scrubbed_data = defined_differences # If no scrubbing requested, pass unmodified difference sequence

            PRQ_array[i, j] = np.mean(scrubbed_data) # The central tendency of the difference sequence is PRQ

    return PRQ_array
def HPC_phi_compute_PRQ_error(PRQ_array, theta_frequency, interference_frequency, theta_amp_min, theta_amp_max, interference_amp_min, interference_amp_max, epsilon=0.2):

    PRQ_array_error = 0
    GT_array = HPC_phi_numerical_GT_probe(
        10000,
        0.01,
        PRQ_array.shape[0],
        theta_frequency,
        interference_frequency,
        epsilon,
        [theta_amp_min, theta_amp_max, PRQ_array.shape[0]],
        [interference_amp_min, interference_amp_max, PRQ_array.shape[0]]
    )

    for i in range(PRQ_array.shape[0]):
        for j in range(PRQ_array.shape[0]):
            GT = GT_array[i, j]
            PRQ = PRQ_array[i, j]
            if GT != 0:
                if GT*PRQ < 0:
                    PRQ_array_error += abs(PRQ)
            else:
                PRQ_array_error += abs(PRQ)

    return PRQ_array_error
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

# Plotting functions
def PRQ_contour(PRQ_array, subset_1, subset_2, subset_1_name, subset_2_name, cmap, ground_truth, mesh_type, mesh_param_1, mesh_param_2, run_number, dbscan_scrub_data=False, numeric_GT_label=False, suppress=False, figure_path=False, print_error=True):

    PRQ_array = np.flipud(PRQ_array)

    if ground_truth == 'recession':
        markers = np.transpose(np.flipud(np.array([-1, -1, -1, -1, -1, -2, -2, -2, -2, -2,
                                                   -1, -1, -1, -1, -2, -2, -2, -2, -2, -2,
                                                   -1, -1, -1, -2, -2, -2, -2, -2, -2, -2,
                                                   -1, -1, -1, -2, -2, -2, -2, -2, -2, -1,
                                                   -1, -1, -2, -2, -2, -2, -2, -1, -1, -1,
                                                   -1, -2, -2, -2, -2, -2, -1, -1, -1, -1,
                                                   -2, -2, -2, -2, -2, -1, -1, -1, -1, 0,
                                                   -2, -2, -2, -2, -1, -1, -1, 0, 0, 0,
                                                   -2, -2, -2, -1, -1, -1, 0, 0, 0, 0,
                                                   -2, -2, -1, -1, 0, 0, 0, 0, 0, 0]).reshape((10, 10))))
    elif ground_truth == 'locking':
        markers = np.zeros((10, 10))
    elif ground_truth == 'precession':
        markers = np.transpose(np.flipud(np.array([-1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 
                                                   -1, -1, -1, 0, 0, 0, 1, 1, 1, 1,
                                                   -1, -1, -1, 0, 0, 1, 1, 1, 1, 1,
                                                   -1, -1, 0, 0, 1, 1, 1, 1, 1, 1,
                                                   -1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                                   -1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                   0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                                                   0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                                   1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                                                   1, 1, 1, 1, 0, 0, 0, 0, 0, 0]).reshape((10, 10))))
    else:
        raise ValueError(f'Argument not recognized, ground_truth: {ground_truth}')

    subset_1_marker_increment = (subset_1[1] - subset_1[0])/11
    subset_2_marker_increment = (subset_2[1] - subset_2[0])/11
    marker_x = np.linspace(subset_1[0] + subset_1_marker_increment, subset_1[-1] - subset_1_marker_increment, num=10)
    marker_y = np.linspace(subset_2[0] + subset_2_marker_increment, subset_2[-1] - subset_2_marker_increment, num=10)

    basic_cols = ['#FF0000', '#000000', '#0000FF']
    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

    plt.close()
    fig, ax = plt.subplots()

    shading = ax.contourf(subset_1, subset_2, PRQ_array, cmap=inferno, vmin=-0.5, vmax=0.5, alpha=0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contour(subset_1, subset_2, PRQ_array, [0], colors='k', linewidths=0.9, linestyle='dashed', alpha=0.75)
    if numeric_GT_label == True:
        for n, i in enumerate(marker_x):
            for m, j in enumerate(marker_y):
                ax.text(i, j, str(markers[n, m]), color='k', fontsize='small', ha='center', va='center')
    plt.colorbar(
        ScalarMappable(norm=shading.norm, cmap=shading.cmap),
        ticks=np.linspace(-0.5, 0.5, 5)
        )

    plt.xlabel(subset_1_name.replace("_", " ").capitalize())
    plt.ylabel(subset_2_name.replace("_", " ").capitalize())

    figure_object = plt.gcf()

    if suppress == False:
        plt.show()
    if figure_path != False:
        if dbscan_scrub_data == True:
            if numeric_GT_error == True:
                figure_object.savefig(Path('/'.join([figure_path, 'contourPlots', f'GT_{ground_truth}_{mesh_type}_{mesh_param_1:.1f}-{mesh_param_2:.1f}_dbscan_labels_{datetime.now().day}-{datetime.now().month}-{datetime.now().year}_{int(run_number)}.svg'])), dpi=400, bbox_inches='tight')
            else:
                figure_object.savefig(Path('/'.join([figure_path, 'contourPlots', f'GT_{ground_truth}_{mesh_type}_{mesh_param_1:.1f}-{mesh_param_2:.1f}_dbscan_{datetime.now().day}-{datetime.now().month}-{datetime.now().year}_{int(run_number)}.svg'])), dpi=400, bbox_inches='tight')
        else:
            if numeric_GT_error == True:
                figure_object.savefig(Path('/'.join([figure_path, 'contourPlots', f'GT_{ground_truth}_{mesh_type}_{mesh_param_1:.1f}-{mesh_param_2:.1f}_labels_{datetime.now().day}-{datetime.now().month}-{datetime.now().year}_{int(run_number)}.svg'])), dpi=400, bbox_inches='tight')
            else:
                figure_object.savefig(Path('/'.join([figure_path, 'contourPlots', f'GT_{ground_truth}_{mesh_type}_{mesh_param_1:.1f}-{mesh_param_2:.1f}_{datetime.now().day}-{datetime.now().month}-{datetime.now().year}_{int(run_number)}.svg'])), dpi=400, bbox_inches='tight')

# Helper functions
def get_centrally_symmetric_indices(subset_1, subset_2, subset_1_num, subset_2_num):

    threshold_of_symmetry_1 = (subset_1[-1] - subset_1[0])/2 + subset_1[0]
    threshold_of_symmetry_2 = (subset_2[-1] - subset_2[0])/2 + subset_2[0]

    symmetric_array_1 = np.linspace(subset_1[0], subset_1[-1], subset_1_num)
    symmetric_array_2 = np.linspace(subset_2[0], subset_2[-1], subset_2_num)

    for index in range(symmetric_array_1.shape[0]):
        if symmetric_array_1[index] < threshold_of_symmetry_1:
            symmetric_array_1[index] = math.floor(symmetric_array_1[index])
        elif symmetric_array_1[index] > threshold_of_symmetry_1:
            symmetric_array_1[index] = math.ceil(symmetric_array_1[index])
        else:
            symmetric_array_1[index] = int(symmetric_array_1[index])
    for index in range(symmetric_array_2.shape[0]):
        if symmetric_array_2[index] < threshold_of_symmetry_2:
            symmetric_array_2[index] = math.floor(symmetric_array_2[index])
        elif symmetric_array_2[index] > threshold_of_symmetry_2:
            symmetric_array_2[index] = math.ceil(symmetric_array_2[index])
        else:
            symmetric_array_2[index] = int(symmetric_array_2[index])

    nearest_indices_1 = []
    for point in symmetric_array_1:
        nearest_indices_1.append((np.abs(subset_1 - point)).argmin())

    nearest_indices_2 = []
    for point in symmetric_array_2:
        nearest_indices_2.append((np.abs(subset_2 - point)).argmin())

    return nearest_indices_1, nearest_indices_2
def handle_sim_encoding(sim_specifications, filepath, V=False, spike_stack=False, encode=False, decode=False):

    if encode == True:
        input_spec_keys = np.array([key for key in sim_specifications.keys()])
        input_spec_values = np.array([value for value in list(sim_specifications.values())[:len(input_spec_keys)]])

        np.savez_compressed(Path(filepath), input_spec_keys=input_spec_keys, input_spec_values=input_spec_values, V=V, spike_stack=spike_stack)

    elif decode == True:
        decoded_message = np.load(Path(filepath))

        spike_stack = decoded_message['spike_stack']

        sim_specifications = {key: value for key, value in np.stack((decoded_message['input_spec_keys'], decoded_message['input_spec_values']), axis=-1)}

        for key in sim_specifications.keys():
            try:
                sim_specifications[key] = float(sim_specifications[key])
            except ValueError:
                continue

        del decoded_message

        return spike_stack, sim_specifications
def central_tendency_selector(data, central_tendency_technique):
    if central_tendency_technique == 'mean':
        CT = np.mean(data)
    elif central_tendency_technique == 'median':
        CT = np.median(data)
    elif central_tendency_technique == 'KDE':
        if data.size == 1:
            CT = data[0]
        else:
            grid = GridSearchCV(
                KernelDensity(kernel='gaussian'),
                {'bandwidth': 10**np.linspace(-1, 1, 100)},
                cv=LeaveOneOut()
            )
            grid.fit(data[:, None])
            optimal_bandwidth = grid.best_params_['bandwidth']

            KDE = KernelDensity(bandwidth=optimal_bandwidth, kernel='gaussian')
            KDE.fit(data[:, None])
            KDE_interval = np.linspace(0, 2*np.pi, num=10000)

            CT = KDE_interval[np.argmax(np.exp(KDE.score_samples(KDE_interval[:, None])))]
    else:
        CT = np.mean(data)
    return CT
def RM_dbscan_scrub(RM_data, RM_coordinates, epsilon=0.5, min_samples=3, suppress=False, dbscan_accumulate_data_fig=False, accumulating_RM_coordinates=None, accumulating_RM_cluster_colors=None, done_accumulating=False):

    if RM_data.ndim == 1:
        RM_data = RM_data.reshape((-1, 1))
 
    db = DBSCAN(eps=0.5, min_samples=3).fit(RM_data)
    labels = np.array(db.labels_)
    num_clusters = np.unique(labels[labels != -1]).shape[0]

    if num_clusters > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cluster_dict = {}
            for i, label in enumerate(np.unique(labels[labels != -1])):
                cluster_array = np.array(list(map(lambda x, y: RM_data[y, 0] if x == i else 0, labels, range(labels.shape[0]))))
                cluster_array = cluster_array[cluster_array != 0]
                cluster_dict.update({
                    f'cluster_{i}': cluster_array
                })

        cluster_abs_means = np.empty((np.unique(labels[labels != -1]).shape[0],))
        for i, key in enumerate(cluster_dict.keys()):
            cluster_abs_means[i] = np.abs(np.mean(cluster_dict[key]))
        
        outlier_cluster = np.argmax(cluster_abs_means)

        scrubbed_data = np.array([])
        for i, key in enumerate(cluster_dict.keys()):
            if i != outlier_cluster:
                scrubbed_data = np.concatenate((scrubbed_data, cluster_dict[key]), axis=0)
    else:
        scrubbed_data = RM_data

    RM_coordinates = np.flip(np.transpose(RM_coordinates), axis=0)

    if num_clusters > 1:
        colors = list(map(lambda x: 'r' if x == outlier_cluster else 'k', labels))
    else: 
        colors = ['k' for i in range(labels.size)]

    if not suppress and not dbscan_accumulate_data_fig:

        plt.close()
        fig, ax = plt.subplots()

        fig.patch.set_alpha(0.0)
        ax.spines['right'].set_visible(False) 
        ax.spines['top'].set_visible(False) 

        ax.scatter(RM_coordinates[:, 0], RM_coordinates[:, 1], color=colors, s=15)

        ax.set_xlabel('$\phi_{i-1}$')
        ax.set_ylabel('$\phi_i$')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        plt.show()

    elif not suppress and dbscan_accumulate_data_fig and done_accumulating == True:

        RM_coordinates = np.concatenate((accumulating_RM_coordinates, RM_coordinates), axis=0)

        colors = np.concatenate((accumulating_RM_cluster_colors, np.array(colors)))

        plt.close()
        fig, ax = plt.subplots()

        fig.patch.set_alpha(0.0)
        ax.spines['right'].set_visible(False) 
        ax.spines['top'].set_visible(False) 

        ax.scatter(RM_coordinates[:, 0], RM_coordinates[:, 1], color=colors, s=15)

        ax.set_xlabel('$\phi_{i-1}$')
        ax.set_ylabel('$\phi_i$')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        plt.show()

    if dbscan_accumulate_data_fig == False:
        return scrubbed_data
    else:
        return scrubbed_data, np.concatenate((accumulating_RM_coordinates, RM_coordinates), axis=0), np.concatenate((accumulating_RM_cluster_colors, np.array(colors)))

def main():

    SIM_SPECS = {
        # Time parameters
        'simulation_duration': 10000, # ms
        'dt': nb.float32(0.01), # ms

        # Neuron biophysical parameters
        'neuron_threshold': -40, # mV
        'neuron_time_constant': 10, # ms
        'rest_V': -75, # mV
        'spike_V': 50, # mV
        'refractory_period_duration': 2, # ms

        # Spike frequency adaptation biophysical parameters
        'adaptation_response_constant': 10, # mV
        'adaptation_decay_constant': 1, # mV
        'adaptation_time_constant': 20, # ms

        # Ornstein-Uhlenbeck process parameters
        'OU_sigma': 80,
        'OU_mu': nb.float32(0.8),
        'OU_time_constant': nb.int32(50),

        # Dual oscillator input parameters
        'theta_amplitude': 35, # mV
        'interference_amplitude': 35, # mV

        'theta_frequency': 10, # Hz
        'interference_frequency': 11 # Hz
    }
    ANALYSIS_SPECS = {
        # Global tracking parameter
        'run_number': 0,

        # PRQ analysis parameters
        'num_cycles': -1,
        'central_tendency_technique': 'mean',
        'dbscan_scrub_data': True,
        'dbscan_epsilon': 0.3,
        'dbscan_min_samples': 10,
        'dbscan_fig_suppress': False,
        'dbscan_accumulate_data_fig': True,
        'numeric_GT_label': False,
        'compute_PRQ_error': False,

        # Just have to go to directory above GT differentiation
        'encode': False,
        'encode_path': '/'.join([os.getcwd(), 'serializedSims']), 

        # Have to point to file directly
        'decode': False,
        'decode_path': False,#'/'.join([os.getcwd(), 'serializedSims', 'GT_precession', 'precession_basemodel_---_1-2-2022_0.npz']),

        # Just have to go to directory above figure type
        'figure_path': False,# '/'.join([os.getcwd(), 'modelFigures', 'GT_precession']),

        'contour_popup_suppress': False,
    }

    single_sim(
        SIM_SPECS,
        ANALYSIS_SPECS,
        'theta_amplitude',
        'interference_amplitude',
        [20, 50, 10],
        [20, 50, 10],
        ARC=False,
        init_nontest=False
        )
    
    # neurodynamics_nested_mesh_sim(
    #     SIM_SPECS,
    #     ANALYSIS_SPECS,
    #     [1, 1000],
    #     [1, 100],
    #     4,
    #     ARC=False,
    #     init_nontest=False
    #     )
    
    # stochastic_nested_mesh_sim(
        # SIM_SPECS,
        # ANALYSIS_SPECS,
        # [0.1, 0.8],
        # [10, 200],
        # 4,
        # ARC=False,
        # init_nontest=False
    # )

if __name__ == '__main__':
    main()
