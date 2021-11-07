import itertools
import math
import os
import time
import warnings
from datetime import datetime, date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm
try:
    import PySimpleGUI as sg
except ImportError:
    pass

from numba_ufuncs import (scalar_add, condit_add,
                                  condit_assign_bool, condit_assign_num,
                                  hadamard, ornstein_uhlenbeck,
                                  subthresh_SASLIF_integrate, integer_add,
                                  stack_topography_flatten)

# Pipeline manager
class PipelineManager():

    def __init__(self, simulation_specifications: dict, analysis_specifications: dict, DOI=True, ARC=False):
        
        self.__dict__.update(simulation_specifications)
        self.__dict__.update(analysis_specifications)

        self.sim_spec_dict = simulation_specifications
        self.num_timesteps = int(self.simulation_duration/self.dt)
        self.num_subsets = 0
        self.DOI = DOI
        self.ARC = ARC
        self.probe_TS = False
        self.probe_RM = False
        self.date_of_sim = datetime.now()

        if self.theta_frequency < self.interference_frequency:
            self.ground_truth = 'precession'
        elif self.theta_frequency > self.interference_frequency:
            self.ground_truth = 'recession'
        else:
            self.ground_truth = 'locking'

        if ARC == True:
            matplotlib.use('Agg')
            self.contour_suppress = True
            self.TM_mesh_suppress = True
            self.RM_mesh_suppress = True
        else:
            sg.theme('Black')
    
    def add_parameter_subset(self, start: float, stop: float, num: int, name: str):
      
        self.__dict__.update({f'subset_{self.num_subsets + 1}': np.linspace(start, stop, num)})
        self.__dict__.update({f'subset_{self.num_subsets + 1}_name': name})
        self.num_subsets += 1
    
    def set_probe_TS(self, coordinate: tuple):
        
        self.probe_TS = True
        self.probe_TS_coordinate = coordinate
        
    def set_probe_RM(self, coordinate: tuple):
        
        self.probe_RM = True
        self.probe_RM_coordinate = coordinate
        
    def initialize_subspace(self, mesh_type: str, reconstruct_from_decode=False):

        if reconstruct_from_decode == True:
            theta_vector = 2*np.pi*self.theta_frequency/1000*self.dt*np.arange(self.num_timesteps)
            theta_rhythm = self.theta_amplitude*np.sin(theta_vector)
            return theta_rhythm


        t_init_start = time.time()
        print('\n')
        mesh_shape = (self.subset_1.size, self.subset_2.size)

        parameter_subspace_mesh = list(itertools.product(*[range(s) for s in mesh_shape]))

        if self.DOI == True:

            theta_vector = 2*np.pi*self.theta_frequency/1000*self.dt*np.arange(self.num_timesteps)
            interference_vector = 2*np.pi*self.interference_frequency/1000*self.dt*np.arange(self.num_timesteps)

            if mesh_type != 'forcing':
                
                theta_rhythm = self.theta_amplitude*np.sin(theta_vector)
                interference_rhythm = self.interference_amplitude*np.sin(interference_vector)
                forcing_function = np.sum((theta_rhythm, interference_rhythm), axis=0)

            else:

                forcing_function_bundle = np.zeros((self.subset_1.size, self.subset_2.size, self.num_timesteps), dtype=np.single)

                for coordinate in enumerate(tqdm(parameter_subspace_mesh, desc='Generating forcing functions'), start=0):

                    theta_rhythm = self.subset_1[coordinate[1][0]]*np.sin(theta_vector)
                    interference_rhythm = self.subset_2[coordinate[1][1]]*np.sin(interference_vector)
                    forcing_function_bundle[coordinate[1][0], coordinate[1][1], :] = np.sum((theta_rhythm, interference_rhythm), axis=0)
        
                t_init_end = time.time() 

                print(f"Time to initialize: {(t_init_end - t_init_start):.2f} s")
        
                return forcing_function_bundle, theta_rhythm

        elif self.DOI == False:
            
            forcing_function = np.ones((self.num_timesteps,))*self.theta_amplitude
            theta_rhythm = forcing_function

        if mesh_type == 'neurodynamics':
    
            neurodynamic_parameter_bundle = np.zeros((self.subset_1.size, self.subset_2.size, 2), dtype=np.single)

            for coordinate in enumerate(tqdm(parameter_subspace_mesh, desc='Generating neurodynamic parameter mesh'), start=0):
                neurodynamic_parameter_bundle[coordinate[1][0], coordinate[1][1], 0] = self.subset_1[coordinate[1][0]] 
                neurodynamic_parameter_bundle[coordinate[1][0], coordinate[1][1], 1] = self.subset_2[coordinate[1][1]]

                t_init_end = time.time() 

                print(f"Time to initialize: {(t_init_end - t_init_start):.2f} s")

            return neurodynamic_parameter_bundle, forcing_function, theta_rhythm

    def execute_pipeline(self, mesh_type: str):
        
        # Initialize simulation file name and encode path
        self.sim_name = f"{self.ground_truth}_{self.subset_1_name}_{self.subset_2_name}_{int(self.subset_1[0])}_{int(self.subset_1[-1])}_{len(self.subset_1)}_{int(self.subset_2[0])}_{int(self.subset_2[-1])}_{len(self.subset_2)}_{self.date_of_sim.day}{self.date_of_sim.month}{self.date_of_sim.year}"
        if self.encode_path != None:
           self.encode_path = '/'.join([self.encode_path, 'GT_' + self.ground_truth, self.sim_name])

        # Ensure a 2d mesh has been specified
        if self.num_subsets != 2:
            raise Exception(f'Must pass 2 parameter subsets, but {self.num_subsets} were specified.')
        
        # Determine meshfig dimensions in advance
        self.mesh_fig_dimensions =(min(4, len(self.subset_1)), min(4, len(self.subset_2)))

        # Simulating new data
        if self.decode == False:

            # Simulating on ARC
            if self.ARC == True:

                # Forcing parameter subspace
                if mesh_type == 'forcing':

                    # Infer subspace sizes from class attribute dictionary
                    with warnings.catch_warnings():
                        warnings.simplefilter(action='ignore', category=FutureWarning)  
                        theta_subset_key = '_'.join(list(self.__dict__.keys())[list(self.__dict__.values()).index('theta_amplitude')].split('_')[:2])
                        interference_subset_key = '_'.join(list(self.__dict__.keys())[list(self.__dict__.values()).index('theta_amplitude')].split('_')[:2])

                    # Synthetic theta rhythm for phase extraction later
                    theta_rhythm = np.sin(2*np.pi*self.theta_frequency/1000*self.dt*np.arange(self.num_timesteps))
 
                    # Create parameter mesh so simulation can generate forcing current internally
                    forcing_amplitude_mesh = np.transpose(np.array(np.meshgrid(self.__dict__[theta_subset_key], self.__dict__[interference_subset_key])))

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

            # Simulating on PC
            elif self.ARC == False:
                
                # Forcing parameter subspace
                if mesh_type == 'forcing':

                    forcing_function_bundle, theta_rhythm = self.initialize_subspace(mesh_type)

                    t0 = time.time()
                    Vm_t, spike_stack = HPC_phi_6_sim_forcing_subspace(
                        forcing_function_bundle,
                        self.subset_1.size,
                        self.subset_2.size,
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

                # Adaptation parameter subspace
                if mesh_type == 'neurodynamics':

                    neurodynamic_parameter_bundle, forcing_function, theta_rhythm = self.initialize_subspace(mesh_type)

                    t0 = time.time()
                    Vm_t, spike_indices = HPC_phi_6_sim_neurodynamic_subspace(
                        forcing_function,
                        neurodynamic_parameter_bundle,
                        self.subset_1.size,
                        self.subset_2.size,
                        self.simulation_duration,
                        self.dt,
                        self.neuron_threshold,
                        self.neuron_time_constant,
                        self.rest_V,
                        self.spike_V,
                        self.refractory_period_duration,
                        self.OU_sigma,
                        self.OU_mu,
                        self.OU_time_constant
                    )
                    t1 = time.time()

                # Summarize simulation output
                print(f"Simulation output construct sizes:\n     - V block: {Vm_t.nbytes/1000000:.2f} MB\n     - Spike times matrix: {spike_indices.nbytes/1000000:.2f} MB")
                print(f"Time to simulate: {(t1 - t0):.2f} s\n")
                print(f'Spike times matrix shape: {spike_indices.shape}\n Spike times matrix sum: {np.sum(spike_indices)}')

                # Plot sample output if requested
                if self.probe_TS == True:

                    probe_time_series(Vm_t, self.simulation_duration, self.dt, self.probe_TS_coordinate, timestamping_block=spike_indices)

                # Encode simulation output if requested
                if self.encode == True:

                    # If on PC, popup to select encode path
                    self.encode_path = sg.popup_get_folder('Select location to save encoded simulation')
                    self.encode_path = Path('/'.join([self.encode_path, self.sim_name]))

                    handle_sim_encoding(self.sim_spec_dict, self.encode_path, V=Vm_t, spike_stack=spike_stack, encode=True, decode=False)
       
       # Decoding previously simulated data
        elif self.decode == True:

            # If on PC, popup to select encode path
            if self.ARC == False:
                self.decode_path = sg.popup_get_file('Select simulation to decode')
                
            # Get simulation data and update the class attributes to reflect state at time of original simulation
            Vm_t, spike_stack, decoded_sim_specifications = handle_sim_encoding(self.sim_spec_dict, self.decode_path, encode=False, decode=True)
            self.__dict__.update(decoded_sim_specifications)

            # Reconstruct theta rhythm using parameters at original time of simulation
            theta_rhythm = self.initialize_subspace(mesh_type, reconstruct_from_decode=True)

        # Analysis pipeline (contingent on there having been a dual-oscillator input (DOI))
        if self.DOI == True:

            # Construct the block of mean phase on each cycle (0 for cycles with no spikes)
            cycle_phi_block, cycle_boundary_indices = HPC_phi_phase_analysis_prototype(spike_stack, hilbert(theta_rhythm))

            # Plot sample return map if requested
            if self.probe_RM == True:
                probe_return_map(cycle_phi_block, self.probe_RM_coordinate)

            # Compute RMQ 
            RMQ_array = HPC_phi_compute_RMQ(self.__dict__, cycle_phi_block)

            # Flip and transpose for ascending parameters along x and y axis
            if self.ARC == True:
                RMQ_array = np.flipud(np.transpose(RMQ_array))

            # Vm_t has value false if absent from decoded file, attempt a reformat
            elif self.ARC == False:
                try:
                    Vm_t = np.flipud(np.transpose(Vm_t, (1, 0, 2)))
                except ValueError:
                    Vm_t = Vm_t 
                RMQ_array = np.flipud(np.transpose(RMQ_array))

            print(f"RMQ array :\n{np.around(RMQ_array, decimals=3)}")

            # Generate and save figures
            RMQ_meshfig(
                RMQ_array, 
                self.subset_1, 
                self.subset_2, 
                self.subset_1_name, 
                self.subset_2_name, 
                figure_path=self.figure_path, 
                suppress=self.contour_suppress
                )
            RM_meshfig(
                cycle_phi_block, 
                self.subset_1, 
                self.subset_2, 
                self.subset_1_name, 
                self.subset_2_name, 
                self.mesh_fig_dimensions[0], 
                self.mesh_fig_dimensions[1], 
                suppress=self.RM_mesh_suppress, 
                figure_path=self.figure_path
                )
            if self.ARC == False: 
                TS_meshfig(
                    Vm_t, 
                    theta_rhythm, 
                    cycle_boundary_indices, 
                    self.simulation_duration,
                    self.dt, 
                    self.subset_1, 
                    self.subset_2, 
                    self.subset_1_name, 
                    self.subset_2_name, 
                    self.mesh_fig_dimensions[0], 
                    self.mesh_fig_dimensions[1], 
                    suppress=self.TS_mesh_suppress, 
                    figure_path=self.figure_path
                    )
        
        # No analysis, just plot output
        elif self.DOI == False and self.ARC == False:

            # Generate dummy output data
            RMQ_array = np.empty((self.subset_1.size, self.subset_2.size))
            RMQ_array[:] = np.nan
            cycle_boundary_indices = np.zeros((1,), dtype=int)

            # Flip and transpose for ascending paramters along x and y axis
            Vm_t = np.flipud(np.transpose(Vm_t, (1, 0, 2)))
            RMQ_array = np.flipud(np.transpose(RMQ_array))

            # Generate and save figures
            TS_meshfig(Vm_t, 
                theta_rhythm, 
                cycle_boundary_indices, 
                self.simulation_duration, 
                self.dt, 
                self.subset_1, 
                self.subset_2, 
                self.subset_1_name, 
                self.subset_2_name, 
                self.mesh_fig_dimensions[0], 
                self.mesh_fig_dimensions[1], 
                suppress=self.TS_mesh_suppress, 
                figure_path=self.figure_path
            )

# Compiled pipeline functions
@nb.njit
def HPC_phi_6_sim_forcing_subspace(
    forcing,
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
    
    Unlike the ARC variant, forcing is passed in as a 3d array and voltage time series are saved.
    """

    # Initialize integer constants
    num_timesteps = int(simulation_duration/dt)
    refractory_period_length = int(refractory_period_duration/dt)

    # Initialize system variable arrays
    V = np.ones((subspace_1_size, subspace_2_size, num_timesteps), dtype=nb.float32)
    V = hadamard(V, rest_V)
    xi = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)
    W = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)
     
    # Initialize neuron state arrays
    spike_state_matrix = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.bool_)
    refractory_period_matrix = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)
    subthreshold_matrix = V[:, :, 0] < neuron_threshold
    suprathreshold_matrix = ~subthreshold_matrix
    
    # Initialize output arrays
    stack_height = 5000 # Maximum number of spikes supported per neuron
    spike_stack = np.zeros((subspace_1_size, subspace_2_size, stack_height), dtype=nb.int32) # To contain indices of each spike along axis 2
    stack_topography = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.int64) # 2d array indicating 'height' of each column when visualized from above

    spike_times_matrix = np.zeros((subspace_1_size, subspace_2_size, num_timesteps), dtype=nb.bool_)

    # Generate stochastic element
    random_numbers = np.random.normal(loc=OU_mu, scale=1.0, size=(subspace_1_size, subspace_2_size, num_timesteps)).astype(nb.float32)

    # Initialize equation constants
    adaptation_decay_rate = adaptation_decay_constant/adaptation_time_constant
    weiner_coefficient = nb.float32(OU_sigma*math.sqrt(dt*OU_time_constant))

    # Initialize mesh state
    spike_state_matrix = condit_assign_bool(spike_state_matrix, suprathreshold_matrix, True)
    spike_state_matrix = condit_assign_bool(spike_state_matrix, subthreshold_matrix, False)
    refractory_period_matrix = condit_add(refractory_period_matrix, spike_state_matrix & (refractory_period_matrix <= refractory_period_length), 1)
    refractory_period_matrix = condit_assign_num(refractory_period_matrix, refractory_period_matrix > refractory_period_length, 0)

    print('\n')
    for step in range(1, num_timesteps):

        # Update adaptation depending on neuron state (subthreshold/suprathreshold)
        W = condit_add(W, spike_state_matrix, nb.float32(dt*adaptation_response_constant))
        W = condit_add(W, True, nb.float32(-dt*adaptation_decay_rate)*W)

        # Update Ornstein-Uhlenback noise contribution
        xi = ornstein_uhlenbeck(xi, OU_mu, OU_time_constant, dt, weiner_coefficient, random_numbers[:, :, step])

        # Conditionally integrate voltage
        V[:, :, step] = subthresh_SASLIF_integrate(V[:, :, step - 1], ~spike_state_matrix, dt, rest_V, forcing[:, :, step], W[:, :], neuron_time_constant, xi[:, :])
        V[:, :, step] = condit_assign_num(V[: , :, step], spike_state_matrix & (refractory_period_matrix <= refractory_period_length), spike_V)
        V[:, :, step] = condit_assign_num(V[: , :, step], spike_state_matrix & (refractory_period_matrix > refractory_period_length), rest_V)

        # Reset refractory period wherever it exceeds the absolute duration
        refractory_period_matrix = condit_assign_num(refractory_period_matrix, (refractory_period_matrix > refractory_period_length), 0)

        # Update neuron state arrays
        subthreshold_matrix = V[:, :, step] < neuron_threshold
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

        # Increment refractory state for neurons which recently fired
        refractory_period_matrix = condit_add(refractory_period_matrix, spike_state_matrix & (refractory_period_matrix <= refractory_period_length), 1)

        # Numba nopython JIT friendly progress bar
        if track_progress == True:
            if step % 100000 == 0:
                print ("\033[A                             \033[A")
                percent = str(int(step/num_timesteps*100)) + '.' + str(int((step/num_timesteps*100) % 1))  
                print('Simulation progress: ', percent, '%')
            elif step == num_timesteps - 1:
                print ("\033[A                             \033[A") 
                print('Simulation progress: 100.0% - COMPLETE')

    return V, spike_times_matrix
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
    stack_height = 5000 # Maximum number of spikes supported per neuron
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
@nb.njit
def HPC_phi_6_sim_neurodynamic_subspace(
    forcing,
    neurodynamic_parameter_bundle,
    subspace_1_size, 
    subspace_2_size, 
    simulation_duration, 
    dt,
    neuron_threshold,
    neuron_time_constant,
    rest_V,
    spike_V,
    refractory_period_duration,
    OU_sigma,
    OU_mu,
    OU_time_constant,
    track_progress=True
    ):

    num_timesteps = int(simulation_duration/dt)
    refractory_period_length = int(refractory_period_duration/dt)

    V = np.ones((subspace_1_size, subspace_2_size, num_timesteps), dtype=nb.float32)
    V = hadamard(V, rest_V)
    xi = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)
    W = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)
     
    spike_state_matrix = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.bool_)
    refractory_period_matrix = np.zeros((subspace_1_size, subspace_2_size), dtype=nb.float32)
    spike_times_matrix = np.zeros((subspace_1_size, subspace_2_size, num_timesteps), dtype=nb.float32)

    random_numbers = np.random.normal(loc=OU_mu, scale=1.0, size=(subspace_1_size, subspace_2_size, num_timesteps)).astype(nb.float32)

    weiner_coefficient = nb.float32(OU_sigma*math.sqrt(dt*OU_time_constant))

    subthreshold_matrix = V[:, :, 0] < neuron_threshold
    suprathreshold_matrix = ~subthreshold_matrix

    spike_state_matrix = condit_assign_bool(spike_state_matrix, suprathreshold_matrix, True)
    spike_state_matrix = condit_assign_bool(spike_state_matrix, subthreshold_matrix, False)

    refractory_period_matrix = condit_add(refractory_period_matrix, spike_state_matrix & (refractory_period_matrix <= refractory_period_length), 1)
    refractory_period_matrix = condit_assign_num(refractory_period_matrix, refractory_period_matrix > refractory_period_length, 0)

    print('\n')
    for step in range(1, num_timesteps):

        W = condit_add(W, spike_state_matrix, dt*neurodynamic_parameter_bundle[:, :, 0])
        W = condit_add(W, True, hadamard(W, -dt*neurodynamic_parameter_bundle[:, :, 1]))

        xi = ornstein_uhlenbeck(xi, OU_mu, OU_time_constant, dt, weiner_coefficient, random_numbers[:, :, step])

        V[:, :, step] = subthresh_SASLIF_integrate(V[:, :, step - 1], ~spike_state_matrix, dt, rest_V, forcing[step], W[:, :], neuron_time_constant, xi[:, :])
        V[:, :, step] = condit_assign_num(V[: , :, step], spike_state_matrix & (refractory_period_matrix <= refractory_period_length), spike_V)
        V[:, :, step] = condit_assign_num(V[: , :, step], spike_state_matrix & (refractory_period_matrix > refractory_period_length), rest_V)

        refractory_period_matrix = condit_assign_num(refractory_period_matrix, (refractory_period_matrix > refractory_period_length), 0)

        subthreshold_matrix = V[:, :, step] < neuron_threshold
        suprathreshold_matrix = ~subthreshold_matrix


        spike_state_matrix = condit_assign_bool(spike_state_matrix, suprathreshold_matrix, True)
        spike_state_matrix = condit_assign_bool(spike_state_matrix, subthreshold_matrix, False)

        spike_times_matrix[:, :, step] = condit_assign_num(spike_times_matrix[:, :, step], spike_state_matrix & (refractory_period_matrix == 0), 1)

        refractory_period_matrix = condit_add(refractory_period_matrix, spike_state_matrix & (refractory_period_matrix <= refractory_period_length), 1)

        if track_progress == True:
            if step % 100000 == 0:
                print ("\033[A                             \033[A")
                percent = str(int(step/num_timesteps*100)) + '.' + str(int((step/num_timesteps*100) % 1))  
                print('Simulation progress: ', percent, '%')
            elif step == num_timesteps - 1:
                print ("\033[A                             \033[A") 
                print('Simulation progress: 100.0% - COMPLETE')

    return V, spike_times_matrix

# Non-compiled pipeline functions
def HPC_phi_phase_analysis_prototype(spike_stack, analytic_signal):
    """
    Return a block of mean phase on each cycle for each neuron.

    spike_stack -- 3d array where axes 0 and 1 give coordinates for a neuron and axis 2 contains int 
                   indices of spikes on time series.
    analytic_signal -- complex 1d array of Hilbert-transformed theta signal.
    """

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

    # Mask zeros to exclude from calculations, truncate the stack where the data ends
    spike_stack = np.ma.masked_equal(spike_stack[:, :, :truncation_index], 0)
    
    # Cycle phi block generation algorithm; spikeless cycles have mean phase of zero
    for i in range(len(cycle_boundary_indices) + 1):
        cycle_phi_slice = np.zeros((spike_stack.shape[0], spike_stack.shape[1], 1)) # Initialize a slice to populate with means
        for row_index in range(spike_stack.shape[0]):
            for column_index in range(spike_stack.shape[1]):
                stack_column = spike_stack[row_index, column_index, :] # The current stack column (spike indices for neuron at coorindate (row_index, column_index)).
                if i == 0: 
                    cycle_phi = theta_phase_time_series[stack_column[(stack_column < cycle_boundary_indices[i]) & (stack_column.mask == False)]]
                    if cycle_phi.size != 0:
                        cycle_phi_slice[row_index, column_index, 0] = np.mean(cycle_phi)
                elif i < len(cycle_boundary_indices):
                    cycle_phi = theta_phase_time_series[stack_column[(stack_column >= cycle_boundary_indices[i - 1]) & (stack_column < cycle_boundary_indices[i]) & (stack_column.mask == False)]]
                    if cycle_phi.size != 0:
                        cycle_phi_slice[row_index, column_index, 0] = np.mean(cycle_phi)
                else:
                    cycle_phi = theta_phase_time_series[stack_column[(stack_column >= cycle_boundary_indices[i - 1]) & (stack_column.mask == False)]]
                    if cycle_phi.size != 0:
                        cycle_phi_slice[row_index, column_index, 0] = np.mean(cycle_phi)
        if i == 0:
            cycle_phi_block = cycle_phi_slice
        else:
            cycle_phi_block = np.concatenate((cycle_phi_block, cycle_phi_slice), axis=2)  

    return cycle_phi_block, cycle_boundary_indices
def HPC_phi_compute_RMQ(post_sim_specifications, cycle_phi_block):
    
    num_cycles = post_sim_specifications['num_cycles']
    if num_cycles <= 0 or num_cycles == 'all':
        num_cycles = cycle_phi_block.shape[2]

    num_cycles = min(num_cycles, cycle_phi_block.shape[2])

    cycle_phi_block_slice = cycle_phi_block[:, :, -num_cycles:] 
    masked_cycle_phi_block_slice = np.ma.masked_equal(cycle_phi_block_slice, 0)
    first_order_differences = np.diff(np.flip(masked_cycle_phi_block_slice, axis=2), axis=2)
    RMQ_array = np.mean(first_order_differences, axis=2)

    return RMQ_array

# Plotting functions
def RMQ_meshfig(RMQ_array, subset_1, subset_2, subset_1_name, subset_2_name, suppress=False, figure_path=False):

    RMQ_array = np.flipud(RMQ_array)

    plt.close()
    fig, ax = plt.subplots()

    shading = ax.contourf(subset_1, subset_2, RMQ_array, cmap='cool', alpha=0.7) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contour(subset_1, subset_2, RMQ_array, [0], colors='k', linewidths=0.9, linestyle='dashed', alpha=0.75)
    plt.colorbar(shading)
    
    plt.xlabel(subset_1_name.replace("_", " ").capitalize())
    plt.ylabel(subset_2_name.replace("_", " ").capitalize())

    figure_object = plt.gcf()

    if suppress == False:
        plt.show()
    if figure_path != False:
        figure_object.savefig(Path('/'.join([figure_path, 'contourPlots', 'HPC_phi_contour_{}{}{}'.format(date.today().day, date.today().month, '.png')])), dpi=400, bbox_inches='tight')  
def TS_meshfig(V, theta_rhythm, cycle_boundaries, simulation_duration, dt, subset_1, subset_2, subset_1_name, subset_2_name, subset_1_num, subset_2_num, suppress=False, figure_path=False):

    nearest_indices_1, nearest_indices_2 = get_centrally_symmetric_indices(subset_1, subset_2, subset_1_num, subset_2_num)

    time_to_plot = min(simulation_duration, 2000)
    time_axis_length = int(time_to_plot/dt)
    time_axis = np.linspace(0, time_to_plot, num=time_axis_length)

    plt.close()
    fig = plt.figure(figsize=(19.2, 10.8))
    fig.patch.set_alpha(0.0)
    root_gridspec = fig.add_gridspec(len(nearest_indices_2), len(nearest_indices_1), wspace=0.5, hspace=0.5)

    for row_number in range(len(nearest_indices_1)):
        for column_number in range(len(nearest_indices_2)):
            sub_gridspec = root_gridspec[row_number, column_number].subgridspec(2, 1, wspace=0.1, hspace=0.4)
            axes = [fig.add_subplot(sub_gridspec[0]), fig.add_subplot(sub_gridspec[1])]
            axes[0].get_shared_x_axes().join(axes[0], axes[1])

            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            axes[0].plot(time_axis, V[row_number, column_number, -time_axis_length:], linewidth=0.25, color='k', label='Vm')
            axes[1].plot(time_axis, theta_rhythm[-time_axis_length:], linewidth=0.25, color='k', label='LFP')
            axes[0].xaxis.set_ticks([])
            axes[1].yaxis.set_ticks([])

            if row_number == len(nearest_indices_1) - 1:
                axes[1].set_xlabel(str(round(subset_1[nearest_indices_1[column_number]], 1)), fontsize='large')
            if column_number == 0:
                axes[0].set_ylabel(str(round(subset_2[-(nearest_indices_2[row_number] + 1)], 1)), fontsize='large')

            vline_bottom = min(theta_rhythm)
            vline_top = max(theta_rhythm)
            for boundary_index in cycle_boundaries[cycle_boundaries <= time_axis_length]:
                axes[1].vlines(time_axis[boundary_index - 1], vline_bottom, vline_top, linestyle='dashed', color='k', linewidth=0.25)

    fig.text(0.5, 0.02, subset_1_name.replace("_", " ").capitalize(), ha='center', fontsize='x-large')
    fig.text(0.04, 0.5, subset_2_name.replace("_", " ").capitalize(), va='center', rotation='vertical', fontsize='x-large')

    figure_object = plt.gcf()

    if suppress == False:
        plt.show()
    if figure_path != False:
        figure_object.savefig(Path('/'.join([figure_path, 'meshTimeSeries', 'HPC_phi_meshTimeSeries_{}{}{}'.format(date.today().day, date.today().month, '.png')])), dpi=400, bbox_inches='tight')  
def RM_meshfig(cycle_phi_block, subset_1, subset_2, subset_1_name, subset_2_name, subset_1_num, subset_2_num, suppress=False, figure_path=False):
          
    line_of_identity = np.linspace(0, 2*np.pi) 
    nearest_indices_1, nearest_indices_2 = get_centrally_symmetric_indices(subset_1, subset_2, subset_1_num, subset_2_num)

    plt.close() 

    fig, axes = plt.subplots(nrows=len(nearest_indices_2), ncols=len(nearest_indices_1)) 
    fig.patch.set_alpha(0.0) 

    for row in axes:
        for col in row:
            col.spines['top'].set_visible(False)
            col.spines['right'].set_visible(False)

    for row_number in range(len(nearest_indices_2)):
        for column_number in range(len(nearest_indices_1)):

            phi_K_previous = np.delete(cycle_phi_block[row_number, column_number, :], -1)
            phi_K = np.delete(cycle_phi_block[row_number, column_number, :], 0)

            axes[row_number, column_number].scatter(phi_K_previous, phi_K, color='k', s=5)
            axes[row_number, column_number].plot(line_of_identity, line_of_identity, linestyle='dashed', color='k', linewidth=0.25)

            axes[row_number, column_number].set_xlim([0, 2*np.pi])
            axes[row_number, column_number].set_ylim([0, 2*np.pi])

            axes[row_number, column_number].set_xticks([])
            axes[row_number, column_number].set_xticks([], minor=True)

            axes[row_number, column_number].set_yticks([])
            axes[row_number, column_number].set_yticks([], minor=True)

            if row_number == len(nearest_indices_2) - 1:
                axes[row_number, column_number].set_xlabel(round(subset_1[column_number], 1), fontsize='large')
            if column_number == 0:
                axes[row_number, column_number].set_ylabel(round(subset_2[-(row_number + 1)], 1), fontsize='large')

            fig.text(0.5, 0.02, subset_1_name.replace("_", " ").capitalize(), ha='center', fontsize='x-large')
            fig.text(0.04, 0.5, subset_2_name.replace("_", " ").capitalize(), va='center', rotation='vertical', fontsize='x-large')

    figure_object = plt.gcf() 

    if suppress == False:
        plt.show()
    if figure_path != False:
        figure_object.savefig(Path('/'.join([figure_path, 'meshReturnMaps', 'HPC_phi_meshReturnMap_{}{}{}'.format(date.today().day, date.today().month, '.png')])), dpi=400, bbox_inches='tight')  

# Utility functions
def probe_time_series(data, simulation_duration, dt, coordinate: tuple, timestamping_block=None):
    
    plt.close()
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['right'].set_visible(False) 
    ax.spines['top'].set_visible(False) 

    time_axis = np.linspace(0, simulation_duration, num=int(simulation_duration/dt))

    ax.plot(time_axis, data[coordinate], linewidth=0.8, color='k')
    if type(timestamping_block) != None:
        timestamping_block[timestamping_block == 0] = np.nan
        ax.scatter(time_axis, timestamping_block[coordinate]*50, color='red')

    ax.set_xlabel('Time ($ms$)')
    ax.set_ylabel('Amplitude ($agnostic$)')

    plt.show()
def probe_return_map(cycle_phi_block, coordinate: tuple):
    
        phi_K_previous = np.delete(cycle_phi_block, -1, axis=2)
        phi_K = np.delete(cycle_phi_block, 0, axis=2)

        line_of_identity = np.linspace(0, 2*np.pi) 

        plt.close()

        fig, ax = plt.subplots()
        fig.patch.set_alpha(0.0)

        ax.spines['right'].set_visible(False) 
        ax.spines['top'].set_visible(False) 

        ax.scatter(phi_K_previous[coordinate[0], coordinate[1], :], phi_K[coordinate[0], coordinate[1], :])
        ax.plot(line_of_identity, line_of_identity, linestyle='dashed', color='k', linewidth=0.8)

        ax.set_xlabel("$\phi_{k-1}$", fontsize=13)
        ax.set_ylabel("$\phi_{k}$", fontsize=13) 
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0, 2*np.pi])

        tickLabelList = [r" ", r"$0$", r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$", r"$\frac{4}{3}\pi$", r"$\frac{5}{3}\pi$", r"$2\pi$"] 
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax.set_xticklabels(tickLabelList) 
        ax.set_yticklabels(tickLabelList)

        plt.show()
def get_centrally_symmetric_indices(subset_1, subset_2, subset_1_num, subset_2_num):

    n = len(subset_1)
    m = len(subset_2)
    line_of_identity = np.linspace(0, 2*np.pi) 

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

        V = decoded_message['V']

        spike_stack = decoded_message['spike_stack']

        sim_specifications = {key: value for key, value in np.stack((decoded_message['input_spec_keys'], decoded_message['input_spec_values']), axis=-1)}

        for key in sim_specifications.keys():
            sim_specifications[key] = float(sim_specifications[key])
        
        del decoded_message
        
        return V, spike_stack, sim_specifications

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
        'adaptation_response_constant': 50, # mV
        'adaptation_decay_constant': 8, # mV
        'adaptation_time_constant': 10, # ms
        
        # Ornstein-Uhlenbeck process parameters
        'OU_sigma': 100,
        'OU_mu': nb.float32(0.3),
        'OU_time_constant': nb.int32(50),

        # Dual oscillator input parameters
        'theta_amplitude': 20, # mV
        'interference_amplitude': 20, # mV

        'theta_frequency': 10, # Hz
        'interference_frequency': 11, # Hz
    }
    ANALYSIS_SPECS = {

        'num_cycles': -1,

        'encode': False,
        'encode_path': '/'.join([os.getcwd(), 'serializedSims']),

        'decode': True,
        'decode_path': '/'.join([os.getcwd(), 'serializedSims', 'GT_precession', 'precession_theta_amplitude_interference_amplitude_20_50_10_20_50_10_7112021.npz']),

        'figure_path': False, #'/'.join([os.getcwd(), 'modelFigures', 'GT_precession']),

        'contour_suppress': False,
        'TS_mesh_suppress': True,
        'RM_mesh_suppress': True,
    }
    
    Simulation = PipelineManager(SIM_SPECS, ANALYSIS_SPECS, DOI=True, ARC=False)

    Simulation.add_parameter_subset(20, 50, 10, 'theta_amplitude')
    Simulation.add_parameter_subset(20, 50, 10, 'interference_amplitude')

    Simulation.execute_pipeline('forcing')

if __name__ == '__main__':
    main()
