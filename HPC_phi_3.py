# Standard library imports
from datetime import date
import itertools
import math
import numpy as np
import os
from pathlib import Path
import sys
import time

# Third party imports
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import mode
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut

# Local application imports: Pylance may be erroneously reporting a missing import, see https://github.com/microsoft/pylance-release/issues/1167#issuecomment-821777661
from HPC_phi_sim_data_pb2 import SimulationData as ProtoBufferInterface

class InputSpecifications():

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.parameter_subspace_names = []
        self.parameter_subspaces = []
        self.sealed = False

    def newParameterSubspace(self, name:str, lambda_function, start:int, stop:int, number_elements:int):
        self.parameter_subspace_names.append(name)
        self.parameter_subspaces.append(list(map(lambda_function, np.linspace(start, stop, number_elements))))
    
    def sealParameterSubspace(self):
        if not self.sealed:
            self.sealed_parameter_subspace = zip(self.parameter_subspace_names, self.parameter_subspaces)
            self.sealed = True
        else:
            pass
class OutputSpecifications():

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
class SimulationIterationManager():

    def __init__(self, InputSpecifications, OutputSpecifications):
        self.__dict__.update(InputSpecifications.__dict__)
        self.__dict__.update(OutputSpecifications.__dict__)
        self.iteration_number = 0

    def initializeOutputArrays(self):

        mesh_shape = tuple(map(lambda x: len(x), self.parameter_subspaces)) # maps the length of the ith iterable to the ith index of a tuple; specifies the shape of output tensors

        self.dimension_lengths = [] # Will contain the shape of the parameter mesh
        self.mesh_size = 1 # Initialize with the identity

        for dimension in self.parameter_subspaces: # For each parameter to iterate over
                self.dimension_lengths.append(len(dimension)) # Take it's length to correspond to the shape in it's dimension
                self.mesh_size *= len(dimension) # The total number of coordinates in parameter space is the product of all axis lengths

        if self.mesh_size == 1:
            self.single_iteration = True
        else:
            self.single_iteration = False

        self.output_mesh = np.zeros(mesh_shape) # To contain return map metric at each coordinate
        self.coordinate_mesh = np.zeros(mesh_shape, dtype=(int, len(self.parameter_subspaces))) # To contain the coordinate at each coordinate
        self.cycle_phi_central_tendencies = np.zeros(mesh_shape, dtype=(list)) # To contain the central tendency of phi on each theta cycle for the simulation at each coordinate

    def generateMeshFigures(self):

        if self.plot_contour == True or self.save_contour_plot == True:

            plt.close()
            fig, ax = plt.subplots()
            shading = ax.contourf(self.parameter_subspaces[0], self.parameter_subspaces[1], np.transpose(self.output_mesh), cmap='cool', alpha=0.7)
            ax.contour(self.parameter_subspaces[0], self.parameter_subspaces[1], np.transpose(self.output_mesh), [0], colors='black', linewidths=0.9, linestyles='dashed', alpha=0.75)
            plt.colorbar(shading)
            plt.xlabel(self.parameterNames[0].replace("_", " ") + " " + self.parameterUnits[0])
            plt.ylabel(self.parameterNames[1].replace("_", " ") + " " + self.parameterUnits[1])

            figureObject = plt.gcf()

            if self.plot_contour == True:
                plt.show()
            if self.save_contour_plot == True:
                figureObject.savefig('{}\\HPC_phi\\modelFigures\\contourPlots\\HPC_phi_contour_{}{}{}'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), date.today().month, date.today().day, self.contour_plot_file_type), bbox_inches='tight')  

        if self.plot_mesh_maps == True or self.save_mesh_fig == True:
            
            line_of_identity = np.linspace(0, 2*np.pi)
            self.cycle_phi_central_tendencies = np.flipud(np.transpose(self.cycle_phi_central_tendencies))

            plt.close()
            fig, axes = plt.subplots(nrows=len(self.parameter_subspaces[1]), ncols=len(self.parameter_subspaces[0]))

            if axes.shape[0] < 7 and axes.shape[1] < 7:
                fontSize = 10
            elif axes.shape[0] < axes.shape[1]:
                fontSize = 20*(math.log(axes.shape[1]))**-1
            else:
                fontSize = 20*(math.log(axes.shape[0]))**-1

            fig.patch.set_alpha(0.0)
            for row in axes:
                for col in row:
                    col.spines['top'].set_visible(False)
                    col.spines['right'].set_visible(False) 

            for row_number, row in enumerate(self.cycle_phi_central_tendencies, start=0):
                for column_number, data in enumerate(row, start=0):
                    phi_K_previous = np.delete(data, -1) # All entries except the Kth one can be the (K - 1)th entry
                    phi_K = np.delete(data, 0) # All entries except the 0th can be the Kth (i.e. next) entry

                    axes[row_number, column_number].scatter(phi_K_previous, phi_K, s=32*(self.mesh_size)**-1, color='deeppink')
                    axes[row_number, column_number].plot(line_of_identity, line_of_identity, linestyle='dashed', color='k', linewidth=0.25) 
                    
                    axes[row_number, column_number].set_xlim([0, 2*np.pi])
                    axes[row_number, column_number].set_ylim([0, 2*np.pi])
                    
                    axes[row_number, column_number].set_xticks([])
                    axes[row_number, column_number].set_xticks([], minor=True)
                   
                    axes[row_number, column_number].set_yticks([])
                    axes[row_number, column_number].set_yticks([], minor=True)

                    if row_number == len(self.cycle_phi_central_tendencies) - 1:
                        axes[row_number, column_number].set_xlabel(int(self.parameter_subspaces[0][column_number]), fontsize=fontSize)
                    if column_number == 0:
                        axes[row_number, column_number].set_ylabel(int(self.parameter_subspaces[1][- row_number - 1]), fontsize=fontSize)
            
            fig.text(0.5, 0.02, self.parameterNames[0].replace("_", " ") + " " + self.parameterUnits[0], ha='center')
            fig.text(0.04, 0.5, self.parameterNames[1].replace("_", " ") + " " + self.parameterUnits[1], va='center', rotation='vertical')

            figureObject = plt.gcf()

            if self.plot_mesh_maps == True:
                plt.show()
            if self.save_mesh_fig == True and self.mesh_fig_file_type == '.svg':
                figureObject.savefig('{}\\HPC_phi\\modelFigures\\meshFigs\\HPC_phi_meshFig_{}{}{}'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), date.today().month, date.today().day, self.mesh_fig_file_type), bbox_inches='tight')
            elif self.save_mesh_fig == True and self.mesh_fig_file_type == '.png':
                figureObject.savefig('{}\\HPC_phi\\modelFigures\\meshFigs\\HPC_phi_meshFig_{}{}{}'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), date.today().month, date.today().day, self.mesh_fig_file_type), dpi=self.dpi, bbox_inches='tight')
    
    def generateProtoBufferFilename(self, update_filename=False): 
        
        if not update_filename:  

            self.proto_buffer_filename = ""

            for parameter_name in self.parameter_subspace_names:
                self.proto_buffer_filename = '_'.join([self.proto_buffer_filename, parameter_name])

            for vector in self.parameter_subspaces:
                self.proto_buffer_filename = '_'.join([self.proto_buffer_filename, str(int(vector[0])), str(int(vector[-1])), str(len(vector))])
            
            d = date.today()
            self.proto_buffer_filename = '_'.join([self.proto_buffer_filename, "{:02d}{:02d}{:02d}".format(d.day, d.month, d.year)])
            self.proto_buffer_filename = '_'.join([self.proto_buffer_filename, "{}.pb.bin".format(self.iteration_number)])

            self.protobuf_encode_path = Path('/'.join([self.protobuf_serialized_data_path, self.proto_buffer_filename]))

        else:

            underscore_separated_segments = self.proto_buffer_filename.split('_')
            period_separated_segments = underscore_separated_segments[-1].split('.')
            period_separated_segments[0] = str(self.iteration_number)

            recombined_period_separated_segments = '.'.join(period_separated_segments)
            underscore_separated_segments[-1] = recombined_period_separated_segments

            self.proto_buffer_filename = '_'.join(underscore_separated_segments)

            self.protobuf_encode_path = Path('/'.join([self.protobuf_serialized_data_path, self.proto_buffer_filename]))
            
    def meshSweep(self):

        self.initializeOutputArrays()

        self.generateProtoBufferFilename(update_filename=False)

        if not self.single_iteration:

            if self.iteration_number != 0:
                self.generateProtoBufferFilename(update_filename=True)

            self.overall_start_time = time.time()

            parameterMesh = itertools.product(*[range(s) for s in self.dimension_lengths])
            for parameter_space_coordinate in parameterMesh: 

                for dimension, subspace_index in enumerate(parameter_space_coordinate): 

                    self.__dict__.update({self.parameter_subspace_names[dimension]: self.parameter_subspaces[dimension][subspace_index]}) 

                self.coordinate_mesh[parameter_space_coordinate] = parameter_space_coordinate
                self.output_mesh[parameter_space_coordinate], self.cycle_phi_central_tendencies[parameter_space_coordinate] = HPC_phi_simulation(self, execute_simulation=True) 

                if self.iteration_number == 0:
                    print("===== {} m {} s to simulate iteration {} of {} =====".format(int((time.time() - self.overall_start_time)/60), round(((time.time() - self.overall_start_time)%60)/1, 2), self.iteration_number + 1, self.mesh_size))
                    subsequent_simulation_start_time = time.time()
                else:
                    print("===== {} m {} s to simulate iteration {} of {} =====".format(int((time.time() - subsequent_simulation_start_time)/60), round(((time.time() - subsequent_simulation_start_time)%60)/1, 2), self.iteration_number + 1, self.mesh_size))
                    subsequent_simulation_start_time = time.time()

                self.iteration_number += 1

            print("\n===== ALL ITERATIONS FINISHED =====")
            print("===== {} m {} s for parameter mesh sweep =====".format(int((time.time() - self.overall_start_time)/60), round(((time.time() - self.overall_start_time)%60)/1, 2)))

            print('Output mesh: \n{}'.format(np.flipud(np.transpose(self.output_mesh))))
            
            if len(self.parameter_subspaces) == 2:
                self.generateMeshFigures()

        else:
            self.overall_start_time = time.time() 
            HPC_phi_simulation(self) 
            print("===== {} m {} s to simulate iteration {} of {} =====".format((time.time() - self.overall_start_time)/60, ((time.time() - self.overall_start_time)%60)/1, 1, 1)) # These magic 1's make print statement indicate that simulation 1 of 1 is complete

    def analyzeProtobufData(self):

        for protobuf_file in self.protobuf_files_to_decode:

            self.protobuf_decode_path = Path('/'.join([self.protobuf_serialized_data_path, protobuf_file]))          
class LIFNeuron:

    def __init__(self, **kwargs):

        model_parameter_names = ['rest_Vm', 'spike_Vm', 'rest_Vm', 'neuron_threshold', 'membrane_time_constant', 'absolute_refractory_period', 'membrane_resistance', 'dt', 'sigma']
        self.__dict__.update({key: kwargs[key] for key in kwargs.keys() if key in model_parameter_names})

        self.Vm = self.rest_Vm

        self.stochastic_input = 0

        self.spikeState = False

        self.refractory_period_length = self.absolute_refractory_period//self.dt 
        self.refractory_period_counter = 0 

    def updateVm(self, time_series_index, forcing, dt):

        if self.Vm >= self.neuron_threshold and self.spikeState == False: 
            
            self.Vm = self.spike_Vm 
            self.spikeState = True 

        elif self.spikeState == True and self.refractory_period_counter < self.refractory_period_length: 
            
            self.refractory_period_counter += 1 
       
        elif self.spikeState == True and self.refractory_period_counter >= self.refractory_period_length: 
            
            self.Vm = self.rest_Vm 
            self.spikeState = False 
            self.refractory_period_counter = 0 
        
        else:

            self.VL = -(self.Vm - self.rest_Vm) 
            self.V_forcing = forcing*self.membrane_resistance
            self.stochastic_input = self.sigma*math.sqrt(self.dt)*np.random.normal(loc=0.0, scale=1.0) 

            voltageSuperposition = self.VL + self.V_forcing + self.stochastic_input

            self.Vm += ((dt*voltageSuperposition)/self.membrane_time_constant) 

    def iterate(self, time_series_index, forcing, dt):

        self.updateVm(time_series_index, forcing, dt) 
class Simulation:

    def __init__(self, LIFNeuron, SimulationIterationManager):
        
        keys_of_interest = ['theta_amplitude', 'interference_amplitude', 'theta_frequency', 'interference_frequency', 'rest_Vm', 'spike_Vm', 'neuron_threshold', 'membrane_time_constant', 'absolute_refractory_period']
        self.parameter_subset_to_print = {key: SimulationIterationManager.__dict__[key] for key in keys_of_interest}
        self.model = LIFNeuron

    def initializeOutput(self, num_timesteps, dt):

        self.timeAxis = np.arange(num_timesteps) * dt 
        self.Vm_t = np.empty(num_timesteps) 
        self.spike_times = np.zeros(num_timesteps)

    def runSim(self, forcingFunction, num_timesteps, dt):

        self.initializeOutput(num_timesteps, dt) 

        print("\nInitialization complete...") #
        print("Simulating {} ms\nParameters: \n{}".format(forcingFunction.shape[0]*dt, self.parameter_subset_to_print))  
        
        for time_series_index in range(num_timesteps):

            self.model.iterate(time_series_index, forcingFunction[time_series_index], dt) 
            self.Vm_t[time_series_index] = self.model.Vm 

            if self.model.spikeState == True and self.model.refractory_period_counter == 0:
                self.spike_times[time_series_index] = 1 
        
        print("Simulation completed") 

def extractSpikePhi(SimulationIterationManager, Simulation, theta_rhythm, execute_simulation=True):

    if execute_simulation == True:

        theta_phase_time_series_INTERMEDIATE = list(map(lambda z: np.arctan2(z.imag, z.real - SimulationIterationManager.LFP_shift), hilbert(theta_rhythm))) # For each element in analytic signal, take arctan2 which outputs angle on interval (-pi, pi]  
        theta_phase_time_series = list(map(lambda phi: phi + 2*np.pi if phi < 0 else phi, theta_phase_time_series_INTERMEDIATE)) # For each angle on (-pi, pi], add 2*pi if phi < 0 to shift the interval to [0, 2*pi)

        del theta_phase_time_series_INTERMEDIATE 


        spike_phi = list(map(lambda i, j: j*theta_phase_time_series[i], range(len(Simulation.spike_times)), Simulation.spike_times))

        if SimulationIterationManager.encode_simulation_output == True:

            SimulationData_message = ProtoBufferInterface()

            for key in Simulation.model.__dict__.keys():

                if key in ['rest_Vm', 'spike_Vm', 'rest_Vm', 'neuron_threshold', 'membrane_time_constant', 'absolute_refractory_period', 'membrane_resistance', 'dt', 'sigma']:
                    
                    parameter = SimulationData_message.parameter.add()
                    parameter.key = key
                    parameter.value = Simulation.model.__dict__[key]

            SimulationData_message.spike_phi[:] = spike_phi

            with open(SimulationIterationManager.protobuf_encode_path, 'wb') as file_directory:

                file_directory.write(SimulationData_message.SerializeToString())

        #if SimulationIterationManager.decode_simulation_output == True:



    return spike_phi, theta_phase_time_series
def computeCyclePhiCentralTendency(SimulationIterationManager, theta_phase_time_series, spike_phi):
    
    spike_phi = np.array(spike_phi)

    central_tendency_measure_dictionary = {"0": 'mean',
                                           "1": 'median',
                                           "2": 'mode',
                                           "3": 'KDE'}

    central_tendency_mode = central_tendency_measure_dictionary[str(SimulationIterationManager.central_tendency_mode)]

    if SimulationIterationManager.theta_cycle_boundary_phase != 0: 

        rotated_theta_phase_time_series_INTERMEDIATE = list(map(lambda phi: phi - SimulationIterationManager.theta_cycle_boundary_phase, theta_phase_time_series)) # Rotate all phases by some angle (radians)
        rotated_theta_phase_time_series = list(map(lambda phi: phi + 2*np.pi if phi < 0 else phi, rotated_theta_phase_time_series_INTERMEDIATE)) # Shift any negative values back to the end of the cycle
        theta_phase_time_series = rotated_theta_phase_time_series # Have this rotated reference frame be the one we analyze further

    wrapped_omega = np.diff(np.array(theta_phase_time_series)) # Wrapped frequency of sinusoid is constant except where trajectory crosses positive x-axis where it is ~ -2*pi
    cycle_boundary_indices = np.where(wrapped_omega < 0)[0] # Hence, wherever frequency is negative we have a new cycle

    if SimulationIterationManager.verbose: print("Cycle boundary indices: {}".format(cycle_boundary_indices))

    if central_tendency_mode == 'mean':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary
            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries 

            if phi_indices.size == 0: # If no spikes occurred on the current cycle

                if SimulationIterationManager.verbose: print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                
                if SimulationIterationManager.verbose: print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

                phi_values = [] # Initialize a list

                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi

                cycle_phi_central_tendencies.append(np.mean(np.array(phi_values))) # Compute mean on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval 

        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            if SimulationIterationManager.verbose: print("final theta cycle contains no spikes") # Notify the operator
        else: # If there were spikes
            
            if SimulationIterationManager.verbose: print("indices of nonzero phi on cycle {}: {}".format(i + 1, phi_indices))

            phi_values = [] # Initialize a list

            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi

            cycle_phi_central_tendencies.append(np.mean(np.array(phi_values))) # Compute mean on all values phi from the end of the last cycle to the end of the time series

    elif central_tendency_mode == 'median':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary
            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

            if phi_indices.size == 0: # If no spikes occurred on the current cycle

                if SimulationIterationManager.verbose: print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes

                if SimulationIterationManager.verbose: print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

                phi_values = [] # Initialize a list

                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi

                cycle_phi_central_tendencies.append(np.median(np.array(phi_values))) # Compute median on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            if SimulationIterationManager.verbose: print("final theta cycle contains no spikes") # Notify the operator

        else: # If there were spikes
            if SimulationIterationManager.verbose: print("indices of nonzero phi on cycle {}: {}".format(i + 1, phi_indices))

            phi_values = [] # Initialize a list

            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi

            cycle_phi_central_tendencies.append(np.median(np.array(phi_values))) # Compute median on all values phi from the end of the last cycle to the end of the time series

    elif central_tendency_mode == 'mode':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary

            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                if SimulationIterationManager.verbose: print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                if SimulationIterationManager.verbose: print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

                phi_values = [] # Initialize a list

                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi

                cycle_phi_central_tendencies.append(mode(np.array(phi_values))[0][0]) # Compute mode on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            if SimulationIterationManager.verbose: print("final theta cycle contains no spikes") # Notify the operator

        else: # If there were spikes
            if SimulationIterationManager.verbose: print("indices of nonzero phi on cycle {}: {}".format(i + 1, phi_indices))

            phi_values = [] # Initialize a list

            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi

            cycle_phi_central_tendencies.append(mode(np.array(phi_values))[0][0]) # Compute mode on all values phi from the end of the last cycle to the end of the time series
    
    elif central_tendency_mode == 'KDE':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary

            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries
            if SimulationIterationManager.verbose: print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                if SimulationIterationManager.verbose: print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(kernelDensityEstimation(phi_values, SimulationIterationManager))

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            if SimulationIterationManager.verbose: print("final theta cycle contains no spikes") # Notify the operator

        else: # If there were spikes
            phi_values = [] # Initialize a list
            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
            cycle_phi_central_tendencies.append(kernelDensityEstimation(phi_values, SimulationIterationManager))
    
    if SimulationIterationManager.verbose: print("intracycle central tendencies: {}".format(cycle_phi_central_tendencies))

    return np.array(cycle_phi_central_tendencies), cycle_boundary_indices
def kernelDensityEstimation(data, SimulationIterationManager, cycle_interval=True):

    data_2D = np.reshape(data, (-1, 1)) # The backend requires a 2D array of shape (N, 1), where N is the number of samples

    if np.size(data_2D) > 1: # If we have greater than 1 spike time then we can do leave-one-out cross-validation

        bandwidth_space = 10**np.linspace(-1, 1, 100) # Allows bandwidths to take on values between 10**-1 and 10**1

        grid = GridSearchCV(KernelDensity(kernel='gaussian'), # Optimize prediction accuracy using the SAME kernel as what the KDE will ultimately use (below)
                                        {'bandwidth': bandwidth_space}, # The bandwidths we'll assess in cross-validation
                                        cv=LeaveOneOut()) # Cross-validation will 'leave-one-out', i.e. only set asside 1 spike to assess how good the KDE is (since we haven't many spikes to begin with)
        
        grid.fit(data_2D) # Perform the optimization

        optimal_bandwidth = grid.best_params_['bandwidth'] # Get the optimal bandwidth for the true KDE

    else: # Otherwise simply arbitrate the bandwidth, we only call it optimal bandwidth so that it's used in the KernelDensity instantiation (just below)
        optimal_bandwidth = SimulationIterationManager.kernel_bandwidth # Generally appears to range from 0.7 to 2.7, no clear relationship between number of spikes and this

    KDE = KernelDensity(bandwidth=optimal_bandwidth, kernel='gaussian') # Instantiate the KDE
    KDE.fit(data_2D) # Optimize it using the intracycle spike phi
    
    if cycle_interval == True:
        KDE_interval = np.reshape(np.linspace(0, 2*np.pi, num=100), (-1, 1)) # A linspace vector which allows us to plot the KDE over [0, 2*pi) and extract a value of phi corresponding to max probability
    else:
        KDE_interval = np.reshape(np.linspace((np.amin(data) - 0.1*(np.amax(data) - np.amin(data))), (np.amax(data) + 0.1*(np.amax(data) - np.amin(data))), num=1000), (-1, 1))
    
    ln_PDF_estimation = KDE.score_samples(KDE_interval) # For some reason you can only extract the ln() of the estimate PDF from the model
    
    if not SimulationIterationManager.central_tendency_PDF_estimation_suppress and cycle_interval == True: 

        plt.close()
        fig, ax = plt.subplots() 
        fig.patch.set_alpha(0.0) 

        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False) 

        ax.plot(KDE_interval, np.exp(ln_PDF_estimation), color='deeppink', linewidth=0.8) 
        ax.set_xlabel("$\phi$ $(radians)$", fontsize=13) 
        ax.set_ylabel("$Probability$", fontsize=13) 

        xLabelList = [r" ", r"$0$", r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$", r"$\frac{4}{3}\pi$", r"$\frac{5}{3}\pi$", r"$2\pi$"] 
        ax.set_xticklabels(xLabelList) 

        plt.show() 

    if not SimulationIterationManager.return_map_PDF_estimation_suppress and cycle_interval == False: 
        
        plt.close()
        fig, ax = plt.subplots() 
        fig.patch.set_alpha(0.0) 

        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False) 

        ax.plot(KDE_interval, np.exp(ln_PDF_estimation), color='deeppink', linewidth=0.8) 
        ax.set_xlabel("$\phi_{k-1} - \phi_k$ $(radians)$", fontsize=13) 
        ax.set_ylabel("$Probability$", fontsize=13)

        plt.show() 

    return KDE_interval[np.argmax(np.exp(ln_PDF_estimation))][0] 
def plotSolution(SimulationIterationManager, Simulation, theta_rhythm, cycle_boundary_indices):

    plt.close()

    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=False) # Two graps, two rows, linked x-axes
    fig.patch.set_alpha(0.0) # Make figure background transparent

    for ax in axes:
        ax.spines['top'].set_visible(False) # Make the top figure border invisible
        ax.spines['right'].set_visible(False)# Make the right figure border invisible

    axes[0].plot(Simulation.timeAxis, Simulation.Vm_t, linewidth=0.8, color='deeppink', label='Vm') # Plot the membrane potential time series
    axes[1].plot(Simulation.timeAxis, theta_rhythm, linewidth=0.8, color='deeppink', label='LFP') # Plot the theta rhythm (or other neural input signal) time series

    for boundary_index in cycle_boundary_indices:
        axes[1].vlines(Simulation.timeAxis[boundary_index], min(theta_rhythm), max(theta_rhythm), linestyle='dashed', color='k', linewidth=0.8)

    axes[0].set_ylabel("$V_m$ $(mV)$") # y-axis label of membrane potential time series, in millivolts
    axes[1].set_ylabel("$Amplitude$ $(pA)$") # y-axis label of theta rhythm
    axes[1].set_xlabel("$Time$ $(ms))$") # Label the x-axis only on the bottommost plot in milliseconds
    
    if SimulationIterationManager.save_sim_fig:
        plt.savefig(Path('/'.join([SimulationIterationManager.saved_figure_path, 'timeSeries', 'HPC_phi_iteration{}.png'.format(SimulationIterationManager.iteration_number)])), dpi=SimulationIterationManager.dpi, bbox_inches='tight') 
    if not SimulationIterationManager.sim_fig_suppress:
        plt.show()
def constructReturnMap(SimulationIterationManager, cycle_phi_central_tendencies, ):

    phi_K_previous = np.delete(cycle_phi_central_tendencies, -1) # All entries except the Kth one can be the (K - 1)th entry
    phi_K = np.delete(cycle_phi_central_tendencies, 0) # All entries except the 0th can be the Kth (i.e. next) entry
    line_of_identity = np.linspace(0, 2*np.pi) # Creates a line of identity with domain that spans the range of phi central tendencies
    phi_map_coordinate_difference = np.subtract(phi_K_previous, phi_K)
    
    if not SimulationIterationManager.return_map_suppress or SimulationIterationManager.save_return_map:

        plt.close()

        fig, ax = plt.subplots() 
        fig.patch.set_alpha(0.0) 

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 

        ax.scatter(phi_K_previous, phi_K, s=6, color='deeppink') 
        ax.plot(line_of_identity, line_of_identity, linestyle='dashed', color='k', linewidth=0.8) 

        ax.set_xlabel("$\phi_{k-1}$", fontsize=13)
        ax.set_ylabel("$\phi_{k}$", fontsize=13) 
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0, 2*np.pi])

        tickLabelList = [r" ", r"$0$", r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$", r"$\frac{4}{3}\pi$", r"$\frac{5}{3}\pi$", r"$2\pi$"] 
        ax.set_xticklabels(tickLabelList) 
        ax.set_yticklabels(tickLabelList)
        
        if SimulationIterationManager.save_return_map:
            plt.savefig(Path('/'.join([SimulationIterationManager.saved_figure_path, 'returnMaps', 'HPC_phi_iteration{}.png'.format(SimulationIterationManager.iteration_number)])), dpi=SimulationIterationManager.dpi, bbox_inches='tight') 
        if not SimulationIterationManager.return_map_suppress:
            plt.show()

    return phi_map_coordinate_difference

def HPC_phi_simulation(SimulationIterationManager, execute_simulation=True):

    if execute_simulation == True:

        num_timesteps = int(SimulationIterationManager.simulation_duration/SimulationIterationManager.dt) 
 
        # Convert desired frequency into radians/ms
        omega_theta = 2*np.pi*((SimulationIterationManager.theta_frequency**-1)*1000)**(-1) # Natural (angular) frequency corresponding to hippocampal theta LFP oscillations (radians/ms); NUMBER IS PERIOD IN ms
        omega_interference = 2*np.pi*((SimulationIterationManager.interference_frequency**-1)*1000)**(-1) # Natural (angular) frequency of interloping oscillator (radians/ms); NUMBER IS PERIOD IN ms

        # Functions defining independent field oscillations using forcing parameters
        theta_function = lambda t: SimulationIterationManager.theta_amplitude*np.sin(omega_theta*t) + SimulationIterationManager.LFP_shift # Corresponds to hippocampal theta sinusoidal oscillations
        interference_function = lambda t: SimulationIterationManager.interference_amplitude*np.sin(omega_interference*t) + SimulationIterationManager.LFP_shift # Corresponds to some interferring oscillation

        # Generate the theta and interference oscillations over specified timeseries, then superimpose the two
        theta_rhythm = list(map(theta_function, list(map(lambda t: t*SimulationIterationManager.dt, range(num_timesteps))))) 
        interference_rhythm = list(map(interference_function, list(map(lambda t: t*SimulationIterationManager.dt, range(num_timesteps))))) 
        field_oscillation = np.array([x + y for x, y in zip(theta_rhythm, interference_rhythm)]) 

        # Instantiate LIFNeuron
        Neuron = LIFNeuron(**SimulationIterationManager.__dict__)

        # Create and run simulation
        NeuronSim = Simulation(Neuron, SimulationIterationManager)
        NeuronSim.runSim(field_oscillation, num_timesteps, SimulationIterationManager.dt)
    
    simulation_quantification, cycle_phi_central_tendencies = simulationAnalysis(SimulationIterationManager, NeuronSim, theta_rhythm, execute_simulation=execute_simulation)

    return simulation_quantification, cycle_phi_central_tendencies
def simulationAnalysis(SimulationIterationManager, Simulation, theta_rhythm, execute_simulation=True):

    # Compute simulation phi sequence
    spike_phi, theta_phase_time_series = extractSpikePhi(SimulationIterationManager, Simulation, theta_rhythm, execute_simulation=execute_simulation)

    # Compute phi central tendencies on each theta cycle and construct return map
    cycle_phi_central_tendencies, cycle_boundary_indices = computeCyclePhiCentralTendency(SimulationIterationManager, theta_phase_time_series, spike_phi)

    # Plot simulation output and Return map
    if not SimulationIterationManager.sim_fig_suppress or SimulationIterationManager.save_sim_fig:
        plotSolution(SimulationIterationManager, Simulation, theta_rhythm, cycle_boundary_indices)

    return_map_coordinate_differences = constructReturnMap(SimulationIterationManager, cycle_phi_central_tendencies)
    simulation_quantification = kernelDensityEstimation(return_map_coordinate_differences, SimulationIterationManager, cycle_interval=False)

    return simulation_quantification, cycle_phi_central_tendencies

def main():

    directory_deepest_level = os.getcwd().split('\\')[-1]
    assert directory_deepest_level.split('.') == ['HPC_phi'], 'Working directory should be to HPC_phi folder;\nCurrent working directory: {}'.format(os.getcwd())

    INPUT_SPECIFICATION_DICTIONARY = {# Time parameters
                                      'dt': 0.01,                                   
                                      'simulation_duration': 1000,  
        
                                      # Dual oscillator input parameters
                                      'theta_amplitude': 300,                       
                                      'interference_amplitude': 300,                
                                      'theta_frequency': 12,                         
                                      'interference_frequency': 13.5,                
                                      'LFP_shift': 0,                                
                 
                                      # Langevin model properties
                                      'rest_Vm': -75,                                
                                      'spike_Vm': 160,                               
                                      'neuron_threshold': -40,                      
                                      'membrane_time_constant': 40,                 
                                      'membrane_resistance': 1,                     
                                      'absolute_refractory_period': 0,              
                                      'sigma': 0.0,

                                      # Analysis parameters
                                      'verbose': False,
                                      'central_tendency_mode': 3,                  
                                      'theta_cycle_boundary_phase': 3*np.pi/2,      
                                      'kernel_bandwidth': 1.5}      
         
    OUTPUT_SPECIFICATION_DICTIONARY = {# Figure popup suppression options 
                                       'sim_fig_suppress': True,                                          
                                       'return_map_suppress': True,                                           
                                       'central_tendency_PDF_estimation_suppress': True,                                       
                                       'return_map_PDF_estimation_suppress': True,                                           
                                       'contour_plot_suppress': False,                                       
                                       'mesh_fig_suppress': False,                                         
                                       
                                       # Figure saving options
                                       'saved_figure_path': '/'.join([os.getcwd(), 'modelFigures']),
                                       'save_sim_fig': False,                       
                                       'save_return_map': False,                     
                                       'save_contour_plot': False,                    
                                       'save_mesh_fig': False,                                        
                                       
                                       # Figure filetype options
                                       'contour_plot_file_type': '.svg',              
                                       'mesh_fig_file_type': '.png',
                                       
                                       # Protocol buffer options
                                       'protobuf_serialized_data_path': '/'.join([os.getcwd(), 'serializedSims']),
                                       'encode_simulation_output': True,
                                       'decode_simulation_output': False,
                                       'protobuf_files_to_decode': ['theta_frequency_interference_frequency_1_3_10_1_3_10_070621_1.pb.bin']} 

    InputSpecificationsObject = InputSpecifications(**INPUT_SPECIFICATION_DICTIONARY)
    InputSpecificationsObject.newParameterSubspace('theta_amplitude', lambda q: q*300, 1, 3, 2)
    InputSpecificationsObject.newParameterSubspace('interference_amplitude', lambda q: q*300, 1, 3, 2)
    InputSpecificationsObject.sealParameterSubspace()

    OutputSpecificationsObject = OutputSpecifications(**OUTPUT_SPECIFICATION_DICTIONARY)

    SimulationSet = SimulationIterationManager(InputSpecificationsObject, OutputSpecificationsObject)
    SimulationSet.meshSweep()

if __name__ == '__main__':
    main()