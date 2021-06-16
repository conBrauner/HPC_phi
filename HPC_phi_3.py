# Standard library imports
from datetime import date
from functools import reduce
import itertools
import math
import numpy as np
import os
from pathlib import Path
import time

# Third party imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import hilbert
from scipy.stats import mode
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut

# Local application imports: Pylance may be erroneously reporting a missing import, see https://github.com/microsoft/pylance-release/issues/1167#issuecomment-821777661
from HPC_phi_sim_data_pb2 import SimulationData as ProtoBufferInterface

class InputSpecifications():
    """Class object primarily serves as a vehicle for all 'input' parameters (those controlling simulation and analysis) by initializing with attributes corresponding to an input dictionary.

    This object's attributes are directly inherited by SimulationIterationManager. The purpose of this initial object is to serve as a separate space for adjusting input parameters.
    """

    def __init__(self, **kwargs) -> None:

        # Assign attributes corresponding to each kwarg from a dictionary
        self.__dict__.update(kwargs)

        # Initialize attributes controlling variable parameters
        self.parameter_subspace_names = [] 
        self.parameter_subspace_units = []
        self.parameter_subspaces = []
        self.sealed = False # True prohibits addition of new variables over current runtime

    def newParameterSubspace(self, name:str, units:str, lambda_function, start:int, stop:int, number_elements:int):

        # Identifies variable parameters by their attribute (str) and appends a vector of parameters to iterate over
        self.parameter_subspace_names.append(name)
        self.parameter_subspace_units.append(units)
        self.parameter_subspaces.append(list(map(lambda_function, np.linspace(start, stop, number_elements)))) # Map a linspace vector to a list by some lambda function
    
    def sealParameterSubspace(self):

        # If the subspace isn't sealed then zip vector list to the list with the corresponding parameter name
        if not self.sealed:
            self.sealed_parameter_subspace = zip(self.parameter_subspace_names, self.parameter_subspaces)
            self.sealed = True
        else:
            pass
class OutputSpecifications():
    """Class object primarily serves as a vehicle for all 'output' parameters (those controlling figures and serialization) by initializing with attributes corresponding to an input dictionary.

    This object's attributes are directly inherited by SimulationIterationManager. The purpose of this initial object is to serve as a separate space for adjusting output parameters.
    """

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
class SimulationIterationManager():
    """Highest level organization for execution, analysis, output and serialization of simulation. Passes itself through a pipeline for organized distribution of parameters and arguments.

    Inherits attributes from InputSpecifications and OutputSpecifications objects. Generates output tensors with shape according to the number and size of parameter subspaces. Iterates over a 
    complete parameter subspace and executes a simulation at each coordinate. Another method allows user to decode serialized simulation data using a protocol buffer to test alternative
    analysis subroutines. If exactly two linear subspaces are identified then a contour plot and meshfigure are generated to summarize a simulation output.
    """

    def __init__(self, InputSpecifications, OutputSpecifications):
        self.__dict__.update(InputSpecifications.__dict__)
        self.__dict__.update(OutputSpecifications.__dict__)
        self.iteration_number = 0

    def initializeOutputArrays(self, fromProtobuf=False):

        if not fromProtobuf:
            # Output tensor will have shape corresponding to the ordered sizes of parameter subspaces.
            mesh_shape = tuple(map(lambda x: len(x), self.parameter_subspaces)) # maps the length of the ith iterable to the ith index of a tuple; specifies the shape of output tensors
            
            self.dimension_lengths = list(mesh_shape) 
            self.mesh_size = reduce(lambda x, y: x*y, self.dimension_lengths) # Reduce has the effect of multiplying all entries and outputting a scalar; indicates total number of simulations

            if self.mesh_size == 1:
                self.single_iteration = True
            else:
                self.single_iteration = False

            self.output_mesh = np.zeros(mesh_shape) # To contain return map metric at each coordinate
            self.coordinate_mesh = np.zeros(mesh_shape, dtype=(int, len(self.parameter_subspaces))) # To contain the coordinate at each coordinate
            self.cycle_phi_central_tendencies = np.zeros(mesh_shape, dtype=(list)) # To contain the central tendency of phi on each theta cycle for the simulation at each coordinate

            if self.mesh_fig_time_series_suppress == False or self.save_mesh_fig_time_series == True:
                self.neuron_time_series_mesh = np.zeros(mesh_shape, dtype=(list))
                self.LFP_time_series_mesh = np.zeros(mesh_shape, dtype=(list))
                self.cycle_boundary_indices_mesh = np.zeros(mesh_shape, dtype=(list))
        
        else:

            self.sorted_protobuf_files_to_decode = [0]*len(self.protobuf_files_to_decode)

            for filename in self.protobuf_files_to_decode:

                underscore_separated_segments = filename.split('_')
                mesh_shape_backwards = []
                decode_iteration = int(underscore_separated_segments[-1].split('.')[0])
                self.sorted_protobuf_files_to_decode[decode_iteration] = filename

                for dimension in range(self.protobuf_mesh_dimension):
                    mesh_shape_backwards.append(int(underscore_separated_segments[-3*(dimension + 1)]))

            self.dimension_lengths = tuple(mesh_shape_backwards.reverse())
            self.sorted_protobuf_files_to_decode = np.reshape(self.sorted_protobuf_files_to_decode, self.dimension_lengths)

            self.output_mesh = np.zeros(self.dimension_lengths) # To contain return map metric at each coordinate
            self.cycle_phi_central_tendencies = np.zeros(self.dimension_lengths, dtype=(list)) # To contain the central tendency of phi on each theta cycle for the simulation at each coordinate

            if self.mesh_fig_time_series_suppress == False or self.save_mesh_fig_time_series == True:
                self.neuron_time_series_mesh = np.zeros(self.dimension_lengths, dtype=(list))
                self.LFP_time_series_mesh = np.zeros(self.dimension_lengths, dtype=(list))
                self.cycle_boundary_indices_mesh = np.zeros(self.dimension_lengths, dtype=(list))

    def generateMeshFigures(self):

        if self.contour_plot_suppress == False or self.save_contour_plot == True:

            plt.close()

            fig, ax = plt.subplots()

            # Generate contour plot
            shading = ax.contourf(self.parameter_subspaces[0], self.parameter_subspaces[1], np.transpose(self.output_mesh), cmap='cool', alpha=0.7) # Colour gradients in background
            ax.contour(self.parameter_subspaces[0], self.parameter_subspaces[1], np.transpose(self.output_mesh), [0], colors='black', linewidths=0.9, linestyles='dashed', alpha=0.75) # Lines indicating sea level; i.e. phase-locking
            plt.colorbar(shading)

            # Parameter subspace names and units specified at newParameterSubspace() call
            plt.xlabel(self.parameter_subspace_names[0].replace("_", " ") + " " + self.parameter_subspace_units[0]) 
            plt.ylabel(self.parameter_subspace_names[1].replace("_", " ") + " " + self.parameter_subspace_units[1])

            figureObject = plt.gcf()

            if self.contour_plot_suppress == False:
                plt.show()
            if self.save_contour_plot == True:
                figureObject.savefig(Path('/'.join([self.saved_figure_path, 'contourPlots', 'HPC_phi_contour_{}{}{}'.format(date.today().day, date.today().month, self.contour_plot_file_type)])), dpi=self.dpi, bbox_inches='tight')  

        if self.mesh_fig_suppress == False or self.save_mesh_fig == True:
            
            line_of_identity = np.linspace(0, 2*np.pi) # Line of phase locking to be plotted in each subfigure
            self.cycle_phi_central_tendencies = np.flipud(np.transpose(self.cycle_phi_central_tendencies)) # Transpose the grid, then reflect in horizontal axis

            plt.close()

            fig, axes = plt.subplots(nrows=len(self.parameter_subspaces[1]), ncols=len(self.parameter_subspaces[0]))
            fig.patch.set_alpha(0.0)

            # For small axis sizes, set fontsize = 10, otherwise set fontsize = 20/ln(size(largest_subspace))
            if axes.shape[0] < 7 and axes.shape[1] < 7:
                font_size = 10
            elif axes.shape[0] < axes.shape[1]:
                font_size = 20*(math.log(axes.shape[1]))**-1
            else:
                font_size = 20*(math.log(axes.shape[0]))**-1
            
            # Make the right and top figure border for each subfigure invisible
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
                        axes[row_number, column_number].set_xlabel(int(self.parameter_subspaces[0][column_number]), fontsize=font_size)
                    if column_number == 0:
                        axes[row_number, column_number].set_ylabel(int(self.parameter_subspaces[1][- row_number - 1]), fontsize=font_size)
            
            fig.text(0.5, 0.02, self.parameter_subspace_names[0].replace("_", " ") + " " + self.parameter_subspace_units[0], ha='center')
            fig.text(0.04, 0.5, self.parameter_subspace_names[1].replace("_", " ") + " " + self.parameter_subspace_units[1], va='center', rotation='vertical')

            figureObject = plt.gcf()

            if self.mesh_fig_suppress == False:
                plt.show()
            if self.save_mesh_fig == True:
                figureObject.savefig(Path('/'.join([self.saved_figure_path, 'meshFigs', 'HPC_phi_meshFig_{}{}{}'.format(date.today().day, date.today().month, self.mesh_fig_file_type)])), dpi=self.dpi, bbox_inches='tight')
    
        if self.mesh_fig_time_series_suppress == False or self.save_mesh_fig_time_series == True:

            self.neuron_time_series_mesh = np.flipud(np.transpose(self.neuron_time_series_mesh)) # Transpose the grid, then reflect in horizontal axis
            self.LFP_time_series_mesh = np.flipud(np.transpose(self.LFP_time_series_mesh)) # Transpose the grid, then reflect in horizontal axis
            time_axis = np.arange(int(self.simulation_duration/self.dt))*self.dt

            plt.close()

            fig = plt.figure()
            fig.patch.set_alpha(0.0)
            root_gridspec = fig.add_gridspec(len(self.parameter_subspaces[1]), len(self.parameter_subspaces[0]), wspace=0.5, hspace=0.5)

            # For small axis sizes, set fontsize = 10, otherwise set fontsize = 20/ln(size(largest_subspace))
            if len(self.parameter_subspaces[1]) < 7 and len(self.parameter_subspaces[0]) < 7:
                font_size = 10
            elif len(self.parameter_subspaces[1]) < len(self.parameter_subspaces[0]):
                font_size = 20*(math.log(len(self.parameter_subspaces[0])))**-1
            else:
                font_size = 20*(math.log(len(self.parameter_subspaces[0])))**-1

            for row_number in range(len(self.parameter_subspaces[1])):
                for column_number in range(len(self.parameter_subspaces[0])):
                    index = (row_number, column_number)
                    sub_gridspec = root_gridspec[index].subgridspec(2, 1, wspace=0.1, hspace=0)
                    axes = [fig.add_subplot(sub_gridspec[0]), fig.add_subplot(sub_gridspec[1])]

                    for ax in axes:
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                    
                    print("Data is: {}".format(self.neuron_time_series_mesh[index]))
                    
                    axes[0].plot(time_axis, self.neuron_time_series_mesh[index], linewidth=0.25, color='deeppink', label='Vm') # Plot the membrane potential time series
                    axes[1].plot(time_axis, self.LFP_time_series_mesh[index], linewidth=0.25, color='deeppink', label='LFP') # Plot the theta rhythm (or other neural input signal) time series

                    for boundary_index in self.cycle_boundary_indices_mesh[index]:
                        axes[1].vlines(time_axis[boundary_index], min(self.LFP_time_series_mesh[index]), max(self.LFP_time_series_mesh[index]), linestyle='dashed', color='k', linewidth=0.25)

                    if index[0] == self.neuron_time_series_mesh.shape[0] - 1:
                        axes[1].set_xlabel("$Time$ $(ms)$", fontsize=font_size)
                    if index[1] == 0:
                        axes[0].set_ylabel("$V_m$ $(mV)$", fontsize=font_size)
                        axes[1].set_ylabel("$LFP$ $(mV)$", fontsize=font_size)
            
            fig.text(0.5, 0.02, self.parameter_subspace_names[0].replace("_", " ") + " " + self.parameter_subspace_units[0], ha='center')
            fig.text(0.04, 0.5, self.parameter_subspace_names[1].replace("_", " ") + " " + self.parameter_subspace_units[1], va='center', rotation='vertical')

            if self.mesh_fig_time_series_suppress == False:
                plt.show()
            if self.save_mesh_fig_time_series == True:
                manager = plt.get_current_fig_manager()
                manager.resize(*manager.window.maxsize())
                figureObject = plt.gcf()
                figureObject.savefig(Path('/'.join([self.saved_figure_path, 'meshTimeSeries', 'HPC_phi_meshTimeSeries_{}{}{}'.format(date.today().day, date.today().month, self.mesh_fig_time_series_file_type)])), dpi=self.dpi, bbox_inches='tight')

    def updateEncodePath(self, update_filename=False): 
        """Generates string filename for serialized data according to format: 
        
        _<parameter_1>_<parameter_2>_..._<parameter_n>_initialParam_1_finalParam_1_subspaceSize_1_initialParam_2_finalParam_2_subspaceSize_2_..._initialParam_n_finalParam_n_subspaceSize_n_ddmmyyyy_iteration_number.pb.bin
        Which looks disgusting but is, in fact, exactly as descriptive as is needed for easy loading and (in the future) using string formatting to load large swathes of saved data.
        """

        # If there is no preexisting filename to simply modify
        if not update_filename:  

            self.proto_buffer_filename = ""

            # Add all parameter subspace names
            for parameter_name in self.parameter_subspace_names:
                self.proto_buffer_filename = '_'.join([self.proto_buffer_filename, parameter_name])

            # Add all parameter subspace maxes, mins, sizes
            for vector in self.parameter_subspaces:
                self.proto_buffer_filename = '_'.join([self.proto_buffer_filename, str(int(vector[0])), str(int(vector[-1])), str(len(vector))])
            
            # Add date information, iteration number and .pb.bin extension
            d = date.today()
            self.proto_buffer_filename = '_'.join([self.proto_buffer_filename, "{:02d}{:02d}{:02d}".format(d.day, d.month, d.year)])
            self.proto_buffer_filename = '_'.join([self.proto_buffer_filename, "{}.pb.bin".format(self.iteration_number)])

            # Update the protocol buffer encode path with the new filename
            self.protobuf_encode_path = Path('/'.join([self.protobuf_serialized_data_path, self.proto_buffer_filename]))

        # If a filename has been previously specified over runtime
        else:
            
            # Split by underscore
            underscore_separated_segments = self.proto_buffer_filename.split('_')

            # Split by periods
            period_separated_segments = underscore_separated_segments[-1].split('.')

            # The first element is the iteration number -- update it
            period_separated_segments[0] = str(self.iteration_number)

            # Recombine segments separated by period
            recombined_period_separated_segments = '.'.join(period_separated_segments)

            # Append period-separated segments to underscore separated segments
            underscore_separated_segments[-1] = recombined_period_separated_segments

            # Recombine segments separated by underscore
            self.proto_buffer_filename = '_'.join(underscore_separated_segments)

            # Update the protocol buffer encode path with the new filename
            self.protobuf_encode_path = Path('/'.join([self.protobuf_serialized_data_path, self.proto_buffer_filename]))
            
    def meshSweep(self):
        
        # Initialize output tensors 
        self.initializeOutputArrays()

        # Initialize a protocol buffer filename and encode path
        self.updateEncodePath(update_filename=False)

        # If the full parameter subspace has size > 1
        if not self.single_iteration:

            self.overall_start_time = time.time()

            # Generate a list of coordinates corresponding to the index of each parameter subspace 
            parameterMesh = itertools.product(*[range(s) for s in self.dimension_lengths])
            
            # Iteratively simulate
            for parameter_space_coordinate in parameterMesh: 
                for dimension, subspace_index in enumerate(parameter_space_coordinate, start=0): # Coordinate index corresponds to single parameter subspace

                    self.__dict__.update({self.parameter_subspace_names[dimension]: self.parameter_subspaces[dimension][subspace_index]}) # Update parameter 
                    self.parameter_space_coordinate = parameter_space_coordinate

                # Update protocol buffer filename and encode path on second and subsequent iterations
                if self.iteration_number != 0:
                    self.updateEncodePath(update_filename=True)

                # Execute the simulation and populate output tensors for the current parameter set
                self.coordinate_mesh[parameter_space_coordinate] = parameter_space_coordinate
                self.output_mesh[parameter_space_coordinate], self.cycle_phi_central_tendencies[parameter_space_coordinate] = HPC_phi_simulation(self, execute_simulation=True) 

                # On the first iteration use the overall_start_time to report progress
                if self.iteration_number == 0:
                    print("===== {} m {} s to simulate iteration {} of {} =====".format(int((time.time() - self.overall_start_time)/60), round(((time.time() - self.overall_start_time)%60)/1, 2), self.iteration_number + 1, self.mesh_size))
                    subsequent_simulation_start_time = time.time() # Initialize a new timer for the second simulation
                
                # On subsequent iterations use and then reset the second timer to report progress
                else:
                    print("===== {} m {} s to simulate iteration {} of {} =====".format(int((time.time() - subsequent_simulation_start_time)/60), round(((time.time() - subsequent_simulation_start_time)%60)/1, 2), self.iteration_number + 1, self.mesh_size))
                    subsequent_simulation_start_time = time.time()

                # Increment iteration number 
                self.iteration_number += 1

            # Report completion and total time after iterating over entire parameter subspace
            print("\n===== ALL ITERATIONS FINISHED =====")
            print("===== {} m {} s for parameter mesh sweep =====".format(int((time.time() - self.overall_start_time)/60), round(((time.time() - self.overall_start_time)%60)/1, 2)))

            # Print the quantification metric in an array where entries preserve relative location of points on contour plot and mesh figures (flipped and transposed)
            print('Output mesh: \n{}'.format(np.flipud(np.transpose(self.output_mesh))))
            
            # If exactly two linear subspaces were specified then the simulation set is eligible for contour plot and mesh figure output
            if len(self.parameter_subspaces) == 2:
                self.generateMeshFigures()

        else:
            self.overall_start_time = time.time() 

            if self.parameter_subspaces:
                for dimension in range(len(self.parameter_subspace_names)):
                    self.__dict__.update({self.parameter_subspace_names[dimension]: self.parameter_subspaces[dimension][0]})

            phase_drift_quantification, cycle_phi_central_tendencies = HPC_phi_simulation(self) 
            print("===== {} m {} s to simulate iteration {} of {} =====".format((time.time() - self.overall_start_time)/60, ((time.time() - self.overall_start_time)%60)/1, 1, 1)) # These magic 1's make print statement indicate that simulation 1 of 1 is complete
            print("Phase drift quantification: {}".format(phase_drift_quantification))

    def analyzeProtobufData(self):

        self.overall_start_time = time.time()
        self.iteration_number = 0

        for coordinate, protobuf_file in np.ndenumerate(self.sorted_protobuf_files_to_decode):

            self.protobuf_decode_path = Path('/'.join([self.protobuf_serialized_data_path, protobuf_file]))  
            self.parameter_space_coordinate = coordinate

            self.output_mesh[self.parameter_space_coordinate], self.cycle_phi_central_tendencies[self.parameter_space_coordinate] = HPC_phi_simulation(self, execute_simulation=False) 

            # On the first iteration use the overall_start_time to report progress
            if self.iteration_number == 0:
                print("===== {} m {} s to decode iteration {} of {} =====".format(int((time.time() - self.overall_start_time)/60), round(((time.time() - self.overall_start_time)%60)/1, 2), self.iteration_number + 1, self.num_files_to_decode))
                subsequent_simulation_start_time = time.time() # Initialize a new timer for the second simulation
            
            # On subsequent iterations use and then reset the second timer to report progress
            else:
                print("===== {} m {} s to decode iteration {} of {} =====".format(int((time.time() - subsequent_simulation_start_time)/60), round(((time.time() - subsequent_simulation_start_time)%60)/1, 2), self.iteration_number + 1, self.num_files_to_decode))
                subsequent_simulation_start_time = time.time()

            # Increment iteration number 
            self.iteration_number += 1  
    
        # Report completion and total time after decoding all requested files
        print("\n===== ALL ITERATIONS FINISHED =====")
        print("===== {} m {} s to decode and analyze all files =====".format(int((time.time() - self.overall_start_time)/60), round(((time.time() - self.overall_start_time)%60)/1, 2)))
 
        # Print the quantification metric in an array where entries preserve relative location of points on contour plot and mesh figures (flipped and transposed)
        print('Output mesh: \n{}'.format(np.flipud(np.transpose(self.output_mesh))))
        
        # If exactly two parameters were included in the decoded simulations, then the simulation set is eligible for contour plot and mesh figure output
        if len(self.protobuf_mesh_dimension) == 2:
            self.generateMeshFigures()    
class LIFNeuron():
    """Linear 'leak' function leaky integrate-and-fire neuron model with (optional) stochastic input. Defined according to Langevin equation [5.3] in https://neuronaldynamics.epfl.ch/online/Ch5.S1.html
    
    Model parameters are passed as an unpacked dictionary upon instantiation. The method .iterate is separate from ._updateVm for future-proofing; maybe we'll want to add more state variables later with 
    separate ._update methods. To eliminate noise, set parameter sigma=0.
    """

    def __init__(self, **kwargs):

        # Specify strings corresponding to attributes in whatever class is being used to initialize the neuron model
        model_parameter_names = ['rest_Vm', 'spike_Vm', 'rest_Vm', 'neuron_threshold', 'membrane_time_constant', 'absolute_refractory_period', 'membrane_resistance', 'dt', 'sigma', 'mu', 'adaptation_decay_constant', 'adaptation_time_constant', 'adaptation_response_constant']
        self.__dict__.update({key: kwargs[key] for key in kwargs.keys() if key in model_parameter_names})

        self.Vm = self.rest_Vm # resting membrane potential starts at resting

        self.stochastic_input = 0 # Stochastic input starts at 0

        self.spike_state = False # Neuron assumed to be subthreshold at initial condition

        self.refractory_period_length = self.absolute_refractory_period//self.dt # Number of timesteps corresponding to absolute refractory period
        self.refractory_period_counter = 0 

        self.adaptation_decay_constant = self.adaptation_decay_constant/self.adaptation_time_constant
        self.W_current = 0

    def updateVm(self, time_series_index, forcing, dt):

        # If the neurn is labelled subthreshold with suprathreshold potential
        if self.Vm >= self.neuron_threshold and self.spike_state == False: 
            
            self.Vm = self.spike_Vm # Make the neuron spike
            self.spike_state = True # label the neuron as suprathreshold

        # If the neuron is labelled as suprathreshold and the absolute refractory period is incomplete
        elif self.spike_state == True and self.refractory_period_counter < self.refractory_period_length: 
            
            self.refractory_period_counter += 1 # Increment the refractory period
       
        # If the neuron is labelled as suprathreshold and the absolute refractory period is complete
        elif self.spike_state == True and self.refractory_period_counter >= self.refractory_period_length: 
            
            self.Vm = self.rest_Vm # Return potential to resting
            self.spike_state = False # Label the neuron as subthreshold
            self.refractory_period_counter = 0 # Reset the refractory period for the next spike

       # If the neuron is labelled as subthreshold with subthreshold potential 
        else:

            self.VL = -(self.Vm - self.rest_Vm) # Compute leak contribution
            self.V_forcing = forcing*self.membrane_resistance # Compute forcing (input) contribution
            self.stochastic_input = self.sigma*math.sqrt(self.dt)*np.random.normal(loc=self.mu, scale=1.0)  # Compute stochastic contribution (Ornstein-Uhlenbeck process)

            voltageSuperposition = self.VL + self.V_forcing - self.W_current*self.membrane_resistance # Combine all contributions to voltage increment, including adaptation for spike frequency attenuation

            self.Vm += ((dt*voltageSuperposition)/self.membrane_time_constant + self.stochastic_input) # Scale by the timestep and membrane time constant, then integrate (in the mathematical sense)

    def updateAdaptation(self, dt):

        if self.spike_state == True:
            self.adaptation_increase = self.adaptation_response_constant
        else:
            self.adaptation_increase = 0

        self.W_current += dt*(-self.adaptation_decay_constant + self.adaptation_increase)

    def iterate(self, time_series_index, forcing, dt):

        # Update adaptation variables
        self.updateAdaptation(dt)

        # Update membrane potential
        self.updateVm(time_series_index, forcing, dt) 
class Simulation():
    """Class which progresses LIFNeuron simulation and stores output.

    Specific to LIFNeuron class in HPC_phi application.
    """
    def __init__(self, LIFNeuron, **kwargs):
        
        if 'dummy_model' in kwargs.keys():

            # Prevents operator from executing simulation without loading a model
            self.disable_simulation = True

        else:

            # Specify keys to be included in the initialization message to the operator
            keys_of_interest = ['theta_amplitude', 'interference_amplitude', 'theta_frequency', 'interference_frequency', 'rest_Vm', 'spike_Vm', 'neuron_threshold', 'membrane_time_constant', 'absolute_refractory_period']
            self.parameter_subset_to_print = {key: kwargs[key] for key in keys_of_interest}

            # Set model substrate of Simulation class instance
            self.model = LIFNeuron
            self.disable_simulation = False

    def initializeOutput(self, num_timesteps, dt):

        # Initialize vectors which will contain the simulation output
        self.time_axis = np.arange(num_timesteps) * dt 
        self.Vm_t = np.empty(num_timesteps) 
        self.spike_times = np.zeros(num_timesteps)

    def runSim(self, forcing_function, theta_rhythm, num_timesteps, dt):
        if not self.disable_simulation:
            # Create vectors to contain output
            self.initializeOutput(num_timesteps, dt) 

            # Keep input functions as attributes for downstream referral
            self.dual_oscillator_rhythm = forcing_function
            self.theta_rhythm = theta_rhythm

            #Print initialization message
            print("\nInitialization complete...") 
            print("Simulating {} ms\nParameters: \n{}".format(forcing_function.shape[0]*dt, self.parameter_subset_to_print))  
            
            # At each timestep
            for time_series_index in range(num_timesteps):

                self.model.iterate(time_series_index, forcing_function[time_series_index], dt) # Integrate model over dt
                self.Vm_t[time_series_index] = self.model.Vm # Keep track of Vm at time t

                # Keep track of when spikes occur (treat as pseudo-instantaneous)
                if self.model.spike_state == True and self.model.refractory_period_counter == 0:
                    self.spike_times[time_series_index] = 1 
            
            print("Simulation completed") 

def extractSpikePhi(SimulationIterationManager, Simulation, execute_simulation=True):
    """Takes spike times from a completed simulation and maps each to a phase value phi on theta_rhthm.

    Assumes that theta_rhythm is a sinusoid, specifically a sin() wave function. Afterwards, uses a protocol buffer to serialize data, if requested. If no simulation was executed and the 
    operator requested, decode one set of phi values via protocol buffer for analysis.
    """

    # If this function is being used in the simulate-and-analyze pipeline
    if execute_simulation == True:

        # Extract phase time series of sinusoid
        theta_phase_time_series_INTERMEDIATE = list(map(lambda z: np.arctan2(z.imag, z.real - SimulationIterationManager.LFP_shift), hilbert(Simulation.theta_rhythm))) # For each element in analytic signal, take arctan2 which outputs angle on interval (-pi, pi]  
        theta_phase_time_series = list(map(lambda phi: phi + 2*np.pi if phi < 0 else phi, theta_phase_time_series_INTERMEDIATE)) # For each angle on (-pi, pi], add 2*pi if phi < 0 to shift the interval to [0, 2*pi)

        del theta_phase_time_series_INTERMEDIATE 

        spike_phi = list(map(lambda i, j: j*theta_phase_time_series[i], range(len(Simulation.spike_times)), Simulation.spike_times))

        # If operator requested a protobuf encode for the simulation
        if SimulationIterationManager.encode_simulation_output == True:

            # Initialize the message as defined in the .proto file
            SimulationData_message = ProtoBufferInterface()

            # All parameters defined in global INPUT_SPECIFICATION_DICTIONARY are encoded as a map via the repeated InputParameter message field
            for key in SimulationIterationManager.__dict__.keys():

                if key in INPUT_SPECIFICATION_DICTIONARY.keys():
                    
                    parameter_message = SimulationData_message.parameters.add()
                    parameter_message.key = key
                    parameter_message.value = SimulationIterationManager.__dict__[key]

            # Encode theta and dual oscillator input time series via the optional InputRhythms message field
            input_rhythms_message = SimulationData_message.input_rhythms
            input_rhythms_message.theta_rhythm[:] = Simulation.theta_rhythm
            input_rhythms_message.dual_oscillator_rhythm[:] = Simulation.dual_oscillator_rhythm

            # Encode the simulation time axis and model membrane potential time series via the optional RawSimulationOutput message field
            raw_simulation_output_message = SimulationData_message.raw_simulation_output
            raw_simulation_output_message.time_axis[:] = Simulation.time_axis
            raw_simulation_output_message.Vm_t[:] = Simulation.Vm_t

            # Encode ordered phi values for each spike time and the theta rhythm phase over time via the optional RefinedSimulation Output message field
            refined_simulation_output_message = SimulationData_message.refined_simulation_output
            refined_simulation_output_message.spike_phi[:] = spike_phi
            refined_simulation_output_message.theta_phase[:] = theta_phase_time_series

            # Open the encode directory and serialize the data in binary
            with open(SimulationIterationManager.protobuf_encode_path, 'wb') as file_directory:
                file_directory.write(SimulationData_message.SerializeToString())

            del SimulationData_message

    # If this function is being used in the decode-and-analyze pipeline
    elif SimulationIterationManager.decode_simulation_output == True:
        
        # Initialize the message as defined in the .proto file
        SimulationData_message = ProtoBufferInterface()

        # Open the encode directory and serialize the data in binary
        with open(SimulationIterationManager.protobuf_decode_path, 'rb') as file_directory:
            SimulationData_message.ParseFromString(file_directory.read())

        # Generate a dictionary containing all of the parameters from the decoded simulation INPUT_PARAMETER_DICTIONARY
        decoded_parameter_dictionary = {}
        for parameter in SimulationData_message.parameters:
            decoded_parameter_dictionary[parameter.key] = parameter.value

        # Update SimulationIterationManager attributes with those from the loaded simulation to be retrieved during analysis
        SimulationIterationManager.__dict__.update(decoded_parameter_dictionary)

        # Update the Simulation object attributes with the model input rhythms to be retrieved during analysis
        Simulation.theta_rhythm = SimulationData_message.input_rhythms.theta_rhythm
        Simulation.dual_oscillator_rhythm = SimulationData_message.input_rhythms.dual_oscillator_rhythm

        # Update the Simulation object attributes to contain the solution and time axis to the decoded simulation (in a way behaving as though it just ran a simulation)
        Simulation.time_axis = SimulationData_message.raw_simulation_output.time_axis
        Simulation.Vm_t = SimulationData_message.raw_simulation_output.Vm_t

        # Decode the typical return variables from this function
        spike_phi = SimulationData_message.refined_simulation_output.spike_phi
        theta_phase_time_series = SimulationData_message.refined_simulation_output.theta_phase

        # Notify the operator of the successful decode
        print("Simulation data decoded from protocol buffer...")
        print("Parameters:\n{}".format(decoded_parameter_dictionary))

        del SimulationData_message

    try:
        return spike_phi, theta_phase_time_series
    except:
        raise Exception('No data for analysis; specify whether to generate or decode data')
def computeCyclePhiCentralTendency(SimulationIterationManager, theta_phase_time_series, spike_phi):
    """For each cycle on the phase time series, use some operator-specified measure of central tendency to return a single value of phi.

    Supports mean, median, mode and kernel density estimation. These choices are mostly identical from an algorithmic standpoint.
    """

    spike_phi = np.array(spike_phi)

    # Dictionary of currently supported measures of central tendency -- algorithmically these are all essentially identical
    central_tendency_measure_dictionary = {"0": 'mean',
                                           "1": 'median',
                                           "2": 'mode',
                                           "3": 'KDE'}

    central_tendency_mode = central_tendency_measure_dictionary[str(SimulationIterationManager.central_tendency_mode)]

    # If the operator would like to rotate the reference frame to adjust the cycle boundaries (which are themselves associated with interval [0, 2*np.pi))
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
    """Uses kernel density estimation to approximate the observation probability density function on some interval, then returns argmax as measure of central tendency.

    Kernel used is Gaussian, bandwidth optimized using leave-one-out cross validation algorithm. If there is only one data point, bandwidth is set to default as specified by operator in
    SimulationIterationManager class attributes.
    """

    data_2D = np.reshape(data, (-1, 1)) # The backend requires a 2D array of shape (N, 1), where N is the number of samples

    if np.size(data_2D) > 1: # If we have greater than 1 value then we can do leave-one-out cross-validation

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
        ax.set_xlabel("$\phi$ $(radians)$", font_size=13) 
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
def plotSolution(SimulationIterationManager, Simulation, cycle_boundary_indices):
    """Simply plots the simulation solution (membrane potential) and theta_rhythm with cycle boundaries demarcated as computed from rotated frame of reference in computePhiCentralTendency().
    
    Optionally, saves the figure to the path specified as a SimulationIterationManager class attribte. File extension is .png, with dpi specified as path.
    """
    if not SimulationIterationManager.sim_fig_suppress or SimulationIterationManager.save_sim_fig:
        
        plt.close()

        fig, axes = plt.subplots(nrows=2, sharex=True, sharey=False) # Two graps, two rows, linked x-axes
        fig.patch.set_alpha(0.0) # Make figure background transparent

        for ax in axes:
            ax.spines['top'].set_visible(False) # Make the top figure border invisible
            ax.spines['right'].set_visible(False)# Make the right figure border invisible

        axes[0].plot(Simulation.time_axis, Simulation.Vm_t, linewidth=0.8, color='deeppink', label='Vm') # Plot the membrane potential time series
        axes[1].plot(Simulation.time_axis, Simulation.theta_rhythm, linewidth=0.8, color='deeppink', label='LFP') # Plot the theta rhythm (or other neural input signal) time series

        for boundary_index in cycle_boundary_indices:
            axes[1].vlines(Simulation.time_axis[boundary_index], min(Simulation.theta_rhythm), max(Simulation.theta_rhythm), linestyle='dashed', color='k', linewidth=0.8)

        axes[0].set_ylabel("$V_m$ $(mV)$") # y-axis label of membrane potential time series, in millivolts
        axes[1].set_ylabel("$Amplitude$ $(pA)$") # y-axis label of theta rhythm
        axes[1].set_xlabel("$Time$ $(ms))$") # Label the x-axis only on the bottommost plot in milliseconds
        
        if SimulationIterationManager.save_sim_fig:
            plt.savefig(Path('/'.join([SimulationIterationManager.saved_figure_path, 'timeSeries', 'HPC_phi_iteration{}.png'.format(SimulationIterationManager.iteration_number)])), dpi=SimulationIterationManager.dpi, bbox_inches='tight') 
        if not SimulationIterationManager.sim_fig_suppress:
            plt.show()
        
    if SimulationIterationManager.mesh_fig_time_series_suppress == False or SimulationIterationManager.save_mesh_fig_time_series == True:
        num_timesteps_to_plot = int(3000/SimulationIterationManager.dt)
        first_cycle_boundary_of_interest = len(Simulation.time_axis) - num_timesteps_to_plot - 1
        if len(Simulation.time_axis) >= num_timesteps_to_plot:
            SimulationIterationManager.neuron_time_series_mesh[SimulationIterationManager.parameter_space_coordinate] = Simulation.Vm_t[-num_timesteps_to_plot:]
            SimulationIterationManager.LFP_time_series_mesh[SimulationIterationManager.parameter_space_coordinate] = Simulation.theta_rhythm[-num_timesteps_to_plot:]
            for index, cycle_boundary in enumerate(cycle_boundary_indices, start=0):
                if cycle_boundary >= first_cycle_boundary_of_interest:
                    start_index = index
                    break
            SimulationIterationManager.cycle_boundary_indices_mesh[SimulationIterationManager.parameter_space_coordinate] = cycle_boundary_indices[start_index:]

        else:
            SimulationIterationManager.neuron_time_series_mesh[SimulationIterationManager.parameter_space_coordinate] = Simulation.Vm_t[:]
            SimulationIterationManager.LFP_time_series_mesh[SimulationIterationManager.parameter_space_coordinate] = Simulation.theta_rhythm[:]
            SimulationIterationManager.cycle_boundary_indices_mesh[SimulationIterationManager.parameter_space_coordinate] = cycle_boundary_indices[:]
def constructReturnMap(SimulationIterationManager, cycle_phi_central_tendencies):
    """Computes vector of phi_{k-1} - phi_{k} for one simulation, then optionally plots these points with line of identity for reference
    """
    try:
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
    except IndexError:
        print("Inputs did not elicit spiking over simulation duration...")
        print("Phase drift quantification not defined")
        return "NO_SPIKES"

def HPC_phi_simulation(SimulationIterationManager, execute_simulation=True):
    """Outer pipeline controls execution of simulation, then calls for analysis and data encoding/decoding
    """
    # If called as part of the simulate-and-analyze pipeline: Simulate!
    if execute_simulation == True:

        num_timesteps = int(SimulationIterationManager.simulation_duration/SimulationIterationManager.dt) 

        # Dual oscillator input 
        if SimulationIterationManager.constant_input == False:
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

        # If we intend to inject a constant current rather than an oscillatory one
        if SimulationIterationManager.constant_input == True:
            field_oscillation = np.array([SimulationIterationManager.theta_amplitude]*num_timesteps)
            theta_rhythm = field_oscillation

        # Instantiate LIFNeuron
        Neuron = LIFNeuron(**SimulationIterationManager.__dict__)

        # Create and run simulation
        NeuronSim = Simulation(Neuron, **SimulationIterationManager.__dict__)
        NeuronSim.runSim(field_oscillation, theta_rhythm, num_timesteps, SimulationIterationManager.dt)
    
    # If called as part of the decode-and-analyze pipeline: Set simulation objects as NoneType and proceed
    else:
        dummy_model = None
        NeuronSim = Simulation(dummy_model, **{"dummy_model": None})
   
    # Run encode/decode and analysis as requested by operator
    simulation_quantification, cycle_phi_central_tendencies = simulationAnalysis(SimulationIterationManager, NeuronSim, execute_simulation=execute_simulation)

    return simulation_quantification, cycle_phi_central_tendencies
def simulationAnalysis(SimulationIterationManager, Simulation, execute_simulation=True):
    """Analysis pipeline extracts phi values, reduces them to a single value per cycle, plots simulation solution/return map and then reduces entire simulation to a single value
    """
    # Compute simulation phi sequence
    spike_phi, theta_phase_time_series = extractSpikePhi(SimulationIterationManager, Simulation, execute_simulation=execute_simulation)

    # Compute phi central tendencies on each theta cycle and construct return map
    cycle_phi_central_tendencies, cycle_boundary_indices = computeCyclePhiCentralTendency(SimulationIterationManager, theta_phase_time_series, spike_phi)

    # Plot simulation output and Return map
    plotSolution(SimulationIterationManager, Simulation, cycle_boundary_indices)

    # Get reverse discrete differences of cycle phi with optional return map plot
    return_map_coordinate_differences = constructReturnMap(SimulationIterationManager, cycle_phi_central_tendencies)

    # If False then no spikes occurred
    if type(return_map_coordinate_differences) != str:

        # Use KDE to reduce reverse differences to a single value: argmax(KDE)
        simulation_quantification = kernelDensityEstimation(return_map_coordinate_differences, SimulationIterationManager, cycle_interval=False)

        return simulation_quantification, cycle_phi_central_tendencies

    # Try returning NoneType objects; most likely will only cause problems during contour and mesh figs   
    else:
        return None, []

def main():
   
    global INPUT_SPECIFICATION_DICTIONARY
    INPUT_SPECIFICATION_DICTIONARY = {# Time parameters
                                      'dt': 0.01,                                   
                                      'simulation_duration': 1000,  
        
                                      # Dual oscillator input parameters
                                      'theta_amplitude': 100,                       
                                      'interference_amplitude': 100,                
                                      'theta_frequency': 8,                         
                                      'interference_frequency': 9,                
                                      'LFP_shift': 0,
                                      'constant_input': False,                                
                 
                                      # Langevin model properties
                                      'rest_Vm': -75,                                
                                      'spike_Vm': 50,                               
                                      'neuron_threshold': -35,                      
                                      'membrane_time_constant': 10,                 
                                      'membrane_resistance': 1,                     
                                      'absolute_refractory_period': 2,              
                                      'sigma': 1.0,
                                      'mu': 0.3,
                                      'adaptation_decay_constant': 1,
                                      'adaptation_time_constant': 10,
                                      'adaptation_response_constant': 3}

    global OUTPUT_SPECIFICATION_DICTIONARY
    OUTPUT_SPECIFICATION_DICTIONARY = {# Analysis parameters
                                       'verbose': False,
                                       'central_tendency_mode': 3,                  
                                       'theta_cycle_boundary_phase': 3*np.pi/2,      
                                       'kernel_bandwidth': 1.5, 

                                       # Figure popup suppression options 
                                       'sim_fig_suppress': True,                                          
                                       'return_map_suppress': True,                                           
                                       'central_tendency_PDF_estimation_suppress': True,                                       
                                       'return_map_PDF_estimation_suppress': True,                                           
                                       'contour_plot_suppress': True,                                       
                                       'mesh_fig_suppress': True, 
                                       'mesh_fig_time_series_suppress': False,                                        
                                       
                                       # Figure saving options
                                       'saved_figure_path': '/'.join([os.getcwd(), 'modelFigures']),
                                       'dpi': 300,
                                       'save_sim_fig': False,                       
                                       'save_return_map': False,                     
                                       'save_contour_plot': False,                    
                                       'save_mesh_fig': False, 
                                       'save_mesh_fig_time_series': True,                                       
                                       
                                       # Figure filetype options
                                       'contour_plot_file_type': '.svg',              
                                       'mesh_fig_file_type': '.svg',
                                       'mesh_fig_time_series_file_type': '.png',
                                       
                                       # Protocol buffer options
                                       'protobuf_serialized_data_path': '/'.join([os.getcwd(), 'serializedSims']),
                                       'encode_simulation_output': False,
                                       'decode_simulation_output': False,
                                       'protobuf_files_to_decode': ['_theta_amplitude_interference_amplitude_10_10_1_10_10_1_09062021_0.pb.bin'],
                                       'protobuf_mesh_dimension': 2}  

    # Instantiate program specification objects
    InputSpecificationsObject = InputSpecifications(**INPUT_SPECIFICATION_DICTIONARY)
    OutputSpecificationsObject = OutputSpecifications(**OUTPUT_SPECIFICATION_DICTIONARY)

    # Add to and seal the parameter subspace to iterate over
    InputSpecificationsObject.newParameterSubspace('theta_amplitude', '(mV)', lambda q: q*10, 1, 1, 2)
    InputSpecificationsObject.newParameterSubspace('interference_amplitude', '(mV)', lambda q: q*10, 1, 1, 2)
    InputSpecificationsObject.sealParameterSubspace()

    # Pass specifications into the iteration manager, then run analysis through the selected 
    SimulationSet = SimulationIterationManager(InputSpecificationsObject, OutputSpecificationsObject)

    if SimulationSet.decode_simulation_output == False:
        SimulationSet.meshSweep()
    else:
        SimulationSet.analyzeProtobufData()

if __name__ == '__main__':
    main()