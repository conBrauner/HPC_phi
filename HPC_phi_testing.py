# %%
import os
import time
from datetime import datetime, date
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from numpy.core.fromnumeric import amax, amin
from scipy.signal import hilbert
from scipy.stats import mode
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut

# Define the neuron and simulation class objects
class LIFNeuron:
    """
    Linear 'leak' function leaky integrate-and-fire neuron model. Defined according to [5.3] in https://neuronaldynamics.epfl.ch/online/Ch5.S1.html
    Model parameters are passed in as a dictionary upon instantiation with keys corresponding to those used in __init__
    The method .iterate is separate from ._updateVm for future-proofing; maybe we'll want to add more state variables later with separate ._update methods
    """
    def __init__(self, parameterDictionary):
        self.modelType = 'LIF' # String recognized by Simulation class object to select spike time determination algorithm
        self.parameters = parameterDictionary # Main argument of class instantiation

        # Initialize model with specified parameters, plainly named
        self.rest_Vm = parameterDictionary['rest_Vm'] 
        self.spike_Vm = parameterDictionary['spike_Vm']
        self.Vm = parameterDictionary['rest_Vm']
        self.neuron_threshold = parameterDictionary['neuron_threshold']
        self.membrane_time_constant = parameterDictionary['membrane_time_constant']
        self.absolute_refractory_period = parameterDictionary['absolute_refractory_period']
        self.membrane_resistance = parameterDictionary['membrane_resistance']
        self.dt = parameterDictionary['dt']

        # Parameters for implementing stochastic input current component, see: https://neuronaldynamics.epfl.ch/online/Ch8.S1.html for technical details (Langevin equation, white noise input)
        self.noise = parameterDictionary['noise']
        self.sigma = parameterDictionary['sigma']
        self.stochastic_input = 0

        # Indicates if a spike occurred at the last time step 
        self.spikeState = False

        # Set parameters of the refractory period
        self.refractory_period_length = self.absolute_refractory_period//self.dt # Use the timstep to convert a desired refractory period duration (ms) to length (indices) 
        self.refractory_period_counter = 0 # Initialize the model as being 'zero steps' along the refractory period
    def _updateVm(self, time_series_index, forcing, dt):
        """
        Increments model membrane potential according to the sum of all inputs.
        - Sets potential to self.spike_Vm if self.neuron_threshold is met or exceeded, then increments the refractory period counter on subsequent iterations
            * After refractory period counter corresponds to a duration equal to or greater than the absolute refractory period, set membrane potential to self.rest_Vm and continue to integrate input
        - Integrates an Ornstein-Uhlenbeck process as a stochastic input current; see equation (8.8): https://neuronaldynamics.epfl.ch/online/Ch8.S1.html 
            * variance of dW is proportionate to dt, so multiplying sqrt(dt) (a standard deviation) with a random number drawn from a Gaussian distribution of mean = 0 and unit variance
        - A noisy LIF neuron with this type of stochastic input is formulated as the Langevin equation, an example of a stochastic differential equation (SDE); see: http://ft-sipil.unila.ac.id/dbooks/AN%20INTRODUCTION%20TO%20STOCHASTIC%20DIFFERENTIAL%20EQUATIONS%20VERSION%201.2.pdf
        """
        if self.Vm >= self.neuron_threshold and self.spikeState == False: # Check if threshold met/exceeded + if the last step was a spike
            self.Vm = self.spike_Vm # Jump to spiking potential if criteria met
            self.spikeState = True # Tag the current time step as having been a spike event
        elif self.spikeState == True and self.refractory_period_counter < self.refractory_period_length: # If the neuron is in the spiking state but has not finished it's refractory period
            self.refractory_period_counter += 1 # Move one step through the refractory period
        elif self.spikeState == True and self.refractory_period_counter >= self.refractory_period_length: # Check if the last time step was a spike and if the refractory period is over
            self.Vm = self.rest_Vm # If so, return to resting potential
            self.spikeState = False # Change the state to reflect the potential reset
            self.refractory_period_counter = 0 # Rest the refractory period
        else: # If the model is currently subthreshold
            self.VL = -(self.Vm - self.rest_Vm) # Compute the decrement in potential change due to the leak
            self.V_forcing = forcing*self.membrane_resistance # Compute contribution from forcing function
            voltageSuperposition = self.VL + self.V_forcing # Add the two contributions
            if self.noise == True:
                self.stochastic_input = self.sigma*math.sqrt(self.dt)*np.random.normal(loc=0.0, scale=1.0)
            self.Vm += ((dt*voltageSuperposition)/self.membrane_time_constant + self.stochastic_input)  # Integrate! (In the mathematical sense)
    def iterate(self, time_series_index, forcing, dt):
        """
        Called by a Simulation object to update all model state variables 
        """
        self._updateVm(time_series_index, forcing, dt) # Increment the membrane potential according to the differential equation
class Simulation:
    """
    Object for running a model simulation. Model of interest is passed into Simulation object upon instantiation
    This class makes reference to hardcoded attributes of it's substrate model, which both limits usability and is a likely source of error during refactoring
    Along with this, the output sequences and spike timing determination method are model specific, and adjustments will have to be made to accomodate unfamiliar models
    """
    def __init__(self, model, parameterDict):
        keys_of_interest = ['theta_amplitude', 'interference_amplitude', 'theta_frequency', 'interference_frequency', 'rest_Vm', 'spike_Vm', 'neuron_threshold', 'membrane_time_constant', 'absolute_refractory_period']
        self.parameter_subset_to_print = {x:parameterDict[x] for x in keys_of_interest}
        self.model = model
    def initializeOutput(self, num_timesteps, dt):
        """
        Initializes arrays to contain temporary attributes of LIFNeuron after each iteration
        """
        self.timeAxis = np.arange(num_timesteps) * dt # Each entry is cumulative time elapsed in ms
        self.Vm_t = np.empty(num_timesteps) # Vm_t here is read: 'membrane potential as a function of time' and is the principle solution of a neuron model
        self.spike_times = np.zeros(num_timesteps) # Binary sequence indicating if a solution index corresponds to a spike; 1 for yes, 0 for no
    def runSim(self, forcingFunction, num_timesteps, dt):
        """
        Initialize the simulation, print a summary, and iterate the model a number of times corresponding to a specified timestep over a specified duration
        """
        self.initializeOutput(num_timesteps, dt) # Initialize output axes to be of the specified length

        print("\nInitialization complete...") # Indicates to operator that model and output axes were defined successfully
        print("Simulating {} ms\nParameters: \n{}".format(forcingFunction.shape[0]*dt, self.parameter_subset_to_print)) # Indicates to operator how many ms are being simulated + the model parameters
        
        for time_series_index in range(num_timesteps): # For each time step in the desired simulation length for a given dt
            self.model.iterate(time_series_index, forcingFunction[time_series_index], dt) # Advance the model state variables by 1 time step
            self.Vm_t[time_series_index] = self.model.Vm # Append current membrane potential to solution
            if self.model.spikeState == True and self.model.refractory_period_counter == 0: # HARDCODED FOR LIF: Refer to model state to determine if current step corresponds to a spike
                self.spike_times[time_series_index] = 1 
        
        print("Simulation completed") # Indicate to operator that simulation is finished
class SimulationIterationManager:
    """
    Takes arbitrarily many strings corresponding to keys in the passed parameter dictionary and lists corresponding to the keys (by order) and containing each parameter to be simulated.
    - Taking each list to be a set of parameters, their cartesian product constitutes a mesh of N-dimensional coordinates to simulate over, where N is the number of variable parameters
    - When N = 2, pass strings indicating units of each variable parameter and corresponding to them, once again, by order. 
        * These units are used for axis labels in contour and mesh figures
    """
    def __init__(self, parameterDict, *args):
       
        self.defaultDictionary = parameterDict # Set the default parameters

        self.plot_contour = not parameterDict['contour_plot_suppress'] # Negation is required since control panel calls for whether or not to suppress the figure, but the class object queries whether or not to plot
        self.save_contour_plot = parameterDict['save_contour_plot'] # Whether or not to save a contour plot to the directory hardcoded in _meshFigs(); does not require that self.plot_contour be true
        self.contour_plot_file_type = parameterDict['contour_plot_file_type'] # A string indicating the file extension for the contour plot; includes the '.'; for example: '.svg', '.png' ...

        self.plot_mesh_maps = not parameterDict['mesh_fig_suppress'] # Negation is required since control panel calls for whether or not to suppress the figure, but the class object queries whether or not to plot
        self.save_mesh_fig = parameterDict['save_mesh_fig'] # Whether or not to save a grid of return maps to the directory hardcoded in _meshFigs(); does not require that self.plot_mesh_maps be true
        self.mesh_fig_file_type = parameterDict['mesh_fig_file_type'] # A string indicating the file extension for the contour plot; includes the '.'; for example: '.svg', '.png' ...
        
        parameter_name_list = [] # Will be filled with strings corresponding to keys in parameterDict which will change between iterations
        parameter_unit_list = []
        parameter_iterable_list = [] # Will be filled with lists where the cartesion product of these represents every coordinate in parameter space to simulate
        for arg in args:
            if type(arg) == str and not "(" in arg: # If the argument is a string
                parameter_name_list.append(arg) # Assume it is a parameterDict key
            elif type(arg) == str and "(" in arg: # If the argument is a string containing a open parenthesis
                parameter_unit_list.append(arg) # Assume the argument indicates the units of a iterable
            elif type(arg) == list: # Otherwise if it's a list
                parameter_iterable_list.append(arg) # Treat it as a set of parameters

        self.parameterNames = parameter_name_list # Set the parameters which will be modified between iterations
        self.parameterUnits = parameter_unit_list # Contains strings identifying units corresponding to each iterable by their order; only used for labeling axes in _meshFigs()
        self.iterables = parameter_iterable_list # Set the list of lists representing the N-dimensional parameter mesh where N is len(parameter_iterable_list)

        assert len(self.parameterNames) == len(self.iterables) # Assert that there is a key for each iterable
        if not self.parameterNames and not self.iterables: # If no variable parameters were specified
            self.singleIteration = True # Run only a single iteration, essentially skipping all class functions
        else:
            self.singleIteration = False # Iterate over the specified parameter space
    def _initializeOutputArrays(self):
        """
        Characterize the dimensions of the parameter space as inducible by the characteristics of the variable parameters; generate tensors to contain the return map metric, coordinates 
        as given by the cartesian product of all iterable indices and the central tendency of phi on each theta cycle for a given simulation.
        """
        mesh_shape = tuple(map(lambda x: len(x), self.iterables)) # maps the length of the ith iterable to the ith index of a tuple; specifies the shape of output tensors

        self.dimension_lengths = [] # Will contain the shape of the parameter mesh
        self.mesh_size = 1 # Initialize with the identity
        for dimension in self.iterables: # For each parameter to iterate over
                self.dimension_lengths.append(len(dimension)) # Take it's length to correspond to the shape in it's dimension
                self.mesh_size *= len(dimension) # The total number of coordinates in parameter space is the product of all axis lengths

        self.output_mesh = np.zeros(mesh_shape) # To contain return map metric at each coordinate
        self.coordinate_mesh = np.zeros(mesh_shape, dtype=(int, len(self.iterables))) # To contain the coordinate at each coordinate
        self.cycle_phi_central_tendencies = np.zeros(mesh_shape, dtype=(list)) # To contain the central tendency of phi on each theta cycle for the simulation at each coordinate
    def _meshFigs(self, dpi=300):
        """
        Plots a contour plot indicating the return map metric for each point in a discrete subset of parameter space, provided this subset is 2D.
        Plots a grid of return maps for each point in a discrete subset of parameter space, provided this subset is 2D
        - Directories are hardcoded in plt.savefig() calls
        """
        if self.plot_contour == True or self.save_contour_plot == True:

            plt.close()
            fig, ax = plt.subplots()
            shading = ax.contourf(self.iterables[0], self.iterables[1], np.transpose(self.output_mesh), cmap='cool', alpha=0.7)
            level_curves = ax.contour(self.iterables[0], self.iterables[1], np.transpose(self.output_mesh), [0], colors='black', linewidths=0.9, linestyles='dashed', alpha=0.75)
            #plt.clabel(level_curves, inline=True, fontsize=10)
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
            fig, axes = plt.subplots(nrows=len(self.iterables[1]), ncols=len(self.iterables[0]))

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
                        axes[row_number, column_number].set_xlabel(int(self.iterables[0][column_number]), fontsize=fontSize)
                    if column_number == 0:
                        axes[row_number, column_number].set_ylabel(int(self.iterables[1][- row_number - 1]), fontsize=fontSize)
            
            fig.text(0.5, 0.02, self.parameterNames[0].replace("_", " ") + " " + self.parameterUnits[0], ha='center')
            fig.text(0.04, 0.5, self.parameterNames[1].replace("_", " ") + " " + self.parameterUnits[1], va='center', rotation='vertical')

            figureObject = plt.gcf()

            if self.plot_mesh_maps == True:
                plt.show()
            if self.save_mesh_fig == True and self.mesh_fig_file_type == '.svg':
                figureObject.savefig('{}\\HPC_phi\\modelFigures\\meshFigs\\HPC_phi_meshFig_{}{}{}'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), date.today().month, date.today().day, self.mesh_fig_file_type), bbox_inches='tight')
            elif self.save_mesh_fig == True and self.mesh_fig_file_type == '.png':
                figureObject.savefig('{}\\HPC_phi\\modelFigures\\meshFigs\\HPC_phi_meshFig_{}{}{}'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), date.today().month, date.today().day, self.mesh_fig_file_type), dpi=dpi, bbox_inches='tight')
    def meshSweep(self):
        self._initializeOutputArrays()
        if not self.singleIteration:
            iteration_number = 0 # Initialize iteration number
            startTime = time.time() # Take the time at which the iteration started
            parameterMesh = itertools.product(*[range(s) for s in self.dimension_lengths]) # Returns tuples of every parameter mesh coordinate

            for parameter_space_coordinate in parameterMesh: # Iterate over every coordinate on the mesh

                for i, j in enumerate(parameter_space_coordinate): # i corresponds to the dimension and it's corresponding key, j corresponds to the index of the dimension's value

                    self.defaultDictionary[self.parameterNames[i]] = self.iterables[i][j] # key parameterDictionary with the ith variable and set to the jth entry of the ith dimension
                    self.defaultDictionary['iteration_number'] = iteration_number # Update iteration_number

                self.output_mesh[parameter_space_coordinate], self.cycle_phi_central_tendencies[parameter_space_coordinate] = HPC_phi_simulation(**self.defaultDictionary) # Run the simulation and analysis subroutine
                self.coordinate_mesh[parameter_space_coordinate] = parameter_space_coordinate

                if iteration_number == 0:
                    print("===== {} m {} s to simulate iteration {} of {} =====".format(int((time.time() - startTime)/60), round(((time.time() - startTime)%60)/1, 2), iteration_number + 1, self.mesh_size))
                    simStartTime = time.time()
                else:
                    print("===== {} m {} s to simulate iteration {} of {} =====".format(int((time.time() - simStartTime)/60), round(((time.time() - simStartTime)%60)/1, 2), iteration_number + 1, self.mesh_size))
                    simStartTime = time.time()

                iteration_number += 1 # Increment iteration index

            print("\n===== ALL ITERATIONS FINISHED =====")
            print("===== {} m {} s for parameter mesh sweep =====".format(int((time.time() - startTime)/60), round(((time.time() - startTime)%60)/1, 2)))

            print('Output mesh: \n{}'.format(np.flipud(np.transpose(self.output_mesh))))
            
            if len(self.iterables) == 2:
                self._meshFigs()
        else:
            startTime = time.time() # Take the time at which the iteration started
            HPC_phi_simulation(**self.defaultDictionary) # Run the simulation and analysis subroutine
            print("===== {} m {} s to simulate iteration {} of {} =====".format((time.time() - startTime)/60, ((time.time() - startTime)%60)/1, 1, 1)) # These magic 1's make print statement indicate that simulation 1 of 1 is complete

# Define functions involved in simulation construction and analysis
def forcingFunction(lambdaFunction, num_timesteps, dt):
    """
    Given a lambda function, passes in N inputs which increment by dt where N = num_timesteps. 
    Output is a list with length num_timesteps where each element solves y = lambdaFunction = f(t) for t on interval [0, dt*(num_timesteps - 1)]
    """
    return list(map(lambdaFunction, list(map(lambda t: t*dt, range(num_timesteps))))) # Note the nested function for scaling each element by dt
def packageParameters_LIFNeuron(rest_Vm, spike_Vm, neuron_threshold, membrane_time_constant, membrane_resistance, absolute_refractory_period, dt, noise, sigma):
    """
    Takes all required parameters for the LIFNeuron class and forms a dictionary with hardcoded keys. 
    Values are assigned as class attributes during __init__(self, *args)
    """
    parameterDictionary = {"rest_Vm": rest_Vm,
                           "spike_Vm": spike_Vm,
                           "neuron_threshold": neuron_threshold,
                           "membrane_time_constant": membrane_time_constant,
                           "membrane_resistance": membrane_resistance,
                           "absolute_refractory_period": absolute_refractory_period,
                           "dt": dt,
                           "noise": noise,
                           "sigma": sigma}
    return parameterDictionary
def plotSolution(SimulationObject, thetaRhythm, cycle_boundary_indices, iteration_number, darkBackground=False, suppress=False, save=False, svg=False):
    """
    Takes the completed simulation and thetaRhythm (really this could be any oscillation, since we merely plot it) and plots these in two rows with shared x axis
    - enabling darkBackground option makes all axes, tick marks, plotted lines and text labels white. Useful for Notion, Manim and dark background slide decks
    """
    if not darkBackground:

        plt.close()
        fig, axes = plt.subplots(nrows=2, sharex=True, sharey=False) # Two graps, two rows, linked x-axes
        fig.patch.set_alpha(0.0) # Make figure background transparent
        for ax in axes:
            ax.spines['top'].set_visible(False) # Make the top figure border invisible
            ax.spines['right'].set_visible(False)# Make the right figure border invisible

        axes[0].plot(SimulationObject.timeAxis, SimulationObject.Vm_t, linewidth=0.8, color='deeppink', label='Vm') # Plot the membrane potential time series
        axes[1].plot(SimulationObject.timeAxis, thetaRhythm, linewidth=0.8, color='deeppink', label='LFP') # Plot the theta rhythm (or other neural input signal) time series
        for boundary_index in cycle_boundary_indices:
            axes[1].vlines(SimulationObject.timeAxis[boundary_index], min(thetaRhythm), max(thetaRhythm), linestyle='dashed', color='k', linewidth=0.8)

        axes[0].set_ylabel("$V_m$ $(mV)$") # y-axis label of membrane potential time series, in millivolts
        axes[1].set_ylabel("$Amplitude$ $(pA)$") # y-axis label of theta rhythm
        axes[1].set_xlabel("$Time$ $(ms))$") # Label the x-axis only on the bottommost plot in milliseconds
        
        if save:
            if not svg:
                plt.savefig('{}\\HPC_phi\\modelFigures\\timeSeries\\HPC_phi_iteration{}.png'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), iteration_number), bbox_inches='tight') 
            else:
                plt.savefig('{}\\HPC_phi\\modelFigures\\timeSeries\\HPC_phi_iteration{}.svg'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), iteration_number), bbox_inches='tight')
        if not suppress:
            plt.show()
    else:

        plt.close()
        fig, axes = plt.subplots(nrows=2, sharex=True, sharey=False)
        fig.patch.set_alpha(0.0) # Make figure background transparent
        for ax in axes:
            ax.patch.set_alpha(0.0) # Make the plot backgrounds transparent
            ax.spines['right'].set_visible(False) # Make the right figure border invisible
            ax.spines['top'].set_visible(False) # Make the top figure border invisible
            ax.spines['left'].set_color('w') # Make y-axis white
            ax.spines['bottom'].set_color('w') # Make x-axis white
            ax.tick_params(axis='x', colors="w") # Make x-axis ticks white
            ax.tick_params(axis='y', colors="w") # Make y-axis ticks white

        axes[0].plot(SimulationObject.timeAxis, SimulationObject.Vm_t, linewidth=0.8, color='w', label='Vm') # Plot the membrane potential time series
        axes[1].plot(SimulationObject.timeAxis, forcingFunction, linewidth=0.8, color='w', label='LFP') # Plot the theta rhythm (or other neural input signal) time series
        for boundary_index in cycle_boundary_indices:
            axes[1].vlines(SimulationObject.timeAxis[boundary_index], min(thetaRhythm), max(thetaRhythm), linestyle='dashed', color='silver', linewidth=0.8)

        axes[0].set_ylabel("$V_m\space (mV)$", color='w') # y-axis label of membrane potential time series, in millivolts (in white)
        axes[1].set_ylabel("$Amplitude$", color='w') # y-axis label of theta rhythm (in white)
        axes[1].set_xlabel("$Time\space (ms)$", color='w') # Label the x-axis only on the bottommost plot in milliseconds (in white)

        if save:
            if not svg:
                plt.savefig('{}\\HPC_phi\\modelFigures\\timeSeries\\HPC_phi_iteration{}.png'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), iteration_number), bbox_inches='tight') 
            else:
                plt.savefig('{}\\HPC_phi\\modelFigures\\timeSeries\\HPC_phi_iteration{}.svg'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), iteration_number), bbox_inches='tight')
        if not suppress:
            plt.show()
def computePhi(thetaRhythm, theta_LFP_shift, SimulationObject=False, realData=False):
    """
    1. Uses Hilbert transform to extract phase as a function of time of the theta rhythm
    2. Extracts spike indices from either a simulation or experimental neural time series
    3. Returns a list of theta phases (phi) where each element phi corresponds to a spike time
    """
    theta_phase_time_series_INTERMEDIATE = list(map(lambda z: np.arctan2(z.imag, z.real - theta_LFP_shift), hilbert(thetaRhythm))) # For each element in analytic signal, take arctan2 which outputs angle on interval (-pi, pi]  
    theta_phase_time_series = list(map(lambda phi: phi + 2*np.pi if phi < 0 else phi, theta_phase_time_series_INTERMEDIATE)) # For each angle on (-pi, pi], add 2*pi if phi < 0 to shift the interval to [0, 2*pi)

    del theta_phase_time_series_INTERMEDIATE # Delete the first list to free up the memory

    if SimulationObject:
        spike_phi = list(map(lambda i, j: j*theta_phase_time_series[i], range(len(SimulationObject.spike_times)), SimulationObject.spike_times))

    return spike_phi, theta_phase_time_series
def kernelDensityEstimation(spike_phi, cycleNumber, kernel_bandwidth=1.5, plot_PDF_estimation=True, verbose=False):
    """
    Kernel density estimation as described and implemented at https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    - First uses leave-one-out cross-validation (leave-one-out works well for when there aren't many data points) to optimize the smoothing parameter (bandwidth) over some domain
    - Performs a kernel density estimation algorithm to create a model probability density function for the intracycle phi over interval phi=[0, 2*pi) 
        - Plots this model PDF over [0, 2*pi) if requested
    - returns the value of phi corresponding to maximum spike probability, the output measure of central tendency
    """
    spike_phi = np.array(spike_phi) # Convert the passed list into an array
    spike_phi_2D = np.reshape(spike_phi, (-1, 1)) # The backend requires a 2D array of shape (N, 1), where N is the number of samples

    if np.size(spike_phi_2D) > 1: # If we have greater than 1 spike time then we can do leave-one-out cross-validation
        if verbose: print("Optimizing kernel bandwidth on theta cycle {}".format(cycleNumber))
        bandwidth_space = 10**np.linspace(-1, 1, 100) # Allows bandwidths to take on values between 10**-1 and 10**1
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), # Optimize prediction accuracy using the SAME kernel as what the KDE will ultimately use (below)
                                        {'bandwidth': bandwidth_space}, # The bandwidths we'll assess in cross-validation
                                        cv=LeaveOneOut()) # Cross-validation will 'leave-one-out', i.e. only set asside 1 spike to assess how good the KDE is (since we haven't many spikes to begin with)
        grid.fit(spike_phi_2D) # Perform the optimization
        optimal_bandwidth = grid.best_params_['bandwidth'] # Get the optimal bandwidth for the true KDE
        if verbose: print("Optimal bandwidth on theta cycle {}: {} Hz".format(cycleNumber, optimal_bandwidth))

    else: # Otherwise simply arbitrate the bandwidth, we only call it optimal bandwidth so that it's used in the KernelDensity instantiation (just below)
        optimal_bandwidth = kernel_bandwidth # Generally appears to range from 0.7 to 2.7, no clear relationship between number of spikes and this
        if verbose: print("Single spike on theta cycle {} precludes optimization; setting bandwidth at: {}".format(cycleNumber, optimal_bandwidth))

    KDE = KernelDensity(bandwidth=optimal_bandwidth, kernel='gaussian') # Instantiate the KDE
    KDE.fit(spike_phi_2D) # Optimize it using the intracycle spike phi

    phi_interval = np.reshape(np.linspace(0, 2*np.pi, num=100), (-1, 1)) # A linspace vector which allows us to plot the KDE over [0, 2*pi) and extract a value of phi corresponding to max probability
    ln_PDF_estimation = KDE.score_samples(phi_interval) # For some reason you can only extract the ln() of the estimate PDF from the model
    
    if plot_PDF_estimation: # If operator requested a figure
        plt.close()
        fig, ax = plt.subplots() # One graph, one figure
        fig.patch.set_alpha(0.0) # Make figure background transparent

        ax.spines['top'].set_visible(False) # Make the top figure border invisible
        ax.spines['right'].set_visible(False) # Make the right figure border invisible

        ax.plot(phi_interval, np.exp(ln_PDF_estimation), color='deeppink', linewidth=0.8) # Plot the KDE as over interval [0, 2*pi)
        ax.set_xlabel("$\phi$ $(radians)$", fontsize=13) # Label x-axis with intracycle phi
        ax.set_ylabel("$Probability$", fontsize=13) # Label y-axis as probability

        xLabelList = [r" ", r"$0$", r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$", r"$\frac{4}{3}\pi$", r"$\frac{5}{3}\pi$", r"$2\pi$"] # Label ticks in radians from [0, 2*pi)
        ax.set_xticklabels(xLabelList) # Set xtick labels as defined in the line above

        plt.show() # Reveal the graph
       
    return phi_interval[np.argmax(np.exp(ln_PDF_estimation))][0] # Index the phi interval of shape (N, 1) where N is the number of samples from [0, 2*pi) with [index of max probability][0], returning intracycle phi of max probability
def cycle_CentralTendency(theta_phase_time_series, spike_phi, kernel_bandwidth=2, central_tendency_mode=0, cycleStart=0, plot_PDF_estimation=True, verbose=False):
    spike_phi = np.array(spike_phi)
    # Dictionary translating user-selected central tendency measure into an explicit string for clarity of reading 
    central_tendency_measure_dictionary = {"0": 'mean',
                                           "1": 'median',
                                           "2": 'mode',
                                           "3": 'KDE'}
    central_tendency_mode = central_tendency_measure_dictionary[str(central_tendency_mode)] # Key the dictionary using the user selection as a string

    if cycleStart != 0: # If we consider phi != 0 to be the cycle boundary
        rotated_theta_phase_time_series_INTERMEDIATE = list(map(lambda phi: phi - cycleStart, theta_phase_time_series)) # Rotate all phases by some angle (radians)
        rotated_theta_phase_time_series = list(map(lambda phi: phi + 2*np.pi if phi < 0 else phi, rotated_theta_phase_time_series_INTERMEDIATE)) # Shift any negative values back to the end of the cycle
        theta_phase_time_series = rotated_theta_phase_time_series # Have this rotated reference frame be the one we analyze further

    wrapped_omega = np.diff(np.array(theta_phase_time_series)) # Wrapped frequency of sinusoid is constant except where trajectory crosses positive x-axis where it is ~ -2*pi
    cycle_boundary_indices = np.where(wrapped_omega < 0)[0] # Hence, wherever frequency is negative we have a new cycle

    if verbose: print("Cycle boundary indices: {}".format(cycle_boundary_indices))

    if central_tendency_mode == 'mean':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary

            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries
            if verbose: print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                if verbose: print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(np.mean(np.array(phi_values))) # Compute mean on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries
        if verbose: print("indices of nonzero phi on cycle {}: {}".format(i + 1, phi_indices))

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            if verbose: print("final theta cycle contains no spikes") # Notify the operator

        else: # If there were spikes
            phi_values = [] # Initialize a list
            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
            cycle_phi_central_tendencies.append(np.mean(np.array(phi_values))) # Compute mean on all values phi from the end of the last cycle to the end of the time series

    elif central_tendency_mode == 'median':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary

            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries
            if verbose: print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                if verbose: print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(np.median(np.array(phi_values))) # Compute median on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            if verbose: print("final theta cycle contains no spikes") # Notify the operator

        else: # If there were spikes
            phi_values = [] # Initialize a list
            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
            cycle_phi_central_tendencies.append(np.median(np.array(phi_values))) # Compute median on all values phi from the end of the last cycle to the end of the time series

    elif central_tendency_mode == 'mode':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary

            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries
            if verbose: print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                if verbose: print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(mode(np.array(phi_values))[0][0]) # Compute mode on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            if verbose: print("final theta cycle contains no spikes") # Notify the operator

        else: # If there were spikes
            phi_values = [] # Initialize a list
            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
            cycle_phi_central_tendencies.append(mode(np.array(phi_values))[0][0]) # Compute mode on all values phi from the end of the last cycle to the end of the time series
    
    elif central_tendency_mode == 'KDE':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary

            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries
            if verbose: print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                if verbose: print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(kernelDensityEstimation(phi_values, i, kernel_bandwidth=kernel_bandwidth, plot_PDF_estimation=plot_PDF_estimation, verbose=verbose))

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            if verbose: print("final theta cycle contains no spikes") # Notify the operator

        else: # If there were spikes
            phi_values = [] # Initialize a list
            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
            cycle_phi_central_tendencies.append(kernelDensityEstimation(phi_values, i + 1, kernel_bandwidth=kernel_bandwidth, plot_PDF_estimation=plot_PDF_estimation, verbose=verbose))
    

    if verbose: print("intracycle central tendencies: {}".format(cycle_phi_central_tendencies))

    return np.array(cycle_phi_central_tendencies), cycle_boundary_indices
def constructReturnMap(cycle_phi_central_tendencies, iteration_number, darkBackground=False, suppress=False, save=False, svg=False):

    phi_K_previous = np.delete(cycle_phi_central_tendencies, -1) # All entries except the Kth one can be the (K - 1)th entry
    phi_K = np.delete(cycle_phi_central_tendencies, 0) # All entries except the 0th can be the Kth (i.e. next) entry
    line_of_identity = np.linspace(0, 2*np.pi) # Creates a line of identity with domain that spans the range of phi central tendencies
    phi_map_coordinate_difference = np.subtract(phi_K_previous, phi_K)
    
    if not darkBackground:
        plt.close()
        fig, ax = plt.subplots() # A single plot
        fig.patch.set_alpha(0.0) # Make figure background transparent

        ax.spines['right'].set_visible(False) # Make the right figure border invisible
        ax.spines['top'].set_visible(False) # Make the top figure border invisible

        ax.scatter(phi_K_previous, phi_K, s=6, color='deeppink') # Plot the 'next' central tendency as a function of the previous (Return map, AKA Poincare map (?))
        ax.plot(line_of_identity, line_of_identity, linestyle='dashed', color='k', linewidth=0.8) # Plot a dashed line indicating the line of identity
        ax.set_xlabel("$\phi_{k-1}$", fontsize=13) # x-axis label for previous/current (depending on perspective) phi central tendency
        ax.set_ylabel("$\phi_{k}$", fontsize=13) # y-axis label for current/next (depending on perspective) phi central tendency
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0, 2*np.pi])

        tickLabelList = [r" ", r"$0$", r"$\frac{1}{3}\pi$", r"$\frac{2}{3}\pi$", r"$\pi$", r"$\frac{4}{3}\pi$", r"$\frac{5}{3}\pi$", r"$2\pi$"] # Label ticks in radians from [0, 2*pi)
        ax.set_xticklabels(tickLabelList) # Set xtick labels as defined in the line above
        ax.set_yticklabels(tickLabelList)
        
        if save:
            if not svg:
                plt.savefig('{}\\HPC_phi\\modelFigures\\returnMaps\\HPC_phi_iteration{}.png'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), iteration_number), bbox_inches='tight')
            else:
                plt.savefig('{}\\HPC_phi\\modelFigures\\returnMaps\\HPC_phi_iteration{}.svg'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), iteration_number), bbox_inches='tight')
        if not suppress:
            plt.show()
    else:
        plt.close()
        fig, ax = plt.subplots() # A single plot
        fig.patch.set_alpha(0.0) # Make figure background transparent

        ax.spines['right'].set_visible(False) # Make the right figure border invisible
        ax.spines['top'].set_visible(False) # Make the top figure border invisible
        ax.spines['left'].set_color('w') # Make y-axis white
        ax.spines['bottom'].set_color('w') # Make x-axis white
        ax.tick_params(axis='x', colors="w") # Make x-axis ticks white
        ax.tick_params(axis='y', colors="w") # Make y-axis ticks white

        ax.scatter(phi_K_previous, phi_K, s=6, color='w') # Plot the 'next' central tendency as a function of the previous (Return map, AKA Poincare map (?))
        ax.plot(line_of_identity, line_of_identity, linestyle='dashed', color='silver') # Plot a dashed line indicating the line of identity
        ax.set_xlabel("$\phi_{k-1}$", fontsize=13, color='w') # x-axis label for previous/current (depending on perspective) phi central tendency
        ax.set_ylabel("$\phi_{k}$", fontsize=13, color='w') # y-axis label for current/next (depending on perspective) phi central tendency

        if save:
            if not svg:
                plt.savefig('{}\\HPC_phi\\modelFigures\\returnMaps\\HPC_phi_iteration{}.png'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), iteration_number), bbox_inches='tight')
            else:
                plt.savefig('{}\\HPC_phi\\modelFigures\\returnMaps\\HPC_phi_iteration{}.svg'.format(os.path.dirname(os.path.realpath("modelFigures")).encode('unicode-escape').decode(), iteration_number), bbox_inches='tight')
        if not suppress:
            plt.show()
    return phi_map_coordinate_difference
def analyzeReturnMap(phi_map_coordinate_differences, iteration_number, kernel_bandwidth=2.0, plot_PDF_estimation=True, verbose=False):
   
    coordinate_differences_2D = np.reshape(phi_map_coordinate_differences, (-1, 1))

    if np.size(coordinate_differences_2D) > 1:
        bandwidth_space = 10**np.linspace(-1, 1, 100) # Allows bandwidths to take on values between 10**-1 and 10**1
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), # Optimize prediction accuracy using the SAME kernel as what the KDE will ultimately use (below)
                                        {'bandwidth': bandwidth_space}, # The bandwidths we'll assess in cross-validation
                                        cv=LeaveOneOut()) # Cross-validation will 'leave-one-out', i.e. only set asside 1 spike to assess how good the KDE is (since we haven't many spikes to begin with)
        grid.fit(coordinate_differences_2D) # Perform the optimization
        optimal_bandwidth = grid.best_params_['bandwidth'] # Get the optimal bandwidth for the true KDE
        if verbose: print("Optimal bandwidth on iteration {} return map: {}".format(iteration_number, optimal_bandwidth))

    else: # Otherwise simply arbitrate the bandwidth, we only call it optimal bandwidth so that it's used in the KernelDensity instantiation (just below)
        optimal_bandwidth = kernel_bandwidth # Generally appears to range from 0.7 to 2.7, no clear relationship between number of spikes and this
        if verbose: print("Only two cycles with spikes on iteration {} precludes optimization; setting bandwidth at: {}".format(iteration_number, optimal_bandwidth))
    
    KDE = KernelDensity(bandwidth=optimal_bandwidth, kernel='gaussian') # Instantiate the KDE
    KDE.fit(coordinate_differences_2D) # Optimize it using the intracycle spike phi

    domain_PDF_estimation = np.reshape(np.linspace((np.amin(phi_map_coordinate_differences) - 0.1*(np.amax(phi_map_coordinate_differences) - np.amin(phi_map_coordinate_differences))), (np.amax(phi_map_coordinate_differences) + 0.1*(np.amax(phi_map_coordinate_differences) - np.amin(phi_map_coordinate_differences))), num=100), (-1, 1))
    ln_PDF_estimation = KDE.score_samples(domain_PDF_estimation)

    if plot_PDF_estimation: # If operator requested a figure
        plt.close()
        fig, ax = plt.subplots() # One graph, one figure
        fig.patch.set_alpha(0.0) # Make figure background transparent

        ax.spines['top'].set_visible(False) # Make the top figure border invisible
        ax.spines['right'].set_visible(False) # Make the right figure border invisible

        ax.plot(domain_PDF_estimation, np.exp(ln_PDF_estimation), color='deeppink', linewidth=0.8) # Plot the KDE as over interval [0, 2*pi)
        ax.set_xlabel("$\phi_{k-1} - \phi_k$ $(radians)$", fontsize=13) # Label x-axis with intracycle phi
        ax.set_ylabel("$Probability$", fontsize=13) # Label y-axis as probability

        plt.show() # Reveal the graph

    phase_drift_quantification = domain_PDF_estimation[np.argmax(np.exp(ln_PDF_estimation))][0]
    if verbose: print("Iteration {} phase drift quantification: {}".format(iteration_number, phase_drift_quantification))

    return  phase_drift_quantification

# Define functions which wrap simulation construction and analysis functions into separate pipelines
def simulationAnalysis(SimulationObject, thetaRhythm, theta_LFP_shift, iteration_number, central_tendency_mode=0, cycleStart=0, kernel_bandwidth=2, plot_central_tendency_PDF_estimation=True, plot_return_map_PDF_estimation=True, sim_fig_suppress=False, sim_fig_dark=False, save_sim_fig=True, return_map_suppress=False, return_map_dark=False, save_return_map=True, svg=False, verbose=False):
    """
    This function simply wraps simulation analysis pipeline into a single function, taking a completed LIFNeuron Simulation and theta LFP (plus its vertical shift):
    1. computePhi() returns phi value of each spike in sequence, the phase time series of theta LFP
    2. cycle_CentralTendency() takes both outputs from (1.) and returns the intracycle phi central tendency and the boundary indices of each cycle (for plotting purposes)
        - This function can make use of kernelDensityEstimation() if central_tendency_mode = 3
    3. Plots the simulation solution figures (if not suppressed) in light/dark background
    4. Plots the return map figure (if not suppressed) in light/dark background
    """

    # Compute simulation phi sequence
    phi_sequence, theta_phase_time_series = computePhi(thetaRhythm, theta_LFP_shift, SimulationObject=SimulationObject)

    # Compute phi central tendencies on each theta cycle and construct return map
    cycle_phi_central_tendencies, cycle_boundary_indices = cycle_CentralTendency(theta_phase_time_series, phi_sequence, kernel_bandwidth=kernel_bandwidth, central_tendency_mode=central_tendency_mode, cycleStart=cycleStart, plot_PDF_estimation=plot_central_tendency_PDF_estimation, verbose=verbose)

    # Plot simulation output and Return map
    plotSolution(SimulationObject, thetaRhythm, cycle_boundary_indices, iteration_number, darkBackground=sim_fig_dark, suppress=sim_fig_suppress, save=save_sim_fig, svg=svg)
    return_map_coordinate_differences = constructReturnMap(cycle_phi_central_tendencies, iteration_number, darkBackground=return_map_dark, suppress=return_map_suppress, save=save_return_map, svg=svg)
    simulation_quantification = analyzeReturnMap(return_map_coordinate_differences, iteration_number, kernel_bandwidth=2, plot_PDF_estimation=plot_return_map_PDF_estimation)

    return simulation_quantification, cycle_phi_central_tendencies
def HPC_phi_simulation(**kwargs):
    """
    Wraps model construction and analysis into a function for iterative testing. Control panel parameters from baseModel are passed as arguments
    """
    simulation_duration = kwargs['simulation_duration']
    dt = kwargs['dt']
    rest_Vm = kwargs['rest_Vm']
    spike_Vm = kwargs['spike_Vm']
    neuron_threshold = kwargs['neuron_threshold']
    membrane_time_constant = kwargs['membrane_time_constant']
    membrane_resistance = kwargs['membrane_resistance']
    absolute_refractory_period = kwargs['absolute_refractory_period']
    noise = kwargs['noise']
    sigma = kwargs['sigma']
    theta_frequency = kwargs['theta_frequency']
    interference_frequency = kwargs['interference_frequency']
    theta_amplitude = kwargs['theta_amplitude']
    interference_amplitude = kwargs['interference_amplitude']
    LFP_shift = kwargs['LFP_shift']
    iteration_number = kwargs['iteration_number']
    central_tendency_mode = kwargs['central_tendency_mode']
    theta_cycle_boundary_phase = kwargs['theta_cycle_boundary_phase']
    kernel_bandwidth = kwargs['kernel_bandwidth']
    plot_central_tendency_PDF_estimation = kwargs['plot_central_tendency_PDF_estimation']
    plot_return_map_PDF_estimation = kwargs['plot_return_map_PDF_estimation']
    sim_fig_suppress = kwargs['sim_fig_suppress']
    sim_fig_dark = kwargs['sim_fig_dark']
    save_sim_fig = kwargs['save_sim_fig']
    return_map_suppress = kwargs['return_map_suppress']
    return_map_dark = kwargs['return_map_dark']
    save_return_map = kwargs['save_return_map']
    svg = kwargs['svg']
    verbose = kwargs['verbose']

    # Compute number of timesteps in simulation, package parameters
    num_timesteps = int(simulation_duration/dt) # Infer number of timesteps from simulation length and dt
    parameterDictionary = packageParameters_LIFNeuron(rest_Vm, spike_Vm, neuron_threshold, membrane_time_constant, membrane_resistance, absolute_refractory_period, dt, noise, sigma)   

    # Convert desired frequency into radians/ms
    omega_theta = 2*np.pi*((theta_frequency**-1)*1000)**(-1) # Natural (angular) frequency corresponding to hippocampal theta LFP oscillations (radians/ms); NUMBER IS PERIOD IN ms
    omega_interference = 2*np.pi*((interference_frequency**-1)*1000)**(-1) # Natural (angular) frequency of interloping oscillator (radians/ms); NUMBER IS PERIOD IN ms

    # Functions defining independent field oscillations using forcing parameters
    thetaFunction = lambda t: theta_amplitude*np.sin(omega_theta*t) + LFP_shift # Corresponds to hippocampal theta sinusoidal oscillations
    interferenceFunction = lambda t: interference_amplitude*np.sin(omega_interference*t) + LFP_shift # Corresponds to some interferring oscillation

    # Generate the theta and interference oscillations over specified timeseries, then superimpose the two
    thetaRhythm = forcingFunction(thetaFunction, num_timesteps, dt) # Generate list of theta  rhythm amplitudes as a function of time
    interferenceRhythm = forcingFunction(interferenceFunction, num_timesteps, dt) # Generates list of interferring oscillation amplitude as a function of time
    fieldOscillation = np.array([x + y for x, y in zip(thetaRhythm, interferenceRhythm)]) # Add the two lists element-wise, then convert to numpy array

    # Instantiate LIFNeuron
    Neuron = LIFNeuron(parameterDictionary)

    # Create and run simulation
    NeuronSim = Simulation(Neuron, kwargs)
    NeuronSim.runSim(fieldOscillation, num_timesteps, dt)

    # Run the simulation analysis pipeline
    simulation_quantification, cycle_phi_central_tendencies = simulationAnalysis(NeuronSim, thetaRhythm, LFP_shift, iteration_number, central_tendency_mode=central_tendency_mode, cycleStart=theta_cycle_boundary_phase, kernel_bandwidth=kernel_bandwidth, plot_central_tendency_PDF_estimation=plot_central_tendency_PDF_estimation, plot_return_map_PDF_estimation=plot_return_map_PDF_estimation, sim_fig_suppress=sim_fig_suppress, sim_fig_dark=sim_fig_dark, save_sim_fig=save_sim_fig, return_map_suppress=return_map_suppress, return_map_dark=return_map_dark, save_return_map=save_return_map, svg=svg, verbose=verbose)

    return simulation_quantification, cycle_phi_central_tendencies

def main():
    # ========== CONTROL PANEL =====================================================================================================================================================================
                    # Model Parameters
    parameterDict = {'theta_amplitude': 300,                        # Amplitude of theta LFP-induced current (pA)
                     'interference_amplitude': 300,                 # Amplitude of interference oscillator LFP-induced current (pA)
                     'theta_frequency': 12,                         # Frequency of theta LFP (Hz)
                     'interference_frequency': 12.5,                # Frequency of interference oscillator (Hz)
                     'LFP_shift': 0,                                # Vertical translation of both the interference and theta oscillation waveform (pA)
                     'dt': 0.01,                                    # Timestep size (ms)
                     'simulation_duration': 1000,                   # Simulation duration (ms)
                     'rest_Vm': -75,                                # Resting membrane potential (mV)
                     'spike_Vm': 80,                                # Spike potential (mV)
                     'neuron_threshold': -40,                       # Spiking threshold (mV)
                     'membrane_time_constant': 100,                 # Membrane time constant
                     'membrane_resistance': 1,                      # Membrane resistance (unspecified, this serves merely as a scaling factor for input current and was consolidated to LFP amplitudes)
                     'absolute_refractory_period': 2,               # Absolute minimum interspike interval (ms)
                     'noise': True,                                 # Indicate whether or not to include stochastic input current in neuron model (Ornstein-Uhlenbeck process)
                     'sigma': 1.0,                                  # Scaling factor which winds up being the amplitude of stochastic input (mV) 
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    # Analysis Parameters
                     'central_tendency_mode': 3,                    # Select measure of central tendency: 0 = mean, 1 = median, 2 = mode (DEFUNCT), 3 = Kernel density estimation (KDE)
                     'theta_cycle_boundary_phase': 3*np.pi/2,       # Analytic signal-derived phase marking the start of a theta cycle; NOTE: sin(0) + HT(sin(0)) = 0 - 1i
                     'kernel_bandwidth': 1.5,                       # Bandwidth of Gaussian kernel used for intracycle phi estimation when leave-one-out cross validation not viable -- i.e. when only one spike occurred on the cycle 
                     'verbose': False,                              # Whether or not to notify operator of individual spike indices, phi values, central tendencies and KDE progress
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                    # Figure Parameters
                     'sim_fig_suppress': True,                      # Whether or not to show/save the simulation time series; HARDCODED IN plotSolution() for now
                     'sim_fig_dark': False,                         # Makes simulation timeseries white on transparent background, suitable for Manim, Notion and dark slide decks -- currently no plt.savefig() in plotSolutions()
                     'save_sim_fig': False,                         # Saves simulation timeseries to directory hardcoded in plotSolutions()
                     'return_map_suppress': True,                   # Whether or not to show/save the simulation time series; HARDCODED IN constructReturnMap() for now
                     'return_map_dark': False,                      # Makes return map white on transparent background, suitable for Manim, Notion and dark slide decks -- currently no plt.savefig() in constructReturnMap()
                     'save_return_map': False,                      # Saves return map to directory hardcoded in constructReturnMap()
                     'plot_central_tendency_PDF_estimation': False, # Whether or not to show the estimated probability density function (PDF) over each theta cycle 
                     'plot_return_map_PDF_estimation': False,       # Whether or not to show the estimated probability density function (PDF) over each return map
                     'svg': True,                                   # Whether or not output figures should be as svg files
                     'iteration_number': 0,                         # Keep track of how many times the model has been run per runTime; mainly for figure naming purposes
                     'contour_plot_suppress': False,                 # Whether or not to show the contour plot for a sweep over a 2D discrete subset of parameter space
                     'save_contour_plot': True,                     # Whether or not to save the contour plot for a sweep over a 2D discrete subset of parameter space
                     'contour_plot_file_type': '.svg',              # String specifying the filetype of the contour plot if it is to be saved
                     'mesh_fig_suppress': False,                     # Whether or not to show a grid of return maps for a sweep over a 2D discrete subset of parameter space
                     'save_mesh_fig': True,                         # Whether or not to save a grid of return maps for a sweep over a 2D discrete subset of parameter space
                     'mesh_fig_file_type': '.png'}                  # String specifying the filetype of the return map mesh figure if it is to be saved
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------            

    # Create N parameter lists for cartesion product (mesh) iteration
    # NOTE: Enter the parameterDict keys into SimulationIterationManager corresponding to order in which each list is entered
    vector_1 = list(map(lambda q: q*300, np.linspace(1, 3, 3)))
    vector_2 = list(map(lambda q: q*300, np.linspace(1, 3, 3))) 

    # ========== SIMULATE AND ANALYZE ==============================================================================================================================================================
    # Constructs a mesh corresponding to the cartesion product of all parameter sets and their specified keys; NOTE: It is crucial that key and parameter order correspond to one another; Ex. SimulationIterationManager('parameter1', 'parameter2',...,'parameterN', list1, list2,..., listN)
    SimulationSet = SimulationIterationManager(parameterDict, "theta_amplitude", "interference_amplitude", vector_1, vector_2, "(pA)", "(pA)")
    SimulationSet.meshSweep()

if __name__ == "__main__":
    main()