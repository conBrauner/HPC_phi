import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from notion.client import NotionClient
from datetime import datetime
from scipy.signal import hilbert

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

        # Indicates if a spike occurred at the last time step 
        self.spikeState = False

    def _updateVm(self, time_series_index, forcing, dt):
        """
        Increments model membrane potential according to the sum of all inputs. Also generates a spike if threshold is met or exceeded, then returns potential to resting at the next step.
        """
        if self.Vm >= self.neuron_threshold and self.spikeState == False: # Check if threshold met/exceeded + if the last step was a spike
            self.Vm = self.spike_Vm # Jump to spiking potential if criteria met
            self.spikeState = True # Tag the current time step as having been a spike event
        elif self.spikeState == True: # Check if the last time step was a spike
            self.Vm = self.rest_Vm # If so, return to resting potential
            self.spikeState = False # Change the state to reflect the potential reset
        else: # If the model is currently subthreshold
            self.VL = -(self.Vm - self.rest_Vm) # Compute the decrement in potential change due to the leak
            self.V_forcing = forcing*self.membrane_resistance # Compute contribution from forcing function
            voltageSuperposition = self.VL + self.V_forcing # Add the two contributions
            self.Vm += (dt*voltageSuperposition)/self.membrane_time_constant # Integrate! (In the mathematical sense)

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
    def __init__(self, model):
        self.model = model
    def initializeOutput(self, num_timesteps, dt):
        self.timeAxis = np.arange(num_timesteps) * dt # Each entry is cumulative time elapsed in ms
        self.Vm_t = np.empty(num_timesteps) # Vm_t here is read: 'membrane potential as a function of time' and is the principle solution of a neuron model
        self.spike_times = np.zeros(num_timesteps) # Binary sequence indicating if a solution index corresponds to a spike; 1 for yes, 0 for no
    def runSim(self, forcingFunction, num_timesteps, dt):
        self.initializeOutput(num_timesteps, dt) # Initialize output axes to be of the specified length
        print("Initialization complete...") # Indicates to operator that model and output axes were defined successfully
        print("Simulating {} ms\nParameters: \n{}".format(forcingFunction.shape[0]*dt, self.model.parameters)) # Indicates to operator how many ms are being simulated + the model parameters
        for time_series_index in range(num_timesteps): # For each time step in the desired simulation length for a given dt
            self.model.iterate(time_series_index, forcingFunction[time_series_index], dt) # Advance the model state variables by 1 time step
            self.Vm_t[time_series_index] = self.model.Vm # Append current membrane potential to solution
            if self.model.spikeState == True: # HARDCODED FOR LIF: Refer to model state to determine if current step corresponds to a spike
                self.spike_times[time_series_index] = 1 
        print("Simulation completed") # Indicate to operator that simulation is finished

def forcingFunction(lambdaFunction, num_timesteps, dt):
    """
    Given a lambda function, passes in N inputs which increment by dt where N = num_timesteps. 
    Output is a list with length num_timesteps where each element solves y = lambdaFunction = f(t) for t on interval [0, dt*(num_timesteps - 1)]
    """
    return list(map(lambdaFunction, list(map(lambda t: t*dt, range(num_timesteps))))) # Note the nested function for scaling each element by dt
def packageParameters_LIFNeuron(rest_Vm, spike_Vm, neuron_threshold, membrane_time_constant, membrane_resistance, absolute_refractory_period):
    """
    Takes all required parameters for the LIFNeuron class and forms a dictionary with hardcoded keys. 
    Values are assigned as class attributes during __init__(self, *args)
    """
    parameterDictionary = {"rest_Vm": rest_Vm,
                           "spike_Vm": spike_Vm,
                           "neuron_threshold": neuron_threshold,
                           "membrane_time_constant": membrane_time_constant,
                           "membrane_resistance": membrane_resistance,
                           "absolute_refractory_period": absolute_refractory_period}
    return parameterDictionary
def plotSolution(SimulationObject, thetaRhythm, darkBackground=False):
    """
    Takes the completed simulation and thetaRhythm (really this could be any oscillation, since we merely plot it) and plots these in two rows with shared x axis
    - enabling darkBackground option makes all axes, tick marks, plotted lines and text labels white. Useful for Notion, Manim and dark background slide decks
    """
    if not darkBackground:

        plt.close()
        fig, axes = plt.subplots(nrows=2, sharex=True, sharey=False) # Two graps, two rows, linked x-axes
        fig.patch.set_alpha(0.0) # Make figure background transparent
        for ax in axes:
            ax.spines['top'].set_visible = False # Make the top figure border invisible
            ax.spines['right'].set_visible = False # Make the right figure border invisible

        axes[0].plot(SimulationObject.timeAxis, SimulationObject.Vm_t, linewidth=0.8, color='deeppink', label='Vm') # Plot the membrane potential time series
        axes[1].plot(SimulationObject.timeAxis, thetaRhythm, linewidth=0.8, color='deeppink', label='LFP') # Plot the theta rhythm (or other neural input signal) time series

        axes[0].set_ylabel("$V_m$ $(mV)$") # y-axis label of membrane potential time series, in millivolts
        axes[1].set_ylabel("$Amplitude$") # y-axis label of theta rhythm
        axes[1].set_xlabel("$Time$ $(ms)$") # Label the x-axis only on the bottommost plot in milliseconds

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

        axes[0].set_ylabel("$V_m\space (mV)$", color='w') # y-axis label of membrane potential time series, in millivolts (in white)
        axes[1].set_ylabel("$Amplitude$", color='w') # y-axis label of theta rhythm (in white)
        axes[1].set_xlabel("$Time\space (ms)$", color='w') # Label the x-axis only on the bottommost plot in milliseconds (in white)

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
        spike_indices = np.where(SimulationObject.spike_times == 1)[0] # Identify all spike times in simulation attribute .spike_times, where 1 is a spike and 0 is subthreshold 

    spike_phi = [theta_phase_time_series[spike_index] for spike_index in spike_indices] # Take all values of phi that co-occur with a spike

    return spike_phi, theta_phase_time_series
def cycle_CentralTendency(theta_phase_time_series, spike_phi, central_tendency_mode=0, cycleStart=0, ):

    # Dictionary translating user-selected central tendency measure into an explicit string for clarity of reading 
    central_tendency_measure_dictionary = {"0": 'mean',
                                           "1": 'median',
                                           "2": 'mode'}
    central_tendency_mode = central_tendency_measure_dictionary[str(central_tendency_mode)] # Key the dictionary using the user selection as a string

    wrapped_omega = np.diff(np.array(theta_phase_time_series)) # Wrapped frequency of sinusoid is constant except where trajectory crosses positive x-axis where it is ~ -2*pi
    cycle_boundary_indices = np.where(wrapped_omega < 0)[0] # Hence, wherever frequency is negative we have a new cycle

    if central_tendency_mode == 'mean':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for cycle_end_index in cycle_boundary_indices: # Take each entry to be a right cycle boundary
            cycle_phi_central_tendencies.append(np.mean(np.where(spike_phi < cycle_end_index and spike_phi >= cycle_start_index)[0])) # Compute mean on all values phi falling on interval [cycle_start, cycle_end)
            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval

        cycle_phi_central_tendencies.append(np.mean(np.where(spike_phi >= cycle_start_index)[0])) # Compute mean on all intervals from the end of the last cycle to the end of the time series
    
    elif central_tendency_mode == 'median':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output median spike_phi for each cycle

        for cycle_end_index in cycle_boundary_indices: # Take each entry to be a right cycle boundary
            cycle_phi_central_tendencies.append(np.mean(np.where(spike_phi < cycle_end_index and spike_phi >= cycle_start_index)[0])) # Compute median on all values phi falling on interval [cycle_start, cycle_end)
            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval

        cycle_phi_central_tendencies.append(np.mean(np.where(spike_phi >= cycle_start_index)[0])) # Compute median on all intervals from the end of the last cycle to the end of the time series

    elif central_tendency_mode == 'mode':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mode of spike_phi for each cycle

        for cycle_end_index in cycle_boundary_indices: # Take each entry to be a right cycle boundary
            cycle_phi_central_tendencies.append(np.mean(np.where(spike_phi < cycle_end_index and spike_phi >= cycle_start_index)[0])) # Compute mode on all values phi falling on interval [cycle_start, cycle_end)
            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval

        cycle_phi_central_tendencies.append(np.mean(np.where(spike_phi >= cycle_start_index)[0])) # Compute mode on all intervals from the end of the last cycle to the end of the time series

    return cycle_phi_central_tendencies

def main():
    # ========== CONTROL PANEL ======================================================================

    # Model Parameters ------------------------------------------------------------------------------
    dt = 0.01 # timestep size in ms
    sim_duration = 1000 # duration of simulation in ms
    num_timesteps = int(sim_duration/dt) # Infer number of timesteps from simulation length and dt
    rest_Vm = -70 # For LIFNeuron this is the starting and reset potential
    spike_Vm = 50 # For LIFNeuron this is the spiking potential
    neuron_threshold = -40 # Whenever LIFNeuron reaches this potential from below the model 'fires' a 'spike'.
    membrane_time_constant = 100
    membrane_resistance = 5
    absolute_refractory_period = 0 # NOT YET SUPPORTED IN LIFNeuron: the duration of spiking events before returning to rest_Vm
    parameterDictionary = packageParameters_LIFNeuron(rest_Vm, spike_Vm, neuron_threshold, membrane_time_constant, membrane_resistance, absolute_refractory_period)

    # Forcing Parameters ----------------------------------------------------------------------------
    omega_theta = 2*np.pi*125**(-1) # Natural (angular) frequency corresponding to hippocampal theta LFP oscillations (radians/ms)
    omega_interference = 2*np.pi*100**(-1) # Natural (angular) frequency of interloping oscillator (radians/ms)

    amplitude_theta = 20.0 # Amplitude of hippocampal theta LFP oscillations (mV)
    amplitude_interference = 20.0 # Amplitude of interloping oscillator (mV)

    theta_LFP_shift = amplitude_theta # Theta local field potential shift, corresponds to a vertical translation of the sinusoid
    interference_LFP_shift = amplitude_interference # Interloping oscillator local field potential shift, corresponds to a vertical translation of the sinusoid

    # Functions defining independent field oscillations using forcing parameters
    thetaFunction = lambda t: amplitude_theta*np.sin(omega_theta*t) + theta_LFP_shift # Corresponds to hippocampal theta sinusoidal oscillations
    interferenceFunction = lambda t: amplitude_interference*np.sin(omega_interference*t) + interference_LFP_shift # Corresponds to some interferring oscillation

    # Generate the theta and interference oscillations over specified timeseries, then superimpose the two
    thetaRhythm = forcingFunction(thetaFunction, num_timesteps, dt) # Generate list of theta  rhythm amplitudes as a function of time
    interferenceRhythm = forcingFunction(interferenceFunction, num_timesteps, dt) # Generates list of interferring oscillation amplitude as a function of time
    fieldOscillation = np.array([x + y for x, y in zip(thetaRhythm, interferenceRhythm)]) # Add the two lists element-wise, then convert to numpy array

    # ========== INSTANTIATE MODEL AND RUN SIMULATION ===============================================

    # Instantiate LIFNeuron
    Neuron = LIFNeuron(parameterDictionary)

    # Create and run simulation
    NeuronSim = Simulation(Neuron)
    NeuronSim.runSim(fieldOscillation, num_timesteps, dt)

    # Plot simulation output
    plotSolution(NeuronSim, thetaRhythm)

    # Compute simulation phi sequence
    phi_sequence, theta_phase_time_series = computePhi(thetaRhythm, theta_LFP_shift, SimulationObject=NeuronSim)

    # Compute phi central tendencies on each theta cycle and construct return map
    cycle_CentralTendency(theta_phase_time_series, phi_sequence)

if __name__ == "__main__":
    main()