import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from notion.client import NotionClient
from datetime import datetime

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
            self.Vm += dt*voltageSuperposition # Integrate! (In the mathematical sense)

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
    return list(map(lambdaFunction, list(map(lambda t: t*dt, range(num_timesteps)))))
def packageParameters(rest_Vm, spike_Vm, neuron_threshold, membrane_time_constant, membrane_resistance, absolute_refractory_period):
    parameterDictionary = {"rest_Vm": rest_Vm,
                           "spike_Vm": spike_Vm,
                           "neuron_threshold": neuron_threshold,
                           "membrane_time_constant": membrane_time_constant,
                           "membrane_resistance": membrane_resistance,
                           "absolute_refractory_period": absolute_refractory_period}
    return parameterDictionary

def main():
    # ========== CONTROL PANEL ======================================================================

    # Model Parameters ------------------------------------------------------------------------------
    dt = 0.001 # timestep size in ms
    sim_duration = 100 # duration of simulation in ms
    num_timesteps = int(sim_duration/dt) # Infer number of timesteps from simulation length and dt
    rest_Vm = -70 # For LIFNeuron this is the starting and reset potential
    spike_Vm = 50 # For LIFNeuron this is the spiking potential
    neuron_threshold = -40 # Whenever LIFNeuron reaches this potential from below the model 'fires' a 'spike'.
    membrane_time_constant = 1
    membrane_resistance = 1
    absolute_refractory_period = 0 # NOT YET SUPPORTED IN LIFNeuron: the duration of spiking events before returning to rest_Vm
    parameterDictionary = packageParameters(rest_Vm, spike_Vm, neuron_threshold, membrane_time_constant, membrane_resistance, absolute_refractory_period)

    # Forcing Parameters ----------------------------------------------------------------------------
    omega_theta = 1 # Natural (angular) frequency corresponding to hippocampal theta LFP oscillations
    omega_interference = 1 # Natural (angular) frequency of interloping oscillator

    amplitude_theta = 1 # Amplitude of hippocampal theta LFP oscillations (units?)
    amplitude_interference = 1 # Amplitude of interloping oscillator

    # Functions defining independent field oscillations using forcing parameters
    thetaFunction = lambda t: amplitude_theta*np.sin(omega_theta*t) # Corresponds to hippocampal theta sinusoidal oscillations
    interferenceFunction = lambda t: amplitude_interference*np.sin(omega_interference*t) # Corresponds to some interferring oscillation

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

if __name__ == "__main__":
    main()