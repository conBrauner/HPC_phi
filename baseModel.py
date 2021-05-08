import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from notion.client import NotionClient
from datetime import datetime

class LIFNeuron:
    def __init__(self, parameterDictionary):
        self.modelType = 'LIF'
        self.parameters = parameterDictionary
        self.rest_Vm = parameterDictionary['rest_Vm']
        self.spike_Vm = parameterDictionary['spike_Vm']
        self.Vm = parameterDictionary['rest_Vm']
        self.neuron_threshold = parameterDictionary['neuron_threshold']
        self.membrane_time_constant = parameterDictionary['membrane_time_constant']
        self.absolute_refractory_period = parameterDictionary['absolute_refractory_period']
        self.membrane_resistance = parameterDictionary['membrane_resistance']
        self.spikeState = False

    def _updateVm(self, time_series_index, forcing, dt):
        if self.Vm >= self.neuron_threshold and self.spikeState == False:
            self.Vm = self.spike_Vm
            self.spikeState = True
        elif self.spikeState == True:
            self.Vm = self.rest_Vm
            self.spikeState = False    
        else:
            self.VL = -(self.Vm - self.rest_Vm)
            self.V_forcing = forcing*self.membrane_resistance
            voltageSuperposition = self.VL + self.V_forcing
            self.Vm += dt*voltageSuperposition

    def iterate(self, time_series_index, forcing, dt):
        self._updateVm(time_series_index, forcing, dt)      
class Simulation:
    def __init__(self, model):
        self.model = model
        self.initializeOutput(0, 0)
    def initializeOutput(self, num_timesteps, dt):
        self.timeAxis = np.arange(num_timesteps) * dt
        self.Vm_t = np.empty(num_timesteps)
        self.spike_times = np.zeros(num_timesteps)  
    def runSim(self, forcingFunction, num_timesteps, dt):
        self.initializeOutput(num_timesteps, dt)
        print("Initialization complete...")
        print("Simulating {} ms\nParameters: \n{}".format(forcingFunction.shape[0]*dt, self.model.parameters))
        for time_series_index in range(num_timesteps):
            self.model.iterate(time_series_index, forcingFunction[time_series_index], dt)
            self.Vm_t[time_series_index] = self.model.Vm
            if self.model.spikeState == True:
                self.spike_times[time_series_index] = 1
        print("Simulation completed")

def main():
    dt = 0.001
    rest_Vm = -70
    spike_Vm = 50
    neuron_threshold = -40
    membrane_time_constant = 1
    membrane_resistance = 1
    absolute_refractory_period = 0
