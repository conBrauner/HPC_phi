import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import mode
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut

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

        # Indicates if a spike occurred at the last time step 
        self.spikeState = False

        # Set parameters of the refractory period
        self.refractory_period_length = self.absolute_refractory_period//self.dt # Use the timstep to convert a desired refractory period duration (ms) to length (indices) 
        self.refractory_period_counter = 0 # Initialize the model as being 'zero steps' along the refractory period


    def _updateVm(self, time_series_index, forcing, dt):
        """
        Increments model membrane potential according to the sum of all inputs. Also generates a spike if threshold is met or exceeded, then returns potential to resting at the next step.
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
            if self.model.spikeState == True and self.model.refractory_period_counter == 0: # HARDCODED FOR LIF: Refer to model state to determine if current step corresponds to a spike
                self.spike_times[time_series_index] = 1 
        print("Simulation completed") # Indicate to operator that simulation is finished

def forcingFunction(lambdaFunction, num_timesteps, dt):
    """
    Given a lambda function, passes in N inputs which increment by dt where N = num_timesteps. 
    Output is a list with length num_timesteps where each element solves y = lambdaFunction = f(t) for t on interval [0, dt*(num_timesteps - 1)]
    """
    return list(map(lambdaFunction, list(map(lambda t: t*dt, range(num_timesteps))))) # Note the nested function for scaling each element by dt
def packageParameters_LIFNeuron(rest_Vm, spike_Vm, neuron_threshold, membrane_time_constant, membrane_resistance, absolute_refractory_period, dt):
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
                           "dt": dt}
    return parameterDictionary
def plotSolution(SimulationObject, thetaRhythm, cycle_boundary_indices, darkBackground=False, suppress=False):
    """
    Takes the completed simulation and thetaRhythm (really this could be any oscillation, since we merely plot it) and plots these in two rows with shared x axis
    - enabling darkBackground option makes all axes, tick marks, plotted lines and text labels white. Useful for Notion, Manim and dark background slide decks
    """
    if not suppress:
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
            for boundary_index in cycle_boundary_indices:
                axes[1].vlines(SimulationObject.timeAxis[boundary_index], min(thetaRhythm), max(thetaRhythm), linestyle='dashed', color='silver', linewidth=0.8)

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
        spike_phi = list(map(lambda i, j: j*theta_phase_time_series[i], range(len(SimulationObject.spike_times)), SimulationObject.spike_times))

    return spike_phi, theta_phase_time_series
def kernelDensityEstimation(spike_phi, cycleNumber, plotPDF_Estimation=True):
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
        print("Optimizing kernel bandwidth on theta cycle {}".format(cycleNumber))
        bandwidth_space = 10**np.linspace(-1, 1, 100) # Allows bandwidths to take on values between 10**-1 and 10**1
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), # Optimize prediction accuracy using the SAME kernel as what the KDE will ultimately use (below)
                                        {'bandwidth': bandwidth_space}, # The bandwidths we'll assess in cross-validation
                                        cv=LeaveOneOut()) # Cross-validation will 'leave-one-out', i.e. only set asside 1 spike to assess how good the KDE is (since we haven't many spikes to begin with)
        grid.fit(spike_phi_2D) # Perform the optimization
        optimal_bandwidth = grid.best_params_['bandwidth'] # Get the optimal bandwidth for the true KDE
        print("Optimal bandwidth on theta cycle {}: {} Hz".format(cycleNumber, optimal_bandwidth))

    else: # Otherwise simply arbitrate the bandwidth, we only call it optimal bandwidth so that it's used in the KernelDensity instantiation (just below)
        optimal_bandwidth = 1.5 # Generally appears to range from 0.7 to 2.7, no clear relationship between number of spikes and this
        print("Single spike on theta cycle {} precludes optimization; setting bandwidth at: {}".format(cycleNumber, optimal_bandwidth))

    KDE = KernelDensity(bandwidth=optimal_bandwidth, kernel='gaussian') # Instantiate the KDE
    KDE.fit(spike_phi_2D) # Optimize it using the intracycle spike phi

    phi_interval = np.reshape(np.linspace(0, 2*np.pi, num=100), (-1, 1)) # A linspace vector which allows us to plot the KDE over [0, 2*pi) and extract a value of phi corresponding to max probability
    ln_PDF_estimation = KDE.score_samples(phi_interval) # For some reason you can only extract the ln() of the estimate PDF from the model
    
    if plotPDF_Estimation: # If operator requested a figure
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
def cycle_CentralTendency(theta_phase_time_series, spike_phi, central_tendency_mode=0, cycleStart=0, plotPDF_Estimation=True):
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

    print("Cycle boundary indices: {}".format(cycle_boundary_indices))

    if central_tendency_mode == 'mean':

        cycle_start_index = 0 # On the first cycle the left boundary is the first element
        cycle_phi_central_tendencies = [] # Will contain the output mean spike_phi for each cycle

        for i, cycle_end_index in enumerate(cycle_boundary_indices, start=0): # Take each entry to be a right cycle boundary

            phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:cycle_end_index] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries
            print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(np.mean(np.array(phi_values))) # Compute mean on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries
        print("indices of nonzero phi on cycle {}: {}".format(i + 1, phi_indices))

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            print("final theta cycle contains no spikes") # Notify the operator

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
            print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(np.median(np.array(phi_values))) # Compute median on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            print("final theta cycle contains no spikes") # Notify the operator

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
            print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(mode(np.array(phi_values))[0][0]) # Compute mode on all values phi falling on interval [cycle_start, cycle_end)

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            print("final theta cycle contains no spikes") # Notify the operator

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
            print("indices of nonzero phi on cycle {}: {}".format(i, phi_indices))

            if phi_indices.size == 0: # If no spikes occurred on the current cycle
                print("theta cycle {} contains no spikes".format(i)) # Notify the operator
                continue # Proceed to the next theta cycle; no central tendency to compute

            else: # If there were spikes
                phi_values = [] # Initialize a list
                for j in phi_indices: # For each index corresponding to a spike
                    phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
                cycle_phi_central_tendencies.append(kernelDensityEstimation(phi_values, i, plotPDF_Estimation=plotPDF_Estimation))

            cycle_start_index = cycle_end_index # The current right boundary index becomes the start of the next theta cycle interval
            
        phi_indices = np.array(list((map(lambda i: i + cycle_start_index, np.where(spike_phi[cycle_start_index:] > 0)[0])))) # These are the indices of spikes with phi values between the cycle boundaries

        if phi_indices.size == 0: # If no spikes occurred on the current cycle
            print("final theta cycle contains no spikes") # Notify the operator

        else: # If there were spikes
            phi_values = [] # Initialize a list
            for j in phi_indices: # For each index corresponding to a spike
                phi_values.append(spike_phi[j]) # Take the spike's phase from spike_phi
            cycle_phi_central_tendencies.append(kernelDensityEstimation(phi_values, i + 1, plotPDF_Estimation=plotPDF_Estimation))
    

    print("intracycle central tendencies: {}".format(cycle_phi_central_tendencies))

    return np.array(cycle_phi_central_tendencies), cycle_boundary_indices
def constructReturnMap(cycle_phi_central_tendencies, darkBackground=False, suppress=False):

    phi_K_previous = np.delete(cycle_phi_central_tendencies, -1) # All entries except the Kth one can be the (K - 1)th entry
    phi_K = np.delete(cycle_phi_central_tendencies, 0) # All entries except the 0th can be the Kth (i.e. next) entry
    line_of_identity = np.arange(min(cycle_phi_central_tendencies)//1, max(cycle_phi_central_tendencies)//1 + 2) # Creates a line of identity with domain that spans the range of phi central tendencies

    if not suppress:
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

            plt.show()
def simulationAnalysis(SimulationObject, thetaRhythm, theta_LFP_shift, central_tendency_mode=0, cycleStart=0, plotPDF_Estimation=True, sim_fig_suppress=False, sim_fig_dark=False, return_map_suppress=False, return_map_dark=False):
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
    cycle_phi_central_tendencies, cycle_boundary_indices = cycle_CentralTendency(theta_phase_time_series, phi_sequence, central_tendency_mode=central_tendency_mode, cycleStart=cycleStart, plotPDF_Estimation=plotPDF_Estimation)

    # Plot simulation output and Return map
    plotSolution(SimulationObject, thetaRhythm, cycle_boundary_indices, darkBackground=sim_fig_dark, suppress=sim_fig_suppress)
    constructReturnMap(cycle_phi_central_tendencies, darkBackground=return_map_dark, suppress=return_map_suppress)

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
    absolute_refractory_period = 2 # The absolute post-spike refractory period of the neuron (ms)
    parameterDictionary = packageParameters_LIFNeuron(rest_Vm, spike_Vm, neuron_threshold, membrane_time_constant, membrane_resistance, absolute_refractory_period, dt)

    # Analysis Parameters ---------------------------------------------------------------------------
    """ 0 = mean; 1 = median; 2 = mode; 3 = kernel density estimation (KDE); """
    central_tendency_mode = 3 # The selected measure of intracycle phi central tendency
    theta_cycle_boundary_phase = 3*np.pi/2 # This value (in radians) when crossed denotes the beginning of a new cycle; sin(t) default is 3*pi/2; cos(t) default is 0*pi
    suppress_simulation_figure = False
    simulation_figure_dark_background = False
    suppress_return_map_figure = False
    return_map_figure_dark_background = False
    plot_KDE = True

    # Forcing Parameters ----------------------------------------------------------------------------
    theta_frequency = 12 # In Hz
    interference_frequency = 12.5 # In Hz

    omega_theta = 2*np.pi*((theta_frequency**-1)*1000)**(-1) # Natural (angular) frequency corresponding to hippocampal theta LFP oscillations (radians/ms); NUMBER IS PERIOD IN ms
    omega_interference = 2*np.pi*((interference_frequency**-1)*1000)**(-1) # Natural (angular) frequency of interloping oscillator (radians/ms); NUMBER IS PERIOD IN ms

    amplitude_theta = 70.0 # Amplitude of hippocampal theta LFP oscillations (mV) might be pA
    amplitude_interference = 70.0 # Amplitude of interloping oscillator (mV)

    theta_LFP_shift = 0 #amplitude_theta # Theta local field potential shift, corresponds to a vertical translation of the sinusoid
    interference_LFP_shift = 0 #amplitude_interference # Interloping oscillator local field potential shift, corresponds to a vertical translation of the sinusoid

    # ========== CREATE FORCING FUNCTIONS ===========================================================

    # Functions defining independent field oscillations using forcing parameters
    thetaFunction = lambda t: amplitude_theta*np.sin(omega_theta*t) + theta_LFP_shift # Corresponds to hippocampal theta sinusoidal oscillations
    interferenceFunction = lambda t: amplitude_interference*np.sin(omega_interference*t) + interference_LFP_shift # Corresponds to some interferring oscillation

    # Generate the theta and interference oscillations over specified timeseries, then superimpose the two
    thetaRhythm = forcingFunction(thetaFunction, num_timesteps, dt) # Generate list of theta  rhythm amplitudes as a function of time
    interferenceRhythm = forcingFunction(interferenceFunction, num_timesteps, dt) # Generates list of interferring oscillation amplitude as a function of time
    fieldOscillation = np.array([x + y for x, y in zip(thetaRhythm, interferenceRhythm)]) # Add the two lists element-wise, then convert to numpy array

    # DEBUGGING: Plot the analytic signal projected onto the complex plane, useful to determine appropriate value of 'theta_cycle_boundary_phase' 
    # plt.close()
    # fig, ax = plt.subplots()
    # ax.scatter(hilbert(thetaRhythm).real[0], hilbert(thetaRhythm).imag[0])
    # plt.show()

    # ========== INSTANTIATE MODEL, RUN SIMULATION AND ANALYZE OUTPUT ===============================

    # Instantiate LIFNeuron
    Neuron = LIFNeuron(parameterDictionary)

    # Create and run simulation
    NeuronSim = Simulation(Neuron)
    NeuronSim.runSim(fieldOscillation, num_timesteps, dt)

    simulationAnalysis(NeuronSim, thetaRhythm, theta_LFP_shift, central_tendency_mode=central_tendency_mode, cycleStart=theta_cycle_boundary_phase, plotPDF_Estimation=plot_KDE, sim_fig_suppress=suppress_simulation_figure, sim_fig_dark=simulation_figure_dark_background, return_map_suppress=suppress_return_map_figure, return_map_dark=return_map_figure_dark_background)

if __name__ == "__main__":
    main()