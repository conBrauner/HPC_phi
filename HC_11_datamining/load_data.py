import mat73

def load_lfp(base_dir, subject_name, broadband=False, narrowband=False, load_all_channels=False):

    # Pick the file with the appropriate channels/bandwidth
    if load_all_channels:
        lfp_array = mat73.loadmat(base_dir + f'/{subject_name}_all_lfp.mat')['lfp']
    else:
        if broadband:
            lfp_array = mat73.loadmat(base_dir + f'/{subject_name}_hightheta_broadband_lfp.mat')['lfp_high_theta']
        elif narrowband:
            lfp_array = mat73.loadmat(base_dir + f'/{subject_name}_hightheta_narrowband_lfp.mat')['lfp_high_theta']
        else:
            lfp_array = mat73.loadmat(base_dir + f'/{subject_name}_hightheta_lfp.mat')['lfp_high_theta']

    return lfp_array

def load_session(base_dir, subject_name, verbose=False):

    # Import summary data
    data_dict = mat73.loadmat(base_dir + f'/{subject_name}_sessInfo.mat')

    # Import position data
    position_array = data_dict['sessInfo']['Position']['OneDLocation']
    position_time_stamps = data_dict['sessInfo']['Position']['TimeStamps']

    # Spike data
    all_spike_times = data_dict['sessInfo']['Spikes']['SpikeTimes'] # .shape == (8414607,)
    all_spike_IDs = data_dict['sessInfo']['Spikes']['SpikeIDs']

    # The following arrays share no elements
    pyramidal_IDs = data_dict['sessInfo']['Spikes']['PyrIDs'] # .shape == (120,)
    interneuron_IDs = data_dict['sessInfo']['Spikes']['IntIDs'] # .shape == (17,)

    # The following indicate epoch boundaries in units seconds
    premaze_epoch = data_dict['sessInfo']['Epochs']['PREEpoch']
    maze_epoch = data_dict['sessInfo']['Epochs']['MazeEpoch']
    REM_epochs = data_dict['sessInfo']['Epochs']['REM']

    # Compile all imports in a dictionary for return
    session_data_dict = {
        'all_spike_times': all_spike_times,
        'all_spike_IDs': all_spike_IDs,
        'pyramidal_IDs': pyramidal_IDs,
        'interneuron_IDs': interneuron_IDs,
        'premaze_epoch': premaze_epoch,
        'maze_epoch': maze_epoch,
        'REM_epochs': REM_epochs,
        'position_array': position_array,
        'position_time_stamps': position_time_stamps
    }

    # State predefined features of data in terminal
    if verbose:
        print(data_dict['sessInfo']['Position'])
        print(f'maze epoch: {maze_epoch[0]}s - {maze_epoch[1]}s')

    return  session_data_dict