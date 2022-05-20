import numpy as np
import numba as nb
from HPC_phi_7 import PipelineManager
import os

def single_sim(
    SIM_SPECS,
    ANALYSIS_SPECS,
    param_1_name: str,
    param_2_name: str,
    param_1_var_params: list,
    param_2_var_params: list,
    mesh_type='forcing',
    ARC=True,
    init_nontest=False
    ):

    if init_nontest == True:
        ANALYSIS_SPECS.update({
            'num_cycles': -1,
            'central_tendency_technique': 'mean',
            'encode': True,
            'encode_path': '/'.join([os.getcwd(), 'serializedSims']), 
            'contour_popup_suppress': True
        })

    Simulation = PipelineManager(SIM_SPECS, ANALYSIS_SPECS, ARC=ARC)

    Simulation.add_parameter_subset(param_1_var_params[0], param_1_var_params[1], param_1_var_params[2], param_1_name)
    Simulation.add_parameter_subset(param_2_var_params[0], param_2_var_params[1], param_2_var_params[2], param_2_name)

    Simulation.execute_pipeline(mesh_type)

def neurodynamics_nested_mesh_sim(
    SIM_SPECS,
    ANALYSIS_SPECS,
    adaptation_response_var_params: list,
    adaptation_time_constant_var_params: list,
    mesh_side_length: int,
    mesh_type='neurodynamics',
    ARC=True,
    init_nontest=True
    ):

    if init_nontest == True:
        ANALYSIS_SPECS.update({
            'num_cycles': -1,
            'central_tendency_technique': 'mean',
            'encode': True,
            'encode_path': '/'.join([os.getcwd(), 'serializedSims']), 
            'contour_popup_suppress': True
        })

    for interference_frequency in [9, 10, 11]:

        run_number = 0

        SIM_SPECS.update({
            'interference_frequency': interference_frequency
        })
        if len(adaptation_response_var_params) != 1:
            adaptation_response_subspace = np.linspace(
                adaptation_response_var_params[0],
                adaptation_response_var_params[1],
                num=mesh_side_length
            )
        else:
            adaptation_response_subspace = adaptation_response_var_params
        
        if len(adaptation_time_constant_var_params) != 1:
            adaptation_time_constant_subspace = np.linspace(
                adaptation_time_constant_var_params[0],
                adaptation_time_constant_var_params[1],
                num=mesh_side_length
            )
        else:
            adaptation_time_constant_subspace = adaptation_time_constant_var_params
        
        for adaptation_response_i in adaptation_response_subspace:
            for adaptation_time_constant_j in adaptation_time_constant_subspace:

                print(f'Adaptation response: {adaptation_response_i}')
                print(f'Adaptation time constant: {adaptation_time_constant_j}')

                ANALYSIS_SPECS.update({
                    'run_number': run_number
                })

                SIM_SPECS.update({
                    'adaptation_response_constant': adaptation_response_i,
                    'adaptation_time_constant': adaptation_time_constant_j
                })

                Simulation = PipelineManager(SIM_SPECS, ANALYSIS_SPECS, ARC=ARC)

                Simulation.add_parameter_subset(20, 50, 40, 'theta_amplitude')
                Simulation.add_parameter_subset(20, 50, 40, 'interference_amplitude')

                Simulation.execute_pipeline(mesh_type)

                run_number += 1

def stochastic_nested_mesh_sim(
    SIM_SPECS,
    ANALYSIS_SPECS,
    OU_mu_var_params: list,
    OU_sigma_var_params: list,
    mesh_side_length: int,
    mesh_type='stochastic',
    ARC=True,
    init_nontest=True
    ):

    if init_nontest == True:
        ANALYSIS_SPECS.update({
            'num_cycles': -1,
            'central_tendency_technique': 'mean',
            'encode': True,
            'encode_path': '/'.join([os.getcwd(), 'serializedSims']), 
            'contour_popup_suppress': True
        })

    for interference_frequency in [9, 10, 11]:

        run_number = 0

        SIM_SPECS.update({
            'interference_frequency': interference_frequency
        })

        OU_mu_subspace = np.linspace(
            OU_mu_var_params[0],
            OU_mu_var_params[1],
            num=mesh_side_length
        )
        OU_sigma_subspace = np.linspace(
            OU_sigma_var_params[0],
            OU_sigma_var_params[1],
            num=mesh_side_length
        )
        
        for OU_mu_i in OU_mu_subspace:
            for OU_sigma_j in OU_sigma_subspace:

                print(f'OU mu: {OU_mu_i}')
                print(f'OU sigma: {OU_sigma_j}')

                ANALYSIS_SPECS.update({
                    'run_number': run_number
                })

                SIM_SPECS.update({
                    'OU_mu': nb.float32(OU_mu_i),
                    'OU_sigma': OU_sigma_j
                })

                Simulation = PipelineManager(SIM_SPECS, ANALYSIS_SPECS, ARC=ARC)

                Simulation.add_parameter_subset(20, 50, 40, 'theta_amplitude')
                Simulation.add_parameter_subset(20, 50, 40, 'interference_amplitude')

                Simulation.execute_pipeline(mesh_type)

                run_number += 1

def single_analysis(
    SIM_SPECS,
    ANALYSIS_SPECS,
    filename,
    mesh_type,
    supradir=False,
    ARC=True,
    init_nontest=True
    ):

    if init_nontest == True:
        ANALYSIS_SPECS.update({
            'num_cycles': -1,
            'central_tendency_technique': 'mean',
            'encode': False,
            'decode': True,
            'contour_popup_suppress': True
        })

    if supradir:
        decode_path = '/'.join([os.getcwd(), 'serializedSims', 'GT_' + filename.split('_')[0], supradir, filename])
    else:
        decode_path = '/'.join([os.getcwd(), 'serializedSims', 'GT_' + filename.split('_')[0], filename])

    ANALYSIS_SPECS.update({
        'decode_path': decode_path
    })

    Simulation = PipelineManager(SIM_SPECS, ANALYSIS_SPECS, ARC=ARC)

    Simulation.add_parameter_subset(20, 50, 40, 'theta_amplitude')
    Simulation.add_parameter_subset(20, 50, 40, 'interference_amplitude')

    Simulation.execute_pipeline(mesh_type)

def set_analysis(
    SIM_SPECS,
    ANALYSIS_SPECS,
    mesh_type,
    GT,
    supradir=False,
    ARC=True,
    init_nontest=True
    ):

    if init_nontest == True:
        ANALYSIS_SPECS.update({
            'num_cycles': -1,
            'central_tendency_technique': 'mean',
            'encode': False,
            'decode': True,
            'contour_popup_suppress': True
        })

    if supradir:
        decode_path_base = '/'.join([os.getcwd(), 'serializedSims', 'GT_' + GT, supradir])
    else:
        decode_path_base = '/'.join([os.getcwd(), 'serializedSims', 'GT_' + GT])

    directory_bytestring = os.fsencode(decode_path_base)

    for file in os.listdir(directory_bytestring):

        filename = os.fsdecode(file)
        decode_path = '/'.join([decode_path_base, filename])

        ANALYSIS_SPECS.update({
            'decode_path': decode_path
        })

        Simulation = PipelineManager(SIM_SPECS, ANALYSIS_SPECS, ARC=ARC)

        Simulation.add_parameter_subset(20, 50, 40, 'theta_amplitude')
        Simulation.add_parameter_subset(20, 50, 40, 'interference_amplitude')

        Simulation.execute_pipeline(mesh_type)







