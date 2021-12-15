#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Jordan Pagé (EN/FR)
jpage019@uottawa.ca
August 24th 2020

Over-hauled by Orad Reshef
orad@reshef.ca
April 2021

Edits and implementation of new functions by Jordan
May 2021

This code generates a multi-layer stack bi-material spaceplate using gradient
descent, where the parameter space consists of all the layer thicknesses.

This version of the spaceplate code maximises 1/RMSE for a constant (target) R.

Say you want to create a 11-layered device. The code starts off by creating a
swarm containing num_points_in_swarm points (devices) where each device has
num_layers layers (in our case 11) of random thickness. The code starts off by
optimizing the first default_start layers of each device. Once those layers have
been optimized, the next next_layers are included in the optimisation process.
This is repeated until all of the layers have been optimized. In our example,
N_layers = 11,  default_start = 3,  next_layers = 1.
The first 3 layers will be optimised to maximise 1/RMSE as much as possible,
the first 4 (default_start + next_layers), then the first 5 (default_start +
next_layers + next_layers) and so on until the thicknesses of all 11 layers
have been optimised.

When a peak is reached, one of two things can happen:

1. If the device does not have a FOM>stage_2_threshold, it is deleted and a new
   device is created to takes its place. This device got stuck at a local maximum.

2. If the device has a FOM>stage_2_threshold, it will be sent to stage 2. Stage 2
   is the same thing as te first stage, but taking much finer steps. If the
   device's FOM makes it above saving_threshold, we save it and we're done.
   Otherwise, a new random device is created in its place and the process continues

# Not implemented:
# In the final step, two arrays containing the rounded layers of the optimized
# device are created; one that contains the layers rounded down to the nearest
# 2nm increment and one that contains the layers rounded up to the nearest 2nm
# increment. All possible combinations are created and checked and the best one
# is saved. This process can really slow down the code for not a lot of gain
# once you have more than say 20 layers.
"""
#######################################################################

import numpy as np
from TMM_subroutines import TMM, import_structure
from spaceplate import fit_spaceplate_toR, fit_spaceplate, plot_spaceplate_phase, plot_structure
from plot_results import plot_sweep_results
from functools import partial
from copy import deepcopy  # for copying 2D lists
from time import time
import concurrent.futures


def main():
    Main_start_time = time()
    # Initialize multi-processing variables:
    num_computer_cores = 2
    num_points_in_swarm = 100  # Note: pick a number divisible by num_computer_cores
    points_per_core = round(num_points_in_swarm / num_computer_cores)

    # Initialize device variables
    num_layers = 13
    n_BG_in = 1  # index of background medium
    n_BG_out = 1  # index of substrate
    n_low = 1.444  # SiO2 at 1550 nm
    n_high = 3.47638  # Si at 1550 nm
    index_list_wBG = create_index_list(num_layers, n_low, n_high, n_BG_in, n_BG_out)  # includes background on both ends

    # Initialize simulation variables
    wavelength = 1550e-9  # in m
    polarization = 'p'  # string, p or s
    max_angle = 10  # in degrees
    theta = np.linspace(0, max_angle, max_angle + 1)  # range of incident angles

    # Generate thickness_list creator function
    min_layer_thickness = 50
    max_layer_thickness = 200

    # Gradient descent parameters
    target_R = 7  # the desired compression factor R
    stage_2_threshold = 600  # when FOM>stage_2_threshold, run again with a smaller step size
    saving_threshold = 700  # when FOM>saving_threshold, save results to file

    # GD parameters:
    step1 = 0.47e-9  # How much the thickness of a layer changes by during
    step2 = 0.2e-9  # gradient descent in stages 1 & 2 respectively.
    derivative_step = 0.05e-9  # Size of dx in derivative

    # optimize_layers is a dict that keeps track of how many layers each device is optimizing.
    # The default number of starting layers is 3 (see variable default_start). Every time a peak
    # is reached, the element in optimize_layers corresponding to that device will increase by
    # next_layers until all layers are being optimized.
    default_start = 3
    optimize_layers_increase = 1  # When the first (default_start) layers have been optimized, the program will
    # add the next (optimize_layers_increase) layer(s) to the optimisation process.

    filename = f'spaceplate_{num_layers}layers_targetR{target_R}_theta{max_angle}'

    swarms = {}
    optimize_layers = {}

    for i in range(num_computer_cores):
        swarms[i] = createSwarm(points_per_core, num_layers, min_layer_thickness, max_layer_thickness)
        optimize_layers[i] = np.ones(num_points_in_swarm, dtype=int) * default_start

    dt = np.dtype([('wavelength', 'f4'), ('theta', (object, len(theta))), ('index_list_wBG', (object, num_layers + 2)),
                   ('num_layers', 'i4'), ('points_per_core', 'i4'), ('default_start', 'i4'),
                   ('optimize_layers_increase', 'i4'),
                   ('step1', 'f4'), ('step2', 'f4'), ('saving_threshold', 'f4'), ('stage_2_threshold', 'f4'),
                   ('filename', 'U' + str(len(filename))),
                   ('polarization', 'U1'), ('min_layer_thickness', 'i4'), ('max_layer_thickness', 'i4'),
                   ('target_R', 'f4'), ('derivative_step', 'f4')])
    variables = np.array(
        (wavelength, [theta], [index_list_wBG], num_layers, points_per_core, default_start, optimize_layers_increase,
         step1, step2, saving_threshold, stage_2_threshold, filename, polarization, min_layer_thickness,
         max_layer_thickness, target_R, derivative_step), dtype=dt)

    if num_computer_cores == 1:
        stage1(swarms[0], optimize_layers[0], variables, 0)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(stage1, swarms[d], optimize_layers[d], variables, d) for d in swarms]

        for f in concurrent.futures.as_completed(results):
            f.result()

    print('Total runtime:', round(time() - Main_start_time, 2))


def create_index_list(N_layers, n_low, n_high, n_BG_in, n_BG_out):
    """Function that specifies what each layers of the structure will be made of.
    N_layers: number of layers the devices in the swarm will have
    n_BG_in: index of the background medium on input side
    n_BG_out: index of the background medium on substrate side"""

    index_list_wBG = np.ones((N_layers + 2, 1))
    index_list_wBG[0] = n_BG_in

    for i in range(1, N_layers + 1):
        if i % 2 == 0:
            index_list_wBG[i] = n_low
        else:
            index_list_wBG[i] = n_high

    index_list_wBG[-1] = n_BG_out

    return index_list_wBG


def create_thickness_list(num_layers, lowerbound, upperbound):
    """createPoint(): creates a random starting point for the swarm.
    IOW, creates a structure with N layers of random thickness
    between lowerbound and upperbound

    num_layers: number of layers
    lowerbound: thinnest value a layer can have (in nm)
    upperbound: thickest value a layer can have (in nm)"""

    layers = np.random.randint(lowerbound, upperbound, size=num_layers).astype(float) * 10 ** -9

    return layers


def calculate_T_from_device(n_BG_in, n_BG_out, thickness_list, index_list, wavelength, theta, polarization):
    _, t, _, T, _ = TMM(n_BG_in, n_BG_out, thickness_list, index_list, wavelength, theta, polarization)
    return t, T


def get_FOM(thickness_list, variables):
    """thickness_list: thickness of each layer
    index_list_wBG: index of each layer
    wavelength: wavelength at the device is designed to work
    theta: the range of angles of incidence
    target_R: The compression factor R that the code is trying to imitate
    polarization: polarization of incident light"""

    # Unapacking required elements from 'variables'
    wavelength = float(variables['wavelength'])
    index_list_wBG = variables['index_list_wBG'][0]
    theta = variables['theta'][0]
    polarization = str(variables['polarization'])
    target_R = float(variables['target_R'])

    max_fit_angle = max(theta)

    device_thickness = sum(thickness_list)
    n_BG_in = index_list_wBG[0]
    index_list = index_list_wBG[1:-1]
    n_BG_out = index_list_wBG[-1]

    t, T = calculate_T_from_device(n_BG_in, n_BG_out, thickness_list, index_list, [wavelength], theta,
                                   polarization)

    RMSE = fit_spaceplate_toR(theta, t, wavelength, max_fit_angle, target_R, device_thickness)
    return 1 / RMSE


def derivativeR(a, i, variables):
    """Returns derivative as function of R for one layer
    a: layers of device
    i: index of the dimension to be derived
    index_list_wBG: composition of the layers of the device
    wavelength: operating wavelength
    theta: the range of angles of incidence
    target_R: The compression factor R that the code is trying to imitate"""

    # Unapacking required elements from 'variables'
    derivative_step = float(variables['derivative_step'])

    # If you don't use deepcopy(), changing b1 and/or b2 will change the original list (a)
    b1 = deepcopy(a)
    b2 = deepcopy(a)

    # Making the i-th layer take a step in the positive and negative direction to calculate the gradient
    b1[i] = b1[i] + derivative_step
    b2[i] = b2[i] - derivative_step

    # df/dx = lim ( f(a+h) - f(a-h) )/2h where h -> 0
    b = ((get_FOM(b1, variables) - get_FOM(b2, variables)) * 10 ** -9) \
        / (2 * derivative_step)

    return b


def save_device(thickness_list, FOM, variables):
    import os

    # Unapacking required elements from 'variables'
    wavelength = float(variables['wavelength'])
    index_list_wBG = variables['index_list_wBG'][0]
    theta = variables['theta'][0]
    filename = str(variables['filename'])
    polarization = str(variables['polarization'])
    target_R = float(variables['target_R'])

    if FOM >= 10:
        FOM_str = str(round(FOM))
    else:
        FOM_str = str(round(FOM, 2)).replace('.', ',')

    filename = f'{filename}_RMSE_' + FOM_str

    if not os.path.exists('gd_spaceplates/'):
        os.makedirs('gd_spaceplates/')

    filename = 'gd_spaceplates/' + filename
    np.savez(filename, index_list_wBG=index_list_wBG, thickness_list=thickness_list, FOM=FOM, target_R=target_R,
             wavelength=wavelength, theta=theta, polarization=polarization)
    print(f'Saved device: {thickness_list * 1e9}')
    print(f'Saved file: {filename}.npz')


def createSwarm(swarm_size, num_layers, lowerbound, upperbound):
    """Creates a list of size swarm_size of randomly generated 'thickness_list's"""
    return [create_thickness_list(num_layers, lowerbound, upperbound) for _ in range(swarm_size)]


def optimize_single_layer(current_device, max_FOM_this_device, gradient, variables, layer_index):
    """Iterates through one layer of the device, and steps towards the maximum FOM"""
    # Unapacking required elements from 'variables'
    step = float(variables['step1'])

    best_device = deepcopy(current_device)
    optimize_next_layer = False
    while optimize_next_layer is False:  # Keep applying gradient descent to the selected layer

        # First, improve on our best device:
        new_device = deepcopy(best_device)
        new_device[layer_index] = take_a_step(best_device[layer_index], gradient, step)
        new_FOM = get_FOM(new_device, variables)
        delta_FOM = new_FOM - max_FOM_this_device

        if delta_FOM > 0.0001:
            # The layer is worth updating again
            max_FOM_this_device = new_FOM
            best_device = new_device

        elif 0 < delta_FOM < 0.0001:
            # If the change in FOM is this small, save this newer structure, but move on.
            # It's not worth our time to keep going.
            max_FOM_this_device = new_FOM
            best_device = new_device
            optimize_next_layer = True

        else:
            # The last device really was better, so keep that one, and move on.
            optimize_next_layer = True

    return best_device, max_FOM_this_device


def stage1(points, optimize_layers_total, variables, core_num):
    # Unapacking required elements from 'variables'
    N_layers = int(variables['num_layers'])
    points_per_core = int(variables['points_per_core'])
    default_start = int(variables['default_start'])
    optimize_layers_increase = int(variables['optimize_layers_increase'])
    stage_2_threshold = float(variables['stage_2_threshold'])
    min_layer_thickness = int(variables['min_layer_thickness'])
    max_layer_thickness = int(variables['max_layer_thickness'])

    FOMs = [0] * points_per_core  # Figure of merit for each device

    Start_time = time()
    Last_save = Start_time
    Last_time = Start_time
    round_num = 0
    break_loop = False
    best_FOM_ever = 0  # initializing variable best_FOM_ever
    while break_loop is False:  # Run forever
        round_num = round_num + 1  # Keep count of round number

        now = time()
        if now - Last_time > 5:  # Update every 5 seconds
            if best_FOM_ever > 10:
                print(
                    f'---- Core #{core_num} -- best FOM: {round(best_FOM_ever)} -- Iteration #{round_num} -- timestamp: {round(now - Start_time, 2)} ----')
            else:
                print(
                    f'---- Core #{core_num} -- best FOM: {round(best_FOM_ever, 2)} -- Iteration #{round_num} -- timestamp: {round(now - Start_time, 2)} ----')
            Last_time = now

        if now - Last_save > 600:  # Save best every 10 minutes
            print(f'---- Time since last save: {round(now - Last_save, 2)} ----')
            save_device(best_device_ever, best_FOM_ever, variables)
            Last_save = now

        for j in range(points_per_core):  # Iterate through the devices for each core
            current_device = deepcopy(points[j])
            max_FOM_this_device = get_FOM(current_device, variables)

            for i in range(optimize_layers_total[j]):  # Iterate through the selected layers of the device
                gradient = derivativeR(current_device, i, variables)  # getting derivative of the i-th dimension
                current_device, max_FOM_this_device = optimize_single_layer(current_device, max_FOM_this_device,
                                                                            gradient, variables, i)

            if max_FOM_this_device > best_FOM_ever:
                best_FOM_ever = max_FOM_this_device
                best_device_ever = deepcopy(current_device)

            FOMs[j] = max_FOM_this_device
            if device_has_changed(points[j], current_device):
                points[j] = current_device

            else:  # We have run out of optimization using just this layer-optimization scheme

                if have_more_layers_to_optimize(optimize_layers_total[j], N_layers):
                    optimize_layers_total[j] = optimize_layers_total[j] + optimize_layers_increase

                else:  # We have run the optimization on ALL the layers! Do something new:
                    if max_FOM_this_device < stage_2_threshold:
                        # This point is stuck in a local max. Kill it an have a new one take its spot
                        # print('NEW POINT CREATED:')
                        points[j] = create_thickness_list(N_layers, min_layer_thickness, max_layer_thickness)
                        FOMs[j] = 0
                        optimize_layers_total[j] = default_start

                    else:
                        print('------------------')
                        print('Core #' + str(core_num) + ' initiating stage 2')
                        stage2_output = stage2(current_device, variables, round_num)
                        break_loop, stage2_device, stage2_FOM = stage2_output

                        if stage2_FOM >= best_FOM_ever:
                            best_device_ever = deepcopy(stage2_device)
                            best_FOM_ever = stage2_FOM

                        if not break_loop:
                            # Once you're done stage 2, if you don't break the loop, then just kill this point and move on.
                            # print('NEW POINT CREATED:')
                            points[j] = create_thickness_list(N_layers, min_layer_thickness, max_layer_thickness)
                            FOMs[j] = 0
                            optimize_layers_total[j] = default_start

    plot_final_device(best_device_ever, variables)


def stage2(current_device, variables, round_num):
    # Unapacking required elements from 'variables'
    threshold = float(variables['saving_threshold'])
    filename = str(variables['filename'])

    filename = f'{filename}_round{round_num}'

    best_FOM_ever = get_FOM(current_device, variables)
    max_FOM_this_device = best_FOM_ever
    break_loop = False
    previous_device = deepcopy(current_device)
    print('Device sent to stage 2:', previous_device * 1e9)
    print('Starting RMSE:', best_FOM_ever)

    while break_loop is False:  # Run forever
        for i, _ in enumerate(current_device):  # Iterate through the selected layers of the device
            gradient = derivativeR(current_device, i, variables)  # getting derivative of the i-th dimension
            current_device, max_FOM_this_device = optimize_single_layer(current_device, max_FOM_this_device,
                                                                        gradient, variables, i)
        if max_FOM_this_device > best_FOM_ever:
            best_FOM_ever = max_FOM_this_device
            print(f'New best FOM: {best_FOM_ever}')

        if device_has_changed(previous_device, current_device):
            previous_device = deepcopy(current_device)
        else:
            break_loop = True
            if max_FOM_this_device > threshold:  # We're done!
                save_device(previous_device, max_FOM_this_device, variables)
                stop_program = True
                final_device = previous_device
                final_FOM = max_FOM_this_device
            else:
                stop_program = False
                final_device = 0
                final_FOM = 0

    return stop_program, final_device, final_FOM


def take_a_step(layer_thickness, gradient, step):
    ## If the x gradient is positive, the maxima must be in the
    ## positive x direction.
    if gradient > 0:
        new_layer_thickness = layer_thickness + step
    else:
        new_layer_thickness = layer_thickness - step
        new_layer_thickness = max(new_layer_thickness, 2e-9)
    return new_layer_thickness


def device_has_changed(old_device, new_device):
    return not all(old_device == new_device)


def have_more_layers_to_optimize(number_of_layers_being_optimized, total_number_of_layers):
    return number_of_layers_being_optimized < total_number_of_layers


def plot_final_device(thickness_list, variables):
    # Unapacking required elements from 'variables'
    wavelength = float(variables['wavelength'])
    N_layers = int(variables['num_layers'])
    index_list_wBG = variables['index_list_wBG'][0]
    angles = variables['theta'][0]
    polarization = str(variables['polarization'])
    target_R = float(variables['target_R'])

    n_BG_in = index_list_wBG[0]
    n_BG_out = index_list_wBG[-1]
    index_list = index_list_wBG[1:-1]

    max_fit_angle = max(angles)
    angles = np.linspace(0, 60, 601)  # we want to see what happens past the max_fit_angle

    r, t, R, T, A = TMM(n_BG_in, n_BG_out, thickness_list, index_list, [wavelength], angles, polarization)
    phase_fit, fit_result, RMSE = fit_spaceplate(angles, t, np.array([wavelength]), max_fit_angle, global_offset=True)

    plot_sweep_results(np.array([wavelength]), angles, R, T, A, max_fit_angle)
    plot_spaceplate_phase(angles, t, phase_fit, max_fit_angle)
    plot_structure(thickness_list)

    d_eff = fit_result.params['d_eff'].value
    t_total = sum(thickness_list)
    compression_R = d_eff / t_total

    print('')
    print('Simulation parameters')
    print('=====================')
    print(f'Operating wavelength: {wavelength * 1e6} um')
    print(f'Polarization: {polarization}')
    print(f'Max fit angle: {max_fit_angle} degs / {np.sin(max_fit_angle * np.pi / 180):.2f} NA')
    print('')
    print('Device parameters')
    print('=====================')
    print(f'{N_layers} layers')
    print(f'Device thickness d:   {t_total * 1e6:.2f} um')
    print(f'Eff. thickness d_eff: {d_eff * 1e6:.2f} um')
    plusminus = u'\u00b1'
    print(f'Compression factor R: {compression_R:.2f}')
    print(f'Fit RMSE:             {RMSE: .2e}')
    print(f'Fit 1/RMSE:           {1 / RMSE: .2f}')


if __name__ == "__main__":
    main()
