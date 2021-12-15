#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to spaceplates
"""

import numpy as np

pi = np.pi


###############################
def fit_spaceplate_toR(angles, t, wavelengths, max_fit_angle, target_R, device_thickness):
    """Takes t as a function of angle, and calculates a fit to the spaceplate transfer function
    FOR A SPECIFIED R.
        phi = -k * d_eff * (1 - np.cos(angle))
        The term of 1 here  ^ sets the global phase offset to 0

    Returns the RMSE."""

    # Calculate the phase, and prepare the data to be fitted
    phase = np.unwrap(np.angle(t)).ravel()
    phase = phase - phase[0]

    angles_rad = np.deg2rad(angles).ravel()

    # Define a new fit region.
    angles_to_keep = angles < max_fit_angle
    angle_fitregion = angles_rad[angles_to_keep]
    phase_fitregion = phase[angles_to_keep]

    # Define the function you want to fit to
    wavelengths = np.array(wavelengths)
    k = 2 * pi / wavelengths
    phi_SP = -k * target_R * device_thickness * (1 - np.cos(angle_fitregion))

    RMSE = np.sqrt(sum((phase_fitregion - phi_SP) ** 2) / len(angle_fitregion))

    return RMSE


def fit_spaceplate(angles, t, wavelengths, max_fit_angle, global_offset=False):
    """Takes t as a function of angle, and calculates a fit to the spaceplate transfer function.
        phi = -k * d_eff * (1 - np.cos(angle))
        The term of 1 here  ^ sets the global phase offset to 0

    If globaleoffset is true, it fits to the following equation instead:
        phi = -phi_global + k * d_eff * np.cos(angle)
    Returns the fitted phi(angle), the fit object, and the RMSE."""

    import lmfit

    # Calculate the phase, and prepare the data to be fitted
    phase = np.unwrap(np.angle(t)).ravel()
    phase = phase - phase[0]

    angles_rad = np.deg2rad(angles).ravel()

    # Define a new fit region.
    angles_to_keep = angles < max_fit_angle
    angle_fitregion = angles_rad[angles_to_keep]
    phase_fitregion = phase[angles_to_keep]

    # Set guesses and limits for the fit parameter:
    d_eff_guess = 5
    d_eff_min = 0
    d_eff_max = 5000

    phi_global_guess = 0
    phi_global_min = -10
    phi_global_max = 10

    # Define the function you want to fit to
    wavelengths = np.array(wavelengths)
    if global_offset:
        def function_to_fit(fit_parameters, x):
            """Given x and the fit parameters, will return f(x)"""
            # Extract the individual parameters from 'params'
            d_eff = fit_parameters['d_eff']
            phi_global = fit_parameters['phi_global']

            # Calculate f(x)
            k = 2 * pi / wavelengths
            model = phi_global - k * d_eff * (1 - np.cos(x))
            return model
    else:
        def function_to_fit(fit_parameters, x):
            """Given x and the fit parameters, will return f(x)"""
            # Extract the individual parameters from 'params'
            d_eff = fit_parameters['d_eff']

            # Calculate f(x)
            k = 2 * pi / wavelengths
            model = -k * d_eff * (1 - np.cos(x))
            return model

    # Define the function to minimize
    def fcn2min(fit_parameters, x, data):
        model = function_to_fit(fit_parameters, x)
        residuals = model - data
        return residuals

    # Define the fit parameters, perform the fit, and calculate the fitted function
    params = lmfit.Parameters()
    params.add('d_eff', value=d_eff_guess, min=d_eff_min, max=d_eff_max)
    if global_offset:
        params.add('phi_global', value=phi_global_guess, min=phi_global_min, max=phi_global_max)
    fit_result = lmfit.Minimizer(fcn2min, params, fcn_args=(angle_fitregion, phase_fitregion)).minimize()
    phase_fit = function_to_fit(fit_result.params, angles_rad)
    RMSE = np.std(fit_result.residual)

    return phase_fit, fit_result, RMSE


def plot_spaceplate_phase(angles, t, phase_fit, max_fit_angle):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rc('font', size=12)
    matplotlib.rc('text', usetex=False)

    angles = angles.ravel()

    phase = np.unwrap(np.angle(t)).ravel()
    phase = phase - phase[0]

    phase_fit = phase_fit.ravel()

    fig, axs = plt.subplots(1)
    plt.axes(axs)
    plt.plot(angles, phase, label='simulation')
    plt.plot(angles, phase_fit, '--', label='fit')

    plt.xlim([angles.min(), angles.max()])
    plt.axvline(max_fit_angle, 0, 1, linestyle='--', color='k', linewidth=1, label='max fit angle')
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('Incident angle (degrees)')
    plt.ylabel('Phase (rad)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('SpaceplatePhase.png')
    plt.show()


def plot_spaceplate_phase_2devices(angles, t, t2, phase_fit, phase_fit2, max_fit_angle):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    matplotlib.rc('font', size=20)
    # matplotlib.rc('text', usetex=True)

    angles = angles.ravel()

    phase = np.unwrap(np.angle(t)).ravel()
    phase = phase - phase[0]
    phase_fit = phase_fit.ravel()

    phase2 = np.unwrap(np.angle(t2)).ravel()
    phase2 = phase2 - phase2[0]
    phase_fit2 = phase_fit2.ravel()

    fig, axs = plt.subplots(1, figsize=(8.5, 8.5))
    plt.axes(axs)
    plt.plot(angles, phase, 'b', label='Device 1', linewidth=4)
    plt.plot(angles, phase2, 'g', label='Device 2', linewidth=4)
    plt.plot(angles, phase_fit, '--', color='k', label='fits', linewidth=3)
    plt.plot(angles, phase_fit2, '--', color='k', linewidth=3)

    # plt.ylim([min(phase_fit), max(phase_fit[0:round(len(phase_fit)*1.2)])])
    # plt.xlim([angles.min(), angles.max()])
    plt.xlim([angles.min(), max_fit_angle * 1.2])
    index = np.where(angles == max_fit_angle * 1.2)
    plt.ylim([phase_fit2[index], max(max(phase_fit2), max(phase_fit))])
    # plt.plot(np.ones(5)*max_fit_angle, np.linspace(0.05,-100,5), '--k', linewidth=2, label='max fit angle')
    axs.add_patch(
        Rectangle((max_fit_angle, 1), max_fit_angle * 0.25, -9, color='gray', alpha=0.5, zorder=10, hatch='///'))
    # plt.axvline(max_fit_angle, 0, 1, color = 'k', linewidth=1, label='max fit angle')

    plt.xlabel('Incident angle (degrees)', fontsize=36)
    plt.ylabel('Phase (radians)', fontsize=36)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.legend(framealpha=0, prop={'size': 36})
    plt.tight_layout()
    plt.savefig('SpaceplatePhase.png')
    plt.show()


def plot_structure(thicknesses):
    """
    Plots the relative thicknesses of each layer of a bimaterial multilayer stack.
    The function assumes the materials are silicon (Si) and silica (SiO2)
    The positioning of the scale bar is approximate, it may need adjusting
    according to the size of the device.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, figsize=(8.5, 8.5))

    total = 0
    for i in range(len(thicknesses)):
        if i % 2 == 1:
            ax.add_patch(Rectangle((total, 0), thicknesses[i], 3, color="grey"))
        else:
            ax.add_patch(Rectangle((total, 0), thicknesses[i], 3, color="black"))
        total += thicknesses[i]

    # These next two lines plot points out of frame to be able to get the labels associated with each material (color)
    ax.plot(thicknesses[0] / 2, 1, linewidth=7, color="grey", label=r"$\mathrm{Si}$")
    ax.plot(thicknesses[1] / 2, 1, linewidth=7, color="black", label=r"$\mathrm{SiO}_2$")

    d = sum(thicknesses)
    if d > 2 and d < 4:
        scale = 1
        plt.text(total - scale * 6 / 5, 0.14, r'${} \mu m$'.format(scale), rotation=0, color='white', size=24)
    elif d <= 2:
        scale = 0.5
        plt.text(total - scale, 0.14, r'${} \mu m$'.format(scale), rotation=0, color='white', size=28)
    elif d > 4:
        scale = 2
        plt.text(total - scale * 0.9, 0.14, r'${} \mu m$'.format(scale), rotation=0, color='white', size=36)

    ax.add_patch(Rectangle((total - scale * 6 / 5, 0.1), scale, 0.02, color="white"))

    # plt.title("Structure to scale")
    plt.xlim(0, total)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right', prop={'size': 36})
    plt.show()


def plot_structure2(thicknesses, indices):
    """
    This function plots devices that were generated by changing the
    index of each layer and not each thickness. It gives each layer
    a colour code depending on it's index

    turning the range of indices into a grayscale value between 0 and 255.
    The highest index will be black and the lightest will be 10/255 (to
    not have a perfectly white layer).
    ie, black = max(index)/max(index) * 245 + 10 = 255
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    grayscale = [1 - (i.real / (round(max(indices.real)))) for i in indices]
    # Use next line to decrease visual range between highest and lowest index
    # log_grayscale = [m.log(1+i) for i in grayscale]

    fig, ax = plt.subplots(1, figsize=(8.5, 8.5))

    total = 0
    ax.add_patch(Rectangle((total, 0), thicknesses[0], 3, color='cornflowerblue'))
    total = thicknesses[0]

    for i in range(len(thicknesses)):
        ax.add_patch(Rectangle((total, 0), thicknesses[i], 3, color=str(grayscale[i])))
        total += thicknesses[i]

    thicknesses_pos = [thicknesses[i] * i for i in range(len(thicknesses))]

    plt.scatter(thicknesses_pos, np.ones(len(thicknesses)) * 5, c=grayscale, cmap='gray', vmin=0,
                vmax=round(max(indices.real)))
    cb = plt.colorbar(label="Index range", orientation="horizontal",
                      ticks=[0, 3.5, 7, 10.5, 14])  # put ticks at these values
    cb.ax.set_xticklabels(['14', '10.5', '7', '3.5', '0'])  # have the ticks in the locations above be the following:
    cb.ax.tick_params(labelsize=24)

    ax.add_patch(Rectangle((total, 0), total + thicknesses[0], 3, color='cornflowerblue'))
    total += thicknesses[0]

    d = sum(thicknesses)
    scale = 0.1
    plt.text(total / 2 + 0.08, 0.14, r'${} \mu m$'.format(scale), rotation=0, color='black', size=24)

    ax.add_patch(Rectangle((total / 2 + 0.08, 0.1), scale, 0.02, color="black"))

    # plt.title("Structure to scale")
    plt.xlim(0, total)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])

    # plt.legend(loc = 'upper right', prop={'size': 36})
    plt.show()
