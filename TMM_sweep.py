#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Orad Reshef // orad@reshef.ca

Calculates the transmission, reflection, and absorption of a thin-film stack using the Transfer Matrix Method (TMM),
-> as a function of incident angle

TMM formalism and notation from Ch 4.9 of this textbook:
Yariv, A. & Yeh, P. Photonics: Optical electronics in modern communications (Oxford University Press, 2007).
"""


def main():
    import numpy as np
    from TMM_subroutines import TMM, import_structure
    from plot_results import plot_sweep_results
    from spaceplate import plot_spaceplate_phase, fit_spaceplate, plot_structure, plot_structure2

    # Set up toggles
    enable_plots = True
    save_data = True
    enable_spaceplate_plots = True
    do_spaceplate_fit = True
    enable_output_text = True
    enable_plot_structure_bimaterial = True  # Use this one if plotting a device with only 2 materials
    enable_plot_structure_multimaterial = False  # Use this one if plotting a device with more tha two materials

    # Define calculation parameters (as numpy arrays)
    wavelengths = np.array([1.55])  # in um
    polarization = 'p'  # string, p or s
    angles = np.linspace(0, 60, 201)  # degrees

    # Spaceplate parameters
    max_fit_angle = 5  # degrees

    # # Define Structure
    n_BG_in = 1  # index of input medium
    n_BG_out = 1  # index of output medium

    thicknesses, indices = import_structure(
        'example_mat&npz_devices_Si_SiO2/spaceplate_15_target16_RMSE1644,24_theta5.mat')

    r, t, R, T, A = TMM(n_BG_in, n_BG_out, thicknesses, indices, wavelengths, angles, polarization)

    if save_data:
        np.savez('TMM_sweep.npz', r=r, t=t, R=R, T=T, A=A, angles=angles, wavelengths=wavelengths,
                 thicknesses=thicknesses, indices=indices, n_BG_in=n_BG_in, n_BG_out=n_BG_out)

    if do_spaceplate_fit:
        phase_fit, fit_result, RMSE = fit_spaceplate(angles, t, wavelengths, max_fit_angle, global_offset=False)

        d_eff = fit_result.params['d_eff'].value
        t_total = sum(thicknesses)
        compression_R = d_eff / t_total
        compression_R_err = fit_result.params['d_eff'].stderr / t_total

        if enable_output_text:
            print('Simulation parameters')
            print('=====================')
            print(f'Operating wavelength: {wavelengths[0]} um')
            print(f'Polarization: {polarization}')
            print(f'Max fit angle: {max_fit_angle} degs / {np.sin(max_fit_angle * np.pi / 180):.2f} NA')
            print('')
            print('Device parameters')
            print('=====================')
            print(f'{len(thicknesses)} layers')
            print(f'Device thickness d:   {t_total:.2f} um')
            print(f'Eff. thickness d_eff: {d_eff:.2f} um')
            plusminus = u'\u00b1'
            print(f'Compression factor R: {compression_R:.2f} {plusminus} {compression_R_err:.2f}')
            print(f'Fit RMSE:             {RMSE: .2f}')
            print(f'Fit 1/RMSE:           {1 / RMSE: .2f}')

    if enable_plots:
        plot_sweep_results(wavelengths, angles, R, T, A, max_fit_angle)

    if enable_spaceplate_plots:
        plot_spaceplate_phase(angles, t, phase_fit, max_fit_angle)

    if enable_plot_structure_bimaterial:
        plot_structure(thicknesses)

    if enable_plot_structure_multimaterial:
        plot_structure2(thicknesses, indices)


if __name__ == "__main__":
    main()
