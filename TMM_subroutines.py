from numba import njit
import numpy as np

# numba jit test results:
# 75 layers, 2 materials, no dispersion
# 21 angles, 1 wavelength, 1 polarizaiton
#
# Nothing jitted:
# 1000 loops, avg: 25.01 ms; best time: 15.43 ms
#
# @njit everything:
# 1000 loops, avg: 1.31 ms; best time: 0.98 ms
#
# Equivalent in MATLAB:
# 1000 loops, avg: 33.9 ms
#
# Equivalent in Lumerical:
# 1000 loops, avg: 5.59 ms


# Set up constants:
pi = np.pi


def TMM(n_BG_in, n_BG_out, thicknesses, indices, wavelengths, angles, polarization):
    """This is the main function of this module.
    It performs the TMM for a complete stack and returns the transmission, reflection and absorption coefficients.
    Can run the calculation for multiple wavlengths and/or incident angles simultaneously.

    Given the parameters of the thin film stack (index_list, thickness_list, and substrate indices), and the
    calculation parameters (incident angle, k0, polarization) this function calculates the transmission, reflection and
    absorption coefficients as a function of wavelength AND incident angle.

    It uses the cos(theta) and sin(theta) Transfer-Matrix formalism.
    See Chapter 10 of D. A. Steck, Classical and Modern Optics, or Ch 2.4 of Macleod for a full derivation of the formalism."""
    n_BG_in_array, n_BG_out_array, angles_rad_array, k0_array, indices_array, thicknesses = \
        initialize_data(n_BG_in, n_BG_out, thicknesses, indices, wavelengths, angles)
    r, t, R, T, A = TMM_calculation_jit(indices_array, thicknesses, angles_rad_array, n_BG_in_array, n_BG_out_array,
                                        k0_array, polarization)
    return r, t, R, T, A


def initialize_data(n_BG_in, n_BG_out, thicknesses, indices, wavelengths, angles):
    """Takes a list of scalars or python lists, and converts them into """
    # %% Convert all input arguments to numpy arrays of the correct dtype:
    wavelengths = np.array(np.real(wavelengths))
    angles = np.array(np.real(angles))
    thicknesses = np.real(np.array(thicknesses))
    indices = np.array(indices, dtype=complex).ravel()

    # %% Calculate useful parameters
    k0 = 2 * pi / wavelengths  # Free space wave-vector, in /um
    num_wavelengths = wavelengths.shape[0]
    num_angles = angles.shape[0]
    num_layers = indices.shape[0]
    angles_rad = np.deg2rad(angles)

    # %% Broadcast to correct shape for the TMM_calculation() function
    # Reshape to num_wavelengths x num_angles arrays
    n_BG_in_array = np.broadcast_to(n_BG_in, (num_wavelengths, num_angles))
    n_BG_out_array = np.broadcast_to(n_BG_out, (num_wavelengths, num_angles))
    angles_rad_array = np.broadcast_to(angles_rad, (num_wavelengths, num_angles))
    k0_array = np.broadcast_to(k0[:, np.newaxis], (num_wavelengths, num_angles))

    # Reshape to num_layers x num_wavelengths
    indices_array = np.broadcast_to(indices, (num_wavelengths, num_layers)).transpose()

    return n_BG_in_array, n_BG_out_array, angles_rad_array, k0_array, indices_array, thicknesses


# %% Functions for the Transfer Matrix Method TMM:
@njit(cache=True)
def TMM_calculation_jit(index_list, thickness_list, incident_angle_rad, n_BG_in, n_BG_out, k0, polarization='p'):
    """This function performs the actual computation using the output of initialize_data() as inputs.
    It is jitted using numba.njit, which makes it fast but makes some functions a bit unpythonic."""

    M = calculate_M_matrix(index_list, thickness_list, n_BG_in, incident_angle_rad, k0, polarization)
    r, t, R, T, A = calculate_RTA(M, k0, incident_angle_rad, n_BG_in, n_BG_out, polarization)
    return r, t, R, T, A


@njit
def calculate_RTA(M, k0, incident_angle_rad, n_BG_in, n_BG_out, polarization):
    """Given the M matrix and a handful of inputs, returns R, T, and A."""
    outgoing_angle_rad = angle_in_medium(n_BG_in, incident_angle_rad, n_BG_out)

    cos_incoming = np.cos(incident_angle_rad)
    cos_outgoing = np.cos(outgoing_angle_rad)

    if polarization == 'p':
        alpha_in = n_BG_in / cos_incoming
        alpha_out = n_BG_out / cos_outgoing
    if polarization == 's':
        alpha_in = n_BG_in * cos_incoming
        alpha_out = n_BG_out * cos_outgoing

    A = M[0, 0, ...]
    B = M[0, 1, ...]
    C = M[1, 0, ...]
    D = M[1, 1, ...]

    t = (2 * alpha_in /
         (alpha_in * A + alpha_in * alpha_out * B + C + alpha_out * D))

    r = ((alpha_in * A + alpha_in * alpha_out * B - C - alpha_out * D) /
         (alpha_in * A + alpha_in * alpha_out * B + C + alpha_out * D))

    k_in = k0 * n_BG_in * cos_incoming
    k_out = k0 * n_BG_out * cos_outgoing

    T = np.abs(t) ** 2 * np.real(k_out / k_in)
    R = np.abs(r) ** 2
    A = 1 - R - T

    return r, t, R, T, A


@njit
def calculate_alpha_phi(n, L, n_BG_in, incident_angle_rad, k0, polarization):
    """Given the index n and thickness L of a layer, calculates the characteristic impedance alpha
    and the phase phi of a given layer as a function of incident angle and k-vector."""
    n = np.expand_dims(n, axis=1)
    cos_angle = np.cos(angle_in_medium(n_BG_in, incident_angle_rad, n))

    alpha = n
    if polarization == 'p':
        alpha = alpha / cos_angle
    if polarization == 's':
        alpha = alpha * cos_angle

    phi = k0 * n * L * cos_angle

    return alpha, phi


@njit
def calculate_F_matrix(alpha, phi, num_wavelengths, num_angles):
    """Given the characteristic impedance alpha and the phase phi of a given layer, calculates
     the characteristic matrix (2 x 2 x num_wavelengths x num_angles) for that layer."""
    F = identity_4D(num_wavelengths, num_angles)
    F[0, 0] = np.cos(phi)  # F11
    F[1, 1] = F[0, 0]  # F22
    F[0, 1] = -1j * np.sin(phi) / alpha  # F12
    F[1, 0] = F[0, 1] * alpha ** 2  # F21
    return F


@njit
def calculate_M_matrix(index_list, thickness_list, n_BG_in, incident_angle_rad, k0, polarization):
    """Calculates the individual characteristic matrices F for each layer. Then it multiplies them
    all up to return the M matrix for the entire stack."""

    # %%Calculate the F matrix for each layer
    num_wavelengths = k0.shape[0]
    num_angles = incident_angle_rad.shape[1]
    num_layers = index_list.shape[0]

    F_list = identity_5D(num_wavelengths, num_angles, num_layers)
    for i, (n, L) in enumerate(zip(index_list, thickness_list)):
        alpha, phi = calculate_alpha_phi(n, L, n_BG_in, incident_angle_rad, k0, polarization)
        F_list[..., i] = calculate_F_matrix(alpha, phi, num_wavelengths, num_angles)

    # %% Multiply up all the Fs, i.e., M = F * F * F...
    # Without numba, we use:
    # M = identity_4D(num_wavelengths, num_angles)
    # for i in range(num_layers):
    #     M = np.einsum('ijml, jkml -> ikml', M, F_list[:, :, :, :, i])  # Equivalent to M @ F

    # Einsum isn't recogognized by numba. To get our speedup, we manually unroll the loop:
    M = F_list[..., 0]
    for i in range(num_wavelengths):
        for j in range(num_angles):
            for k in range(1, num_layers):
                A = M[:, :, i, j].copy()  # Critical
                B = F_list[:, :, i, j, k]
                M[0, 0, i, j] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
                M[0, 1, i, j] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
                M[1, 0, i, j] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
                M[1, 1, i, j] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    return M


@njit
def angle_in_medium(n_BG, theta, n):
    """Takes the incident angle (from a medium of index n_BG_in) and converts it into the
     angle inside that of a medium of refractive index n."""
    theta_in_medium = np.arcsin(n_BG / n * np.sin(theta))
    theta_in_medium = correct_arcsin_branch(theta_in_medium, n)
    return theta_in_medium


@njit
def correct_arcsin_branch(theta, n):
    """Tests for stability, and fixes instable points to the correct branch (theta -> pi - theta)
    For more info, see Appendix D of arXiv:1603.02720 [physics.comp-ph]"""

    theta = np.asarray(theta)  # ensures the function will also work if there's just 1 item in the list
    unstable_angles = np.imag(np.conj(n) * np.cos(theta)) < 0

    # theta[unstable_angles] = pi - theta[unstable_angles] # Unsupported by numba, unfortunately, so behold this monstrosity:
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if unstable_angles[i, j]:
                theta[i, j] = pi - theta[i, j]

    return theta


@njit
def identity_4D(y, z):
    """Creates a 4D identity matrix of dimension dim x dim x y x z"""
    out = np.array([[1 + 0j, 0], [0, 1 + 0j]])
    out = out.repeat(y * z).reshape((2, 2, y, z))
    return out


@njit
def identity_5D(x, y, z):
    """Creates a 5D identity matrix of dimension dim x dim x y x z"""
    out = np.array([[1 + 0j, 0], [0, 1 + 0j]])
    out = out.repeat(x * y * z).reshape((2, 2, x, y, z))
    return out


def import_from_csv(filename):
    """Imports csv files where the first column is a list of layer thicknesses (in um) and the second column
    is a complex list of refractive indices"""

    with open(filename) as f:
        thicknesses, indices = np.genfromtxt(f, delimiter=',', dtype=np.complex, skip_header=1, unpack=True)

    thicknesses = np.real(thicknesses)
    return thicknesses, indices


def import_from_mat(filename):
    import mat73
    data_dict = mat73.loadmat(filename)
    thicknesses = data_dict['layers'] # units: m
    thicknesses = thicknesses*1e6 # units: um
    thicknesses = thicknesses[1:-1] # Lumerical included first and last layers of thickness 0. Python doesn't use these
    indices = data_dict['n2']
    indices = indices[1:-1] # Lumerical included first and last layers of index 1. Python doesn't use these
    return thicknesses, indices


def import_from_npz(filename):
    GD_data = np.load(filename)
    thicknesses = GD_data['thickness_list']*1e6
    indices_wBG = GD_data['index_list_wBG']
    indices = indices_wBG[1:-1]
    return thicknesses, indices


def import_structure(filename):
    """

    """
    if filename.endswith('.mat'):
        thicknesses, indices = import_from_mat(filename)
    elif filename.endswith('.npz'):
        thicknesses, indices = import_from_npz(filename)
    elif filename.endswith('.csv'):
        thicknesses, indices = import_from_csv(filename)
    else:
        raise Exception("The filetype you entered is not recognized. Only the following filetypes are supported: *.mat  *.csv  *.npz")
    return thicknesses, indices