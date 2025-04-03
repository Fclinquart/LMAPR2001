import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Standardize plot settings
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

def refraction_index(filename):
    """
    Load refractive index data from a file.

    Parameters:
    filename (str): Path to the file containing refractive index data.

    Returns:
    tuple: A tuple containing:
        - lambda_um (numpy.ndarray): Wavelengths in micrometers.
        - n0 (float): Refractive index of air (assumed to be 1).
        - n1 (numpy.ndarray): Complex refractive index of the metal (silver).
        - n2 (numpy.ndarray): Complex refractive index of the glass.
    """
    data = pd.read_csv("Data/n_k_combined.txt", delim_whitespace=True, skiprows=1, header=None)
    lambda_um = data[0].values  # Wavelengths in micrometers
    n_glass = data[1].values  # Refractive index of glass
    kappa_glass = data[2].values  # Extinction coefficient of glass
    n_silver = data[3].values  # Refractive index of silver
    kappa_silver = data[4].values  # Extinction coefficient of silver
    n0 = 1  # Refractive index of air
    n1 = n_silver - 1j * kappa_silver  # Complex refractive index of silver
    n2 = n_glass - 1j * kappa_glass  # Complex refractive index of glass

    return lambda_um, n0, n1, n2

def n_k(filename):
    """
    Load refractive index and extinction coefficient data from a file.

    Parameters:
    filename (str): Path to the file containing refractive index and extinction coefficient data.

    Returns:
    tuple: A tuple containing:
        - lambda_um (numpy.ndarray): Wavelengths in micrometers.
        - n (numpy.ndarray): Refractive index.
        - k (numpy.ndarray): Extinction coefficient.
    """
    data = pd.read_csv(filename, delim_whitespace=True, skiprows=1, header=None)
    lambda_um = data[0].values  # Wavelengths in micrometers
    n = data[1].values  # Refractive index
    k = data[2].values  # Extinction coefficient
    # replace Nan by 0
    k = np.nan_to_num(k)
    
    return lambda_um, n, k

def snells(n0, n1, phi0):
    """
    Calculate the angle of refraction using Snell's law.

    Parameters:
    n0 (float): Refractive index of the first medium.
    n1 (float): Refractive index of the second medium.
    phi0 (float): Angle of incidence in degrees.

    Returns:
    float: Angle of refraction in degrees.
    """
    print("Task 1 : Calculating angle of refraction...")
    phi0 = np.radians(phi0)
    phi1 = np.arcsin(n0 * np.sin(phi0) / n1)
    return phi1

def snells_law(n0, n1, n2, phi0):
    """
    Calculate the angle of refraction using Snell's law for two interfaces.

    Parameters:
    n0 (float): Refractive index of the first medium.
    n1 (float): Refractive index of the second medium.
    n2 (float): Refractive index of the third medium.
    phi0 (float): Angle of incidence in degrees.

    Returns:
    tuple: A tuple containing:
        - phi1 (float): Angle of refraction at the first interface in degrees.
        - phi2 (float): Angle of refraction at the second interface in degrees.
    """
    print("Task 2 : Calculating angle of refraction...")
    phi0 = np.radians(phi0)
    phi1 = np.arcsin(n0 * np.sin(phi0) / n1)
    phi2 = np.arcsin(n0 * np.sin(phi0) / n2)
    return phi1, phi2

def reflectivity_semi_infinite_layer(n0, n1, phi0):
    """
    Calculate the reflectivity of a semi-infinite layer.

    Parameters:
    n0 (float): Refractive index of the first medium.
    n1 (float): Refractive index of the second medium.
    phi0 (float): Angle of incidence in degrees.

    Returns:
    float: Reflectivity of the material.
    """
    print("Task 1 : Calculating reflectivity for semi-infinite layer...")
    phi0 = np.radians(phi0)
    phi1 = snells(n0, n1, phi0)
    
    # Calculate the reflection coefficients for s- and p-polarized light
    r_s = (n0 * np.cos(phi0) - n1 * np.cos(phi1)) / (n0 * np.cos(phi0) + n1 * np.cos(phi1))
    r_p = (n1 * np.cos(phi0) - n0 * np.cos(phi1)) / (n1 * np.cos(phi0) + n0 * np.cos(phi1))
    
    # Calculate the reflectivity as the average of |r_s|^2 and |r_p|^2
    R = (np.abs(r_s)**2 + np.abs(r_p)**2) / 2
        
    return R

def compute_R_T_circular(n0, n1, n2, d1, wavelength, phi0):
    """
    Compute reflection (R) and transmission (T) coefficients for circularly polarized light.

    Parameters:
    n0, n1, n2 (complex): Complex refractive indices of the three layers (air, metal, substrate).
    d1 (float): Thickness of the metal layer (in µm).
    wavelength (float): Wavelength of light (in µm).
    phi0 (float): Angle of incidence in radians (in medium 0, e.g., air).

    Returns:
    tuple: A tuple containing:
        - R (float): Reflectivity.
        - T (float): Transmissivity.
        - A (float): Absorbance.
    """
    
    phi0 = np.radians(phi0)
    
    sin_phi0 = np.sin(phi0)
    sin_phi1 = (n0 / n1) * sin_phi0
    phi1 = np.arcsin(sin_phi1)
    sin_phi2 = (n1 / n2) * np.sin(phi1)
    phi2 = np.arcsin(sin_phi2)

    # Compute cosines of angles
    cos_phi0 = np.cos(phi0)
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)

    # Fresnel coefficients for p-polarization (equations 4.41-4.46)
    r01p = (n1 * cos_phi0 - n0 * cos_phi1) / (n1 * cos_phi0 + n0 * cos_phi1)
    r12p = (n2 * cos_phi1 - n1 * cos_phi2) / (n2 * cos_phi1 + n1 * cos_phi2)

    t01p = (2 * n0 * cos_phi0) / (n1 * cos_phi0 + n0 * cos_phi1)
    t12p = (2 * n1 * cos_phi1) / (n2 * cos_phi1 + n1 * cos_phi2)

    # Fresnel coefficients for s-polarization (equations 4.43-4.48)
    r01s = (n0 * cos_phi0 - n1 * cos_phi1) / (n0 * cos_phi0 + n1 * cos_phi1)
    r12s = (n1 * cos_phi1 - n2 * cos_phi2) / (n1 * cos_phi1 + n2 * cos_phi2)

    t01s = (2 * n0 * cos_phi0) / (n0 * cos_phi0 + n1 * cos_phi1)
    t12s = (2 * n1 * cos_phi1) / (n1 * cos_phi2 + n2 * cos_phi1)

    # Phase thickness beta (equation 4.32 corrected)
    beta = 2 * np.pi * (d1 / wavelength) * np.sqrt(n1**2 - n0**2 * sin_phi0**2)

    beta = 2 * np.pi * d1 * n1 * cos_phi1 / wavelength

    # Reflection coefficients (equations 4.37-4.38)
    R_p = (r01p + r12p * np.exp(-2j * beta)) / (1 + r01p * r12p * np.exp(-2j * beta))
    R_s = (r01s + r12s * np.exp(-2j * beta)) / (1 + r01s * r12s * np.exp(-2j * beta))

    # Transmission coefficients (equations 4.39-4.40)
    T_p = (t01p * t12p * np.exp(-1j * beta)) / (1 + r01p * r12p * np.exp(-2j * beta))
    T_s = (t01s * t12s * np.exp(-1j * beta)) / (1 + r01s * r12s * np.exp(-2j * beta))

    # Circular polarization: average s and p components
    R = (np.abs(R_p)**2 + np.abs(R_s)**2) / 2
    T = (np.abs(T_p)**2 + np.abs(T_s)**2) / 2

    correction_factor = (n2 * cos_phi2) / (n0 * cos_phi0)

    A = 1 - R - T * correction_factor

    return R, correction_factor * T, A

def plot_R_T_A_fixed_phi0_and_d(n0, n1, n2, d1, lambda_um, phi0, title="", save=False):
    """
    Plot reflectivity, transmissivity, and absorbance for fixed angle of incidence and thickness.

    Parameters:
    n0 (float): Refractive index of air.
    n1 (numpy.ndarray): Complex refractive index of the metal.
    n2 (numpy.ndarray): Complex refractive index of the glass.
    d1 (float): Thickness of the metal layer (in µm).
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    phi0 (float): Angle of incidence in degrees.
    title (str): Title of the plot.
    save (bool): Whether to save the plot as a file.
    """
    R, T, A = compute_R_T_circular(n0, n1, n2, d1, lambda_um, phi0)
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_um, R, label="Reflectivity")
    plt.plot(lambda_um, T, label="Transmissivity")
    plt.plot(lambda_um, A, label="Absorbance")
    plt.xlabel("Wavelength (µm)")
    plt.xscale('log')
    plt.axvspan(0.38, 0.8, color="yellow", alpha=0.2, label="Visible Spectrum")
    plt.axvspan(0.2, 0.38, color="purple", alpha=0.2, label="UV Spectrum")
    plt.axvspan(0.8, 20, color="red", alpha=0.2, label="IR Spectrum")
    plt.ylabel("R, T, A")
    plt.legend()
    if title != "":
        plt.title(title)
    if save:
        plt.savefig("Output/RTA_phi0_d/RTA_phi0_{}nm_{}.png".format(d1 * 1e3, phi0))
    else:
        plt.show()

def power_spectrum(lambda_um, lambda_min, lambda_max, T, I0=1000):
    """
    Calculate the power spectrum within a given wavelength range.

    Parameters:
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    lambda_min (float): Minimum wavelength of the range.
    lambda_max (float): Maximum wavelength of the range.
    T (numpy.ndarray): Transmissivity values.
    I0 (float): Incident power (default is 1000 W/m²).

    Returns:
    float: Integrated power within the specified wavelength range.
    """
    mask = (lambda_um >= lambda_min) & (lambda_um <= lambda_max)
    integrand = np.real(T[mask]) * I0 * lambda_um[mask]
    return np.trapz(integrand, lambda_um[mask])

def power_spectrum_solar(lambda_um, lambda_min, lambda_max, I, T):
    """
    Calculate the solar power spectrum within a given wavelength range.

    Parameters:
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    lambda_min (float): Minimum wavelength of the range.
    lambda_max (float): Maximum wavelength of the range.
    I (numpy.ndarray): Solar irradiance values.
    T (numpy.ndarray): Transmissivity values.

    Returns:
    float: Integrated solar power within the specified wavelength range.
    """
    I = I  # Convert I from W/nm·m² to W/µm·m²
    mask = (lambda_um >= lambda_min) & (lambda_um <= lambda_max)
    lambda_filtered = lambda_um[mask]
    I_filtered = I[mask]
    T_filtered = T[mask]
    integrand = np.real(T_filtered) * I_filtered * lambda_filtered
    power = np.trapz(integrand, lambda_filtered)
    return power

def power_ratio(lambda_um, lambda_min, lambda_max, T, I0=1000):
    """
    Calculate the power ratio within a given wavelength range.

    Parameters:
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    lambda_min (float): Minimum wavelength of the range.
    lambda_max (float): Maximum wavelength of the range.
    T (numpy.ndarray): Transmissivity values.
    I0 (float): Incident power (default is 1000 W/m²).

    Returns:
    float: Power ratio within the specified wavelength range.
    """
    power_spec = power_spectrum(lambda_um, lambda_min, lambda_max, T, I0)
    integrand = np.real(T) * I0 * lambda_um
    total_power = np.trapz(integrand, lambda_um)
    return np.real(power_spec / total_power)

def power_ratio_solar(lambda_um, lambda_min, lambda_max, I, T):
    """
    Calculate the solar power ratio within a given wavelength range.

    Parameters:
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    lambda_min (float): Minimum wavelength of the range.
    lambda_max (float): Maximum wavelength of the range.
    I (numpy.ndarray): Solar irradiance values.
    T (numpy.ndarray): Transmissivity values.

    Returns:
    float: Solar power ratio within the specified wavelength range.
    """
    power_spec = power_spectrum_solar(lambda_um, lambda_min, lambda_max, I, T)
    integrand = I * lambda_um * np.real(T)
    total_power = np.trapz(integrand, lambda_um)
    print("total power", total_power)
    return np.real(power_spec / total_power)

def optimal_thickness_d(n0, n1, n2, lambda_um, phi0, I, plot=False):
    """
    Find the optimal thickness of the metal layer that maximizes the power ratio in the visible spectrum.

    Parameters:
    n0 (float): Refractive index of air.
    n1 (numpy.ndarray): Complex refractive index of the metal.
    n2 (numpy.ndarray): Complex refractive index of the glass.
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    phi0 (float): Angle of incidence in degrees.
    I (numpy.ndarray): Solar irradiance values.
    plot (bool): Whether to plot the power ratio versus thickness.

    Returns:
    float: Optimal thickness in micrometers.
    """
    d_values = np.linspace(0, 100e-3, 1000)
    power_ratios = []
    for d in d_values:
        R, T, A = compute_R_T_circular(n0, n1, n2, d, lambda_um, phi0)
        power_ratios.append(power_ratio_solar(lambda_um, 0.4, 0.8, T, I))

    if plot:
        plt.plot(d_values, power_ratios)
        plt.xlabel("Thickness (µm)")
        plt.ylabel("Power ratio")
        plt.show()

    return d_values[np.argmax(power_ratios)]

def Solar_spectrum(filename):
    """
    Load the solar spectrum data from a file.

    Parameters:
    filename (str): Path to the file containing solar spectrum data.

    Returns:
    numpy.ndarray: Solar irradiance values.
    """
    data = pd.read_csv(filename, delim_whitespace=True, skiprows=1, header=None)
    I = data[5].values * 1e3  # Convert to W/m²/µm
    return I

def plot_solar_spectrum(lambda_um, I):
    """
    Plot the solar spectrum.

    Parameters:
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    I (numpy.ndarray): Solar irradiance values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_um, I, label="AM1.5 Global Spectrum ASTM G-173", color="orange")
    plt.xlabel("Wavelength (µm)", fontsize=14)
    plt.ylabel("Intensity (W/m²/µm)", fontsize=14)
    plt.xlim(0.2, 3)
    plt.legend()
    plt.show()

def plot_n_k(lambda_um, n, k, title="", log =True):
    """
    Plot the refractive index and extinction coefficient.

    Parameters:
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    n (numpy.ndarray): Refractive index.
    k (numpy.ndarray): Extinction coefficient.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_um, n, label="Refractive index")
    plt.plot(lambda_um, k, label="Extinction coefficient")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("n, k")
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.legend()
    if title != "":
        plt.title(title)
    plt.show()

def spectral_RTA(spectrum: str, lambda_um, n0, n1, n2, d_list: list, phi0, Irradiance):
    """
    Calculate the reflectivity, transmissivity, and absorbance for a given spectrum.

    Parameters:
    spectrum (str): Spectrum type ('UV', 'Visible', or 'IR').
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    n0 (float): Refractive index of air.
    n1 (numpy.ndarray): Complex refractive index of the metal.
    n2 (numpy.ndarray): Complex refractive index of the glass.
    d_list (list): List of thicknesses of the metal layer.
    phi0 (float): Angle of incidence in degrees.
    Irradiance (numpy.ndarray): Solar irradiance values.

    Returns:
    tuple: A tuple containing:
        - R_l (list): Reflectivity values.
        - T_l (list): Transmissivity values.
        - A_l (list): Absorbance values.
    """
    R_l, T_l, A_l = [], [], []
    if spectrum == 'UV':
        mask = (lambda_um >= 0.2) & (lambda_um <= 0.38)
    elif spectrum == 'Visible':
        mask = (lambda_um >= 0.38) & (lambda_um <= 0.8)
    elif spectrum == 'IR':
        mask = (lambda_um >= 0.8) & (lambda_um <= 20)
    else:
        print("Invalid spectrum")

    for d in d_list:
        R_tot, T_tot, A_tot = 0, 0, 0
        for i, wl in enumerate(lambda_um):
            R, T, A = compute_R_T_circular(n0, n1[i], n2[i], d, wl, phi0)
            if mask[i]:
                R_tot += R * Irradiance[i]
                T_tot += np.real(T) * Irradiance[i]
                A_tot += np.real(A) * Irradiance[i]
        I = np.sum(Irradiance[mask])
        R_l.append(R_tot / I * 100)
        T_l.append(T_tot / I * 100)
        A_l.append(A_tot / I * 100)

    return R_l, T_l, A_l

def plot_I_vs_d(lambda_um, n0, n1, n2, d_list, phi0, Irradiance):
    """
    Plot the reflectivity, transmissivity, and absorbance versus thickness for different spectra.

    Parameters:
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    n0 (float): Refractive index of air.
    n1 (numpy.ndarray): Complex refractive index of the metal.
    n2 (numpy.ndarray): Complex refractive index of the glass.
    d_list (list): List of thicknesses of the metal layer.
    phi0 (float): Angle of incidence in degrees.
    Irradiance (numpy.ndarray): Solar irradiance values.
    """
    # UV spectrum
    R_UV, T_UV, A_UV = spectral_RTA('UV', lambda_um, n0, n1, n2, d_list, phi0, Irradiance)
    plt.figure(figsize=(10, 6))
    plt.plot(d_list, R_UV, label="Reflectivity (UV Spectrum)")
    plt.plot(d_list, T_UV, label="Transmissivity (UV Spectrum)")
    plt.plot(d_list, A_UV, label="Absorbance (UV Spectrum)")
    plt.xlabel("Thickness (µm)")
    plt.xscale('log')
    plt.ylabel("R, T, A (%)")
    plt.legend()
    plt.savefig("Output/Fraction_of_spectra_vs_thickness/UV.png")

    # Visible spectrum
    R_Visible, T_Visible, A_Visible = spectral_RTA('Visible', lambda_um, n0, n1, n2, d_list, phi0, Irradiance)
    plt.figure(figsize=(10, 6))
    plt.plot(d_list, R_Visible, label="Reflectivity (Visible Spectrum)")
    plt.plot(d_list, T_Visible, label="Transmissivity (Visible Spectrum)")
    plt.plot(d_list, A_Visible, label="Absorbance (Visible Spectrum)")
    plt.xlabel("Thickness (µm)")
    plt.xscale('log')
    plt.ylabel("R, T, A (%)")
    plt.legend()
    plt.savefig("Output/Fraction_of_spectra_vs_thickness/Visible.png")

    # IR spectrum
    R_IR, T_IR, A_IR = spectral_RTA('IR', lambda_um, n0, n1, n2, d_list, phi0, Irradiance)
    plt.figure(figsize=(10, 6))
    plt.plot(d_list, R_IR, label="Reflectivity (IR Spectrum)")
    plt.plot(d_list, T_IR, label="Transmissivity (IR Spectrum)")
    plt.plot(d_list, A_IR, label="Absorbance (IR Spectrum)")
    plt.xlabel("Thickness (µm)")
    plt.xscale('log')
    plt.ylabel("R, T, A (%)")
    plt.legend()
    plt.savefig("Output/Fraction_of_spectra_vs_thickness/IR.png")

    # visible vs IR
    plt.figure(figsize=(10, 6))
    plt.plot(d_list, T_Visible, label="Transmissivity (Visible Spectrum)")
    plt.plot(d_list, R_IR, label="Reflectivity (IR Spectrum)")
    plt.xlabel("Thickness (µm)")
    plt.xscale('log')
    plt.ylabel("R, T (%)")
    plt.vlines(Optimal_thickness(d_list, lambda_um, n0, n1, n2, phi0, Irradiance), 0, 100, color="red", label="Optimal thickness")
    plt.legend()
    plt.savefig("Output/Fraction_of_spectra_vs_thickness/Visible_IR.png")

def plot_R_T_A_vs_d(lambda_um, n0, n1, n2, d_list, phi0):
    """
    Plot reflectivity, transmissivity, and absorbance versus wavelength for different thicknesses.

    Parameters:
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    n0 (float): Refractive index of air.
    n1 (numpy.ndarray): Complex refractive index of the metal.
    n2 (numpy.ndarray): Complex refractive index of the glass.
    d_list (list): List of thicknesses of the metal layer.
    phi0 (float): Angle of incidence in degrees.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

    for d in d_list:
        R, _, _ = compute_R_T_circular(n0, n1, n2, d, lambda_um, phi0)
        axs[0].plot(lambda_um, R, label=f"d = {d * 1e3:.2f} nm")
    axs[0].set_ylabel("Reflectivity")
    axs[0].set_xscale('log')
    axs[0].legend()
    axs[0].axvspan(0.38, 0.8, color="yellow", alpha=0.2, label="Visible Spectrum")
    axs[0].axvspan(0.2, 0.38, color="purple", alpha=0.2, label="UV Spectrum")
    axs[0].axvspan(0.8, 20, color="red", alpha=0.2, label="IR Spectrum")

    for d in d_list:
        _, T, _ = compute_R_T_circular(n0, n1, n2, d, lambda_um, phi0)
        axs[1].plot(lambda_um, T, label=f"d = {d * 1e3:.2f} nm")
    axs[1].set_ylabel("Transmissivity")
    axs[1].set_xscale('log')
    axs[1].legend()
    axs[1].axvspan(0.38, 0.8, color="yellow", alpha=0.2, label="Visible Spectrum")
    axs[1].axvspan(0.2, 0.38, color="purple", alpha=0.2, label="UV Spectrum")
    axs[1].axvspan(0.8, 20, color="red", alpha=0.2, label="IR Spectrum")

    for d in d_list:
        _, _, A = compute_R_T_circular(n0, n1, n2, d, lambda_um, phi0)
        axs[2].plot(lambda_um, A, label=f"d = {d * 1e3:.2f} nm")
    axs[2].set_xlabel("Wavelength (µm)")
    axs[2].set_ylabel("Absorbance")
    axs[2].set_xscale('log')
    axs[2].legend()
    axs[2].axvspan(0.38, 0.8, color="yellow", alpha=0.2, label="Visible Spectrum")
    axs[2].axvspan(0.2, 0.38, color="purple", alpha=0.2, label="UV Spectrum")
    axs[2].axvspan(0.8, 20, color="red", alpha=0.2, label="IR Spectrum")

    plt.tight_layout()
    plt.savefig("Output/RTA_vs_d/RTA_combined_{}.png".format(phi0))

def Optimal_thickness(d_list, lambda_um, n0, n1, n2, phi0, Irradiance):
    """
    Find the optimal thickness that maximizes the product of reflectivity in the IR spectrum and transmissivity in the visible spectrum.

    Parameters:
    d_list (list): List of thicknesses of the metal layer.
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    n0 (float): Refractive index of air.
    n1 (numpy.ndarray): Complex refractive index of the metal.
    n2 (numpy.ndarray): Complex refractive index of the glass.
    phi0 (float): Angle of incidence in degrees.
    Irradiance (numpy.ndarray): Solar irradiance values.

    Returns:
    float: Optimal thickness in micrometers.
    """
    R_IR, T_IR, _ = spectral_RTA('IR', lambda_um, n0, n1, n2, d_list, phi0, Irradiance)
    R_Vis, T_Vis, _ = spectral_RTA('Visible', lambda_um, n0, n1, n2, d_list, phi0, Irradiance)
    d_opt = []
    for i, d in enumerate(d_list):
        d_opt.append(R_IR[i] * T_Vis[i])
    return d_list[np.argmax(d_opt)]

def ellipsometry_theoretical(n0, n1, n2, phi0, d1, wavelength):
    """
    Calculate the reflectance and transmittance for a thin film using ellipsometry.

    Parameters:
    n0 (float): Refractive index of the first medium.
    n1 (float): Refractive index of the second medium.
    n2 (float): Refractive index of the third medium.
    phi0 (float): Angle of incidence in degrees.
    d1 (float): Thickness of the thin film in micrometers.
    wavelength (float): Wavelength of light in micrometers.

    Returns:
    tuple: A tuple containing:
        - R (float): Reflectance.
        - T (float): Transmittance.
    """
    print("Task 2 : Calculating")
    phi0 = np.radians(phi0)
    
    sin_phi0 = np.sin(phi0)
    sin_phi1 = (n0 / n1) * sin_phi0
    phi1 = np.arcsin(sin_phi1)
    sin_phi2 = (n1 / n2) * np.sin(phi1)
    phi2 = np.arcsin(sin_phi2)

    # Compute cosines of angles
    cos_phi0 = np.cos(phi0)
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)

    # Fresnel coefficients for p-polarization (equations 4.41-4.46)
    r01p = (n1 * cos_phi0 - n0 * cos_phi1) / (n1 * cos_phi0 + n0 * cos_phi1)
    r12p = (n2 * cos_phi1 - n1 * cos_phi2) / (n2 * cos_phi1 + n1 * cos_phi2)

    # Fresnel coefficients for s-polarization (equations 4.43-4.48)
    r01s = (n0 * cos_phi0 - n1 * cos_phi1) / (n0 * cos_phi0 + n1 * cos_phi1)
    r12s = (n1 * cos_phi1 - n2 * cos_phi2) / (n1 * cos_phi1 + n2 * cos_phi2)

    t01s = (2 * n0 * cos_phi0) / (n0 * cos_phi0 + n1 * cos_phi1)
    t12s = (2 * n1 * cos_phi1) / (n1 * cos_phi2 + n2 * cos_phi1)

    # Phase thickness beta (equation 4.32 corrected)

    beta = 2 * np.pi * d1 * n1 * cos_phi1 / wavelength

    # Reflection coefficients (equations 4.37-4.38)
    R_p = (r01p + r12p * np.exp(-2j * beta)) / (1 + r01p * r12p * np.exp(-2j * beta))
    R_s = (r01s + r12s * np.exp(-2j * beta)) / (1 + r01s * r12s * np.exp(-2j * beta))

    rho = R_p/R_s
    psi = np.arctan(np.abs(rho))
    delta = np.angle(rho)

    return np.degrees(psi), np.degrees(delta)

def plot_ellipsometry_theoretical(n0, n1, n2, phi0, d1, lambda_um, metalname="Silver"):
    """
    Plot the ellipsometry parameters (psi and delta) versus wavelength.

    Parameters:
    n0 (float): Refractive index of the first medium.
    n1 (float): Refractive index of the second medium.
    n2 (float): Refractive index of the third medium.
    phi0 (float): Angle of incidence in degrees.
    d1 (float): Thickness of the thin film in micrometers.
    lambda_um (numpy.ndarray): Wavelengths in micrometers.
    """
    psi, delta = ellipsometry_theoretical(n0, n1, n2, phi0, d1, lambda_um)
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_um, psi, label=r"$\Psi$")
    plt.plot(lambda_um, delta, label=r"$\Delta$")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel(r"$\Psi$, $\Delta$")
    plt.xscale('log')
    plt.legend()
    plt.savefig("Output/Ellipsometry/{}_ellipsometry_theoretical_psi_delta_vs_lambda_for_{}nm_and_{}°.png".format(metalname, d1 * 1e3, phi0))

    plt.figure(figsize=(10, 6))
    plt.plot(psi, delta)
    plt.xlabel(r"$\Psi$")
    plt.ylabel(r"$\Delta$")
    plt.savefig("Output/Ellipsometry/{}_ellipsometry_theoretical_psi_vs_delta_for_{}nm_and_{}°.png".format(metalname, d1 * 1e3,phi0))

def read_plot_FTIR(filename):
    """
    Read and plot the FTIR spectrum.

    Parameters:
    filename (str): Path to the file containing the FTIR spectrum.
    metalname (str): Name of the metal.
    """
    data = pd.read_csv(filename)
    wl = data.iloc[:, 0]  
    I = data.iloc[:, 1] 
    data = pd.read_csv("Data/FTIRref_or.CSV")
    wl_ref = data.iloc[:, 0]
    I_ref = data.iloc[:, 1]

    data = pd.read_csv("Data/FTIRref_verre.CSV")
    wl_glass = data.iloc[:, 0]
    I_glass = data.iloc[:, 1]

    max_glass = np.max(I_glass)
    I_glass = I_glass/max_glass
    max_ref = np.max(I_ref)
    I_ref = I_ref/max_ref
    max_I = np.max(I)
    I = I/max_I

  

    plt.figure(figsize=(8, 5))
    plt.plot(wl, I, label='Sample 1', color='b')
    plt.plot(wl_ref, I_ref, label='Reference', color='r')
    plt.plot(wl_glass, I_glass, label='Glass', color='g')
    plt.xlabel('Wave number (cm$^{-1}$)')
    plt.ylabel('reflectance')
    plt.legend()
    
    plt.savefig("Output/FTIR/Sample1_FTIR_linear.png")

    wl = 1e4/wl
    wl_ref = 1e4/wl_ref
    wl_glass = 1e4/wl_glass

    plt.figure(figsize=(8, 5))
    plt.semilogx(wl, I, label='Sample 1', color='b')
    plt.semilogx(wl_ref, I_ref, label='Reference', color='r')
    plt.semilogx(wl_glass, I_glass, label='Glass', color='g')
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('reflectance')
    plt.xlim(0.2, 20)
    plt.legend()
    plt.savefig("Output/FTIR/Sample1_FTIR_log.png")






if __name__ == "__main__":
    print ("Task 2 :")
    filename = "Data/n_k_combined.txt"
    




