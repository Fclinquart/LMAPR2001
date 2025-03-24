import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 
import os
import Extraction
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Task 2'))
sys.path.append(parent_dir)
import task2 # type: ignore

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
     # alpha for transparency
    
})


def layers(config, debug = False):
    """
    Generates the multilayer system based on the given configuration.

    Parameters:
    -----------
    config : list of tuples
        A list where each element is a tuple containing:
        - material (str): The name of the material (e.g., "air", "ZnS", "Cu", "glass").
        - thickness (float): The thickness of the layer in nanometers.
    debug : bool, optional
        If True, prints debugging information (default is False).

    Returns:
    --------
    list of tuples:
        Each tuple represents a layer and contains:
        - real(n): Real part of the refractive index.
        - -imag(n): Negative imaginary part of the refractive index (extinction coefficient).
        - thickness (float): Thickness of the layer in nanometers.

    Notes:
    ------
    - The function reads refractive index data from external files specific to each material.
    - The function raises a ValueError if an unknown material is encountered.
    - The refractive index values are extracted using the `Extraction.n_k_wl_trilayer` function.
    - The generated layer stack is used for optical calculations in subsequent functions.

    """
    print("Task 3 : Generation of the multilayer system...")
    layers = []
    filename = []
    for material, thickness in config:
        if debug :
            print("#" * 50)
            print("Debug information:")
            print(f"Material: {material}, Thickness: {thickness}")
            print("Number of layers: ", len(layers))
            print("#" * 50)


        if material == "air":
            layers.append(thickness)
        elif material == "ZnS":
           filename.append("Data/ZnS_Querry.txt")
           layers.append(thickness)
        elif material == "Cu":
            filename.append("Data/n_k_copper.txt")
            layers.append(thickness)
        elif material == "glass":
            filename.append("Data/Glass_Palik.txt")
            layers.append(thickness)
        else:
            raise ValueError(f"Unknown material {material}")
        
    
    wl , n0, n1 , n2,  n3 , n_glass = Extraction.n_k_wl_trilayer(filename[0], filename[1], filename[2], filename[3], 0.2, 20)
    if debug:
        print("#" * 50)
        print("Taille des couches: ", len(layers))
        print("Tailles des couches en nm : ", layers)
        print("#" * 50)
    print("Task 3 : Generation of the multilayer system : Done")

    return [ (np.real(n1), -np.imag(n1),layers[0]), (np.real(n2), -np.imag(n2),layers[1]), (np.real(n3), -np.imag(n3),layers[2]), (np.real(n_glass), -np.imag(n_glass),layers[3])]

def calculate_RTA_multilayer(layers, wl, phi0=0):
    """
    Computes the Reflectivity (R), Transmissivity (T), and Absorbance (A) 
    of a multilayer optical system for a given range of wavelengths.

    Parameters:
    -----------
    layers : list of tuples
        A list where each tuple represents a layer and contains:
        - n (float): Real part of the refractive index.
        - kappa (float): Extinction coefficient (negative imaginary part of n).
        - d (float): Thickness of the layer in micrometers.
    wl : array-like
        Wavelengths in micrometers at which R, T, and A will be computed.
    phi0 : float, optional
        Angle of incidence in degrees (default is 0°).

    Returns:
    --------
    tuple of arrays:
        - R (array): Reflectivity as a function of wavelength.
        - T (array): Transmissivity as a function of wavelength.
        - A (array): Absorbance as a function of wavelength.

    Notes:
    ------
    - The function uses a transfer matrix approach to compute R, T, and A.
    - The multilayer structure is defined by its optical constants (n, kappa) 
      and thicknesses (d) for each layer.
    - The transmissivity is corrected to account for the refractive index of the substrate.
    - The final values of R, T, and A are averaged over s- and p-polarizations.
    """

    phi0 = np.radians(phi0)
    n0 = 1.0

    # Initialize the scattering matrix S as the identity matrix for each wavelength
    S_p = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, len(wl)))
    S_s = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, len(wl)))


    # Iterate over each layer
    for i, (n, kappa, d) in enumerate(layers):
        N_layer = n - 1j * kappa
        
    
        # Calculate the angle of propagation in the current layer
        sin_theta_layer = n0 * np.sin(phi0) / N_layer
        cos_theta_layer = np.sqrt(1 - sin_theta_layer**2)
        
    
        # Calculate the phase shift beta for all wavelengths
        beta = 2 * np.pi * d * N_layer * cos_theta_layer / wl
        
    
        # Create the layer matrix L as a 3D array
        L = np.zeros((2, 2, len(wl)), dtype=complex)
        L[0, 0, :] = np.exp(1j * beta)  # Forward propagation
        L[1, 1, :] = np.exp(-1j * beta)  # Backward propagation
        
        
        
        # Calculate the Fresnel coefficients for p and s polarizations
        if i == 0:
            N_prev = n0
        else:
            N_prev = layers[i-1][0] - 1j * layers[i-1][1]
        
        r_p = (N_layer * np.cos(phi0) - N_prev * cos_theta_layer) / (N_layer * np.cos(phi0) + N_prev * cos_theta_layer)
        r_s = (N_prev * np.cos(phi0) - N_layer * cos_theta_layer) / (N_prev * np.cos(phi0) + N_layer * cos_theta_layer)
        
        t_p = (2 * N_prev * np.cos(phi0)) / (N_layer * np.cos(phi0) + N_prev * cos_theta_layer)
        t_s = (2 * N_prev * np.cos(phi0)) / (N_prev * np.cos(phi0) + N_layer * cos_theta_layer)
        
        
        
        # Interface matrix I for each wavelength
        I_p = np.zeros((2, 2, len(wl)), dtype=complex)
        I_p[0, 0, :] = 1 / t_p
        I_p[0, 1, :] = r_p / t_p
        I_p[1, 0, :] = r_p / t_p
        I_p[1, 1, :] = 1 / t_p
        
        I_s = np.zeros((2, 2, len(wl)), dtype=complex)
        I_s[0, 0, :] = 1 / t_s
        I_s[0, 1, :] = r_s / t_s
        I_s[1, 0, :] = r_s / t_s
        I_s[1, 1, :] = 1 / t_s
        
        
        
        # Update the scattering matrix S for each wavelength
        for l in range(len(wl)):
            S_p[:, :, l] = np.dot(S_p[:, :, l], np.dot(I_p[:, :, l], L[:, :, l]))
            S_s[:, :, l] = np.dot(S_s[:, :, l], np.dot(I_s[:, :, l], L[:, :, l]))
        
        

    # Calculate the reflection and transmission coefficients for each wavelength
    R_p = np.abs(S_p[1, 0, :] / S_p[0, 0, :])**2
    R_s = np.abs(S_s[1, 0, :] / S_s[0, 0, :])**2

    T_p = np.abs(1 / S_p[0, 0, :])**2
    T_s = np.abs(1 / S_s[0, 0, :])**2


    # Correction factor for transmissivity
    n_substrate = layers[-1][0]
    correction_factor = (n_substrate * np.cos(phi0)) / (n0 * np.cos(phi0))
    T_p_corrected = T_p * correction_factor
    T_s_corrected = T_s * correction_factor



    # Average R and T for unpolarized light
    R = (R_p + R_s) / 2
    T = (T_p_corrected + T_s_corrected) / 2
    A = 1 - R - T

   
    
    return R, T, A

def plot_R_T_A_fixed_phi0_and_d_multilayer(layers, wl,  phi0, title="", save=False):
    """
    Plots the Reflectivity (R), Transmissivity (T), and Absorbance (A) 
    of a multilayer optical system for a fixed angle of incidence.

    Parameters:
    -----------
    layers : list of tuples
        A list where each tuple represents a layer and contains:
        - n (float): Real part of the refractive index.
        - kappa (float): Extinction coefficient (negative imaginary part of n).
        - d (float): Thickness of the layer in micrometers.
    wl : array-like
        Wavelengths in micrometers at which R, T, and A will be computed.
    phi0 : float
        Angle of incidence in degrees.
    title : str, optional
        Title of the plot (default is an empty string).
    save : bool, optional
        If True, saves the plot as an image file instead of displaying it (default is False).

    Returns:
    --------
    None

    Notes:
    ------
    - The function computes R, T, and A using `calculate_RTA_multilayer` 
      and plots them as a function of wavelength.
    - The plot highlights the visible (0.38-0.8 µm), UV (0.2-0.38 µm), and 
      IR (0.8-20 µm) spectral regions for better visualization.
    - If `save=True`, the plot is saved in the "Output/RTA_phi0_d" directory 
      with a filename based on the angle of incidence.
    """

    R, T, A = calculate_RTA_multilayer(layers, wl, phi0)
    plt.figure(figsize=(10, 6))
    plt.plot(wl, R, label="Reflectivity")
    plt.plot(wl, T, label="Transmissivity")
    plt.plot(wl, A, label="Absorbance")
    plt.xlabel("Wavelength (µm)")
    plt.xscale('log')
    plt.axvspan(0.38, 0.8, color="yellow", alpha=0.1, label="Visible Spectrum")
    plt.axvspan(0.2, 0.38, color="purple", alpha=0.1, label="UV Spectrum")
    plt.axvspan(0.8, 20, color="red", alpha=0.1, label="IR Spectrum")
    plt.ylabel("R, T, A")
    plt.legend()
    if title != "":
        plt.title(title)
    if save:
        plt.savefig("Output/RTA_phi0_d/RTA_multilayer_{}.png".format(phi0))
    else:
        plt.show()


if __name__ == "__main__":

    config =[ ("ZnS", 20e-3), ("Cu", 30e-3), ("ZnS", 20e-3), ("glass", 0.5)]
    wl, _, _, _,_,_ = Extraction.n_k_wl_trilayer("Data/ZnS_Querry.txt", "Data/Cu_Querry.txt", "Data/ZnS_Querry.txt", "Data/Glass_Palik.txt", 0.2, 20)
    

    lambda_um, n_air, n_zns, n_cu, n_zns, n_glass = Extraction.n_k_wl_trilayer("Data/ZnS_Querry.txt", "Data/Cu_Querry.txt", "Data/ZnS_Querry.txt", "Data/Glass_Palik.txt", 0.2, 20)

    layers = layers(config, debug = False)
    plot_R_T_A_fixed_phi0_and_d_multilayer(layers, wl, 0, title="Reflectivity, Transmissivity, and Absorbance of a Multilayer System", save=False)