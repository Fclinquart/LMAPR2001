import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 
import os
import Extraction
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Task 2'))
sys.path.append(parent_dir)
import task2 # type: ignore

wl_glass, n0, n1, n2, n3, n_glass = Extraction.n_k_wl_trilayer("Data/ZnS_Querry.txt", "Data/Cu_Hagemann.txt", "Data/ZnS_Querry.txt", "Data/Glass_Palik.txt", 0.2, 20)



def compute_R_T_circular_trilayer(n0, n1, n2, n3, d1, d2, d3, wavelength, phi0):
    """
    Compute reflection (R) and transmission (T) coefficients for circularly polarized light in a trilayer system.

    Parameters:
    n0, n1, n2, n3 (complex): Complex refractive indices of the four layers (air, ZnS, Cu, ZnS, glass).
    d1, d2, d3 (float): Thicknesses of the ZnS, Cu, and ZnS layers (in µm).
    wavelength (float): Wavelength of light (in µm).
    phi0 (float): Angle of incidence in degrees (in medium 0, e.g., air).

    Returns:
    tuple: A tuple containing:
        - R (float): Reflectivity.
        - T (float): Transmissivity.
        - A (float): Absorbance.
    """
    print("Task 2 : Calculating reflection and transmission coefficients for circular polarization in a trilayer system...")
    phi0 = np.radians(phi0)  # Convert angle of incidence to radians

    # Compute angles in each layer using Snell's law
    sin_phi0 = np.sin(phi0)
    sin_phi1 = (n0 / n1) * sin_phi0
    phi1 = np.arcsin(sin_phi1)
    sin_phi2 = (n1 / n2) * np.sin(phi1)
    phi2 = np.arcsin(sin_phi2)
    sin_phi3 = (n2 / n3) * np.sin(phi2)
    phi3 = np.arcsin(sin_phi3)

    # Compute cosines of angles
    cos_phi0 = np.cos(phi0)
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)
    cos_phi3 = np.cos(phi3)

    # Fresnel coefficients for p-polarization
    r01p = (n1 * cos_phi0 - n0 * cos_phi1) / (n1 * cos_phi0 + n0 * cos_phi1)
    r12p = (n2 * cos_phi1 - n1 * cos_phi2) / (n2 * cos_phi1 + n1 * cos_phi2)
    r23p = (n3 * cos_phi2 - n2 * cos_phi3) / (n3 * cos_phi2 + n2 * cos_phi3)

    t01p = (2 * n0 * cos_phi0) / (n1 * cos_phi0 + n0 * cos_phi1)
    t12p = (2 * n1 * cos_phi1) / (n2 * cos_phi1 + n1 * cos_phi2)
    t23p = (2 * n2 * cos_phi2) / (n3 * cos_phi2 + n2 * cos_phi3)

    # Fresnel coefficients for s-polarization
    r01s = (n0 * cos_phi0 - n1 * cos_phi1) / (n0 * cos_phi0 + n1 * cos_phi1)
    r12s = (n1 * cos_phi1 - n2 * cos_phi2) / (n1 * cos_phi1 + n2 * cos_phi2)
    r23s = (n2 * cos_phi2 - n3 * cos_phi3) / (n2 * cos_phi2 + n3 * cos_phi3)

    t01s = (2 * n0 * cos_phi0) / (n0 * cos_phi0 + n1 * cos_phi1)
    t12s = (2 * n1 * cos_phi1) / (n1 * cos_phi1 + n2 * cos_phi2)
    t23s = (2 * n2 * cos_phi2) / (n2 * cos_phi2 + n3 * cos_phi3)

    # Phase thickness for each layer
    beta1 = 2 * np.pi * d1 * n1 * cos_phi1 / wavelength
    beta2 = 2 * np.pi * d2 * n2 * cos_phi2 / wavelength
    beta3 = 2 * np.pi * d3 * n3 * cos_phi3 / wavelength

    # Reflection coefficients for the trilayer system
    R_p = (r01p + r12p * np.exp(-2j * beta1) + r23p * np.exp(-2j * (beta1 + beta2))) / \
          (1 + r01p * r12p * np.exp(-2j * beta1) + r01p * r23p * np.exp(-2j * (beta1 + beta2)) + r12p * r23p * np.exp(-2j * beta2))
    R_s = (r01s + r12s * np.exp(-2j * beta1) + r23s * np.exp(-2j * (beta1 + beta2))) / \
          (1 + r01s * r12s * np.exp(-2j * beta1) + r01s * r23s * np.exp(-2j * (beta1 + beta2)) + r12s * r23s * np.exp(-2j * beta2))

    # Transmission coefficients for the trilayer system
    T_p = (t01p * t12p * t23p * np.exp(-1j * (beta1 + beta2 + beta3))) / \
          (1 + r01p * r12p * np.exp(-2j * beta1) + r01p * r23p * np.exp(-2j * (beta1 + beta2)) + r12p * r23p * np.exp(-2j * beta2))
    T_s = (t01s * t12s * t23s * np.exp(-1j * (beta1 + beta2 + beta3))) / \
          (1 + r01s * r12s * np.exp(-2j * beta1) + r01s * r23s * np.exp(-2j * (beta1 + beta2)) + r12s * r23s * np.exp(-2j * beta2))

    # Circular polarization: average s and p components
    R = (np.abs(R_p)**2 + np.abs(R_s)**2) / 2
    T = (np.abs(T_p)**2 + np.abs(T_s)**2) / 2

    # Correction factor for transmission (energy conservation)
    correction_factor = (n3 * cos_phi3) / (n0 * cos_phi0)

    # Absorbance
    A = 1 - R - T * correction_factor

    return R, correction_factor * T, A


