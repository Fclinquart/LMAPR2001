import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 
import os
import Extraction
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Task 2'))
sys.path.append(parent_dir)
import task2 # type: ignore

wl_glass, n_air, n_ZnS, n_Cu, n_ZnS, n_glass = Extraction.n_k_wl_trilayer("Data/ZnS_Querry.txt", "Data/Cu_Querry.txt", "Data/ZnS_Querry.txt", "Data/Glass_Palik.txt", 0.2, 20)

# task2.plot_n_k(wl_glass, np.real(n_ZnS), np.imag(n_ZnS),"ZnS")
# R = task2.reflectivity_semi_infinite_layer(n_air, n_ZnS, 0)
# plt.plot(wl_glass, R, label="Reflectivity")
# plt.xscale('log')
# plt.show()  ### Question pour le prof !!!

task2.plot_R_T_A_fixed_phi0_and_d(n0=n_air, n1=n_ZnS, n2=n_glass, d1=14e-3, lambda_um =wl_glass, phi0=0)




def compute_R_T_A_trilayer(n0, n1, n2, n3, n_glass, d1, d2, d3, wavelength, phi0):
    """
    Compute reflection (R), transmission (T), and absorbance (A) for a trilayer structure
    on a semi-infinite glass substrate.

    Parameters:
    n0, n1, n2, n3, n_glass (complex): Refractive indices of air, layer 1, layer 2, layer 3, and the glass.
    d1, d2, d3 (float): Thicknesses of layers 1, 2, and 3 (in µm).
    wavelength (float): Wavelength of light (in µm).
    phi0 (float): Angle of incidence in degrees (in medium 0, e.g., air).

    Returns:
    tuple: (R, T, A) - Reflectivity, Transmissivity, Absorbance.
    """
    phi0 = np.radians(phi0)  
    sin_phi1 = (n0 / n1) * np.sin(phi0)
    phi1 = np.arcsin(sin_phi1)
    sin_phi2 = (n1 / n2) * np.sin(phi1)
    phi2 = np.arcsin(sin_phi2)
    sin_phi3 = (n2 / n3) * np.sin(phi2)
    phi3 = np.arcsin(sin_phi3)
    sin_phi_glass = (n3 / n_glass) * np.sin(phi3)
    phi_glass = np.arcsin(sin_phi_glass)
    cos_phi0 = np.cos(phi0)
    cos_phi1 = np.cos(phi1)
    cos_phi2 = np.cos(phi2)
    cos_phi3 = np.cos(phi3)
    cos_phi_glass = np.cos(phi_glass)
    def fresnel_coefficients(n_i, n_j, cos_phi_i, cos_phi_j):
        rs = (n_i * cos_phi_i - n_j * cos_phi_j) / (n_i * cos_phi_i + n_j * cos_phi_j)
        rp = (n_j * cos_phi_i - n_i * cos_phi_j) / (n_j * cos_phi_i + n_i * cos_phi_j)
        ts = 2 * n_i * cos_phi_i / (n_i * cos_phi_i + n_j * cos_phi_j)
        tp = 2 * n_i * cos_phi_i / (n_j * cos_phi_i + n_i * cos_phi_j)
        return rs, rp, ts, tp
    r01s, r01p, t01s, t01p = fresnel_coefficients(n0, n1, cos_phi0, cos_phi1)
    r12s, r12p, t12s, t12p = fresnel_coefficients(n1, n2, cos_phi1, cos_phi2)
    r23s, r23p, t23s, t23p = fresnel_coefficients(n2, n3, cos_phi2, cos_phi3)
    r3Gs, r3Gp, t3Gs, t3Gp = fresnel_coefficients(n3, n_glass, cos_phi3, cos_phi_glass)
    beta1 = 2 * np.pi * d1 * n1 * cos_phi1 / wavelength
    beta2 = 2 * np.pi * d2 * n2 * cos_phi2 / wavelength
    beta3 = 2 * np.pi * d3 * n3 * cos_phi3 / wavelength

    def multilayer_reflection_transmission(r01, r12, r23, r3G, t01, t12, t23, t3G, beta1, beta2, beta3):
        M1 = np.array([[np.exp(-1j * beta1), r12 * np.exp(-1j * beta1)], [r12 * np.exp(-1j * beta1), np.exp(1j * beta1)]])
        M2 = np.array([[np.exp(-1j * beta2), r23 * np.exp(-1j * beta2)], [r23 * np.exp(-1j * beta2), np.exp(1j * beta2)]])
        M3 = np.array([[np.exp(-1j * beta3), r3G * np.exp(-1j * beta3)], [r3G * np.exp(-1j * beta3), np.exp(1j * beta3)]])
        
        
        M_total = M1 @ M2 @ M3

        r_total = (r01 + M_total[1, 0] / (1 - r01 * M_total[1, 1]))
        t_total = (t01 * t12 * t23 * t3G * np.exp(-1j * (beta1 + beta2 + beta3))) / (1 - r01 * M_total[1, 1])
        
        return r_total, t_total

    R_s, T_s = multilayer_reflection_transmission(r01s, r12s, r23s, r3Gs, t01s, t12s, t23s, t3Gs, beta1, beta2, beta3)
    R_p, T_p = multilayer_reflection_transmission(r01p, r12p, r23p, r3Gp, t01p, t12p, t23p, t3Gp, beta1, beta2, beta3)

    
    R = (np.abs(R_s)**2 + np.abs(R_p)**2) / 2
    T = (np.abs(T_s)**2 + np.abs(T_p)**2) / 2

    correction_factor = (n_glass * cos_phi_glass) / (n0 * cos_phi0)

    A = 1 - R - T * correction_factor

    return R, T, A

R, T, A = compute_R_T_A_trilayer(n0=n_air, n1=n_ZnS, n2=n_Cu, n3=n_ZnS, n_glass=n_glass, d1=14e-3, d2=100e-9, d3=14e-3, wavelength=0.55, phi0=0)
print("R : ", R)
