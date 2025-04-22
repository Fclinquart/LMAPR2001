import numpy as np
import matplotlib.pyplot as plt
import Extraction
import sys 
import os
import scipy.optimize as opt
from scipy.optimize import minimize


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
   
    phi0 = np.radians(phi0)
    phi1 = snells(n0, n1, phi0)
    
    # Calculate the reflection coefficients for s- and p-polarized light
    r_s = (n0 * np.cos(phi0) - n1 * np.cos(phi1)) / (n0 * np.cos(phi0) + n1 * np.cos(phi1))
    r_p = (n1 * np.cos(phi0) - n0 * np.cos(phi1)) / (n1 * np.cos(phi0) + n0 * np.cos(phi1))
    
    # Calculate the reflectivity as the average of |r_s|^2 and |r_p|^2
    R = (np.abs(r_s)**2 + np.abs(r_p)**2) / 2
        
    return R
def Extract_n_k(material, wl_interp):
    """
    Extracts the refractive index (n) and extinction coefficient (k) from a material file.

    Parameters:
    material (str): Path to the file containing the material data.

    Returns:
    tuple: Wavelength, n, k arrays.
    """
   
    material_files = {
        "ZnS": "Data/ZnS_Querry.txt",
        "Cu": "Data/Cu_Querry.txt",
        "glass": "Data/Glass_Palik.txt",
        "Ag": "Data/Ag_Hagemann.txt",
        "Ag_UV": "Data/Ag_Werner.txt",
        "Ag Christy": "Data/Ag_Christy.txt",
        "Ag Yang": "Data/Ag_Yang.txt",
        "TiO2": "Data/TiO2_Franta.txt",
        "SiO2": "Data/SiO2_Franta.txt",
        "Al": "Data/Al_rakic.txt",
        "PMMA": "Data/PMMA_Zhang.txt",
        "ZnO": "Data/ZnO_Bond.txt",
        "VO2": "Data/VO2_Beaini.txt",
        "SiO": "Data/SiO_Hass.txt",
        "MgF2": "Data/MgF2_Franta.txt",
        "PC": "Data/PC_Zhang.txt",
        "PDMS": "Data/PDMS_Zhang.txt",
        "PVC": "Data/PVC_Zhang.txt",
        "SiC": "Data/SiC.txt",
        "Si3N4":"Data/Si3N4.txt",
        "In":"Data/In.txt",

    }

    # Check if the material is in the dictionary
    if material not in material_files:
        raise ValueError(f"Material '{material}' is not supported.")
    return Extraction.extract_nk(material_files[material], wl_interp)


def Plasma_frequency(N,m_0,eV = False):
    """
    Calculate the plasma frequency of a material.

    Parameters:
    N (float): Carrier concentration in cm^-3.
    m_0 (float): Effective mass of the carriers in units of electron mass.
    eV (bool): If True, return the plasma frequency in eV. Default is False.

    Returns:
    float: Plasma frequency in THz or eV.
    """
    
    # Constants
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
    e = 1.602176634e-19  # Elementary charge in C
    m_e = 9.10938356e-31  # Electron mass in kg
    h = 6.62607015e-34  # Planck's constant in J*s
    # Convert N to m^-3
    N_m3 = N * 1e6

    # Calculate plasma frequency
    omega_p = np.sqrt(N_m3 * e**2 / (epsilon_0 * m_0 * m_e))

    if eV:
        wl = Omega_to_wl(omega_p)
        omega_p = wl_to_ev(wl)
        return omega_p  # Convert to eV

    return  omega_p / (2 * np.pi )  # Convert to THz


def wl_to_Omega(wl):
    """
    Convert wavelength to angular frequency.

    Parameters:
    wl (float): Wavelength in µm.

    Returns:
    float: Angular frequency in THz.
    """
    
    # Convert wavelength to meters
    wl = wl * 1e-6  # Convert to meters
    Omega = (2 * np.pi * 3e8) / wl
    return Omega

def wl_to_ev(wl):
    """
    Convert wavelength to energy in eV.

    Parameters:
    wl (float): Wavelength in nm.

    Returns:
    float: Energy in eV.
    """
    
    # Convert wavelength to meters
    wl = wl * 1e-9  # Convert to meters
    E = (6.62607015e-34 * 3e8) / wl
    return E / (1.602176634e-19)  # Convert to eV

def Omega_to_wl(Omega):
    """
    Convert angular frequency to wavelength.

    Parameters:
    Omega (float): Angular frequency in THz.

    Returns:
    float: Wavelength in nm.
    """
    
    # Convert angular frequency to wavelength
    wl = (2 * np.pi * 3e8) / Omega
    return wl * 1e9  # Convert to nm


def Reflectivity_angular_frequency(n0, n1, phi0, wl):
    """
    Calculate the reflectivity as a function of angle and frequency.

    Parameters:
    n0 (float): Refractive index of the first medium.
    n1 (float): Refractive index of the second medium.
    phi0 (float): Angle of incidence in degrees.
    wl (float): Wavelength in nm.

    Returns:
    float: Reflectivity at the given wavelength.
    """
    
    # Convert wavelength to meters
    
    # Calculate frequency
    Omega_p = Plasma_frequency(N, m_0, eV=True) # eV
    E = wl_to_ev(wl)/(1000) # eV
    R = reflectivity_semi_infinite_layer(n0, n1, phi0)
    plt.figure()
    plt.plot(E, R, color = 'black', linewidth = 0.7, marker = 'd', markersize = 8, markerfacecolor = 'white', markevery = 50)
    plt.vlines(x=Omega_p, ymin=0, ymax=1, color='red', linestyle='--', label='Plasma frequency', linewidth=0.75)
    plt.vlines(x=3.8, ymin=0, ymax=1, color='purple', linestyle='--', label='Interband transition', linewidth=0.75)
    plt.legend(loc='upper right', fontsize=8)
    plt.xscale('log')
    plt.xlabel('Photon energy (eV)', fontsize = 8)
    plt.ylabel('Reflectivity R', fontsize = 8)
    plt.savefig("Output/Reflectivity.png", dpi=300, bbox_inches='tight')



def Drude_model(wl, Omega_p, gamma, plot = True):
    Omega = wl_to_Omega(wl)  # Convert wavelength to angular frequency
    espilon_r = 1 - Omega_p**2 / (Omega**2 + 1j * gamma * Omega)
    n, k = Extract_n_k("Ag_UV", wl)
    e1 = n**2 - k**2
    e2 = 2 * n * k
    if plot:
        wl_ev = wl_to_ev(wl) / 1000  # Convert wavelength to eV
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Plot real part
        axs[0].plot(wl_ev, -e1, label='Real part', color='blue')
        axs[0].set_xlabel('Photon energy (eV)')
        axs[0].set_ylabel('Real part of Epsilon_r')
        axs[0].legend()
        
        axs[0].set_xlim([0, 5])

        # Plot imaginary part
        axs[1].plot(wl_ev, e2, label='Imaginary part', color='red')
        axs[1].set_xlabel('Photon energy (eV)')
        axs[1].set_ylabel('Imaginary part of Epsilon_r')
        axs[1].legend()
        axs[1].set_xlim([0, 5])
        plt.tight_layout()
        plt.suptitle('Drude Model', y=1.02)
        plt.show()
    return espilon_r

def Plot_area_R(wl):
    n, k = Extract_n_k("Ag_UV", wl)
    n0 = 1.0
    n1 = n - 1j*k
    R_1 = reflectivity_semi_infinite_layer(n0, n1, 0)
    n, k = Extract_n_k("Ag", wl)
    n0 = 1.0
    n1 = n - 1j*k
    R_2 = reflectivity_semi_infinite_layer(n0, n1, 0)
    n, k = Extract_n_k("Ag Christy", wl)
    n0 = 1.0
    n1 = n - 1j*k
    R_3 = reflectivity_semi_infinite_layer(n0, n1, 0)
    n, k = Extract_n_k("Ag Yang", wl)
    n0 = 1.0
    n1 = n - 1j*k
    R_4 = reflectivity_semi_infinite_layer(n0, n1, 0)
    # Plot the reflectivity for different materials
    plt.figure()
    plt.plot(wl, R_1, label='Werner et al, 2009 ', color = 'blue', linewidth = 0.75, marker = 'o', markersize = 4, markerfacecolor = 'white', markevery = 50)
    plt.plot(wl, R_2, label='Hagemann et al, 1974', color = 'red', linewidth = 0.75, marker = 's', markersize = 4, markerfacecolor = 'white', markevery = 50)
    plt.plot(wl, R_3, label='Johnson and Christy, 1972 ', color = 'green', linewidth = 0.75, marker = 'x', markersize = 4, markerfacecolor = 'white', markevery = 50)
    plt.plot(wl, R_4, label='Yang et al, 2014', color = 'purple', linewidth = 0.75, marker = 'd', markersize = 4, markerfacecolor = 'white', markevery = 50)
    plt.fill_between(wl, R_1, R_2, where=(R_1 > R_2), color='gray', alpha=0.5)
    plt.fill_between(wl, R_2, R_3, where=(R_2 > R_3), color='gray', alpha=0.5)
    plt.fill_between(wl, R_3, R_4, where=(R_3 > R_4), color='gray', alpha=0.5)
    plt.fill_between(wl, R_4, R_1, where=(R_4 > R_1), color='gray', alpha=0.5)
    plt.xlabel('Wavelength (µm)', fontsize = 8)
    plt.ylabel('Reflectivity R', fontsize = 8)
    plt.ylim(0, 1)
    plt.legend(loc='lower right', fontsize=8)
    plt.xscale('log')
    plt.savefig("Output/Reflectivity_area.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # for Silver 
    m_0 = 1.0
    N = 5.86 * 1e22 # cm^-3
    Omega_p = Plasma_frequency(N, m_0, eV=True) # eV
    
    print("Plasma frequency: ", Omega_p)
    Omega_p =  Plasma_frequency(N, m_0) # THz
    
    print("Plasma frequency: ", Omega_p)
    wl = np.linspace(0.1, 4, 1000) # µm
    n, k = Extract_n_k("Ag_UV", wl)
    n0 = 1.0
    n1 = n - 1j*k
    phi0 = 0
    
    Reflectivity_angular_frequency(n0, n1, phi0, wl)
    Drude_model(wl, Omega_p, 0)

    Plot_area_R(wl)

    