import numpy as np
import matplotlib.pyplot as plt
import Extraction
import sys 
import os
import scipy.optimize as opt
from scipy.optimize import minimize
import task3 # type: ignore
import string
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import (MultipleLocator, 
                               FormatStrFormatter, 
                               AutoMinorLocator)

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

def plot_solar_spectrum(filename, filename_2=None, filename_3=None, save=False):
    """
    Plots the solar spectrum from a given file and optionally other spectra.

    Parameters:
    filename (str): Path to the file containing the solar spectrum data.
    filename_2 (str, optional): Path to the file containing another spectrum data. Defaults to None.
    filename_3 (str, optional): Path to the file containing a third spectrum data. Defaults to None.

    Returns:
    None
    """
    

    # Load the solar spectrum data
    data = np.loadtxt(filename)
    wavelength = data[:, 0]
    flux = data[:, 1]

    # Plot the solar spectrum
    plt.plot(wavelength, flux, label='ASTM0 Extraterrestrial', color='orange')

    # If a second file is provided, load and plot its data
    if filename_2 is not None:
        data_2 = np.loadtxt(filename_2)
        wavelength_2 = data_2[:, 0]
        flux_2 = data_2[:, 1]
        plt.plot(wavelength_2, flux_2, label='AMST1.5 Global irradiance', color='blue')

    # If a third file is provided, load and plot its data
    if filename_3 is not None:
        data_3 = np.loadtxt(filename_3)
        wavelength_3 = data_3[:, 0]
        flux_3 = data_3[:, 1]
        plt.plot(wavelength_3, flux_3, label='Third Spectrum', color='green')

    # Set plot labels and title
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Solar Irradiance (W/m²/nm)')
    plt.legend()
    
    if save:
        # Save the plot as a PNG file
        plt.savefig('Output/Solar_Spectrum/solar_spectrum_plot.png', dpi=300)
    # Show the plot
    else: 
        plt.show()

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
        "Si":"Data/Si.txt",
        "Si-InP":"Data/Si-InP.txt",
        "Ta2O5":"Data/ta5O5.txt",

    }

    # Check if the material is in the dictionary
    if material not in material_files:
        raise ValueError(f"Material '{material}' is not supported.")
    return Extraction.extract_nk(material_files[material], wl_interp)

def calculate_refractive_index_air(wl, plot=False):
    """
    Calculate the refractive index (n) for a given wavelength (wl) in micrometers.
    
    Parameters:
        wl (float or numpy.ndarray): Wavelength in micrometers.
    
    Returns:
        float or numpy.ndarray: Refractive index (n).
    """
    term1 = 0.05792105 / (238.0185 - wl**(-2))
    term2 = 0.00167917 / (57.362 - wl**(-2))
    n = 1 + term1 + term2
    k = np.zeros_like(n) 
    
    if plot :
        plt.figure(figsize=(10, 6))
        plt.plot(wl, n, label='Air')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Refractive Index (n)')
        # manual y axis value 

        plt.xscale('log')
       
        plt.savefig('Output/Solar_Spectrum/air_refractive_index.png', dpi=300)
         
    return n

def irradiance_black_body(wl, T,thetha):
    """
    Calculate the irradiance of a black body at a given temperature (T) for a range of wavelengths (wl).
    
    Parameters:
        wl (numpy.ndarray): Wavelengths in nm.
        T (float): Temperature in Kelvin.
    
    Returns:
        numpy.ndarray: Irradiance values.
    """
    h = 6.62607015e-34  # Planck's constant (J·s)
    c = 3e8  # Speed of light (m/s)
    k = 1.380649e-23  # Boltzmann constant (J/K)
   

    wl_m = wl*1e-6  # Convert wavelength from nm to m
    irradiance = (2 * h * c**2) / (wl_m**5) * (1 / (np.exp((h * c) / (wl_m * k * T)) - 1)) 
    
    return irradiance * np.cos(thetha) * 1e-9  # Convert to W/m²/nm

def plot_black_body_spectrum(wl, T, thetha):
    """
    Plot the black body spectrum for a given temperature (T) and angle (thetha).
    
    Parameters:
        wl (numpy.ndarray): Wavelengths in nm.
        T (float): List of Temperature in Kelvin.
        thetha (float): Angle in radians.
    
    Returns:
        None
    """
    
    
    plt.figure(figsize=(10, 6))
    for i, temp in enumerate(T):
        irrandiance = irradiance_black_body(wl, temp + 273.15, thetha)
        plt.plot(wl, irrandiance, label='T = {} °C'.format(temp))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Irradiance (W/m²sr nm)')
   
    
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig('Output/Black_Body/black_body_spectrum{}.png'.format(T), dpi=300)

def plot_solar_irrandiance_vs_black_body(filename, filename2, wl, T, thetha):
    """
    Plot the solar irradiance and black body spectrum for a given temperature (T) and angle (thetha).
    
    Parameters:
        wl (numpy.ndarray): Wavelengths in nm.
        T (float): Temperature in Kelvin.
        thetha (float): Angle in radians.
    
    Returns:
        None
    """
    data = np.loadtxt(filename)
    wavelength = data[:, 0]/1000
    flux = data[:, 1]

    data2 = np.loadtxt(filename2)
    wavelength2 = data2[:, 0]/1000
    flux2 = data2[:, 1]
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux, label='ASTM0 Extraterrestrial', color='orange')
    plt.plot(wavelength2, flux2, label='AMST1.5 Global irradiance', color='blue')
    i = irradiance_black_body(wl, T, thetha)/1e4
    plt.plot(wl, i, label='Black Body Spectrum {} K'.format(T), color='red')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Irradiance (W/m²/nm)')
    
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig('Output/Black_Body/solar_irradiance_vs_black_body.png', dpi=300)

def plot_n_k(material_list, wl_interp, log=False):
    """
    Plot the refractive index (n) and extinction coefficient (k) for a list of materials in separate subplots.
    
    Parameters:
        material_list (list): List of strings containing material names.
        wl_interp (numpy.ndarray): Wavelengths in nm.
        log (bool): If True, plot on a logarithmic scale.
    
    Returns:
        None
    """
    
    # 2  subplots for all materials, one for n and one for k

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 20))
    
    # use a different marker for each material, markerface color = 'white', line color = 'black', line width = 0.7, sans serif, fontsize = 10, no title
    markers = ['o', 's', 'D', '^', 'v', 'x', '+', '*', 'P', 'H']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, material in enumerate(material_list):
        n, k = Extract_n_k(material, wl_interp)
        R, T, A = task3.calculate_RTA_multilayer(task3.layers([(material, 1)], wl_interp), wl_interp, 0, False)
        
        if log:
            axs[0].plot(wl_interp, n, label=material, marker=markers[i % len(markers)], markerfacecolor='white', color=colors[i % len(colors)], linewidth=0.7, markersize=8, markevery=(np.logspace(0, np.log10(len(wl_interp)), num=50, dtype=int) - 1), markeredgecolor=colors[i % len(colors)])
            axs[1].plot(wl_interp, k, label=material, marker=markers[i % len(markers)], markerfacecolor='white', color=colors[i % len(colors)], linewidth=0.7, markersize=8, markevery=(np.logspace(0, np.log10(len(wl_interp)), num=50, dtype=int) - 1), markeredgecolor=colors[i % len(colors)])
            axs[2].plot(wl_interp, A, label=material, marker=markers[i % len(markers)], markerfacecolor='white', color=colors[i % len(colors)], linewidth=0.7, markersize=8, markevery=(np.logspace(0, np.log10(len(wl_interp)), num=50, dtype=int) - 1), markeredgecolor=colors[i % len(colors)])
        else:
            axs[0].plot(wl_interp, n, label=material)
            axs[1].plot(wl_interp, k, label=material)
            axs[2].plot(wl_interp, A, label=material)

    # Add zoom-in inset on the third subplot
    axins = inset_axes(axs[2], width="20%", height="50%", loc='upper right', borderpad=0)
   
    axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    x1, x2 = 8, 13  # Define the x-axis range for the zoom
    y1, y2 = 0.4, 1  # Define the y-axis range for the zoom
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    for i, material in enumerate(material_list):
        _, _, A = task3.calculate_RTA_multilayer(task3.layers([(material, 1)], wl_interp), wl_interp, 0, False)
        axins.plot(wl_interp, A, label=material, marker=markers[i % len(markers)], markerfacecolor='white', color=colors[i % len(colors)], linewidth=0.7, markersize=8, markevery=(np.logspace(0, np.log10(len(wl_interp)), num=50, dtype=int) - 1), markeredgecolor=colors[i % len(colors)])
    mark_inset(axs[2], axins, loc1=2, loc2=1, fc="None", ec="red", lw=2)

    axs[0].set_ylabel('Refractive Index (n)')
    axs[1].set_ylabel('Extinction Coefficient (k)')
    axs[2].set_ylabel('Absorbance (A)')
    axs[2].set_xlabel('Wavelength (µm)')
    
    
    # Highlight the area between 8 and 13 µm
    for n, ax in enumerate(axs):
        ax.axvspan(8, 13, color='blue', alpha=0.05, label='Transparancy Window of the sun')
        ax.text(0.05, 0.9, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=20, weight='bold')
    
    # Add a single legend for all subplots behind the plots
   
    axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
          fancybox=True, shadow=True, ncol=5, fontsize=10)
    
    if log:
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_xscale('log')
        axs[1].set_ylim(10e-2,10)
        axs[2].set_ylim(0,1)
    plt.tight_layout()
    plt.savefig('Output/Choice/n_k_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

   
def generate_config(material_list, wl_interp):
    config = []
    config.append(("air", 0))
    for i, material in enumerate(material_list):
        config.append((material_list[i],0.1))
    config.append(("glass",1))

    I = Extraction.solar_interpolation("Data/ASTM1.5Global.txt",wl_interp)
    print(config)
    d_1, d_2, _ = task3.optimize_layer_thicknesses(config, wl_interp, I, 0, Radiative=True)

    config[1] = (config[1][0], d_1)
    config[2] = (config[2][0], d_2)
    config[3] = (config[3][0], d_1)
    return config
       
def plot_R_T_A_fixed_phi0_and_d_multilayer_6configs(configs, wl, Irradiance=False, phi0=0, titles=None, save=False):
    """
    Plots R, T, A for 6 multilayer configs using subplots.

    Parameters:
    -----------
    configs : list of list of tuples
        List of 6 configurations (each is a list of (material, thickness) tuples).
    wl : array-like
        Wavelengths (µm).
    Irradiance : array-like or False
        Solar spectrum data or False to ignore.
    phi0 : float
        Angle of incidence in degrees.
    titles : list of str or None
        Titles for each subplot.
    save : bool
        Whether to save the figure.

    """
    if titles is None:
        titles = [f"Config {i+1}" for i in range(len(configs))]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axs = axs.ravel()

    if Irradiance is not False:
        wl_sol, Irradiance = Extraction.extract_solar_irrandiance("Data/ASTM1.5Global.txt", plot=False)

    for idx, config in enumerate(configs):
        ax1 = axs[idx]
        l = task3.layers(config, wl)
        R, T, A = task3.calculate_RTA_multilayer(l, wl, phi0)

        # R, T, A
        ax1.plot(wl, R, 'r-', label="R", linewidth=0.75)
        ax1.plot(wl, T, 'g-', label="T")
        ax1.plot(wl, A, 'b-', label="A")

        # Visible, UV, IR bands
        ax1.axvspan(wl[0], 0.7, color="yellow", alpha=0.05)
        ax1.axvspan(0.2, 0.4, color="purple", alpha=0.05)
        ax1.axvspan(0.7, 8, color="red", alpha=0.05)
        ax1.axvspan(13, wl[-1], color="red", alpha=0.05)
        ax1.axvspan(8, 13, color="blue", alpha=0.05)

        # Labeling
        ax1.set_xscale('log')
        ax1.set_title(titles[idx], fontsize=12)
        ax1.set_xlabel("λ (µm)")
        ax1.set_ylabel("R, T, A")
        ax1.set_ylim(0, 1)  # Limit y-axis between 0 and 1

       

        # Solar Irradiance (secondary y-axis)
        if Irradiance is not False:
            ax2 = ax1.twinx()
            ax2.plot(wl_sol, Irradiance, 'k--', alpha=0.3, label="Solar Irradiance")
            ax2.set_ylabel("Irradiance", color='k')
            ax2.tick_params('y', colors='k')

    # Global legend (first axis only)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig("Output/multilayer_RTA6_{}.png".format(config[1][0]), dpi=300)
    else:
        plt.show()

# P=2S∫∞0IλB(T)2π∫π/20ϵλ(θ)cosθsinθdθdϕdλ


from scipy.constants import h, c, k
from scipy.integrate import simps

def radiated_power(T, wl, config, phi0=0, Atmosphere=False):
    """
    Compute radiated power per square meter of a multilayer system at temperature T.

    Parameters:
        T (float): Temperature in Kelvin.
        wl (numpy.ndarray): Wavelengths in micrometers.
        config (list): List of tuples (material, thickness) for each layer.
        phi0 (float): Angle of incidence in degrees (used in RTA computation).
        Atmosphere (bool): If True, apply atmospheric transparency window mask.

    Returns:
        P (float): Radiated power in W/m².
    """
    # Apply atmospheric transparency window mask if Atmosphere is True
    if Atmosphere:
        mask = (wl > 8) & (wl < 13)
        wl = wl[mask]
    wl_m = wl * 1e-6  # Convert wavelength to meters for Planck's law

    # Planck spectral radiance (per wavelength), unpolarized
    I_lambda = (2 * h * c**2) / (wl_m**5 * (np.exp(h * c / (wl_m * k * T)) - 1))  # in W·m⁻²·nm⁻¹·sr⁻¹

    # Compute directional emissivity ε = 1 - R - T at each wavelength
    layers = task3.layers(config, wl)
    R, T_, A = task3.calculate_RTA_multilayer(layers, wl, phi0=phi0)
    

    # Angular integration factor for hemispheric emission (∫cosθ sinθ dθ dφ from 0 to π = π)
    angular_factor = np.pi

    # Spectral power per unit area (W/m²·µm)
    spectral_power = A * I_lambda * angular_factor  # W·m⁻²·nm⁻¹

    # Integrate over wavelength (in µm)
    P = simps(spectral_power, wl_m)  # Result in W/m²

    return P

def power_save(config,wl,I, phi= 0):
    R, T, A = task3.calculate_RTA_multilayer(task3.layers(config, wl), wl, phi)
    return np.trapz(I * R, wl)  # Integrate over wavelength


def temperature_change(T_init_array, P_net, mass, cp, time_hours):
    """
    Compute temperature change after a given cooling time for an array of initial temperatures.

    Parameters:
        T_init_array (array-like): Array of initial temperatures in Kelvin.
        P_net (float): Net radiative power per m² (positive if losing energy), in W/m².
        mass (float): Mass of the system in kg.
        cp (float): Heat capacity in J/kg/K.
        time_hours (float): Total time of cooling in hours.

    Returns:
        delta_T_array (numpy.ndarray): Array of temperature changes in Kelvin.
    """
    T_init_array = np.array(T_init_array)  # Ensure input is an array
    time_seconds = time_hours * 3600       # Convert hours to seconds
    delta_T = -(P_net * time_seconds) / (mass * cp)  # Scalar ΔT for given P_net

    # Create an array of the same shape as T_init_array
    delta_T_array = np.full_like(T_init_array, delta_T)

    return delta_T_array


if __name__ == "__main__":
    wl = np.linspace(0.2, 50, 10000)
    I = Extraction.solar_interpolation("Data/ASTM1.5Global.txt",wl)
   
    
    

    material = [ "PMMA", "PC", "PDMS", "PVC","SiC", "SiO", "SiO2","Si3N4", "Ta2O5"]
#     #TiO2
# ZnO
# ZnS
# poly(methylmethacrylate)
# poly(etherimide)
# poly(carbonate)
   
   
   
    config3 = generate_config(["TiO2", "PDMS","TiO2"], wl)
   

   

    T =[300,310,320,330,340,350]
    T = np.array(T)
    for i in T:
        P = radiated_power(i, wl, config3, phi0=0, Atmosphere=True)
        print("Radiated power for config3: ", P)

    P = [51.08770967063887, 59.910041989515186, 69.60168029225552, 80.17390195902188, 91.63380598982508, 103.98469298157967]
    P = np.array(P)
    mass = 1  # kg
    cp = 840e3    # J/kg/K
    time_hours = 8

    delta_T_array = temperature_change(T, P, mass, cp, time_hours)
    print("Temperature change: ", delta_T_array)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(T-273.15, P, label='Radiated Power', color='black', marker='o', markersize=8, markerfacecolor='white', markeredgecolor='black', linewidth=0.7)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('Radiated Power (W/m²)', fontsize=10)

    ax2 = ax.twinx()
    ax2.plot(T-273.15, -delta_T_array, label='$\Delta T$', color='black', marker='D', markersize=8, markerfacecolor='white', markeredgecolor='red', linewidth=0.7)
    ax2.set_ylabel('Temperature Change $\Delta T$ (°C)', fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x))) 
    ax2.yaxis.set_major_locator(MultipleLocator(1))

    fig.legend(loc='lower right', fontsize=10, frameon=False)
    
    
    fig.tight_layout()
    plt.savefig('Output/Black_Body/radiated_power_vs_temperature_change.png', dpi=300)


    
   
    
   
    

    