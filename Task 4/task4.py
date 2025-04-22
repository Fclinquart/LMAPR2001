import numpy as np
import matplotlib.pyplot as plt
import Extraction
import sys 
import os
import scipy.optimize as opt
from scipy.optimize import minimize
import task3 # type: ignore

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
    import plotly.graph_objects as go

    fig = go.Figure()

    for idx, material in enumerate(material_list):
        n, k = Extract_n_k(material, wl_interp)
        
        # Add refractive index (n) trace
        fig.add_trace(go.Scatter(
            x=wl_interp,
            y=n,
            mode='lines',
            name=f'{material} (n)',
            line=dict(width=2)
        ))
        
        # Add extinction coefficient (k) trace
        fig.add_trace(go.Scatter(
            x=wl_interp,
            y=k,
            mode='lines',
            name=f'{material} (k)',
            line=dict(width=2, dash='dot')
        ))

    if log:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log', title_text='Refractive Index (n) / Extinction Coefficient (k)')
    else:
        fig.update_yaxes(title_text='Refractive Index (n) / Extinction Coefficient (k)', secondary_y=False)

    fig.update_xaxes(title_text='Wavelength (nm)')
    fig.add_vrect(x0=8, x1=13, fillcolor="yellow", opacity=0.1, line_width=0)

    fig.update_layout(
        title='Refractive Index and Extinction Coefficient',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    fig.write_image('Output/Choice/refractive_index_and_extinction_coefficient_plotly.png')
    fig.show()

def plot_transmittance_semi_infinite_layer(material_list, wl):
    """
    Plot the transmittance of a semi-infinite layer for a list of materials.

    Parameters:
        material_list (list): List of strings containing material names.
        wl (numpy.ndarray): Wavelengths in nm.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(material_list)))  # Generate distinct colors for each material

    for idx, material in enumerate(material_list):
        n, k = Extract_n_k(material, wl)
        n_air = calculate_refractive_index_air(wl)
        phi0 = 0  # Normal incidence
        R = reflectivity_semi_infinite_layer(n_air, n + 1j * k, phi0)
        T = 1 - R  # Transmittance is 1 - Reflectance
        plt.plot(wl, T, label=material, linewidth=2, color=colors[idx])

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance')
    plt.axvspan(8, 13, color='yellow', alpha=0.1)
    plt.xscale("log")
    plt.title('Transmittance of Semi-Infinite Layers')
    plt.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.savefig('Output/Choice/transmittance_semi_infinite_layer.png', dpi=300)
    plt.show()

    
    
    


if __name__ == "__main__":
    wl = np.linspace(0.2, 50, 10000)
    I = Extraction.solar_interpolation("Data/ASTM1.5Global.txt",wl)
    n,k =Extract_n_k("SiO", wl)
    print("n: ", n)
    print("k: ", k)
    plot_n_k(["Ag"], wl, log=True)
    

    material = [ "PMMA", "PC", "PDMS", "PVC","SiC", "SiO", "SiO2","Si3N4"]
    
    config =[
        ('air',0),
       
        ('PVC',0.025),
        ('SiO',0.025),
        ('In',0.001),
        ('PVC',0.025),
        ('SiO',0.025),
        ("glass",0.025)

    ]

    # task3.plot_R_T_A_fixed_phi0_and_d_multilayer(config,wl,True)
    
    
