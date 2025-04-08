import numpy as np
import matplotlib.pyplot as plt
import Extraction


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
        "SiO2": "Data/Glass_Palik.txt",
        "Al": "Data/Al_rakic.txt",
        "PMMA": "Data/PMMA_Zhang.txt",
        "ZnO": "Data/ZnO_Bond.txt",
        "VO2": "Data/VO2_Beaini.txt",
    }

    # Check if the material is in the dictionary
    if material not in material_files:
        raise ValueError(f"Material '{material}' is not supported.")
    # Load the material data
    wl, n, k = Extraction.extract_wl_n_k(material_files[material])
    # Interpolate n and k values to match the wavelength range
    return Extraction.interpolate(wl_interp,wl, n, k)

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


    
    


if __name__ == "__main__":
    wl = np.linspace(0.2, 20, 100000)
    plot_black_body_spectrum(wl, [-50,25,100], 0)
    plot_solar_irrandiance_vs_black_body('Data/ASTM0.txt', 'Data/ASTM1.5Global.txt', wl, 6000,0)
    
