import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def refraction_index(filename):
    data = pd.read_csv("Data/n_k_combined.txt", delim_whitespace=True, skiprows=1, header=None)
    lambda_um = data[0].values # Wavelengths in micrometers
    n_glass = data[1].values # Refractive index of glass
    kappa_glass = data[2].values # Extinction coefficient of glass
    n_silver = data[3].values # Refractive index of silver
    kappa_silver = data[4].values # Extinction coefficient
    n0 = 1 
    n1 = n_silver - 1j * kappa_silver
    n2 = n_glass - 1j * kappa_glass
    plot_n_k(lambda_um, n_silver, kappa_silver)

    return lambda_um, n0, n1, n2

def snells_law(n0, n1, phi0):
    """
    Calculate the angle of refraction using Snell's law.
    
    Parameters:
    n0 : float
        Refractive index of the first medium.
    n1 : float
        Refractive index of the second medium.
    phi0 : float
        Angle of incidence in degrees.
    
    Returns:
    float
        Angle of refraction in degrees.
    """
    print("Task 1 : Calculating angle of refraction...")
    phi0 = np.radians(phi0)
    phi1 = np.arcsin(n0 * np.sin(phi0) / n1)
    return phi1

def snells_law(n0, n1, n2, phi0):
    """
    Calculate the angle of refraction using Snell's law.
    
    Parameters:
    n0 : float
        Refractive index of the first medium.
    n1 : float
        Refractive index of the second medium.
    phi0 : float
        Angle of incidence in degrees.
    
    Returns:
    float
        Angle of refraction in degrees.
    """
    print("Task 2 : Calculating angle of refraction...")
    phi0 = np.radians(phi0)
    phi1 = np.arcsin(n0 * np.sin(phi0) / n1)
    phi2 = np.arcsin(n0 * np.sin(phi0) / n2)
    return phi1, phi2

def reflectivity_semi_infinite_layer(n0, n1, phi0):
    """
    Calculate the reflectivity of a material given its refractive index (n), extinction coefficient (k), and angle of incidence (theta_deg).
    
    Parameters:
    n : float
        Refractive index of the material.
    k : float
        Extinction coefficient of the material.
    theta_deg : float
        Angle of incidence in degrees.
    
    Returns:
    float
        Reflectivity of the material.
    """
    print("Task 1 : Calculating reflectivity for semi-infinite layer...")

    phi0 = np.radians(phi0)
    phi1 = snells_law(n0, n1, phi0)
    
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
        R_circular (complex): Complex reflection coefficient for circular polarization.
        T_circular (complex): Complex transmission coefficient for circular polarization.
    """
    print("Task 2 : Calculating reflection and transmission coefficients for circular polarization...")
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

    A = 1 - R - T*correction_factor

    return R, correction_factor * T , A 

def plot_R_T_A_fixed_phi0_and_d(n0, n1, n2, d1, lambda_um, phi0, Tilte = "", save = False):
    """
    Plot reflectivity, transmissivity, and absorbance for fixed angle of incidence and thickness.

    """

    R, T, A = compute_R_T_circular(n0, n1, n2, d1, lambda_um, phi0)
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_um, R, label="Reflectivity")
    plt.plot(lambda_um, T, label = "Transmissivity")
    plt.plot(lambda_um, A, label = "Absorbance")
    plt.xlabel("Wavelength (µm)")
    plt.xscale('log')
    plt.axvspan(0.38, 0.8, color="yellow", alpha=0.2, label="Visible Spectrum")
    plt.axvspan(0.2, 0.38, color="purple", alpha=0.2, label="UV Spectrum")
    plt.axvspan(0.8, 20, color="red", alpha=0.2, label="IR Spectrum")
    plt.ylabel("R, T, A")
    plt.legend()
    if Tilte != "":
        plt.title(Tilte)
    if save:
        plt.savefig("R_T_A.png")
    else : 
        plt.show()

def power_spectrum(lambda_um, lambda_min, lambda_max,T, I0=1000):
    mask = (lambda_um >= lambda_min) & (lambda_um <= lambda_max)
    
    integrand = np.real(T[mask]) * I0 * lambda_um[mask]
    
    return np.trapz(integrand, lambda_um[mask])

def power_spectrum_solar(lambda_um, lambda_min, lambda_max, I, T):
    # Convert I from W/nm·m² to W/µm·m² (since lambda_um is in µm)
    I = I  # 1 nm = 1e-3 µm

    # Filter the data to include only the wavelengths within [lambda_min, lambda_max]
    mask = (lambda_um >= lambda_min) & (lambda_um <= lambda_max)
    lambda_filtered = lambda_um[mask]
    I_filtered = I[mask]
    T_filtered = T[mask]

    # Compute the integrand: T * I * lambda
    integrand = np.real(T_filtered) * I_filtered * lambda_filtered

    # Perform the integration using np.trapz
    power = np.trapz(integrand, lambda_filtered)

    return power
  
def power_ratio(lambda_um, lambda_min, lambda_max, T, I0 = 1000):
    power_spec = power_spectrum(lambda_um, lambda_min, lambda_max, T, I0)
    integrand = np.real(T) * I0 * lambda_um
    total_power = np.trapz(integrand, lambda_um)
    return np.real(power_spec / total_power)

def power_ratio_solar(lambda_um, lambda_min, lambda_max, I, T):
    power_spec = power_spectrum_solar(lambda_um, lambda_min, lambda_max, I, T)
    integrand =  I * lambda_um * np.real(T)
    total_power = np.trapz(integrand, lambda_um)
    print("total power", total_power)
    return np.real(power_spec / total_power)

def optimal_thickness_d(n0, n1, n2, lambda_um, phi0, I, plot = False):
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
    data = pd.read_csv(filename, delim_whitespace=True, skiprows=1, header=None)
    # https://www.pveducation.org/pvcdrom/appendices/standard-solar-spectra 
    I = data[5].values * 1e3 
    return I

def plot_n_k(lambda_um, n, k):
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_um, n, label="Refractive index")
    plt.plot(lambda_um, k, label="Extinction coefficient")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("n, k")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    lambda_um, n0, n1, n2 = refraction_index("Data/n_k_combined_Sun.txt")
    
    phi0 = 0
    d1 = 14e-3
    plot_R_T_A_fixed_phi0_and_d(n0, n1, n2, d1, lambda_um, phi0, "Reflectivity, transmissivity, and absorbance for 0° and thickness", save = False)
    plot_R_T_A_fixed_phi0_and_d(n0, n1, n2, d1, lambda_um, 28.7, "Reflectivity, transmissivity, and absorbance for solar noon of incidence and thickness", save= False)
    I = Solar_spectrum("Data/n_k_combined_Sun.txt")
    print("The power ratio is maximized when the thickness of the metal layer is {} µm".format(optimal_thickness_d(n0, n1, n2, lambda_um, phi0, I, plot = False)*1e3))
  
    R, T, A = compute_R_T_circular(n0, n1, n2, d1, lambda_um, phi0)

    print("The power intensity of the solar spectrum is : ", power_spectrum_solar(lambda_um, 0.2, 20, I, T))
    print("The power intensity of UV, visible, and IR light :")
    print("The power ratio of the solar spectrum is : ", power_ratio_solar(lambda_um, 0.2, 20, I, T))
    print("The power ratio of the solar spectrum is : ", power_ratio(lambda_um, 0.2, 20, I, T))
    print("The power ratio of the UV light is : ", power_ratio_solar(lambda_um, 0.2, 0.4, I, T))
    print("The power ratio of the visible light is : ", power_ratio_solar(lambda_um, 0.4, 0.8, I, T))
    print("The power ratio of the IR light is : ", power_ratio_solar(lambda_um, 0.8, 20, I, T))

    # plot the solar spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_um, I)
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Power intensity (W/nm·m²)")
    plt.axvspan(0.38, 0.8, color="yellow", alpha=0.2, label="Visible Spectrum")
    plt.axvspan(0.2, 0.38, color="purple", alpha=0.2, label="UV Spectrum")
    plt.axvspan(0.8, 20, color="red", alpha=0.2, label="IR Spectrum")
    plt.legend()
    plt.xscale('log')
    plt.show()


