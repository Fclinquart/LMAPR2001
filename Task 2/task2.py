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

    return lambda_um, n0, n1, n2

def n_k(filename):
    data = pd.read_csv(filename, delim_whitespace=True, skiprows=1, header=None)
    lambda_um = data[0].values # Wavelengths in micrometers
    n = data[1].values # Refractive index of glass
    k = data[2].values # Extinction coefficient of glass
    return lambda_um, n, k

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

def compute_spectral_RTA(lambda_um, n0, n1, n2, d_metal_values, I):
    """
    Compute reflectivity (R), transmissivity (T), and absorbance (A) for visible, UV, and IR spectra.
    
    Parameters:
        lambda_um (array): Wavelengths in micrometers.
        n0, n1, n2 (complex): Refractive indices of air, metal, and substrate.
        d_metal_values (array): Thickness values of the metal layer (in µm).
        I (array): Solar irradiance data (in W/nm·m²).
    
    Returns:
        d_metal_values (array): Thickness values of the metal layer.
        R_visible, T_visible, A_visible (arrays): Reflectivity, transmissivity, and absorbance for visible light.
        R_UV, T_UV, A_UV (arrays): Reflectivity, transmissivity, and absorbance for UV light.
        R_IR, T_IR, A_IR (arrays): Reflectivity, transmissivity, and absorbance for IR light.
    """
    R_visible, T_visible, A_visible = [], [], []
    R_UV, T_UV, A_UV = [], [], []
    R_IR, T_IR, A_IR = [], [], []
    
    # Define wavelength masks
    visible_mask = (lambda_um >= 0.38) & (lambda_um <= 0.75)
    uv_mask = (lambda_um < 0.38)
    ir_mask = (lambda_um > 0.75)
    
    for d_metal in d_metal_values:
        R_total_vis, T_total_vis, A_total_vis = 0, 0, 0
        R_total_UV, T_total_UV, A_total_UV = 0, 0, 0
        R_total_IR, T_total_IR, A_total_IR = 0, 0, 0
        
        for i, lambda_val in enumerate(lambda_um):
            R, T, A = compute_R_T_circular(n0, n1, n2, d_metal, lambda_val, 0)  # Angle of incidence = 0°
            
            if visible_mask[i]:
                R_total_vis += R * I[i]
                T_total_vis += T * I[i]
                A_total_vis += A * I[i]
            elif uv_mask[i]:
                R_total_UV += R * I[i]
                T_total_UV += T * I[i]
                A_total_UV += A * I[i]
            elif ir_mask[i]:
                R_total_IR += R * I[i]
                T_total_IR += T * I[i]
                A_total_IR += A * I[i]
        
        # Normalize by the total irradiance in each range
        I_visible = np.sum(I[visible_mask])
        I_UV = np.sum(I[uv_mask])
        I_IR = np.sum(I[ir_mask])
        
        R_visible.append(R_total_vis / I_visible * 100) # Convert to percentage
        T_visible.append(T_total_vis / I_visible * 100) # Convert to percentage
        A_visible.append(A_total_vis / I_visible * 100) # Convert to percentage
        
        R_UV.append(R_total_UV / I_UV * 100)
        T_UV.append(T_total_UV / I_UV * 100)
        A_UV.append(A_total_UV / I_UV * 100)
        
        R_IR.append(R_total_IR / I_IR * 100)
        T_IR.append(T_total_IR / I_IR * 100)
        A_IR.append(A_total_IR / I_IR * 100)

    


    
    return  R_visible, T_visible, A_visible, R_UV, T_UV, A_UV, R_IR, T_IR, A_IR

def plot_percentage_vs_thickness(d_metal_values, R_vis, T_vis, A_vis,  R_UV, T_UV, A_UV, R_IR, T_IR, A_IR, metal_name="metal"):
    """
    Plot reflectivity, transmissivity, and absorbance as a function of metal thickness.
    
    Parameters:
        d_metal_values (array): Thickness values of the metal layer.
        R_vis, T_vis, A_vis (arrays): Reflectivity, transmissivity, and absorbance for visible light.
        R_UV, T_UV, A_UV (arrays): Reflectivity, transmissivity, and absorbance for UV light.
        R_IR, T_IR, A_IR (arrays): Reflectivity, transmissivity, and absorbance for IR light.
        metal_name (str): Name of the metal (e.g., "silver").
    """
    # Plot for visible light
    plt.figure(figsize=(10, 5))
    plt.plot(d_metal_values, R_vis, 'r-', label='Reflected (Visible)')
    plt.plot(d_metal_values, T_vis, 'g-', label='Transmitted (Visible)')
    plt.plot(d_metal_values, A_vis, 'b-', label='Absorbed (Visible)')
    
    plt.xscale('log')
    plt.xlabel("Metal film thickness (µm)")
    plt.ylabel("Percentage (%)")
    plt.title(f"Effect of {metal_name} thickness on RTA Properties (Visible light)")
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    
    # Plot for UV and IR light
    plt.figure(figsize=(10, 5))
    plt.plot(d_metal_values, R_UV, 'r--', label='Reflected (UV)')
    plt.plot(d_metal_values, T_UV, 'g--', label='Transmitted (UV)')
    plt.plot(d_metal_values, A_UV, 'b--', label='Absorbed (UV)')
    
    plt.plot(d_metal_values, R_IR, 'r-.', label='Reflected (IR)')
    plt.plot(d_metal_values, T_IR, 'g-.', label='Transmitted (IR)')
    plt.plot(d_metal_values, A_IR, 'b-.', label='Absorbed (IR)')
    
    plt.xscale('log')
    plt.xlabel("Metal film thickness (µm)")
    plt.ylabel("Percentage (%)")
    plt.title(f"Effect of {metal_name} thickness on RTA properties (UV & IR)")
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def theoretical_optimal_thickness(T_percentages_visible, R_percentages_UV, R_percentages_IR, d_metal_values, T_threshold, angle_incidence=0, metal_name="metal"):
    """
    Calculate the optimal metal film thickness based on a desired visible light transmissivity (T) threshold.
    
    Parameters:
        T_percentages_visible (list): Transmission percentages for visible light.
        R_percentages_UV (list): Reflection percentages for UV light.
        R_percentages_IR (list): Reflection percentages for IR light.
        d_metal_values (list): Thickness values of the metal layer.
        T_threshold (float): Desired transmissivity threshold (%) for visible light.
        angle_incidence (float): Angle of incidence of light (in degrees).
        metal_name (str): Name of the metal (e.g., "silver").
    
    Returns:
        optimal_thickness_visible (float): Optimal thickness for visible light.
        R_UV_at_optimal (float): Reflectivity for UV at the optimal thickness.
        R_IR_at_optimal (float): Reflectivity for IR at the optimal thickness.
    """
    # Find the minimum thickness that achieves the desired visible light transmissivity threshold
    optimal_thickness_visible = None
    for i, T_visible in enumerate(T_percentages_visible):
        if T_visible >= T_threshold:
            optimal_thickness_visible = d_metal_values[i]
            break
    
    if optimal_thickness_visible is None:
        print(f"No thickness achieves the transmissivity threshold of {T_threshold}% for visible light.")
        return None, None, None
    
    # Use the optimal thickness for visible light to calculate the RTA contributions in UV and IR
    idx_optimal = np.where(d_metal_values == optimal_thickness_visible)[0][0]
    R_UV_at_optimal = R_percentages_UV[idx_optimal]
    R_IR_at_optimal = R_percentages_IR[idx_optimal]
    T_visible_at_optimal = T_percentages_visible[idx_optimal]
    
    # Display the results
    print(f"For {metal_name} (Angle of incidence = {angle_incidence}°):")
    print(f"- Optimal thickness to achieve {T_threshold}% transmissivity in visible light: {optimal_thickness_visible:.2f} µm")
    print(f"- UV reflectivity at this thickness: {R_UV_at_optimal:.2f}%")
    print(f"- IR reflectivity at this thickness: {R_IR_at_optimal:.2f}%")
    
    return optimal_thickness_visible, R_UV_at_optimal, R_IR_at_optimal

if __name__ == "__main__":
    lambda_um, n0, n1, n2 = refraction_index("Data/n_k_combined_Sun.txt")
    I = Solar_spectrum("Data/n_k_combined_Sun.txt")
    
    # Define thickness values to test
    d_metal_values = [10e-3, 100e-3]
    
    # Compute RTA for different thicknesses
    R_vis, T_vis, A_vis, R_UV, T_UV, A_UV, R_IR, T_IR, A_IR = compute_spectral_RTA(lambda_um, n0, n1, n2, d_metal_values, I)

    

    
    # Plot RTA vs thickness
    plot_percentage_vs_thickness(d_metal_values, R_vis, T_vis, A_vis, R_UV, T_UV, A_UV, R_IR, T_IR, A_IR, metal_name="silver")
    
  