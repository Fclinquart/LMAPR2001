import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 
import os
import Extraction
import scipy.optimize as opt
from scipy.optimize import minimize

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Task 2'))
sys.path.append(parent_dir)
import task2 # type: ignore



def layers(config, wl_interp, debug=False):
    """Génère le système multicouche basé sur la configuration donnée."""
    layers = []
    filenames = []
    
    material_files = {
        "ZnS": "Data/ZnS_Querry.txt",
        "Cu": "Data/n_k_copper.txt",
        "glass": "Data/Glass_Palik.txt",
        "Ag": "Data/n_k_silver.txt"  # Exemple pour un autre matériau
    }
    
    for material, thickness in config:
        if material == "air":
            continue
        if material in material_files:
            filenames.append(material_files[material])
            layers.append(thickness)
        else:
            raise ValueError(f"Unknown material {material}")
    
    extracted_data = [Extraction.extract_wl_n_k(file) for file in filenames]
    interpolated_data = [Extraction.interpolate(wl_interp, *data) for data in extracted_data]
    
    layer_tuples = [(n_interp, k_interp, layers[i]) for i, (n_interp, k_interp) in enumerate(interpolated_data)]
    return layer_tuples

def calculate_RTA_multilayer(layers, wl, phi0=0):
    """
    Calculate Reflectivity, Transmissivity, and Absorbance for a multi-layered system.
    
    Parameters:
    layers: list of tuples, each tuple contains (n, kappa, d) for each layer
            where n is the refractive index, kappa is the extinction coefficient,
            and d is the thickness in micrometers.
    lambda_um: array-like, wavelength in micrometers.
    angle_incidence_deg: angle of incidence in degrees.
    
    Returns:
    R: Reflectivity
    T: Transmissivity
    A: Absorbance
    """
    angle_incidence = np.radians(phi0)
    N_air = 1.0

    # Initialize the scattering matrix S as the identity matrix for each wavelength
    S_p = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, len(wl)))
    S_s = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, len(wl)))

    # Iterate over each layer
    for i, (n, kappa, d) in enumerate(layers):
        N_layer = n - 1j * kappa

        # Calculate the angle of propagation in the current layer
        sin_theta_layer = N_air * np.sin(angle_incidence) / N_layer
        cos_theta_layer = np.sqrt(1 - sin_theta_layer**2)

        # Calculate the phase shift beta for all wavelengths
        beta = 2 * np.pi * d * N_layer * cos_theta_layer / wl

        # Create the layer matrix L as a 3D array
        L = np.zeros((2, 2, len(wl)), dtype=complex)
        L[0, 0, :] = np.exp(1j * beta)  # Forward propagation
        L[1, 1, :] = np.exp(-1j * beta)  # Backward propagation

        # Calculate the Fresnel coefficients for p and s polarizations
        if i == 0:
            N_prev = N_air
        else:
            N_prev = layers[i-1][0] - 1j * layers[i-1][1]

        r_p = (N_layer * np.cos(angle_incidence) - N_prev * cos_theta_layer) / (N_layer * np.cos(angle_incidence) + N_prev * cos_theta_layer)
        r_s = (N_prev * np.cos(angle_incidence) - N_layer * cos_theta_layer) / (N_prev * np.cos(angle_incidence) + N_layer * cos_theta_layer)

        t_p = (2 * N_prev * np.cos(angle_incidence)) / (N_layer * np.cos(angle_incidence) + N_prev * cos_theta_layer)
        t_s = (2 * N_prev * np.cos(angle_incidence)) / (N_prev * np.cos(angle_incidence) + N_layer * cos_theta_layer)

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
        for i in range(len(wl)):
            S_p[:, :, i] = np.dot(S_p[:, :, i], np.dot(I_p[:, :, i], L[:, :, i]))
            S_s[:, :, i] = np.dot(S_s[:, :, i], np.dot(I_s[:, :, i], L[:, :, i]))

    # Calculate the reflection and transmission coefficients for each wavelength
    R_p = np.abs(S_p[1, 0, :] / S_p[0, 0, :])**2
    R_s = np.abs(S_s[1, 0, :] / S_s[0, 0, :])**2

    T_p = np.abs(1 / S_p[0, 0, :])**2
    T_s = np.abs(1 / S_s[0, 0, :])**2

    # Correction factor for transmissivity
    n_substrate = layers[-1][0]
    correction_factor = (n_substrate * np.cos(angle_incidence)) / (N_air * np.cos(angle_incidence))
    T_p_corrected = T_p * correction_factor
    T_s_corrected = T_s * correction_factor

    # Average R and T for unpolarized light
    R = (R_p + R_s) / 2
    T = (T_p_corrected + T_s_corrected) / 2
    A = 1 - R - T

    return R, T, A

def objective_function(thicknesses, layers_config, wl, Irradiance, phi0=0, Spectrum_UV_IR = False):
    """Fonction objectif pour l'optimisation avec irradiance."""
    # Mise à jour des épaisseurs
    updated_config = []
    thickness_idx = 0
    
    for material, _ in layers_config:
        if material == "air":
            updated_config.append((material, 0))
        elif material == "glass":
            updated_config.append((material, 0.5))
        else:
            updated_config.append((material, thicknesses[thickness_idx]))
            thickness_idx += 1
    
    # Calcul des propriétés optiques
    optical_layers = layers(updated_config)
    R, T, A = calculate_RTA_multilayer(optical_layers, wl, phi0)
    
    # Calcul du score pondéré par l'irradiance
    mask_T = (wl >= 0.4) & (wl <= 0.7)
    if Spectrum_UV_IR : 
        mask_R = ~mask_T
    else :
        mask_R = (wl > 0.7) & (wl <= 20)
    
    # Intégration pondérée par l'irradiance
    T_integrated = np.trapz(T[mask_T] * Irradiance[mask_T], wl[mask_T])
    R_integrated = np.trapz(R[mask_R] * Irradiance[mask_R], wl[mask_R])
    
    # Score à minimiser (on veut maximiser T dans le visible et R ailleurs)
    score = 1/(T_integrated * R_integrated + 1e-10)  # +1e-10 pour éviter division par zéro
    
    return score

def optimize_layer_thicknesses(layers_config, wl, Irradiance, phi0=0, bounds=None, Spectrum_UV_IR = False):
    """
    Optimise les épaisseurs en tenant compte de l'irradiance solaire.
    """
    # Extraire les épaisseurs initiales (sauf air et verre)
    initial_thicknesses = [thickness for material, thickness in layers_config 
                         if material not in ("air", "glass")]
    
    # Vérification
    if len(initial_thicknesses) != 3:
        raise ValueError("La configuration doit contenir exactement 3 couches à optimiser (ZnS, Cu, ZnS)")
    
    # Bornes par défaut (1nm à 1µm)
    if bounds is None:
        bounds = [(0.0001, 4000) for _ in initial_thicknesses]
    
    # Optimisation
    result = minimize(
        objective_function,
        initial_thicknesses,
        args=(layers_config, wl, Irradiance, phi0),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 100, 'disp': True}
    )
    
    if result.success:
        return result.x[0], result.x[1], result.x[2]  # ZnS1, Cu, ZnS2
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")

def plot_R_T_A_fixed_phi0_and_d_multilayer(config, wl, Irradiance = False, phi0=0, title="", save=False):
    """
    Plots the Reflectivity (R), Transmissivity (T), and Absorbance (A) 
    of a multilayer optical system for a fixed angle of incidence,
    along with the solar irradiance spectrum.
    
    Parameters:
    -----------
    layers : list of tuples
        A list where each tuple represents a layer and contains:
        - n (float): Real part of the refractive index.
        - kappa (float): Extinction coefficient.
        - d (float): Thickness of the layer in micrometers.
    wl : array-like
        Wavelengths in micrometers.
    Irradiance : array-like
        Solar irradiance data corresponding to wavelengths.
    phi0 : float, optional
        Angle of incidence in degrees (default is 0°).
    title : str, optional
        Title of the plot.
    save : bool, optional
        If True, saves the plot as an image file.

    """
    l = layers(config, wl)
    R, T, A = calculate_RTA_multilayer(l, wl, phi0)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot RTA on primary y-axis
    ax1.plot(wl, R, 'r-', label="Reflectivity", linewidth=2)
    ax1.plot(wl, T, 'g-', label="Transmissivity", linewidth=2)
    ax1.plot(wl, A, 'b-', label="Absorbance", linewidth=2)
    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel("R, T, A")
    ax1.set_xscale('log')
    
    # Create secondary y-axis for irradiance
    if Irradiance is not False:
        wl_sol, Irradiance = Extraction.extract_solar_irrandiance("Data/ASTM1.5Global.txt", plot=False)

        ax2 = ax1.twinx()
        ax2.plot(wl_sol, Irradiance, 'k--', alpha=0.5, label="Solar Irradiance")
        ax2.set_ylabel("Irradiance (W/m²/µm)", color='k')
        ax2.tick_params('y', colors='k')
    
    # Add spectral regions
    ax1.axvspan(0.38, 0.8, color="yellow", alpha=0.05, label="Visible")
    ax1.axvspan(0.2, 0.38, color="purple", alpha=0.05, label="UV")
    ax1.axvspan(0.8, 20, color="red", alpha=0.05, label="IR")
    
    # Add a legend for the system configuration
     # Add a legend for the system configuration
    # system_legend = "System Configuration:\n"
    # for i, layer in enumerate(layer_configs):
    #     system_legend += f"Layer {i+1}: {layer['material']} ({layer['thickness']*1000:.2f} nm)\n"
    
    # # Place the system configuration legend in the plot
    # plt.text(0.02, 0.98, system_legend.strip(), transform=plt.gca().transAxes,
    #          fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    legend = ""
    for i, (material, thickness) in enumerate(config):
        if material == "air":
            continue
        if material == "glass":
            legend += f"{material} ({thickness/1000:.2f} mm)\n"
            continue
        legend += f"{material} ({thickness*1000:.2f} nm)\n"
    plt.text(0.85, 0.98, legend.strip(), transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8, pad=3))

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if Irradiance is not False:
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    if title:
        ax1.set_title(title)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"Output/RTA_phi0_d/{title}.png")
    else:
        plt.show()
        
def plot_optimization_landscape(layers_config, wl, d_ZnS_range, d_Cu_range, Irradiance, phi0=0):
    """
    Trace le paysage d'optimisation en 2D avec irradiance.
    """
    # Création du meshgrid
    D_ZnS, D_Cu = np.meshgrid(d_ZnS_range, d_Cu_range)
    scores = np.zeros_like(D_ZnS)
    
    # Calcul du score pour chaque combinaison
    for i in range(len(d_ZnS_range)):
        for j in range(len(d_Cu_range)):
            current_thicknesses = [d_ZnS_range[i], d_Cu_range[j], d_ZnS_range[i]]
            scores[j,i] = objective_function(current_thicknesses, layers_config, wl, Irradiance, phi0)
    
    # Normalisation logarithmique pour meilleure visualisation
    scores = np.log(scores)
    
    # Plot
    plt.figure(figsize=(12, 8))
    contour = plt.contourf(D_ZnS, D_Cu, scores, levels=50, cmap='viridis')
    
    # Ajout du point optimal si disponible
    try:
        opt_ZnS, opt_Cu, _ = optimize_layer_thicknesses(layers_config, wl, Irradiance, phi0)
        plt.scatter(opt_ZnS, opt_Cu, c='red', marker='x', s=100, 
                   label=f'Optimum ({opt_ZnS:.2f} µm, {opt_Cu:.2f} µm)')
        plt.legend()
    except Exception as e:
        print(f"Could not plot optimum: {str(e)}")
    
    plt.colorbar(contour, label='Score (log scale)')
    plt.xlabel('Épaisseur ZnS (µm)')
    plt.ylabel('Épaisseur Cu (µm)')
    plt.title('Paysage d\'optimisation (plus bas = mieux)')
    plt.show()

def comparaison(config1, config2, config3, wl, phi0 = 0, debug = False):
    """
    Compare the performance of three distinct configurations : "
    "1. the 3-layers "
    "2. the system with simple metal "
    "3. the bare glass"
    """

    R, T, A = calculate_RTA_multilayer(layers(config1), wl, phi0)
    R_metal, T_metal, A_metal = calculate_RTA_multilayer(layers(config2), wl, phi0)


if __name__ == "__main__":
    # Configuration initiale
    config = [
        ("air", 0),
        ("ZnS", 0.01),
        ("Cu", 0.01),
        ("ZnS", 0.01),
        ("glass", 0.5)
        
    ]

    wl = np.linspace(0.2, 20, 1000)  

    l = layers(config,wl) 

    plot_R_T_A_fixed_phi0_and_d_multilayer(config, wl, Irradiance = False, phi0=0, title="Initial Configuration", save=False)



    