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
    n_values = []
    k_values = []
    
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
    
    for material, thickness in config:
        if material == "air":
            continue
        elif material == "aerogel":
            # Cas spécial : Aerogel -> n = 1, k = 0 pour toutes les longueurs d'onde
            n_values.append(np.ones_like(wl_interp))  # n = 1
            k_values.append(np.zeros_like(wl_interp))  # k = 0
            layers.append(thickness)  # Ajout de l'épaisseur de l'aerogel
        elif material in material_files:
            filenames.append(material_files[material])
            layers.append(thickness)
        else:
            raise ValueError(f"Unknown material {material}")
    
 
    interpolated_data = [Extraction.extract_nk(file,wl_interp) for file in filenames]
    
    # Ajout des matériaux standards aux listes n_values et k_values
    for n_interp, k_interp in interpolated_data:
        n_values.append(n_interp)
        k_values.append(k_interp)
    
    layer_tuples = [(n_values[i], k_values[i], layers[i]) for i in range(len(layers))]
    
    return layer_tuples

def calculate_RTA_multilayer(layers, wl, phi0=0, ellipsometry=False):
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

    
    if ellipsometry:
        rho = (S_p[1, 0, :] / S_p[0, 0, :]) / (S_s[1, 0, :] / S_s[0, 0, :])
        return rho

    return R, T, A

def objective_function(thicknesses, layers_config, wl, Irradiance, phi0=0, Spectrum_UV_IR = False, Radiative = False):
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
    optical_layers = layers(updated_config,wl)
    R, T, A = calculate_RTA_multilayer(optical_layers, wl, phi0)
    
    # Calcul du score pondéré par l'irradiance
    mask_T = (wl >= 0.4) & (wl <= 0.7)
    if Spectrum_UV_IR : 
        mask_R = ~mask_T
        A_integrated = 1
    elif Radiative:
        mask_R = (wl > 0.2) & (wl <= 8)
        mask_A = (wl > 8) & (wl <= 13)
        A_integrated = np.trapz(A[mask_A] * Irradiance[mask_A], wl[mask_A])
    else :
        mask_R = (wl > 0.7) & (wl <= 20)
        A_integrated = 1

    
    # Intégration pondérée par l'irradiance
    T_integrated = np.trapz(T[mask_T] * Irradiance[mask_T], wl[mask_T])
    R_integrated = np.trapz(R[mask_R] * Irradiance[mask_R], wl[mask_R])
    
    
    # Score à minimiser (on veut maximiser T dans le visible et R ailleurs)
    score = 1/(T_integrated * R_integrated * A_integrated + 1e-10)  # +1e-10 pour éviter division par zéro
    if Radiative:
        score = 1/(T_integrated * A_integrated * R_integrated + 1e-10)
    
    
    return score

def optimize_layer_thicknesses(layers_config, wl, Irradiance, phi0=0, bounds=None, Spectrum_UV_IR = False, Radiative = False):
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
        args=(layers_config, wl, Irradiance, phi0, Spectrum_UV_IR, Radiative),
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
    ax1.plot(wl, R, 'r-', label="Reflectivity", linewidth=0.75)
    ax1.plot(wl, T, 'g-', label="Transmissivity")
    ax1.plot(wl, A, 'b-', label="Absorbance")
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
    ax1.axvspan(wl[0], 0.7, color="yellow", alpha=0.05, label="Visible")
    ax1.axvspan(0.2, 0.4, color="purple", alpha=0.05, label="UV")
    ax1.axvspan(0.7, wl[-1], color="red", alpha=0.05, label="IR")
    
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
        plt.savefig(f"Output/{title}_{config[1][0]}_{config[2][0]}_{config[3][0]}.png")
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
    contour = plt.contourf(D_ZnS, D_Cu, scores, levels=50, cmap='plasma')
    
    # Ajout du point optimal si disponible
    try:
        opt_ZnS, opt_Cu, _ = optimize_layer_thicknesses(layers_config, wl, Irradiance, phi0)
        plt.scatter(opt_ZnS, opt_Cu, c='red', marker='x', s=100, 
                   label=f'Optimum ({opt_ZnS:.2f} µm, {opt_Cu:.2f} µm)')
        plt.legend()
    except Exception as e:
        print(f"Could not plot optimum: {str(e)}")
    
    plt.colorbar(contour, label='Score (log scale)')
    plt.xlabel('Thickness ZnS (µm)')
    plt.ylabel('Thickness Cu (µm)')
    plt.title('Optimal Thickness Landscape')
    plt.savefig("Output/Optimal_thickness/Optimization_Landscape_{}.png".format(len(d_ZnS_range)))
    plt.show()

def comparaison(config1, config2, config3, wl, config4=None, config5 =None, Irrandiance = False,  phi0 = 0, title = "",  save = False, debug = False):
    """
    Compare the performance of three distinct configurations : "
    "1. the 3-layers "
    "2. the system with simple metal "
    "3. the bare glass"
    """

    R_multi, T_multi, A_multi = calculate_RTA_multilayer(layers(config1,wl), wl, phi0)
    R_metal, T_metal, A_metal = calculate_RTA_multilayer(layers(config2,wl), wl, phi0)
    R_glass, T_glass, A_glass = calculate_RTA_multilayer(layers(config3,wl), wl, phi0)
    if config4 is not None:
        R_config4, T_config4, A_config4 = calculate_RTA_multilayer(layers(config4,wl), wl, phi0)
    if config5 is not None:
        R_config5, T_config5, A_config5 = calculate_RTA_multilayer(layers(config5,wl), wl, phi0)

    if Irrandiance:
        I = Extraction.solar_interpolation("Data/ASTM1.5Global.txt", wl)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)
    
    axs[0].plot(wl, R_multi, label="{}/{}/{}".format(config1[1][0],config1[2][0],config1[3][0]), color='blue')
    axs[0].plot(wl, R_metal, label="{}/{}/{}".format(config2[1][0],config2[2][0],config2[3][0]), color='red')
    axs[0].plot(wl, R_glass, label="{}/{}/{}".format(config3[1][0],config3[2][0],config3[3][0]), color='green')
    if config4 is not None:
        axs[0].plot(wl, R_config4, label="{}/{}/{}".format(config4[1][0],config4[2][0],config4[3][0]), color='purple')
    if config5 is not None:
        axs[0].plot(wl, R_config5, label="{}/{}/{}".format(config5[1][0],config5[2][0],config5[3][0]), color='orange')
    axs[0].axvspan(0.4, 0.7, color="yellow", alpha=0.05)
    axs[0].axvspan(0.2, 0.4, color="purple", alpha=0.05)
    axs[0].axvspan(0.7, 20, color="red", alpha=0.05)
    axs[0].legend(loc = "upper right", fontsize = 10)
    if Irrandiance:
        ax0 = axs[0].twinx()
        ax0.plot(wl, I, label="Irradiance", color='orange', alpha=0.5)
        ax0.set_ylabel("Irradiance (W/m²/µm)")
        
        ax1 = axs[1].twinx()
        ax1.plot(wl, I, label="Irradiance", color='orange', alpha=0.5)
        ax1.set_ylabel("Irradiance (W/m²/µm)")
        
        ax2 = axs[2].twinx()
        ax2.plot(wl, I, label="Irradiance", color='orange', alpha=0.5)
        ax2.set_ylabel("Irradiance (W/m²/µm)")
        
    axs[0].set_ylabel("R")
    axs[0].set_title("Reflectance")
    axs[0].set_xscale('log')
    axs[1].plot(wl, T_multi, label="Multilayer", color='blue')
    axs[1].plot(wl, T_metal, label="Metallic layer", color='red')
    axs[1].plot(wl, T_glass, label="Bare Glass", color='green')
    if config4 is not None:
        axs[1].plot(wl, T_config4, label="Config4", color='purple')
    if config5 is not None:
        axs[1].plot(wl, T_config5, label="Config5", color='orange')
    axs[1].set_ylabel("T")
    
    axs[1].set_title("Transmissivity")
    axs[1].set_xscale('log')
    axs[1].axvspan(0.4, 0.7, color="yellow", alpha=0.05, label="Visible")
    axs[1].axvspan(0.2, 0.4, color="purple", alpha=0.05, label="UV")
    axs[1].axvspan(0.7, 20, color="red", alpha=0.05, label="IR")
  
    axs[2].plot(wl, A_multi, label="Multilayer", color='blue')
    axs[2].plot(wl, A_metal, label="Metallic layer", color='red')
    axs[2].plot(wl, A_glass, label="Bare Glass", color='green')
    if config4 is not None:
        axs[2].plot(wl, A_config4, label="Config4", color='purple')
    if config5 is not None:
        axs[2].plot(wl, A_config5, label="Config5", color='orange')
    axs[2].set_ylabel("A")
    axs[2].set_title("Absorbance")
    axs[2].set_xscale('log')
    axs[2].set_xlabel("Wavelength (µm)")
    axs[2].axvspan(0.4, 0.7, color="yellow", alpha=0.05, label="Visible")
    axs[2].axvspan(0.2, 0.4, color="purple", alpha=0.05, label="UV")
    axs[2].axvspan(0.7, 20, color="red", alpha=0.05, label="IR")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig("Output/Comparison/{}.png".format(title))
    else:
        plt.show()

def power_from_the_sun(wl, Irradiance):
    """
    Calculate the energy from the sun for a given wavelength range and solar irradiance.
    
    Parameters:
    wl : array-like
        Wavelengths in micrometers.
    Irradiance : array-like
        Solar irradiance data corresponding to wavelengths.
    
    Returns:
    float : Energy from the sun in Joules.
    """
    
    
    # Calculate energy from the sun
    energy = np.trapz(Irradiance, wl)
    
    return energy

def power_reflected_and_transmitted_absorbed(wl, Irradiance, R, T):
    """
    Calculate the power reflected and transmitted for a given wavelength range and solar irradiance.
    
    Parameters:
    wl : array-like
        Wavelengths in micrometers.
    Irradiance : array-like
        Solar irradiance data corresponding to wavelengths.
    R : array-like
        Reflectivity values for each wavelength.
    T : array-like"
    "    Transmissivity values for each wavelength."
    """ 
    # Calculate power reflected and transmitted
    power_reflected = np.trapz(Irradiance * R, wl)
    power_transmitted = np.trapz(Irradiance * T, wl)
    power_absorbed = np.trapz(Irradiance * (1 - R - T), wl)
    
    return power_reflected, power_transmitted, power_absorbed

def power_save(wl, Irradiance, R, T, A, debug=False):
    """
    Calculate the power saved for a given wavelength range and solar irradiance.
    
    Parameters:
    wl : array-like
        Wavelengths in micrometers.
    Irradiance : array-like
        Solar irradiance data corresponding to wavelengths.
    R : array-like
        Reflectivity values for each wavelength.
    T : array-like
        Transmissivity values for each wavelength.
    A : array-like
        Absorbance values for each wavelength.
    debug : bool, optional
        If True, prints debug information.
    
    Returns:
    float : Power saved in Watt.
    """
    power_r, power_t, power_a = power_reflected_and_transmitted_absorbed(wl, Irradiance, R, T)
    power_sun = power_from_the_sun(wl, Irradiance)
    if debug:
        print(("#" * 20) + "Debug Information : power_save" + ("#" * 20))
        print(f"Power from the sun: {power_sun} W")
        print(f"Power reflected: {power_r} W")
        print(f"Power transmitted: {power_t} W")
        print(f"Power absorbed: {power_a} W")
        print("L'énergie transmisse et l'énergie absorbé rechauffe la piece car l'énergie absorbée par la vitre est thermalisé")
        print("#" * 60)
    print(f"Pourcentage de l'énergie reflechie : {power_r/power_sun*100:.2f}%")
    return power_r

def create_aerogel_dielectric_multilayer(num_bilayers, zns_thickness, aerogel_thickness):
    """
    Create a multilayer system of ZnS and aerogel with a glass substrate.
    """
    layer_configs = []
    for _ in range(num_bilayers):
        layer_configs.append(("ZnS", zns_thickness))
        layer_configs.append(("aerogel", aerogel_thickness))
        
    layer_configs.append(("glass", 0.5))  # substrate
    
    return layer_configs

def explore_multilayer_performance(wl, num_bilayers=10):
    """
    Evaluate and plot performance of ZnS/aerogel multilayers: 
    - Reflectivity in IR
    - Transmissivity in visible
    - Product of both
    """
    # Load solar spectrum
    I = Extraction.solar_interpolation("Data/ASTM1.5Global.txt", wl)
    
    zns_thicknesses = np.linspace(0.001, 1, 40)
    aerogel_thicknesses = np.linspace(0.001, 1, 40)

    T_visible = np.zeros((len(zns_thicknesses), len(aerogel_thicknesses)))
    R_infrared = np.zeros((len(zns_thicknesses), len(aerogel_thicknesses)))
    TR_product = np.zeros((len(zns_thicknesses), len(aerogel_thicknesses)))

    mask_visible = (wl >= 0.4) & (wl <= 0.7)
    mask_infrared = (wl > 0.7)

    for i, zns_th in enumerate(zns_thicknesses):
        for j, aero_th in enumerate(aerogel_thicknesses):
            config = create_aerogel_dielectric_multilayer(num_bilayers, zns_th, aero_th)
            optical_layers = layers(config, wl)
            R, T, A = calculate_RTA_multilayer(optical_layers, wl, phi0=0)
            T_visible[i, j] = np.trapz(T[mask_visible] * I[mask_visible], wl[mask_visible])
            R_infrared[i, j] = np.trapz(R[mask_infrared] * I[mask_infrared], wl[mask_infrared])
            TR_product[i, j] = T_visible[i, j] * R_infrared[i, j]
    
    # Plot 1 - Visible Transmittivity
    plt.figure(figsize=(8, 6))
    plt.imshow(T_visible, extent=[aerogel_thicknesses[0], aerogel_thicknesses[-1],
                                  zns_thicknesses[0], zns_thicknesses[-1]],
               origin='lower', aspect='auto', cmap='viridis')
    plt.title(f"Transmittivity Visible ({num_bilayers} bilayers)")
    plt.xlabel("Aerogel thickness (µm)")
    plt.ylabel("ZnS thickness (µm)")
    plt.colorbar(label="Transmittivity")
    plt.tight_layout()
    plt.savefig("Output/Aerogel/ZnS_aerogel_transmittivity.png")
    plt.show()

    # Plot 2 - IR Reflectivity
    plt.figure(figsize=(8, 6))
    plt.imshow(R_infrared, extent=[aerogel_thicknesses[0], aerogel_thicknesses[-1],
                                   zns_thicknesses[0], zns_thicknesses[-1]],
               origin='lower', aspect='auto', cmap='plasma')
    plt.title(f"Reflectivity Infrared ({num_bilayers} bilayers)")
    plt.xlabel("Aerogel thickness (µm)")
    plt.ylabel("ZnS thickness (µm)")
    plt.colorbar(label="Reflectivity")
    plt.tight_layout()
    plt.savefig("Output/Aerogel/ZnS_aerogel_reflectivity.png")
    plt.show()

    # Plot 3 - T_visible × R_IR
    plt.figure(figsize=(8, 6))
    plt.imshow(TR_product, extent=[aerogel_thicknesses[0], aerogel_thicknesses[-1],
                                   zns_thicknesses[0], zns_thicknesses[-1]],
               origin='lower', aspect='auto', cmap='inferno')
    plt.title(f"T × R Product ({num_bilayers} bilayers)")
    plt.xlabel("Aerogel thickness (µm)")
    plt.ylabel("ZnS thickness (µm)")
    plt.colorbar(label="T × R")
    plt.tight_layout()
    plt.savefig("Output/Aerogel/ZnS_aerogel_TR_product.png")
    plt.show()


def Psi_Delta_theory(config, wl, phi0):
    rho = calculate_RTA_multilayer(layers(config, wl), wl, phi0=phi0, ellipsometry=True)
    psi = np.arctan(np.abs(rho))
    delta = np.angle(rho)
    return np.degrees(psi), np.degrees(delta)

def extract_experimental_ellipsometry():
    """
    Extract experimental ellipsometry data from files and return interpolated values.
    
    Returns:
    tuple: (wl_exp, Psi_exp, Delta_exp, Psi65_exp, Delta_exp65) where wavelengths are in micrometers
    """
    # Extract data from files
    Psi45, Delta45 = Extraction.read_and_plot_ellipsometry("Data/FRANCOIS_45.txt",debug=False)
    Psi65, Delta65 = Extraction.read_and_plot_ellipsometry("Data/FRANCOIS_65.txt",debug=False)
    
    return Psi45, Delta45, Psi65, Delta65

def compare_ellipsometry(Psi_exp, Delta_exp, Psi_exp65, Delta_exp65, config, save=False):
    """
    Compare experimental ellipsometry data with theoretical values for a given configuration.
    
    Parameters:
    Psi_exp : array-like
        Experimental Psi values at 45 degrees.
    Delta_exp : array-like
        Experimental Delta values at 45 degrees.
    Psi_exp65 : array-like
        Experimental Psi values at 65 degrees.
    Delta_exp65 : array-like
        Experimental Delta values at 65 degrees.
    config : list of tuples
        Configuration of the multilayer system.
    save : bool, optional
        If True, saves the plot as an image file.
    filename : str, optional
        Filename for saving the plot.
    """
    wl = np.linspace(400, 700, 31)
    Psi_theory, Delta_theory = Psi_Delta_theory(config, wl, phi0=45)
    Psi_theory_65, Delta_theory_65 = Psi_Delta_theory(config, wl, phi0=65)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Delta vs Wavelength for 45°
    axs[0, 0].plot(wl, Delta_exp, 'b-', label="Experiment")
    axs[0, 0].plot(wl, Delta_theory, 'r--', label="Theory")
    axs[0, 0].set_xlabel("Wavelength (nm)")
    axs[0, 0].set_ylabel("Delta (deg)")
    axs[0, 0].set_title("Delta vs Wavelength (45°)")
    axs[0, 0].grid(True)
    axs[0, 0].legend(loc='upper right', fontsize=8)

    # Psi vs Wavelength for 45°
    axs[0, 1].plot(wl, Psi_exp, 'b-', label="Experiment")
    axs[0, 1].plot(wl, Psi_theory, 'r--', label="Theory")
    axs[0, 1].set_xlabel("Wavelength (nm)")
    axs[0, 1].set_ylabel("Psi (deg)")
    axs[0, 1].set_title("Psi vs Wavelength (45°)")
    axs[0, 1].grid(True)
    axs[0, 1].legend(loc='upper right', fontsize=8)

    # Delta vs Wavelength for 65°
    axs[1, 0].plot(wl, Delta_exp65, 'b-', label="Experiment")
    axs[1, 0].plot(wl, Delta_theory_65, 'r--', label="Theory")
    axs[1, 0].set_xlabel("Wavelength (nm)")
    axs[1, 0].set_ylabel("Delta (deg)")
    axs[1, 0].set_title("Delta vs Wavelength (65°)")
    axs[1, 0].grid(True)
    axs[1, 0].legend(loc='upper right', fontsize=8)

    # Psi vs Wavelength for 65°
    axs[1, 1].plot(wl, Psi_exp65, 'b-', label="Experiment")
    axs[1, 1].plot(wl, Psi_theory_65, 'r--', label="Theory")
    axs[1, 1].set_xlabel("Wavelength (nm)")
    axs[1, 1].set_ylabel("Psi (deg)")
    axs[1, 1].set_title("Psi vs Wavelength (65°)")
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc='upper right', fontsize=8)


    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save:
        filename =''
        for i, (material, thickness) in enumerate(config):
            filename += f"{material}_{thickness*1000:.2f}nm-"
        plt.savefig("Output/Ellipsometry/{}.png".format(filename))
    else:
        plt.show()

def fit_oxide_metal_stack_symmetric(
    Psi_Delta_theory,
    extract_experimental_ellipsometry,
    config_template=None,
    bounds=None,
    initial_guess=None
):
    """
    Fit the ellipsometry data (Psi, Delta at 45° and 65°) assuming the same oxide thickness above and below the metal layer.

    Parameters:
    - Psi_Delta_theory: function(config, wl, phi0) → (Psi, Delta)
    - extract_experimental_ellipsometry: function() → (wl, Psi_45, Delta_45, Psi_65, Delta_65)
    - config_template: list of (material, thickness)
    - bounds: list of 2-tuples [(oxide_min, oxide_max), (metal_min, metal_max)]
    - initial_guess: [oxide_thickness, metal_thickness]

    Returns:
    - optimized_config: list of (material, thickness)
    """

    if config_template is None:
        config_template = [
            ("air", 0),
            ("Al2O3", 0.010),  # Top oxide
            ("Ag", 0.012),     # Metal
            ("Al2O3", 0.010),  # Bottom oxide (same material)
            ("glass", 4.0)
        ]

    if bounds is None:
        bounds = [(0.001, 4000), (0.001, 4000)]  # µm

    if initial_guess is None:
        initial_guess = [0.030, 0.012]  # [oxide_thickness, metal_thickness]

    # Load experimental data
    wl = np.linspace(400, 700, 31)  # Wavelength range in nm
    Psi_45, Delta_45, Psi_65, Delta_65 = extract_experimental_ellipsometry()

    def objective(thicknesses):
        oxide_thickness, metal_thickness = thicknesses
        config = config_template.copy()
        config[1] = (config[1][0], oxide_thickness)   # top oxide
        config[2] = (config[2][0], metal_thickness)   # metal
        config[3] = (config[3][0], oxide_thickness)   # bottom oxide (same thickness)

        psi_45, delta_45 = Psi_Delta_theory(config, wl, phi0=45)
        psi_65, delta_65 = Psi_Delta_theory(config, wl, phi0=65)

        error = (
            np.mean((Psi_45 - psi_45)**2) +
            np.mean((Delta_45 - delta_45)**2) +
            np.mean((Psi_65 - psi_65)**2) +
            np.mean((Delta_65 - delta_65)**2)
        )
        return error

    result = minimize(
        objective,
        initial_guess,
        bounds=bounds,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': 300}
    )

    # Update config with optimized thicknesses
    optimized_config = config_template.copy()
    optimized_config[1] = (optimized_config[1][0], result.x[0])
    optimized_config[2] = (optimized_config[2][0], result.x[1])
    optimized_config[3] = (optimized_config[3][0], result.x[0])

    return optimized_config
    
def make_config(number_of_layers, initial_config):
    """
    Create a configuration for a multilayer system, of a repeation of a dielectic and a metal layer.
    The configuration is a list of tuples, where each tuple contains the material name and its thickness.
    tickness contains the thickness of the dielectric and the metal layer. Config is the initial configuration is a trilayer system.
    The function returns a list of tuples representing the configuration of the multilayer system.
    """
    config = []
    config.append((initial_config[0]))  # Add the air layer
    for i in range(number_of_layers):
        # Add the dielectric layer
        config.append((initial_config[1]))
        # Add the metal layer
        config.append((initial_config[2]))
    
    config.append((initial_config[3]))  # Add the glass layer
    return config

def pourcentage(config, wl, Irradiance, phi0=0):
    """
    Calculate the percentage of power reflected and transmitted for a given configuration.
    
    Parameters:
    config : list of tuples
        Configuration of the multilayer system.
    wl : array-like
        Wavelengths in micrometers.
    Irradiance : array-like
        Solar irradiance data corresponding to wavelengths.
    phi0 : float, optional
        Angle of incidence in degrees (default is 0°).  
    
        
    Returns:
    tuple : (R_percentage, T_percentage, A_percentage)
        Percentages of power reflected, transmitted, and absorbed.
    """
    # Calculate R, T, A for the given configuration
    R, T, A = calculate_RTA_multilayer(layers(config, wl), wl, phi0)
    
    # Calculate power reflected and transmitted
    power_reflected = np.trapz(Irradiance * R, wl)
    power_transmitted = np.trapz(Irradiance * T, wl)
    power_absorbed = np.trapz(Irradiance * (1 - R - T), wl)
    
    # Calculate total power from the sun
    power_sun = np.trapz(Irradiance, wl)
    
    # Calculate percentages
    R_percentage = (power_reflected / power_sun) * 100
    T_percentage = (power_transmitted / power_sun) * 100
    A_percentage = (power_absorbed / power_sun) * 100
    
    return R_percentage, T_percentage, A_percentage


if __name__ == "__main__":

    wl = np.linspace(0.2, 20, 1000)
    I = Extraction.solar_interpolation("Data/ASTM1.5Global.txt", wl)
    
    
    

    power_saving_information= False
    aerogel = False
    optim_d = False
    ZnS_info = False
    Cu_info = False
    Test = False
    Cu_dielec = False
    Ag_dielec = False
    comparaison_multi = False
    plot_ten_layer = False
    ellipsometry = True
    if ellipsometry:
        Psi, Delta, Psi65, Delta65 = extract_experimental_ellipsometry()

        config = [
            ("air", 0),
            ("TiO2", 0.05),
            ("Ag", 0.009),
            ("TiO2", 0.005),
            ("glass", 0.5)
        ]

        d_TiO2, d_Ag, d_TiO2 = optimize_layer_thicknesses(config, wl, I, phi0=0, Spectrum_UV_IR=True)
        print("Optimized thicknesses: ", d_TiO2, d_Ag, d_TiO2)
        config = [
            ("air", 0),
            ("TiO2", d_TiO2),
            ("Ag", d_Ag),
            ("TiO2", d_TiO2),
            ("glass", 0.5)
        ]
        config_ACG = fit_oxide_metal_stack_symmetric(Psi_Delta_theory, extract_experimental_ellipsometry, config_template=config)
        print("Optimized configuration: ", config_ACG)
        compare_ellipsometry(Psi, Delta, Psi65, Delta65, config, save = True)

        R_pourcentage, T_pourcentage, A_pourcentage = pourcentage(config, wl, I, phi0=0)
        print(f"Reflectance: {R_pourcentage:.2f}%")
        print(f"Transmittance: {T_pourcentage:.2f}%")
        print(f"Absorbance: {A_pourcentage:.2f}%")
        
        


    if plot_ten_layer:
        config = create_aerogel_dielectric_multilayer(10, 0.3, 0.3)
        
        plot_R_T_A_fixed_phi0_and_d_multilayer(config, wl, Irradiance=I, phi0=0, title="aerogel", save=True)
    

    if Cu_dielec:  # Add a condition here if needed
        # Configuration initiale
        config = [
            ("air", 0),
            ("ZnS", 0.032),
            ("Cu", 0.022),
            ("ZnS", 0.032),
            ("glass", 10)
        ]

        config_2 = [
            ("air", 0),
            ("VO2", 0.022),
            ("Cu", 0.022),
            ("VO2", 0.022),
            ("glass", 10)
        ]

        d_Zn0, d_Cu, d_ZnO = optimize_layer_thicknesses(config_2, wl, I, phi0=0, Spectrum_UV_IR=True)

        config_2 = [
            ("air", 0),
            ("VO2", d_Zn0),
            ("Cu", d_Cu),
            ("VO2", d_ZnO),
            ("glass", 10)
        ]

        config_3 = [
            ("air", 0),
            ("TiO2", 0.032),
            ("Cu", 0.022),
            ("TiO2", 0.032),
            ("glass", 10)
        ]

        d_TiO2, d_Cu, d_TiO2 = optimize_layer_thicknesses(config_3, wl, I, phi0=0, Spectrum_UV_IR=True)

        config_3 = [
            ("air", 0),
            ("TiO2", d_TiO2),
            ("Cu", d_Cu),
            ("TiO2", d_TiO2),
            ("glass", 10)
        ]

        config_4 = [
            ("air", 0),
            ("SiO2", 0.032),
            ("Cu", 0.022),
            ("SiO2", 0.032),
            ("glass", 10)
        ]

        d_SiO2, d_Cu, d_SiO2 = optimize_layer_thicknesses(config_4, wl, I, phi0=0, Spectrum_UV_IR=True)

        config_4 = [
            ("air", 0),
            ("SiO2", d_SiO2),
            ("Cu", d_Cu),
            ("SiO2", d_SiO2),
            ("glass", 10)
        ]

        config_5 = [
            ("air", 0),
            ("PMMA", 0.032),
            ("Cu", 0.022),
            ("PMMA", 0.032),
            ("glass", 10)
        ]

        d_PMMA, d_Cu, d_PMMA = optimize_layer_thicknesses(config_5, wl, I, phi0=0, Spectrum_UV_IR=True)
        config_5 = [
            ("air", 0),
            ("PMMA", d_PMMA),
            ("Cu", d_Cu),
            ("PMMA", d_PMMA),
            ("glass", 10)
        ]

        comparaison(config, config_2, config_3, wl, config_4, config_5, Irrandiance=True, phi0=0, title="ZnS-VO2-SiO2-TiO2-PMMA_Copper", save=True)
        if Ag_dielec:  # Add a condition here if needed
            # Configuration initiale
            config = [
                ("air", 0),
                ("ZnS", 0.032),
                ("Ag", 0.022),
                ("ZnS", 0.032),
                ("glass", 10)
            ]

            config_2 = [
                ("air", 0),
                ("VO2", 0.022),
                ("Ag", 0.022),
                ("VO2", 0.022),
                ("glass", 10)
            ]

            d_Zn0, d_Ag, d_ZnO = optimize_layer_thicknesses(config_2, wl, I, phi0=0, Spectrum_UV_IR=True)

            config_2 = [
                ("air", 0),
                ("VO2", d_Zn0),
                ("Ag", d_Ag),
                ("VO2", d_ZnO),
                ("glass", 10)
            ]

            config_3 = [
                ("air", 0),
                ("TiO2", 0.032),
                ("Ag", 0.022),
                ("TiO2", 0.032),
                ("glass", 10)
            ]

            d_TiO2, d_Ag, d_TiO2 = optimize_layer_thicknesses(config_3, wl, I, phi0=0, Spectrum_UV_IR=True)

            config_3 = [
                ("air", 0),
                ("TiO2", d_TiO2),
                ("Ag", d_Ag),
                ("TiO2", d_TiO2),
                ("glass", 10)
            ]

            config_4 = [
                ("air", 0),
                ("SiO2", 0.032),
                ("Ag", 0.022),
                ("SiO2", 0.032),
                ("glass", 10)
            ]

            d_SiO2, d_Ag, d_SiO2 = optimize_layer_thicknesses(config_4, wl, I, phi0=0, Spectrum_UV_IR=True)

            config_4 = [
                ("air", 0),
                ("SiO2", d_SiO2),
                ("Ag", d_Ag),
                ("SiO2", d_SiO2),
                ("glass", 10)
            ]

            config_5 = [
                ("air", 0),
                ("PMMA", 0.032),
                ("Ag", 0.022),
                ("PMMA", 0.032),
                ("glass", 10)
            ]

            d_PMMA, d_Ag, d_PMMA = optimize_layer_thicknesses(config_5, wl, I, phi0=0, Spectrum_UV_IR=True)
            config_5 = [
                ("air", 0),
                ("PMMA", d_PMMA),
                ("Ag", d_Ag),
                ("PMMA", d_PMMA),
                ("glass", 10)
            ]

            comparaison(config, config_2, config_3, wl, config_4, config_5, Irrandiance=True, phi0=0, title="ZnS-VO2-SiO2-TiO2-PMMA_Silver", save=True)
    if ZnS_info :
        
        wl = np.linspace(0.2, 40, 1000)
        n, k = Extraction.interpolate(wl, *Extraction.extract_wl_n_k("Data/ZnS_Querry.txt"))
        n1 = n - 1j * k
        n0 = 1.0
        phi0 = 0
        R = task2.reflectivity_semi_infinite_layer(n0, n1, phi0)
        print("Calculating the reflectivity of ZnS")
        plt.figure(figsize=(10, 6))
        plt.plot(wl, R, label="R", linewidth=2)
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Reflectivity")
        plt.title("Reflectivity of ZnS")
        plt.xscale('log')
        plt.legend()
        plt.axvspan(0.4, 0.7, color="yellow", alpha=0.05, label="Visible")
        plt.axvspan(0.2, 0.4, color="purple", alpha=0.05, label="UV")
        plt.axvspan(0.7, 20, color="red", alpha=0.05, label="IR")
        plt.show()

        task2.plot_n_k(wl, n, k, title="ZnS", log=True)

        config = [
            ("air", 0),
            ("ZnS", 1),
    
        ]

        
    if Cu_info :
       
        n, k = Extraction.interpolate(wl, *Extraction.extract_wl_n_k("Data/Cu_Querry.txt"))
        n1 = n - 1j * k
        n0 = 1.0
        phi0 = 0
        R = task2.reflectivity_semi_infinite_layer(n0, n1, phi0)
        print("Calculating the reflectivity of Cu")
        plt.figure(figsize=(10, 6))
        plt.plot(wl, R, label="R", linewidth=2)
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Reflectivity")
        plt.title("Reflectivity of Cu") 
        plt.xscale('log')
        plt.legend()
        plt.axvspan(0.4, 0.7, color="yellow", alpha=0.05, label="Visible")
        plt.axvspan(0.2, 0.4, color="purple", alpha=0.05, label="UV")
        plt.axvspan(0.7, 20, color="red", alpha=0.05, label="IR")
        plt.savefig("Output/Cu/Cu_Transmittance.png")
        plt.show()
        task2.plot_n_k(wl, n, k, title="Cu", log=True)
    # Plot the ellipsometry data
    
    if power_saving_information :
        config = [
            ("air", 0),
            ("ZnS", 0.032),
            ("Cu", 0.022),
            ("ZnS", 0.032),
            ("glass", 10)
        ]

        config_metal = [
            ("air", 0),
            ("Ag", 0.009),
            ("glass", 10)
        ]
        config_copper = [
            ("air", 0),
            ("Cu", 0.009),
            ("glass", 10)
        ]
        config_2 = [
            ("air", 0),
            ("ZnS", 0.032),
            ("Ag", 0.022),
            ("ZnS", 0.032),
            ("glass", 10)
        ]

        d_ZnS1, d_Ag, d_ZnS2 = optimize_layer_thicknesses(config_2, wl, I, phi0=0, Spectrum_UV_IR=True)
        config_2 = [
            ("air", 0),
            ("ZnS", d_ZnS1),
            ("Ag", d_Ag),
            ("ZnS", d_ZnS2),
            ("glass", 10)
        ]
        
        R, T, A = calculate_RTA_multilayer(layers(config, wl), wl)
        P = power_save(wl, I, R, T, A, False)
        print("The power saved in the case of the multilayer system  is :", P, " w")

       

        R_metal, T_metal, A_metal = calculate_RTA_multilayer(layers(config_metal, wl), wl)
        P_metal = power_save(wl, I, R_metal, T_metal, A_metal, False)
        print("The power saved by the silver layer is :", P_metal, " w")

        R_copper, T_copper, A_copper = calculate_RTA_multilayer(layers(config_copper, wl), wl)
        P_copper = power_save(wl, I, R_copper, T_copper, A_copper, False)
        print("The power saved by the copper layer is :", P_copper, " w")

        R_2, T_2, A_2 = calculate_RTA_multilayer(layers(config_2, wl), wl)
        P_2 = power_save(wl, I, R_2, T_2, A_2, False)
        print("The power saved in the case of the multilayer system with Ag is :", P_2, " w")
    
    d = np.linspace(0.001, 0.04, 40)

    if optim_d : 
        print("Optimizing the thicknesses of the multilayer system")
        d_zns, d_cu, d_zns = optimize_layer_thicknesses(config, wl, I, phi0=0, Spectrum_UV_IR= True)

        config = [
            ("air", 0),
            ("ZnS", d_zns),
            ("Cu", d_cu),
            ("ZnS", d_zns),
            ("glass", 10)
        ]
        print(f"Optimized thicknesses: ZnS1={d_zns*1000:.2f} µm, Cu={d_cu*1000:.2f} µm, ZnS2={d_zns*1000:.2f} µm")
        config_metal = [
            ("air", 0),
            ("Ag", 0.009),
            ("glass", 10)
        ]
        config_glass = [
            ("air", 0),
            ("glass", 1)
        ]
        comparaison(config, config_metal, config_glass, wl, True, phi0=0, title="Opt_Ag", save=True)

        config_metal = [
            ("air", 0),
            ("Cu", 0.009),
            ("glass", 10)
        ]

        comparaison(config, config_metal, config_glass, wl, True, phi0=0, title="Opt_Cu", save=True)

    #10 bilayer of ZnS/Air 
    if aerogel:
        print("Optimizing the thicknesses of the aerogel system")
        wl = np.linspace(0.2, 20, 1000)
        explore_multilayer_performance(wl, num_bilayers=10)
    
    
    if comparaison_multi:  # Add a condition here if needed
        # Define initial configurations for ZnS/Al/ZnS, ZnS/Cu/ZnS, and ZnS/Ag/ZnS
        configs = {
            "ZnS/Al/ZnS": [("air", 0), ("ZnS", 0.032), ("Al", 0.022), ("ZnS", 0.032), ("glass", 10)],
            "ZnS/Cu/ZnS": [("air", 0), ("ZnS", 0.032), ("Cu", 0.022), ("ZnS", 0.032), ("glass", 10)],
            "ZnS/Ag/ZnS": [("air", 0), ("ZnS", 0.032), ("Ag", 0.022), ("ZnS", 0.032), ("glass", 10)],
        }

        optimized_configs = {}

        # Optimize thicknesses for each configuration
        for name, config in configs.items():
            d_ZnS1, d_metal, d_ZnS2 = optimize_layer_thicknesses(config, wl, I, phi0=0, Spectrum_UV_IR=True)
            optimized_configs[name] = [
                ("air", 0),
                ("ZnS", d_ZnS1),
                (config[2][0], d_metal),  # Use the metal from the original config
                ("ZnS", d_ZnS2),
                ("glass", 10),
            ]

        # Compare the optimized configurations
        comparaison(
            optimized_configs["ZnS/Al/ZnS"],
            optimized_configs["ZnS/Cu/ZnS"],
            optimized_configs["ZnS/Ag/ZnS"],
            wl,
            Irrandiance=True,
            phi0=0,
            title="Comparison_ZnS_Al_Cu_Ag",
            save=True,
        )


    