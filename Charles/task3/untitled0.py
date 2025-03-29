import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.interpolate import interp1d
import pandas as pd
import re
from scipy.optimize import minimize
from scipy.integrate import simps


plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})



def read_nk_file(filename, debug=False):
    """Lit un fichier nk et extrait les données sous forme de dictionnaire {wl: (n, k)} ou pour un fichier avec 3 colonnes."""
    data = {}
    lambda_um, n, k = [], [], []

    with open(filename, "r") as f:
        lines = f.readlines()

    wl = None
    mode = None  # "n" ou "k"

    if debug:
        print(f"Début de la lecture du fichier : {filename}")
    
    # Lecture ligne par ligne
    for i, line in enumerate(lines):
        parts = line.split()

        # Vérifie si la ligne est dans un format avec 3 colonnes (lambda, n, k)
        if len(parts) == 3:
            try:
                # Extraire les valeurs des 3 colonnes et les ajouter aux listes
                lambda_um.append(float(parts[0]))
                n.append(float(parts[1]))
                k.append(float(parts[2]))
                if debug:
                    print(f"  -> Méthode 3 colonnes : λ = {parts[0]}, n = {parts[1]}, k = {parts[2]}")
            except ValueError:
                if debug:
                    print(f"  Ignoré (ligne mal formée) : {line.strip()}")
                continue  # Ignorer les lignes mal formées
            continue

        # Cas d'une section "wl n" ou "wl k"
        if len(parts) == 2 and parts[0].lower() == "wl":
            mode = parts[1].lower()  # On sait si on est dans un bloc "n" ou "k"
            if debug:
                print(f"  Section détectée : wl {mode}")
            continue  # Passer à la ligne suivante

        # Cas pour extraire les valeurs de n ou k dans le format de longueurs d'onde
        if len(parts) == 2:
            try:
                wl = float(parts[0])
                value = float(parts[1])

                if wl not in data:
                    data[wl] = {"n": None, "k": None}  # Initialisation

                if mode == "n":
                    data[wl]["n"] = value
                    if debug:
                        print(f"  → Méthode wl n/k : wl = {wl} µm, n = {value}")
                elif mode == "k":
                    data[wl]["k"] = value
                    if debug:
                        print(f"  → Méthode wl n/k : wl = {wl} µm, k = {value}")

            except ValueError:
                if debug:
                    print(f"  Ignoré (ligne mal formée ou donnée invalide) : {line.strip()}")
                continue  # Ignorer les lignes mal formées

    # Filtrage des valeurs valides pour la première approche (format "wl n/k")
    for wl, values in data.items():
        if values["n"] is not None and values["k"] is not None:
            lambda_um.append(wl)
            n.append(values["n"])
            k.append(values["k"])

    if debug:
        print(f"\n--- Résumé des données extraites pour le fichier {filename} ---")
        print(f"Longueurs d'onde valides : {len(lambda_um)}")
        for wl, n_val, k_val in zip(lambda_um, n, k):
            print(f"  λ = {wl} µm, n = {n_val}, k = {k_val}")

    # Conversion des listes en arrays NumPy
    lambda_um = np.array(lambda_um)
    n = np.array(n)
    k = np.array(k)

    # Retourner les résultats finaux
    return lambda_um, n, k

def interpolate_data(lambda_um, n, k, lambda_common):
    """Interpolate n and k data to a common wavelength grid."""
    interp_n = interp1d(lambda_um, n, kind='linear', fill_value="extrapolate")
    interp_k = interp1d(lambda_um, k, kind='linear', fill_value="extrapolate")
    n_interp = interp_n(lambda_common)
    k_interp = interp_k(lambda_common)
    return n_interp, k_interp


def initialize_layers(metals, lambda_um, layer_configs, n_glass_interp, kappa_glass_interp, debug=False):
    """
    Initialise les couches pour une structure multicouche en utilisant les données interpolées de n et k.
    
    Parameters:
    metals: dict, dictionnaire contenant les données interpolées de n et k pour chaque métal.
    lambda_um: float, longueur d'onde en micromètres pour laquelle les valeurs de n et k sont interpolées.
    layer_configs: list of dict, liste de configurations de couches. Chaque configuration est un dictionnaire
                   contenant les clés 'material', 'thickness', et éventuellement d'autres paramètres.
    n_glass_interp: array-like, indice de réfraction interpolé du verre sur une grille de longueurs d'onde commune.
    kappa_glass_interp: array-like, coefficient d'extinction interpolé du verre sur une grille de longueurs d'onde commune.
    debug: bool, si True, affiche des informations de débogage (uniquement les 3 premières valeurs de n et kappa).
    
    Returns:
    layers: list of tuples, liste de couches sous la forme (n, kappa, d) pour chaque couche.
    """
    layers = []
    
    if debug:
        print("Configuration du système multicouche :")
        print("=" * 40)
    
    for i, config in enumerate(layer_configs):
        material = config["material"]
        thickness = config["thickness"]
        
        if material in metals:
            # Récupérer les valeurs interpolées de n et k pour la longueur d'onde donnée
            n = metals[material]["n_interp"][np.searchsorted(lambda_common, lambda_um)]
            kappa = metals[material]["k_interp"][np.searchsorted(lambda_common, lambda_um)]
            
            if debug:
                # Afficher uniquement les 3 premières valeurs de n et kappa
                print(f"Couche {i+1}:")
                print(f"  Matériau: {material}")
                print(f"  Épaisseur: {thickness} µm")
                print(f"  n (3 premières valeurs): {metals[material]['n_interp'][:3]}")
                print(f"  kappa (3 premières valeurs): {metals[material]['k_interp'][:3]}")
                print("-" * 40)
            
            # Ajouter la couche à la liste
            layers.append((n, kappa, thickness))
        elif material == "Air":
            # Cas de l'air (n=1, kappa=0)
            if debug:
                print(f"Couche {i+1}:")
                print(f"  Matériau: Air")
                print(f"  Épaisseur: {thickness} µm")
                print(f"  n: 1.0")
                print(f"  kappa: 0.0")
                print("-" * 40)
            
            layers.append((1.0, 0.0, thickness))
        elif material == "Glass":
            # Cas du verre (n et kappa interpolés)
            n = n_glass_interp[np.searchsorted(lambda_common, lambda_um)]
            kappa = kappa_glass_interp[np.searchsorted(lambda_common, lambda_um)]
            
            if debug:
                print(f"Couche {i+1}:")
                print(f"  Matériau: Glass")
                print(f"  Épaisseur: {thickness} µm")
                print(f"  n (3 premières valeurs): {n_glass_interp[:3]}")
                print(f"  kappa (3 premières valeurs): {kappa_glass_interp[:3]}")
                print("-" * 40)
            
            layers.append((n, kappa, thickness))
        else:
            raise ValueError(f"Matériau non reconnu : {material}")
    
    if debug:
        print("=" * 40)
        print("Fin de la configuration du système multicouche")
        print("\n")
    
    return layers

def plot_multilayer_structure(layer_configs):
    """
    Trace un diagramme représentant les différentes couches du système multicouche avec leurs épaisseurs à l'échelle,
    en les affichant verticalement. Affiche également le pourcentage que représente chaque couche par rapport à l'épaisseur totale.
    
    Parameters:
    layer_configs: list of dict, configurations des couches. Chaque dictionnaire doit contenir "material" et "thickness".
    """
    fig, ax = plt.subplots(figsize=(6, 10))  # Taille adaptée pour un affichage vertical
    
    # Position horizontale de départ pour les couches
    x_start = 0
    
    # Couleurs pour les différents matériaux (vous pouvez les personnaliser)
    material_colors = {
        "Air": "lightblue",
        "Gold": "gold",
        "Silver": "silver",
        "Copper": "peru",
        "Iron": "gray",
        "Nickel": "darkgray",
        "Glass": "lightgreen",
    }
    
    # Calculer l'épaisseur totale des couches (en nm)
    total_thickness = sum(config["thickness"] * 1000 for config in layer_configs)
    
    # Parcourir chaque couche et dessiner un rectangle
    for i, config in enumerate(layer_configs):
        material = config["material"]
        thickness = config["thickness"] * 1000  # Convertir en nm
        
        # Couleur du matériau
        color = material_colors.get(material, "white")  # Par défaut, blanc si le matériau n'est pas dans le dictionnaire
        
        # Dessiner un rectangle pour la couche (inversion des axes x et y)
        rect = patches.Rectangle((x_start, 0), thickness, 1, linewidth=1, edgecolor='black', facecolor=color, label=material)
        ax.add_patch(rect)
        
        # Calculer le pourcentage que représente cette couche
        percentage = (thickness / total_thickness) * 100
        
        # Ajouter une annotation pour le matériau, l'épaisseur et le pourcentage
        ax.text(
            x_start + thickness / 2, 0.5, 
            f"{material}\n{thickness:.1f} nm ({percentage:.1f}%)", 
            ha='center', va='center', fontsize=12, color='black', rotation=90
        )
        
        # Mettre à jour la position horizontale pour la prochaine couche
        x_start += thickness
    
    # Configurer les axes et le titre
    ax.set_ylim(0, 1)
    ax.set_xlim(0, x_start)
    ax.set_yticks([])  # Masquer les ticks de l'axe y
    ax.set_xlabel("Thickness (nm)", fontsize=12)
    ax.set_title("Multilayer structure optimized", fontsize=14)
    
    # Ajouter une légende pour les matériaux
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Éviter les doublons dans la légende
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    
    # Afficher le graphique
    plt.show()



def calculate_RTA_multilayer(layers, lambda_um, angle_incidence_deg=0, debug=False):
    """
    Calculate Reflectivity, Transmissivity, and Absorbance for a multi-layered system.
    
    Parameters:
    layers: list of tuples, each tuple contains (n, kappa, d) for each layer
            where n is the refractive index, kappa is the extinction coefficient,
            and d is the thickness in micrometers.
    lambda_um: array-like, wavelength in micrometers.
    angle_incidence_deg: angle of incidence in degrees.
    debug: boolean, if True, print detailed debug information.
    
    Returns:
    R: Reflectivity
    T: Transmissivity
    A: Absorbance
    psi_deg: ellipsometric parameter psi in degrees
    delta_deg: ellipsometric parameter delta in degrees
    """
    angle_incidence = np.radians(angle_incidence_deg)
    N_air = 1.0
    
    if debug:
        print("Debug mode activated.")
        print(f"Angle of incidence (radians): {angle_incidence}")
        print(f"Wavelengths (um): {lambda_um}")
        print("Layers (n, kappa, d):")
        for i, layer in enumerate(layers):
            print(f"Layer {i+1}: n = {layer[0]}, kappa = {layer[1]}, d = {layer[2]} um")
    
    # Initialize the scattering matrix S as the identity matrix for each wavelength
    S_p = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, len(lambda_um)))
    S_s = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, len(lambda_um)))
    
    if debug:
        print("\nInitial scattering matrices (S_p and S_s):")
        print("S_p:", S_p)
        print("S_s:", S_s)
    
    # Iterate over each layer
    for i, (n, kappa, d) in enumerate(layers):
        N_layer = n - 1j * kappa
        
        if debug:
            print(f"\nProcessing Layer {i+1}: n = {n}, kappa = {kappa}, d = {d} um")
            print(f"Complex refractive index of layer {i+1}: {N_layer}")
        
        # Calculate the angle of propagation in the current layer
        sin_theta_layer = N_air * np.sin(angle_incidence) / N_layer
        cos_theta_layer = np.sqrt(1 - sin_theta_layer**2)
        
        if debug:
            print(f"sin(theta_layer): {sin_theta_layer}")
            print(f"cos(theta_layer): {cos_theta_layer}")
        
        # Calculate the phase shift beta for all wavelengths
        beta = 2 * np.pi * d * N_layer * cos_theta_layer / lambda_um
        
        if debug:
            print(f"Phase shift beta: {beta}")
        
        # Create the layer matrix L as a 3D array
        L = np.zeros((2, 2, len(lambda_um)), dtype=complex)
        L[0, 0, :] = np.exp(1j * beta)  # Forward propagation
        L[1, 1, :] = np.exp(-1j * beta)  # Backward propagation
        
        if debug:
            print(f"Layer matrix L for layer {i+1}:")
            print(L)
        
        # Calculate the Fresnel coefficients for p and s polarizations
        if i == 0:
            N_prev = N_air
        else:
            N_prev = layers[i-1][0] - 1j * layers[i-1][1]
        
        r_p = (N_layer * np.cos(angle_incidence) - N_prev * cos_theta_layer) / (N_layer * np.cos(angle_incidence) + N_prev * cos_theta_layer)
        r_s = (N_prev * np.cos(angle_incidence) - N_layer * cos_theta_layer) / (N_prev * np.cos(angle_incidence) + N_layer * cos_theta_layer)
        
        t_p = (2 * N_prev * np.cos(angle_incidence)) / (N_layer * np.cos(angle_incidence) + N_prev * cos_theta_layer)
        t_s = (2 * N_prev * np.cos(angle_incidence)) / (N_prev * np.cos(angle_incidence) + N_layer * cos_theta_layer)
        
        if debug:
            print(f"Fresnel coefficients for layer {i+1}:")
            print(f"r_p: {r_p}")
            print(f"r_s: {r_s}")
            print(f"t_p: {t_p}")
            print(f"t_s: {t_s}")
        
        # Interface matrix I for each wavelength
        I_p = np.zeros((2, 2, len(lambda_um)), dtype=complex)
        I_p[0, 0, :] = 1 / t_p
        I_p[0, 1, :] = r_p / t_p
        I_p[1, 0, :] = r_p / t_p
        I_p[1, 1, :] = 1 / t_p
        
        I_s = np.zeros((2, 2, len(lambda_um)), dtype=complex)
        I_s[0, 0, :] = 1 / t_s
        I_s[0, 1, :] = r_s / t_s
        I_s[1, 0, :] = r_s / t_s
        I_s[1, 1, :] = 1 / t_s
        
        if debug:
            print(f"Interface matrix I_p for layer {i+1}:")
            print(I_p)
            print(f"Interface matrix I_s for layer {i+1}:")
            print(I_s)
        
        # Update the scattering matrix S for each wavelength
        for wl in range(len(lambda_um)):
            S_p[:, :, wl] = np.dot(S_p[:, :, wl], np.dot(I_p[:, :, wl], L[:, :, wl]))
            S_s[:, :, wl] = np.dot(S_s[:, :, wl], np.dot(I_s[:, :, wl], L[:, :, wl]))
        
        if debug:
            print(f"Updated scattering matrix S_p after layer {i+1}:")
            print(S_p)
            print(f"Updated scattering matrix S_s after layer {i+1}:")
            print(S_s)
    
    # Calculate the reflection and transmission coefficients for each wavelength
    R_p = np.abs(S_p[1, 0, :] / S_p[0, 0, :])**2
    R_s = np.abs(S_s[1, 0, :] / S_s[0, 0, :])**2
    
    T_p = np.abs(1 / S_p[0, 0, :])**2
    T_s = np.abs(1 / S_s[0, 0, :])**2
    
    if debug:
        print("\nReflection and transmission coefficients:")
        print(f"R_p: {R_p}")
        print(f"R_s: {R_s}")
        print(f"T_p: {T_p}")
        print(f"T_s: {T_s}")
    
    # Correction factor for transmissivity
    n_substrate = layers[-1][0]
    correction_factor = (n_substrate * np.cos(angle_incidence)) / (N_air * np.cos(angle_incidence))
    T_p_corrected = T_p * correction_factor
    T_s_corrected = T_s * correction_factor
    
    if debug:
        print(f"\nCorrection factor for transmissivity: {correction_factor}")
        print(f"Corrected T_p: {T_p_corrected}")
        print(f"Corrected T_s: {T_s_corrected}")
    
    # Average R and T for unpolarized light
    R = (R_p + R_s) / 2
    T = (T_p_corrected + T_s_corrected) / 2
    A = 1 - R - T
    
    if debug:
        print("\nFinal Reflectivity (R), Transmissivity (T), and Absorbance (A):")
        print(f"R: {R}")
        print(f"T: {T}")
        print(f"A: {A}")
    
    # Calculate ellipsometric parameters
    rho = (S_p[1, 0, :] / S_p[0, 0, :]) / (S_s[1, 0, :] / S_s[0, 0, :])
    psi = np.arctan(np.abs(rho))
    delta = np.angle(rho)
    
    psi_deg = np.degrees(psi)
    delta_deg = np.degrees(delta)
    
    if debug:
        print("\nEllipsometric parameters:")
        print(f"psi (degrees): {psi_deg}")
        print(f"delta (degrees): {delta_deg}")
    
    return R, T, A, psi_deg, delta_deg


def plot_RTA(lambda_um, R, T, A, layer_configs):
    """
    Plot Reflectivity (R), Transmissivity (T), and Absorbance (A) as a function of wavelength.
    
    Parameters:
    lambda_um: array-like, wavelength in micrometers.
    R: Reflectivity values.
    T: Transmissivity values.
    A: Absorbance values.
    layer_configs: list of dictionaries, each containing "material" and "thickness" for each layer.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Reflectivity (R)
    plt.plot(lambda_um, R, label="Reflectivity (R)", color="blue", linewidth=2)
    
    # Plot Transmissivity (T)
    plt.plot(lambda_um, T, label="Transmissivity (T)", color="green", linewidth=2)
    
    # Plot Absorbance (A)
    plt.plot(lambda_um, A, label="Absorbance (A)", color="orange", linewidth=2)
    
    # Generate the title dynamically based on the layer materials
    materials = [layer["material"] for layer in layer_configs]
    title = f"RTA of Multilayer System ({' / '.join(materials)})"
    
    # Add labels, title, and legend
    plt.xscale("log")
    plt.xlabel("Wavelength (µm)", fontsize=12)
    plt.ylabel("R, T, A", fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add a legend for the system configuration
    system_legend = "System Configuration:\n"
    for i, layer in enumerate(layer_configs):
        system_legend += f"Layer {i+1}: {layer['material']} ({layer['thickness']*1000:.2f} nm)\n"
    
    # Place the system configuration legend in the plot
    plt.text(0.02, 0.98, system_legend.strip(), transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a legend for the curves (R, T, A)
    plt.legend(fontsize=12, loc='lower right')
    
    # Add grid
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Set x-axis limits based on the wavelength range
    plt.xlim(min(lambda_um), max(lambda_um))
    
    # Show the plot
    plt.show()
    
def objective_function(thicknesses, metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp):
    """
    Fonction objectif pour l'optimisation des épaisseurs des couches.
    Elle maximise T dans la plage 0.4-0.7 µm et R dans les autres zones.
    
    Parameters:
    thicknesses: array-like, épaisseurs des couches à optimiser.
    metals: dict, dictionnaire contenant les données interpolées de n et k pour chaque métal.
    lambda_common: array-like, grille de longueurs d'onde commune.
    layer_configs: list of dict, configurations des couches.
    n_glass_interp: array-like, indice de réfraction interpolé du verre.
    kappa_glass_interp: array-like, coefficient d'extinction interpolé du verre.
    
    Returns:
    score: float, score à minimiser (1 - T dans 0.4-0.7 µm + R dans les autres zones).
    """
    # Mettre à jour les épaisseurs dans la configuration des couches
    thickness_index = 0  # Index pour parcourir thicknesses
    for config in layer_configs:
        if config["material"] != "Air":  # On ne modifie pas l'épaisseur de l'air
            config["thickness"] = thicknesses[thickness_index]
            thickness_index += 1
    
    # Initialiser les couches avec les nouvelles épaisseurs
    layers = initialize_layers(metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp)
    
    # Calculer R, T, A
    R, T, A, _, _ = calculate_RTA_multilayer(layers, lambda_common, angle_incidence_deg=0)
    
    # Séparer les longueurs d'onde dans la plage 0.4-0.7 µm et les autres
    mask_T = (lambda_common >= 0.4) & (lambda_common <= 0.7)
    mask_R = ~mask_T
    
    # Calculer la moyenne de T dans la plage 0.4-0.7 µm et R dans les autres zones
    T_mean = np.mean(T[mask_T])
    R_mean = np.mean(R[mask_R])
    
    # Score à minimiser (1 - T_mean + (1 - R_mean))
    score = (1 - T_mean) + (1 - R_mean)
    
    return score

def optimize_layer_thicknesses(metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp):
    """
    Optimise les épaisseurs des couches pour maximiser T dans 0.4-0.7 µm et R dans les autres zones.
    
    Parameters:
    metals: dict, dictionnaire contenant les données interpolées de n et k pour chaque métal.
    lambda_common: array-like, grille de longueurs d'onde commune.
    layer_configs: list of dict, configurations des couches.
    n_glass_interp: array-like, indice de réfraction interpolé du verre.
    kappa_glass_interp: array-like, coefficient d'extinction interpolé du verre.
    
    Returns:
    optimized_thicknesses: array, épaisseurs optimisées des couches.
    """
    # Initial guess pour les épaisseurs (en µm)
    initial_thicknesses = [config["thickness"] for config in layer_configs if config["material"] != "Air"]
    
    # Bornes pour les épaisseurs (en µm)
    bounds = [(0.0001, 4000) for _ in initial_thicknesses]  # Exemple de bornes, à ajuster
    
    # Optimisation
    result = minimize(
        objective_function,
        initial_thicknesses,
        args=(metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp),
        bounds=bounds,
        method='L-BFGS-B'  # Méthode d'optimisation adaptée aux problèmes avec bornes
    )
    
    if result.success:
        optimized_thicknesses = result.x
        print("Optimisation réussie. Épaisseurs optimisées :", optimized_thicknesses)
    else:
        raise ValueError("L'optimisation a échoué :", result.message)
    
    return optimized_thicknesses

def plot_ftir_data(file_trilayer):
    """
    Trace la réflectance normalisée en fonction de la longueur d'onde en µm
    pour les données FTIR d'un fichier trilayer.

    Paramètres :
    - file_trilayer : chemin du fichier CSV contenant les données du trilayer
    """
    # Charger les données
    data_trilayer = np.loadtxt(file_trilayer, delimiter=',')

    # Extraire les nombres d'onde et la réflectance
    wavenumber_trilayer = data_trilayer[:, 0]
    reflectance_trilayer = data_trilayer[:, 1]

    # Convertir le nombre d'onde en longueur d'onde (µm)
    wavelength_trilayer = 1e4 / wavenumber_trilayer

    # Normaliser la réflectance entre 0 et 1
    max_reflectance_trilayer = np.max(reflectance_trilayer)
    reflectance_trilayer_normalized = reflectance_trilayer / max_reflectance_trilayer

    # Tracer les données
    plt.figure(figsize=(10, 6))
    plt.semilogx(wavelength_trilayer, reflectance_trilayer_normalized, label='Trilayer Reflectivity')

    # Limiter l'axe des x entre 0.2 µm et 20 µm
    plt.xlim(0.2, 20)

    # Ajouter des labels et une légende
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Reflectivity')
    plt.title('FTIR Results: Reflectivity vs Wavelength')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Afficher le plot
    plt.show()


def weighted_RTA(lambda_um, R, T, A, solar_spectrum_file):
    """
    Calcule la réflectivité, transmissivité et absorbance pondérées et normalisées par le spectre solaire à chaque longueur d'onde.
    
    Parameters:
    lambda_um (array): Longueurs d'onde en micromètres.
    R (array): Réflectivité.
    T (array): Transmissivité.
    A (array): Absorbance.
    solar_spectrum_file (str): Chemin du fichier contenant le spectre solaire.
    
    Returns:
    dict: Valeurs pondérées et normalisées de R, T et A sous forme de tableaux.
    """
    # Charger le spectre solaire
    wavelengths, irradiance = np.loadtxt(solar_spectrum_file, skiprows=15, unpack=True)
    wavelengths_um = wavelengths / 1000  # Convertir nm -> µm
    
    # Interpolation du spectre solaire sur la grille de lambda_um
    interp_solar = np.interp(lambda_um, wavelengths_um, irradiance, left=0, right=0)
    
    # Calcul des valeurs pondérées
    R_weighted = R * interp_solar
    T_weighted = T * interp_solar
    A_weighted = A * interp_solar
    
    # Normalisation par l'intégrale du spectre solaire
    solar_integral = np.trapz(interp_solar, lambda_um)
    R_weighted /= solar_integral
    T_weighted /= solar_integral
    A_weighted /= solar_integral
    
    return {"R_weighted": R_weighted, "T_weighted": T_weighted, "A_weighted": A_weighted, "lambda_um": lambda_um}

def plot_weighted_RTA(weighted_results, layer_configs):
    """
    Plot the solar spectrum-weighted Reflectivity (R), Transmissivity (T), and Absorbance (A) curves 
    with a logarithmic scale for the x-axis.
    
    Parameters:
    weighted_results (dict): Dictionary containing lambda_um, R_weighted, T_weighted, and A_weighted.
    layer_configs (list of dict): Configuration of the layers, each containing "material" and "thickness".
    """
    lambda_um = weighted_results["lambda_um"]
    R_weighted = weighted_results["R_weighted"]
    T_weighted = weighted_results["T_weighted"]
    A_weighted = weighted_results["A_weighted"]
    
    # Generate the title dynamically based on the layer materials
    materials = [layer["material"] for layer in layer_configs]
    title = f"Solar Spectrum-Weighted RTA of multilayer system ({' / '.join(materials)})"
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_um, R_weighted, label="Reflectivity (R)", color="blue")
    plt.plot(lambda_um, T_weighted, label="Transmissivity (T)", color="green")
    plt.plot(lambda_um, A_weighted, label="Absorbance (A)", color="orange")
    
    plt.xscale("log")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Weighted Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Add a text box with the system configuration
    system_legend = "System Configuration:\n"
    for i, layer in enumerate(layer_configs):
        system_legend += f"Layer {i+1}: {layer['material']} ({layer['thickness']*1000:.2f} nm)\n"
    
    plt.figtext(0.6, 0.5, system_legend.strip(), fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
    
def compare_performance(metals, lambda_common, n_glass_interp, kappa_glass_interp, weighted=False):
    """
    Compare the performance of the optimal window (TiO2-Copper-TiO2-Glass) with bare glass (Glass) 
    and a simple metal film (Copper-Glass) for normal incidence, using optimized thicknesses.
    
    Parameters:
    metals: dict, dictionary containing interpolated n and k data for each metal.
    lambda_common: array-like, common wavelength grid.
    n_glass_interp: array-like, interpolated refractive index of glass.
    kappa_glass_interp: array-like, interpolated extinction coefficient of glass.
    weighted: bool, if True, calculate weighted R, T, A based on the solar spectrum.
    
    Returns:
    results: dict, dictionary containing R, T, A for each system.
    optimized_thicknesses: dict, dictionary containing optimized thicknesses for each system.
    """
    # Define the layer configurations for each system
    system_configs = {
        "Multilayer system ": [
            {"material": "ZnO", "thickness": 0.02},
            {"material": "Copper", "thickness": 0.03},
            {"material": "ZnO", "thickness": 0.02},
            {"material": "Glass", "thickness": 5},
        ],
        "Bare glass": [
            {"material": "Glass", "thickness": 0.1},
        ],
        "Simple layer system": [
            {"material": "Gold", "thickness": 0.03},
            {"material": "Glass", "thickness": 5},
        ],
    }
    
    # Initialize dictionaries to store the results and optimized thicknesses
    results = {}
    optimized_thicknesses = {}
    
    # Calculate R, T, A for each system
    for system_name, layer_configs in system_configs.items():
        # Optimize thicknesses for systems with multiple layers (excluding bare glass)
        if system_name != "Bare glass":
            optimized_thicknesses[system_name] = optimize_layer_thicknesses(
                metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp
            )
            
            # Update the layer configurations with optimized thicknesses
            thickness_index = 0
            for config in layer_configs:
                if config["material"] != "Air" and config["material"] != "Glass":  # Ne pas modifier l'épaisseur du verre
                    config["thickness"] = optimized_thicknesses[system_name][thickness_index]
                    thickness_index += 1
        
        # Initialize layers and calculate R, T, A
        layers = initialize_layers(metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp)
        R, T, A, _, _ = calculate_RTA_multilayer(layers, lambda_common, angle_incidence_deg=0)
        
        if weighted:
            # Calculate weighted R, T, A based on the solar spectrum
            weighted_results = weighted_RTA(lambda_common, R, T, A, "data/solar_irradiation.txt")
            R, T, A = weighted_results["R_weighted"], weighted_results["T_weighted"], weighted_results["A_weighted"]
        
        results[system_name] = {"R": R, "T": T, "A": A}
    
    return results, optimized_thicknesses

def plot_performance_comparison(lambda_common, results, optimized_thicknesses):
    """
    Plot the performance comparison (R, T, A) for the different systems, including optimized thicknesses.
    
    Parameters:
    lambda_common: array-like, common wavelength grid.
    results: dict, dictionary containing R, T, A for each system.
    optimized_thicknesses: dict, dictionary containing optimized thicknesses for each system.
    """
    plt.figure(figsize=(14, 8))
    
    # Plot Reflectivity (R)
    plt.subplot(3, 1, 1)
    for system_name, data in results.items():
        plt.plot(lambda_common, data["R"], label=f"{system_name}", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Reflectivity (R)")
    plt.title("Reflectivity Comparison")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Plot Transmissivity (T)
    plt.subplot(3, 1, 2)
    for system_name, data in results.items():
        plt.plot(lambda_common, data["T"], linewidth=2)
    plt.xscale("log")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Transmissivity (T)")
    plt.title("Transmissivity Comparison")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Plot Absorbance (A)
    plt.subplot(3, 1, 3)
    for system_name, data in results.items():
        plt.plot(lambda_common, data["A"], linewidth=2)
    plt.xscale("log")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Absorbance (A)")
    plt.title("Absorbance Comparison")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Add a text box with optimized thicknesses
    thickness_text = "Optimized thicknesses:\n"
    for system_name, thicknesses in optimized_thicknesses.items():
        thickness_text += f"{system_name}:\n"
        for i, thickness in enumerate(thicknesses):
            thickness_text += f"  Layer {i+1}: {thickness:.4f} µm\n"
    
    plt.figtext(0.85, 0.2, thickness_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def calculate_energy_savings(results, solar_spectrum_file):
    """
    Calculate energy savings by comparing the solar-weighted reflectivity of the multilayer system with bare glass,
    focusing only on wavelengths above 700 nm (infrared) to estimate heating-related energy savings.
    
    Parameters:
    results: dict, dictionary containing R, T, A for each system.
    solar_spectrum_file: str, path to the file containing the solar spectrum.
    
    Returns:
    energy_savings: float, energy savings in W/m² (for heating only).
    """
    print("\n=== Starting energy savings calculation (heating only, λ > 700 nm) ===")
    
    # Load the solar spectrum
    print("1. Loading the solar spectrum...")
    wavelengths, irradiance = np.loadtxt(solar_spectrum_file, skiprows=15, unpack=True)
    wavelengths_um = wavelengths / 1000  # Convert nm -> µm
    print(f"   - Loaded wavelength range: {wavelengths_um.min():.2f} µm to {wavelengths_um.max():.2f} µm")
    print(f"   - Maximum irradiance: {irradiance.max():.2f} W/m²/nm")
    
    # Filter wavelengths to keep only those above 700 nm (0.7 µm)
    print("\n2. Filtering wavelengths to keep only infrared (λ > 700 nm)...")
    mask = wavelengths_um > 0.7  # Mask for wavelengths > 700 nm
    wavelengths_um_filtered = wavelengths_um[mask]
    irradiance_filtered = irradiance[mask]
    print(f"   - Filtered wavelength range: {wavelengths_um_filtered.min():.2f} µm to {wavelengths_um_filtered.max():.2f} µm")
    print(f"   - Maximum irradiance in filtered range: {irradiance_filtered.max():.2f} W/m²/nm")
    
    # Calculate the total solar irradiance for the filtered spectrum (infrared only)
    print("\n3. Calculating total solar irradiance for infrared (λ > 700 nm)...")
    solar_integral_filtered = np.trapz(irradiance_filtered, wavelengths_um_filtered)
    print(f"   - Total solar irradiance (infrared only): {solar_integral_filtered:.2f} W/m²")
    
    # Retrieve the weighted reflectivity values for the multilayer system and bare glass
    print("\n4. Retrieving weighted reflectivity values for infrared (λ > 700 nm)...")
    # Interpolate reflectivity values to match the solar spectrum's wavelength grid
    R_multilayer_interp = np.interp(wavelengths_um_filtered, lambda_common, results["Multilayer system "]["R"])
    R_bare_glass_interp = np.interp(wavelengths_um_filtered, lambda_common, results["Bare glass"]["R"])
    
    R_multilayer = np.mean(R_multilayer_interp)
    R_bare_glass = np.mean(R_bare_glass_interp)
    print(f"   - Average reflectivity of the multilayer system (λ > 700 nm): {R_multilayer:.4f}")
    print(f"   - Average reflectivity of bare glass (λ > 700 nm): {R_bare_glass:.4f}")
    
    # Calculate energy savings for heating (infrared only)
    print("\n5. Calculating energy savings for heating (λ > 700 nm)...")
    energy_savings = solar_integral_filtered * (R_multilayer - R_bare_glass)
    print(f"   - Estimated energy savings for heating: {energy_savings:.2f} W/m²")
    
    # Interpretation of the results
    print("\n=== Interpretation of results ===")
    if energy_savings > 0:
        print(f"   - The multilayer system reflects more infrared energy than bare glass.")
        print(f"   - This reduces the thermal load on the building, leading to energy savings for heating.")
    elif energy_savings < 0:
        print(f"   - The multilayer system reflects less infrared energy than bare glass.")
        print(f"   - This may increase the thermal load on the building, requiring more energy for heating.")
    else:
        print(f"   - The multilayer system reflects the same amount of infrared energy as bare glass.")
        print(f"   - No impact on the thermal load or energy required for heating.")
    
    print("\n=== Energy savings calculation completed ===")
    
    return energy_savings

def sweep_multilayer_dzno_daero(
    metals,
    lambda_common,
    n_glass_interp,
    kappa_glass_interp,
    thickness_list_aero,       # list/array of aerogel thicknesses (µm)
    thickness_list_dielectric, # list/array of ZnO thicknesses (µm)
    N_bilayers=10,
    angle_incidence_deg=0,
    lambda_min_band=0.4,
    lambda_max_band=0.7
):
    """
    2D sweep of the thicknesses of the aerogel layer (n=1, k=0) and 
    the dielectric layer (ZnO), for a given number of bilayers.

    The average transmission T (spectral average) is computed
    over the spectral band lambda_min_band - lambda_max_band.
    
    Parameters
    ----------
    metals : dict
        Dictionary containing n and k interpolations for each material.
    lambda_common : array
        Wavelength grid (µm) on which R, T, A are evaluated.
    n_glass_interp : array
        Interpolated refractive index n of the glass over lambda_common.
    kappa_glass_interp : array
        Interpolated extinction coefficient k of the glass over lambda_common.
    thickness_list_aero : array
        List of aerogel thicknesses to test (in microns).
    thickness_list_dielectric : array
        List of ZnO thicknesses to test (in microns).
    N_bilayers : int
        Number of bilayers (default is 10).
    angle_incidence_deg : float
        Incidence angle in degrees (0 = normal incidence).
    lambda_min_band, lambda_max_band : float
        Lower and upper bounds (in µm) of the spectral band over which
        the average transmission (or another criterion) is computed.
    
    Returns
    -------
    T_map : 2D np.array
        Matrix [len(thickness_list_aero), len(thickness_list_dielectric)]
        containing the computed average transmission values.
    """
    
    # Prepare the matrix to store the performance metric
    T_map = np.zeros((len(thickness_list_aero), len(thickness_list_dielectric)))
    
    # To speed up index searches in lambda_common
    # We define a mask for the relevant spectral band (e.g., 0.4–0.7 µm)
    mask_band = (lambda_common >= lambda_min_band) & (lambda_common <= lambda_max_band)
    
    # Double loop: for each (aerogel thickness, ZnO thickness) pair
    for i_da, da in enumerate(thickness_list_aero):
        for j_dz, dz in enumerate(thickness_list_dielectric):
            
            # Dynamically construct the layer stack
            # Each bilayer = [ZnO(dz), Aerogel(da)], repeated N_bilayers times
            layer_configs_bilayer = []
            for _ in range(N_bilayers):
                layer_configs_bilayer.append({"material": "ZnO", "thickness": dz})
                # We approximate the aerogel by "Air"
                layer_configs_bilayer.append({"material": "Air", "thickness": da})
            
            # End with a glass layer (substrate)
            layer_configs_bilayer.append({"material": "Glass", "thickness": 1.0})
            
            # Initialize layers from configs
            layers = initialize_layers(
                metals,
                lambda_common,
                layer_configs_bilayer,
                n_glass_interp,
                kappa_glass_interp,
                debug=False
            )
            
            # Compute R, T, A
            R, T, A, _, _ = calculate_RTA_multilayer(
                layers, 
                lambda_common, 
                angle_incidence_deg=angle_incidence_deg,
                debug=False
            )
            
            # Criterion example = average T over the band [lambda_min_band, lambda_max_band]
            T_mean = np.mean(T[mask_band])
            
            # Store the performance metric
            T_map[i_da, j_dz] = T_mean
    
    return T_map


def plot_map_2D(thickness_list_aero, thickness_list_dielectric, T_map):
    """
    Displays the 2D map (heatmap) of performance T_map as a function of
    aerogel thickness (X-axis) and ZnO thickness (Y-axis).

    Parameters
    ----------
    thickness_list_aero : array
        Aerogel thicknesses (X-axis).
    thickness_list_dielectric : array
        ZnO thicknesses (Y-axis).
    T_map : 2D np.array
        Matrix containing the performance metric (e.g., T_mean).
    """
    
    plt.figure(figsize=(8, 6))
    # Transpose T_map depending on how we want the X/Y axes
    # Here, X-axis = aerogel, Y-axis = ZnO
    plt.imshow(
        T_map.T,  # transpose so j_dz is on the vertical axis
        origin='lower',
        aspect='auto',
        extent=[
            thickness_list_aero[0], thickness_list_aero[-1],
            thickness_list_dielectric[0], thickness_list_dielectric[-1]
        ],
        cmap='turbo'
    )
    plt.colorbar(label="Average transmission (in defined band)")
    plt.xlabel("Aerogel thickness (µm)")
    plt.ylabel("ZnO thickness (µm)")
    plt.title("2D Map of Average Transmission\n(as a function of daerogel and dZnO, 10 bilayers)")
    plt.show()

def sweep_multilayer_dzno_daero_reflectivity(
    metals,
    lambda_common,
    n_glass_interp,
    kappa_glass_interp,
    thickness_list_aero,       # list/array of aerogel thicknesses (µm)
    thickness_list_dielectric, # list/array of ZnO thicknesses (µm)
    N_bilayers=10,
    angle_incidence_deg=0,
    lambda_min_band=0.7,
    lambda_max_band=10
):
    """
    2D sweep of aerogel and ZnO thicknesses based on AVERAGE REFLECTIVITY
    in the given spectral band.
    
    Returns
    -------
    R_map : 2D np.array
        Matrix [len(thickness_list_aero), len(thickness_list_dielectric)]
        containing the computed average reflectivity values.
    """
    
    R_map = np.zeros((len(thickness_list_aero), len(thickness_list_dielectric)))
    mask_band = (lambda_common >= lambda_min_band) & (lambda_common <= lambda_max_band)
    
    for i_da, da in enumerate(thickness_list_aero):
        for j_dz, dz in enumerate(thickness_list_dielectric):
            layer_configs_bilayer = []
            for _ in range(N_bilayers):
                layer_configs_bilayer.append({"material": "ZnO", "thickness": dz})
                layer_configs_bilayer.append({"material": "Air", "thickness": da})
            layer_configs_bilayer.append({"material": "Glass", "thickness": 1.0})
            
            layers = initialize_layers(
                metals,
                lambda_common,
                layer_configs_bilayer,
                n_glass_interp,
                kappa_glass_interp,
                debug=False
            )
            
            R, T, A, _, _ = calculate_RTA_multilayer(
                layers,
                lambda_common,
                angle_incidence_deg=angle_incidence_deg,
                debug=False
            )
            
            R_mean = np.mean(R[mask_band])
            R_map[i_da, j_dz] = R_mean
    
    return R_map

def plot_map_2D_reflectivity(thickness_list_aero, thickness_list_dielectric, R_map):
    """
    Affiche la carte 2D de réflectivité moyenne en fonction des épaisseurs.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(
        R_map.T,
        origin='lower',
        aspect='auto',
        extent=[
            thickness_list_aero[0], thickness_list_aero[-1],
            thickness_list_dielectric[0], thickness_list_dielectric[-1]
        ],
        cmap='plasma'
    )
    plt.colorbar(label="Average reflectivity (in defined band)")
    plt.xlabel("Aerogel thickness (µm)")
    plt.ylabel("ZnO thickness (µm)")
    plt.title("2D Map of Average Reflectivity\n(as a function of daerogel and dZnO, 10 bilayers)")
    plt.show()


if __name__ == "__main__":
    debug = False
    file_nkglass = 'data/n_k_glass.txt'
    lambda_um_glass, n_glass, k_glass = read_nk_file(file_nkglass, debug)
    lambda_common = np.linspace(0.2, 20, num=1000)
    n_glass_interp, kappa_glass_interp = interpolate_data(lambda_um_glass, n_glass, k_glass, lambda_common)
    
     # Dictionnaire pour stocker les informations des métaux
    metals = {
        "Gold": {
            "file_nk": 'data/n_k_gold.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "Copper": {
            "file_nk": 'data/n_k_copper.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "Silver": {
            "file_nk": 'data/n_k_silver.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "Iron": {
            "file_nk": 'data/n_k_iron.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "Nickel": {
            "file_nk": 'data/n_k_nickel.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "TitaniumDioxide": {
            "file_nk": 'data/n_k_titaniumdioxide.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "ZnO":{
            "file_nk": 'data/n_k_Zincmono.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "ZnS":{
            "file_nk": 'data/ZnS_Querry.txt',
            "n_interp": None,
            "k_interp": None,
        },
        
    
    }
    
    # Charger et interpoler les données pour chaque métal
    for metal_name, metal_data in metals.items():
        lambda_um, n, k = read_nk_file(metal_data["file_nk"], debug)
        metal_data["n_interp"], metal_data["k_interp"] = interpolate_data(lambda_um, n, k, lambda_common)
    
    layer_configs = [
        #{"material": "Air", "thickness": 0},
        {"material": "ZnS", "thickness": 0.035},
        {"material": "Copper", "thickness": 0.025},
        {"material": "ZnS", "thickness": 0.035},
        {"material": "Glass", "thickness": 0.5},
    ]
    

    
    lambda_min = 0.2
    lambda_max = 20.0
    
    angles_incidence = [0]
    debug = True
    
    
    #plot_ftir_data('data/trilayers.CSV')
    layers = initialize_layers(metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp)
    R, T, A, _, _ = calculate_RTA_multilayer(layers, lambda_common, angle_incidence_deg=0)
    plot_RTA(lambda_common, R, T, A, layer_configs)
        # Optimiser les épaisseurs des couches
    optimized_thicknesses = optimize_layer_thicknesses(metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp)
    
    # Mettre à jour les épaisseurs dans la configuration des couches
    thickness_index = 0
    for config in layer_configs:
        if config["material"] != "Air":
            config["thickness"] = optimized_thicknesses[thickness_index]
            thickness_index += 1
    
    # Calculer et afficher les résultats avec les épaisseurs optimisées
    layers = initialize_layers(metals, lambda_common, layer_configs, n_glass_interp, kappa_glass_interp)
    R, T, A, _, _ = calculate_RTA_multilayer(layers, lambda_common, angle_incidence_deg=0)
    plot_RTA(lambda_common, R, T, A, layer_configs)
    plot_multilayer_structure(layer_configs)
    
    weighted_RTA_results = weighted_RTA(lambda_common, R, T, A, 'data/solar_irradiation.txt')
    plot_weighted_RTA(weighted_RTA_results,layer_configs=layer_configs)
    results,optimized_thick = compare_performance(metals, lambda_common, n_glass_interp, kappa_glass_interp,weighted=True)
    plot_performance_comparison(lambda_common, results, optimized_thick)
    calculate_energy_savings(results, 'data/solar_irradiation.txt')
    
    # Exemple de grilles d’épaisseur (en microns)
    thickness_list_aero = np.linspace(0.001, 0.05, 20)  # de 1 nm à 50 nm, par ex.
    thickness_list_dielectric = np.linspace(0.01, 0.1, 20)  # de 10 nm à 100 nm
    
    # Balayage 2D
    # T_map = sweep_multilayer_dzno_daero(
    #     metals,
    #     lambda_common,
    #     n_glass_interp,
    #     kappa_glass_interp,
    #     thickness_list_aero,
    #     thickness_list_dielectric,
    #     N_bilayers=10,              # Par exemple 10 bicouches
    #     angle_incidence_deg=0,      # Incidence normale
    #     lambda_min_band=0.4, 
    #     lambda_max_band=0.7
    # )
    
    # # Tracé de la heatmap
    # plot_map_2D(thickness_list_aero, thickness_list_dielectric, T_map)

    # R_map = sweep_multilayer_dzno_daero_reflectivity(
    # metals,
    # lambda_common,
    # n_glass_interp,
    # kappa_glass_interp,
    # thickness_list_aero,
    # thickness_list_dielectric,
    # N_bilayers=10,
    # angle_incidence_deg=0,
    # lambda_min_band=0.7,
    # lambda_max_band=10
    # )

    # plot_map_2D_reflectivity(thickness_list_aero, thickness_list_dielectric, R_map)


