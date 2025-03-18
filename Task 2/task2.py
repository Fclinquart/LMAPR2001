import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import re
from scipy.optimize import minimize
from scipy.integrate import simpson

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

def calculate_RTA(n_glass, kappa_glass, n_metal, k_metal, d_metal, lambda_um, angle_incidence_deg=0):
    """Calculate Reflectivity, Transmissivity, and Absorbance."""
    angle_incidence = np.radians(angle_incidence_deg)
    N_glass = n_glass - 1j * kappa_glass
    N_metal = n_metal - 1j * k_metal
    N_air = 1.0
    
    sin_theta_metal = N_air * np.sin(angle_incidence) / N_metal
    cos_theta_metal = np.sqrt(1 - sin_theta_metal**2)
    
    beta = 2 * np.pi * d_metal * N_metal * cos_theta_metal / lambda_um
    
    r_p_air_metal = (N_metal * np.cos(angle_incidence) - N_air * cos_theta_metal) / (N_metal * np.cos(angle_incidence) + N_air * cos_theta_metal)
    r_s_air_metal = (N_air * np.cos(angle_incidence) - N_metal * cos_theta_metal) / (N_air * np.cos(angle_incidence) + N_metal * cos_theta_metal)
    
    t_p_air_metal = (2 * N_air * np.cos(angle_incidence)) / (N_metal * np.cos(angle_incidence) + N_air * cos_theta_metal)
    t_s_air_metal = (2 * N_air * np.cos(angle_incidence)) / (N_air * np.cos(angle_incidence) + N_metal * cos_theta_metal)
    
    r_p_metal_glass = (N_glass * cos_theta_metal - N_metal * np.cos(angle_incidence)) / (N_glass * cos_theta_metal + N_metal * np.cos(angle_incidence))
    r_s_metal_glass = (N_metal * np.cos(angle_incidence) - N_glass * cos_theta_metal) / (N_metal * np.cos(angle_incidence) + N_glass * cos_theta_metal)
    
    t_p_metal_glass = (2 * N_metal * np.cos(angle_incidence)) / (N_glass * cos_theta_metal + N_metal * np.cos(angle_incidence))
    t_s_metal_glass = (2 * N_metal * np.cos(angle_incidence)) / (N_metal * np.cos(angle_incidence) + N_glass * cos_theta_metal)
    
    denominator_p = 1 + r_p_air_metal * r_p_metal_glass * np.exp(-2j * beta)
    R_p = np.abs((r_p_air_metal + r_p_metal_glass * np.exp(-2j * beta)) / denominator_p)**2
    T_p = np.abs((t_p_air_metal * t_p_metal_glass * np.exp(-1j * beta)) / denominator_p)**2
    
    denominator_s = 1 + r_s_air_metal * r_s_metal_glass * np.exp(-2j * beta)
    R_s = np.abs((r_s_air_metal + r_s_metal_glass * np.exp(-2j * beta)) / denominator_s)**2
    T_s = np.abs((t_s_air_metal * t_s_metal_glass * np.exp(-1j * beta)) / denominator_s)**2
    
    R = (R_p + R_s) / 2
    T = (T_p + T_s) / 2
    A = 1 - R - T
    
    R_p_complex = (r_p_air_metal + r_p_metal_glass * np.exp(-2j * beta)) / denominator_p
    R_s_complex = (r_s_air_metal + r_s_metal_glass * np.exp(-2j * beta)) / denominator_s
    
    rho = R_p_complex / R_s_complex
    psi = np.arctan(np.abs(rho))
    delta = np.angle(rho)
    
    psi_deg = np.degrees(psi)
    delta_deg = np.degrees(delta)
    
    return R, T, A, psi_deg, delta_deg

def plot_RTA_vs_wavelength(d_metal_values, n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, lambda_common, angle_incidence=0, metal_name="metal"):
    """Plot Reflectivity (R), Transmissivity (T), and Absorbance (A) vs Wavelength for different film thicknesses."""
    for d_metal in d_metal_values:
        R, T, A, psi, delta = calculate_RTA(n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, d_metal, lambda_common, angle_incidence)
        
        plt.figure(figsize=(10, 6))
        plt.plot(lambda_common, R, label='Reflectivity (R)', linewidth=2)
        plt.plot(lambda_common, T, label='Transmissivity (T)', linewidth=2)
        plt.plot(lambda_common, A, label='Absorbance (A)', linewidth=2)
        
        plt.xscale("log")
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('R, T, A')
        plt.title(f'Reflectivity, transmissivity, and absorbance vs wavelength for {metal_name}\n(d_metal = {d_metal} nm, angle of incidence = {angle_incidence}°)')
        plt.legend()
        plt.xticks()
        plt.yticks()
        plt.grid()
        plt.tight_layout()
        plt.show()

def plot_R_vs_wavelength(d_metal_values, n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, lambda_common, angle_incidence=0, metal_name="metal"):
    """
    Plot Reflectivity (R) vs Wavelength for different film thicknesses on the same plot.
    
    Parameters:
        d_metal_values (list): List of film thicknesses (in nm).
        n_glass_interp (array): Interpolated refractive index of glass.
        kappa_glass_interp (array): Interpolated extinction coefficient of glass.
        n_metal_interp (array): Interpolated refractive index of metal.
        k_metal_interp (array): Interpolated extinction coefficient of metal.
        lambda_common (array): Common wavelength range (in µm).
        angle_incidence (float): Angle of incidence in degrees (default: 0°).
        metal_name (str): Name of the metal (default: "metal").
    """
    plt.figure(figsize=(10, 6))
    
    for d_metal in d_metal_values:
        # Calculate R, T, A, psi, delta
        R, T, A, psi, delta = calculate_RTA(n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, d_metal, lambda_common, angle_incidence)
        
        # Plot Reflectivity (R) vs Wavelength
        plt.plot(lambda_common, R, label=f'd_metal = {d_metal} nm', linewidth=2)
    
    # Set plot properties
    plt.xscale("log")
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Reflectivity (R)')
    plt.title(f'Reflectivity vs Wavelength for {metal_name}\n(Angle of incidence = {angle_incidence}°)')
    plt.legend()
    plt.xticks()
    plt.yticks()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Fonction pour calculer le pourcentage de lumière réfléchie, transmise et absorbée dans une plage spécifique
def calculate_percentages(R, T, A, lambda_common, range_start, range_end):
    # Masquer les longueurs d'onde dans la plage spécifiée
    mask = (lambda_common >= range_start) & (lambda_common <= range_end)
    lambda_range = lambda_common[mask]
    R_range = R[mask]
    T_range = T[mask]
    A_range = A[mask]
    
    # Intégrer R, T et A sur la plage de longueurs d'onde en utilisant la règle du trapèze
    R_integral = simpson(R_range)
    T_integral = simpson(T_range)
    A_integral = simpson(A_range)
    
    total_integral = R_integral + T_integral + A_integral
    
    # Calculer les pourcentages
    R_percentage = (R_integral / total_integral) * 100
    T_percentage = (T_integral / total_integral) * 100
    A_percentage = (A_integral / total_integral) * 100
    
    return R_percentage, T_percentage, A_percentage

# Calculer les pourcentages pour différentes épaisseurs de film
def calculate_RTA_percentages(d_metal_values, n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, lambda_common, angle_incidence=0):
    R_percentages_visible = []
    T_percentages_visible = []
    A_percentages_visible = []
    
    R_percentages_UV = []
    T_percentages_UV = []
    A_percentages_UV = []
    
    R_percentages_IR = []
    T_percentages_IR = []
    A_percentages_IR = []
    
    for d_metal in d_metal_values:
        # Calculer R, T, A pour l'épaisseur de film actuelle
        R, T, A, psi, delta = calculate_RTA(n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, d_metal, lambda_common, angle_incidence)
        
        # Calculer les pourcentages pour la lumière visible
        R_visible, T_visible, A_visible = calculate_percentages(R, T, A, lambda_common, visible_range[0], visible_range[1])
        R_percentages_visible.append(R_visible)
        T_percentages_visible.append(T_visible)
        A_percentages_visible.append(A_visible)
        
        # Calculer les pourcentages pour la lumière UV
        R_UV, T_UV, A_UV = calculate_percentages(R, T, A, lambda_common, non_visible_UV_range[0], non_visible_UV_range[1])
        R_percentages_UV.append(R_UV)
        T_percentages_UV.append(T_UV)
        A_percentages_UV.append(A_UV)
        
        # Calculer les pourcentages pour la lumière IR
        R_IR, T_IR, A_IR = calculate_percentages(R, T, A, lambda_common, non_visible_IR_range[0], non_visible_IR_range[1])
        R_percentages_IR.append(R_IR)
        T_percentages_IR.append(T_IR)
        A_percentages_IR.append(A_IR)
    
    return (R_percentages_visible, T_percentages_visible, A_percentages_visible,
            R_percentages_UV, T_percentages_UV, A_percentages_UV,
            R_percentages_IR, T_percentages_IR, A_percentages_IR)

def print_lenth(file_elli45):
    psi_exp, delta_exp = read_and_plot_ellipsometry(file_elli45, debug)
    xp = np.arange(400, 690, 10)
    fp = psi_exp

    print(f"Longueur de xp : {len(xp)}")
    print(f"Longueur de fp : {len(fp)}")

# Trouver l'épaisseur qui maximise la réflectivité pour la lumière visible, UV et IR
def theoretical_optimal_thickness(R_percentages_visible, R_percentages_UV, R_percentages_IR, d_metal_values, angle_incidence=0, metal_name="metal"):
    # Use np.argmax to find the index of the first maximum reflectivity value
    optimal_thickness_visible = d_metal_values[np.argmax(R_percentages_visible)]
    optimal_thickness_UV = d_metal_values[np.argmax(R_percentages_UV)]
    optimal_thickness_IR = d_metal_values[np.argmax(R_percentages_IR)]
    
    # Display the optimal thicknesses with the angle of incidence and metal name
    print(f"The optimal thickness for visible light reflectivity for {metal_name} (Angle of incidence = {angle_incidence}°) is: {optimal_thickness_visible:.2f} nm")
    print(f"The optimal thickness for UV light reflectivity for {metal_name} (Angle of incidence = {angle_incidence}°) is: {optimal_thickness_UV:.2f} nm")
    print(f"The optimal thickness for IR light reflectivity for {metal_name} (Angle of incidence = {angle_incidence}°) is: {optimal_thickness_IR:.2f} nm\n")
    print()
    
    return optimal_thickness_visible, optimal_thickness_UV, optimal_thickness_IR

# Plot the results
def plot_percentages_vs_thickness(d_metal_values, R_percentages_visible, T_percentages_visible, A_percentages_visible,
                                  R_percentages_UV, T_percentages_UV, A_percentages_UV,
                                  R_percentages_IR, T_percentages_IR, A_percentages_IR, angle_incidence=0, metal_name="metal"):
    """
    Plot Reflectivity, Transmissivity, and Absorbance percentages vs Film Thickness.

    Parameters:
        d_metal_values (list): List of film thicknesses (in nanometers) to plot.
        R_percentages_visible (list): Reflectivity percentages for visible light.
        T_percentages_visible (list): Transmissivity percentages for visible light.
        A_percentages_visible (list): Absorbance percentages for visible light.
        R_percentages_UV (list): Reflectivity percentages for UV light.
        T_percentages_UV (list): Transmissivity percentages for UV light.
        A_percentages_UV (list): Absorbance percentages for UV light.
        R_percentages_IR (list): Reflectivity percentages for IR light.
        T_percentages_IR (list): Transmissivity percentages for IR light.
        A_percentages_IR (list): Absorbance percentages for IR light.
        angle_incidence (float): Angle of incidence in degrees (default is 0°).
        metal_name (str): Name of the metal (default: "metal").
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the percentages for visible light
    plt.plot(d_metal_values, R_percentages_visible, label='Reflectivity (Visible)', linewidth=2)
    plt.plot(d_metal_values, T_percentages_visible, label='Transmissivity (Visible)', linewidth=2)
    plt.plot(d_metal_values, A_percentages_visible, label='Absorbance (Visible)', linewidth=2)
    
    # Plot the percentages for UV light
    plt.plot(d_metal_values, R_percentages_UV, label='Reflectivity (UV)', linestyle='--', linewidth=2)
    plt.plot(d_metal_values, T_percentages_UV, label='Transmissivity (UV)', linestyle='--', linewidth=2)
    plt.plot(d_metal_values, A_percentages_UV, label='Absorbance (UV)', linestyle='--', linewidth=2)
    
    # Plot the percentages for IR light
    plt.plot(d_metal_values, R_percentages_IR, label='Reflectivity (IR)', linestyle=':', linewidth=2)
    plt.plot(d_metal_values, T_percentages_IR, label='Transmissivity (IR)', linestyle=':', linewidth=2)
    plt.plot(d_metal_values, A_percentages_IR, label='Absorbance (IR)', linestyle=':', linewidth=2)
    
    # Logarithmic scale for thickness
    plt.xscale('log')
    
    # Customize labels and title
    plt.xlabel('Film Thickness (nm)')
    plt.ylabel('Percentage (%)')
    plt.title(f'Reflectivity, transmissivity, and absorbance vs film thickness for {metal_name}\n(Angle of incidence = {angle_incidence}°)')
    
    # Customize legend
    plt.legend(loc='upper right')
    
    # Customize axis ticks
    plt.xticks()
    plt.yticks()
    
    # Add grid and adjust layout
    plt.grid(True)
    plt.tight_layout()
    
    # Display the plot
    plt.show()

def read_and_plot_ellipsometry(file_elli45, debug=False):
    """Reads an ellipsometry file and returns experimental Psi and Delta curves."""
    def debug_message(message, debug=False):
        if debug:
            print(f"[DEBUG] {message}")
    
    debug_message(f"Reading file: {file_elli45}", debug)
    
    angle = re.search(r'\d+', file_elli45)
    angle = angle.group() if angle else "Unknown"
    
    try:
        with open(file_elli45, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        debug_message("File read with UTF-8 encoding.", debug)
    except UnicodeDecodeError:
        with open(file_elli45, 'r', encoding='latin-1') as file:
            lines = file.readlines()
        debug_message("File read with Latin-1 encoding.", debug)
    
    data = []
    for line in lines:
        if line.strip() and not line.startswith('#') and not line.startswith('nm'):
            parts = line.split()
            if len(parts) == 3:
                try:
                    nm = float(parts[0])
                    psi = float(parts[1])
                    delta = float(parts[2])
                    data.append((nm, psi, delta))
                except ValueError:
                    debug_message(f"Ignored line (conversion failed): {line.strip()}", debug)
                    continue
    
    if not data:
        debug_message("No valid data found in the file.", debug)
        return None, None
    
    data = data[:-2]
    debug_message("Last point excluded from data.", debug)
    
    df = pd.DataFrame(data, columns=['nm', 'Psi', 'Delta'])
    debug_message(f"Number of data points extracted: {len(df)}", debug)
    debug_message(f"Data extracted:\n{df}", debug)
    
    if debug:
        # Plot the graphs
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax1.plot(df['nm'], df['Psi'], label='Psi', color='blue')
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Psi (deg)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.tick_params(axis='x')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot(df['nm'], df['Delta'], label='Delta', color='red')
        ax2.set_ylabel("Delta (deg)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f"Ellipsometry - Psi and Delta vs. Wavelength (Incidence: {angle}°)")
        fig = plt.gcf()
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.tight_layout()
        plt.show()
        debug_message("Plot displayed successfully.", debug)
    
    return df['Psi'].values, df['Delta'].values

def compare_psi_delta(file_elli45, n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, lambda_common, d_metal, angle_incidence=45, metal_name="metal", debug=False):
    """Compare experimental and theoretical Psi and Delta."""
    psi_exp, delta_exp = read_and_plot_ellipsometry(file_elli45, debug)
    
    lambda_compare = np.arange(400, 690, 10) / 1000.0
    
    R, T, A, psi_theo, delta_theo = calculate_RTA(n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, d_metal, lambda_common, angle_incidence)
    interp_psi = interp1d(lambda_common, psi_theo, kind='linear', fill_value="extrapolate")
    interp_delta = interp1d(lambda_common, delta_theo, kind='linear', fill_value="extrapolate")
    psi_theo_interp = interp_psi(lambda_compare)
    delta_theo_interp = interp_delta(lambda_compare)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(lambda_compare * 1000, psi_exp, 'o-', label='Experimental $\psi$', color='red')
    plt.plot(lambda_compare * 1000, psi_theo_interp, '-', label='Theoretical $\psi$', color='green')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('$\psi$')
    plt.legend()
    plt.title(f'$\psi$ for {metal_name}; d = {d_metal:.2f}nm; Angle : {angle_incidence}°')
    
    plt.subplot(1, 2, 2)
    plt.plot(lambda_compare * 1000, delta_exp, 'o-', label='Experimental $\Delta$', color='blue')
    plt.plot(lambda_compare * 1000, delta_theo_interp, '-', label='Theoretical $\Delta$', color='green')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('$\Delta$')
    plt.legend()
    plt.title(f'$\Delta$ for {metal_name}; d = {d_metal:.2f}nm; Angle : {angle_incidence}°')
    
    plt.tight_layout()
    plt.show()

def find_optimal_thickness(file_elli45, n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, lambda_common, angle_incidence=45, debug=False):
    """Find the optimal metal thickness that minimizes the error between experimental and theoretical $\psi$ and $\Delta$."""
    psi_exp, delta_exp = read_and_plot_ellipsometry(file_elli45, debug)
    
    lambda_compare = np.arange(400, 690, 10) / 1000.0
    
    def interp_exp_psi_delta(lambda_compare):
        psi_interp = np.interp(lambda_compare * 1000, np.arange(400, 690, 10), psi_exp)
        delta_interp = np.interp(lambda_compare * 1000, np.arange(400, 690, 10), delta_exp)
        return psi_interp, delta_interp
    
    psi_exp_interp, delta_exp_interp = interp_exp_psi_delta(lambda_compare)
    
    def objective(d_metal):
        R, T, A, psi_theo, delta_theo = calculate_RTA(n_glass_interp, kappa_glass_interp, n_metal_interp, k_metal_interp, d_metal, lambda_common, angle_incidence)
        interp_psi = interp1d(lambda_common, psi_theo, kind='linear', fill_value="extrapolate")
        interp_delta = interp1d(lambda_common, delta_theo, kind='linear', fill_value="extrapolate")
        psi_theo_interp = interp_psi(lambda_compare)
        delta_theo_interp = interp_delta(lambda_compare)
        
        error_psi = np.mean((psi_exp_interp - psi_theo_interp) ** 2)
        error_delta = np.mean((delta_exp_interp - delta_theo_interp) ** 2)
        return error_psi + error_delta
    
    result = minimize(objective, x0=0.1, bounds=[(0.001, 100)], method='L-BFGS-B')
    
    if debug:
        print(f"Optimal thickness found: {result.x[0]} nm")
    
    return result.x[0]

def plot_ftir_data(file_verre, file_or):
    """
    Trace la réflectance normalisée en fonction de la longueur d'onde en µm
    pour les données FTIR du verre et de l'or.

    Paramètres :
    - file_verre : chemin du fichier CSV contenant les données du verre
    - file_or : chemin du fichier CSV contenant les données de l'or
    """
    # Charger les données
    data_verre = np.loadtxt(file_verre, delimiter=',')
    data_or = np.loadtxt(file_or, delimiter=',')

    # Extraire les nombres d'onde et la réflectance
    wavenumber_verre = data_verre[:, 0]
    reflectance_verre = data_verre[:, 1]

    wavenumber_or = data_or[:, 0]
    reflectance_or = data_or[:, 1]

    # Convertir le nombre d'onde en longueur d'onde (µm)
    wavelength_verre = 1e4 / wavenumber_verre
    wavelength_or = 1e4 / wavenumber_or

    # Normaliser la réflectance entre 0 et 1
    max_reflectance_verre = np.max(reflectance_verre)
    max_reflectance_or = np.max(reflectance_or)
    reflectance_verre_normalized = reflectance_verre / max_reflectance_verre
    reflectance_or_normalized = reflectance_or/ max_reflectance_or

    # Tracer les données
    plt.figure(figsize=(10, 6))
    plt.semilogx(wavelength_verre, reflectance_verre_normalized, label='Ref glass')
    plt.semilogx(wavelength_or, reflectance_or_normalized, label='Ref gold')

    # Limiter l'axe des x entre 0.2 µm et 20 µm
    plt.xlim(0.2, 20)

    # Ajouter des labels et une légende
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Reflectivity')
    plt.title('FTIR results : Reflectivity vs wavelength')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Afficher le plot
    plt.show()

# Main execution
if __name__ == "__main__":
    debug = False
    file_nkglass = 'n_k_glass.txt'
    file_elli45 = 'Ellipsometrie_45.txt'
    file_elli65 = 'Ellipsometrie_65.txt'
    
    
    
    lambda_um_glass, n_glass, k_glass = read_nk_file(file_nkglass, debug)
    lambda_common = np.linspace(0.2, 20, num=1000)
    n_glass_interp, kappa_glass_interp = interpolate_data(lambda_um_glass, n_glass, k_glass, lambda_common)
    visible_range = (0.4, 0.7)  # Plage de longueurs d'onde pour la lumière visible (en µm)
    non_visible_UV_range = (0.01, 0.4)  # Plage de longueurs d'onde pour l'UV (en µm)
    non_visible_IR_range = (0.7, 1000)  # Plage de longueurs d'onde pour l'IR (en µm)
    # Définir différentes épaisseurs de film (en nanomètres)
    d_metal_values_log = np.logspace(-3, 4, 50)  # Épaisseur de 0.001 nm à 10000 nm (échelle logarithmique)
    # Dictionnaire pour stocker les informations des métaux
    metals = {
        "Gold": {
            "file_nk": 'n_k_gold.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "Silver": {
            "file_nk": 'n_k_silver.txt',
            "n_interp": None,
            "k_interp": None,
        },
        "Nickel": {
            "file_nk": 'n_k_nickel.txt',
            "n_interp": None,
            "k_interp": None,
        },
    
    }
    
    # Charger et interpoler les données pour chaque métal
    for metal_name, metal_data in metals.items():
        lambda_um, n, k = read_nk_file(metal_data["file_nk"], debug)
        metal_data["n_interp"], metal_data["k_interp"] = interpolate_data(lambda_um, n, k, lambda_common)
    
    lambda_min = 0.2
    lambda_max = 20.0
    

    # Épaisseurs de film à tester
    d_metal_values = [0.00000001,0.001, 0.01, 0.1, 1, 10, 100]

    # Angles d'incidence à utiliser
    angles_incidence = [0,27.1]  # Exemple d'angles
    ellipsometry_angles = [45, 65] # Angles spécifiques pour l'ellipsométrie
    
    #options de plot
    RTA = False
    percentage_thikness = True
    RvsWavelenth = False
    ellipsometry = True
    ftir = False
    
    plot = True
    
    if plot:
        # Liste des métaux à traiter
        metals_to_plot = ["Gold", "Silver", "Nickel"]
        #
        print_lenth(file_elli45)
        #
        for metal_name in metals_to_plot:
            for angle in angles_incidence:
                if RTA:
                    plot_RTA_vs_wavelength(d_metal_values, n_glass_interp, kappa_glass_interp, metals[metal_name]["n_interp"], metals[metal_name]["k_interp"], lambda_common, angle_incidence=angle, metal_name=metal_name)
                
                if percentage_thikness:
                    R_percentages_visible, T_percentages_visible, A_percentages_visible, R_percentages_UV, T_percentages_UV, A_percentages_UV, R_percentages_IR, T_percentages_IR, A_percentages_IR = calculate_RTA_percentages(d_metal_values_log, n_glass_interp, kappa_glass_interp, metals[metal_name]["n_interp"], metals[metal_name]["k_interp"], lambda_common, angle_incidence=angle)
                    optimal_thickness_visible, optimal_thickness_UV, optimal_thickness_IR = theoretical_optimal_thickness(R_percentages_visible, R_percentages_UV, R_percentages_IR, d_metal_values_log, angle, metal_name=metal_name)
                    plot_percentages_vs_thickness(d_metal_values_log, R_percentages_visible, T_percentages_visible, A_percentages_visible, R_percentages_UV, T_percentages_UV, A_percentages_UV, R_percentages_IR, T_percentages_IR, A_percentages_IR, angle, metal_name=metal_name)
                
                if RvsWavelenth:
                    plot_R_vs_wavelength(d_metal_values, n_glass_interp, kappa_glass_interp, metals[metal_name]["n_interp"], metals[metal_name]["k_interp"], lambda_common, angle_incidence=angle, metal_name=metal_name)
            
            if ellipsometry:
                for angle in ellipsometry_angles:
                    file_elli = file_elli45 if angle == 45 else file_elli65
                    optimal_thickness = find_optimal_thickness(file_elli, n_glass_interp, kappa_glass_interp, metals[metal_name]["n_interp"], metals[metal_name]["k_interp"], lambda_common, angle_incidence=angle)
                    print(f"Optimal metal thickness for {angle}° incidence ({metal_name}): {optimal_thickness} nm")
                    compare_psi_delta(file_elli, n_glass_interp, kappa_glass_interp, metals[metal_name]["n_interp"], metals[metal_name]["k_interp"], lambda_common, optimal_thickness, angle_incidence=angle, metal_name=metal_name)
        
        if ftir:
            # Plot the FTIR reflectivity for our sample
            plot_ftir_data('FTIRref_verre.CSV', 'FTIRref_or.CSV')
            
        

        