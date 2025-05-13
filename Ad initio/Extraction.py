import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Task 2'))
sys.path.append(parent_dir)
import task2 # type: ignore

def extract_wl_n_k(file_path):
    """
    Extract wavelength (wl), refractive index (n), and extinction coefficient (k) from a .txt file.

    Parameters:
    file_path (str): Path to the .txt file.

    Returns:
    tuple: A tuple containing:
        - wl (list): List of wavelengths.
        - n (list): List of refractive indices.
        - k (list): List of extinction coefficients.
    """
    wl = []  # Wavelength
    n = []   # Refractive index
    k = []   # Extinction coefficient
    if file_path == "Data/Glass_Palik.txt":
        wl, n, k = task2.n_k(file_path)
        return wl, n, k
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Flags to identify sections
        read_wl_n = False
        read_wl_k = False

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Check for the start of the wl and n section
            if "wl\tn" in line:
                read_wl_n = True
                read_wl_k = False
                continue

            # Check for the start of the wl and k section
            if "wl\tk" in line:
                read_wl_k = True
                read_wl_n = False
                continue

            # Read wl and n values
            if read_wl_n:
                parts = line.split()
                if len(parts) == 2:
                    wl.append(float(parts[0]))
                    n.append(float(parts[1]))

            # Read wl and k values
            if read_wl_k:
                parts = line.split()
                if len(parts) == 2:
                    # Ensure the wl values match between n and k sections
                    if len(wl) > len(k):
                        k.append(float(parts[1]))


        

    return wl, n, k

def n_k_wl_trilayer(file_path1, file_path2, file_path3, file_path_glass, wl_min, wl_max):
    """
    Extract refractive indices and extinction coefficients for the trilayer system within a specified wavelength range.

    Parameters:
    file_path1 (str): Path to the .txt file containing the refractive index and extinction coefficient for ZnS.
    file_path2 (str): Path to the .txt file containing the refractive index and extinction coefficient for Cu.
    file_path3 (str): Path to the .txt file containing the refractive index and extinction coefficient for ZnS.
    file_path_glass (str): Path to the .txt file containing the refractive index and extinction coefficient for glass.
    wl_min (float): Minimum wavelength to consider.
    wl_max (float): Maximum wavelength to consider.

    Returns:
    tuple: A tuple containing:
        - n0, n1, n2, n3 (complex): Complex refractive indices of the four layers.
    """
    wl1, n1, k1 = extract_wl_n_k(file_path1)
    wl2, n2, k2 = extract_wl_n_k(file_path2)
    wl3, n3, k3 = extract_wl_n_k(file_path3)
    wl_glass, n_glass, k_glass = task2.n_k(file_path_glass)
    # Filter wavelengths within the specified range
    wl1, n1, k1 = zip(*[(wl, n, k) for wl, n, k in zip(wl1, n1, k1) if wl_min <= wl <= wl_max])
    wl2, n2, k2 = zip(*[(wl, n, k) for wl, n, k in zip(wl2, n2, k2) if wl_min <= wl <= wl_max])
    wl3, n3, k3 = zip(*[(wl, n, k) for wl, n, k in zip(wl3, n3, k3) if wl_min <= wl <= wl_max])
    wl_glass, n_glass, k_glass = zip(*[(wl, n, k) for wl, n, k in zip(wl_glass, n_glass, k_glass) if wl_min <= wl <= wl_max])
    wl_glass = np.array(wl_glass) 
    
    wl= np.linspace(wl_min, wl_max, 1000)
    n0 = np.ones(len(wl))
    n1 = np.interp(wl, wl1, n1) - 1j * np.interp(wl, wl1, k1)
    n2 = np.interp(wl, wl2, n2) - 1j * np.interp(wl, wl2, k2)
    n3 = np.interp(wl, wl3, n3) - 1j * np.interp(wl, wl3, k3)
    n_glass = np.interp(wl, wl_glass, n_glass) - 1j * np.interp(wl, wl_glass, k_glass)

    
    return wl, n0, n1, n2, n3, n_glass

def extract_solar_irrandiance(file_path, plot=False):
    """
    Extract solar irradiance data from a .txt file.

    Parameters:
    file_path (str): Path to the .txt file.

    Returns:
    tuple: A tuple containing:
        - wl (list): List of wavelengths in micrometers.
        - solar_irradiance (list): List of solar irradiance values.
    """
    wl = []  # Wavelength
    solar_irradiance = []  # Solar irradiance

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) == 2:
                wl.append(float(parts[0])/1000)
                solar_irradiance.append(float(parts[1])*1000)
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(wl, solar_irradiance)
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Solar Irradiance (W/m²/µm)")
        plt.title("Solar Irradiance Spectrum")
        plt.xscale("log")
        plt.savefig("Output/Solar Spectrum - ASTM1.5/ASTM1.5Global.png")

    return wl, solar_irradiance

def solar_interpolation(file_path, wl):
    """
    Interpolate solar irradiance data to match the wavelength range of the trilayer system.

    Parameters:
    file_path (str): Path to the .txt file containing solar irradiance data.
    wl (numpy.ndarray): Wavelength range of the trilayer system.

    Returns:
    numpy.ndarray: Interpolated solar irradiance values.
    """
    wl_solar, solar_irradiance = extract_solar_irrandiance(file_path)
    solar_irradiance_interp = np.interp(wl, wl_solar, solar_irradiance)
    return solar_irradiance_interp

def interpolate(wl_interp, wl, n, k):
    """
    Interpolate the refractive index and extinction coefficient to match the wavelength range.

    Parameters:
    wl (numpy.ndarray): Wavelength range of the trilayer system.
    n (numpy.ndarray): Refractive index values.
    k (numpy.ndarray): Extinction coefficient values.

    Returns:
    tuple: A tuple containing:
        - n_interp (numpy.ndarray): Interpolated refractive index values.
   
    Args:
        file_path (str): Chemin vers le fichier .txt contenant les données
    
    Returns:        - k_interp (numpy.ndarray): Interpolated extinction coefficient values.
    """
    n_interp = np.interp(wl_interp, wl, n)
    k_interp = np.interp(wl_interp, wl, k)
    return n_interp, k_interp

def extract_spectral_data(file_path, phi0 = 45):
    """
    Extrait les données spectrales d'un fichier texte ellipsométrique
 
        tuple: (wl, Psi, Delta) où:
            - wl: liste des longueurs d'onde en nm
            - Psi: liste des angles Psi en degrés
            - Delta: liste des angles Delta en degrés
    """
    wl = []
    Psi = []
    Delta = []
    
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                
                # Recherche du début des données
                data_start = False
                for line in lines:
                    if "# DATA:" in line:
                        data_start = True
                        continue
                    
                    if data_start and line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                wl.append(float(parts[0]))
                                Psi.append(float(parts[1]))
                                Delta.append(float(parts[2]))
                            except (ValueError, IndexError):
                                continue
                    
                
                if wl:  # Si on a trouvé des données, on sort
                    break
                
                    
        except UnicodeDecodeError:
            continue
        
       
            
            
    return wl[:-2], Psi[:-2], Delta[:-2]

from scipy.interpolate import interp1d

def extract_nk(filename, wl_interp):
    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d

    if filename == "Data/Glass_Palik.txt":
        wl, n, k = task2.n_k(filename)
        return interpolate(wl_interp,wl,n,k)

    # Lire toutes les lignes
    with open(filename, "r") as f:
        lines = f.readlines()

    # Trouver l’index du 2ᵉ header (où la colonne 'wl' recommence pour k)
    second_table_start = None
    for i, line in enumerate(lines):
        if line.strip() == "wl\tk":
            second_table_start = i
            break
                
    if second_table_start is None:
        raise ValueError("Impossible de trouver le deuxième tableau (wl-k) dans le fichier.")

    # Lire les deux tableaux avec pandas
    n_data = pd.read_csv(filename, sep="\t", skiprows=1, nrows=second_table_start - 2, names=["wl", "n"])
    k_data = pd.read_csv(filename, sep="\t", skiprows=second_table_start + 1, names=["wl", "k"])

    # Convertir en float (sécurité)
    n_data = n_data.astype(float)
    k_data = k_data.astype(float)

    # Créer les fonctions d'interpolation
    n_interp_func = interp1d(n_data["wl"], n_data["n"], kind='linear', bounds_error=False, fill_value="extrapolate")
    k_interp_func = interp1d(k_data["wl"], k_data["k"], kind='linear', bounds_error=False, fill_value="extrapolate")

    # Appliquer l'interpolation
    n_interp = n_interp_func(wl_interp)
    k_interp = k_interp_func(wl_interp)

    return n_interp, k_interp

