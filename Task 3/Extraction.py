import numpy as np
import pandas as pd
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
        - n0, n1, n2, n3 (complex): Complex refractive indices of the four layers (air, ZnS, Cu, ZnS, glass).
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

    n0 = 1.0  # Refractive index of air
    n1 = np.interp(wl_glass, wl1, n1) + 1j * np.interp(wl_glass, wl1, k1)
    n2 = np.interp(wl_glass, wl2, n2) + 1j * np.interp(wl_glass, wl2, k2)
    n3 = np.interp(wl_glass, wl3, n3) + 1j * np.interp(wl_glass, wl3, k3)
    n_glass = n_glass 

    return wl_glass, n0, n1, n2, n3, n_glass
