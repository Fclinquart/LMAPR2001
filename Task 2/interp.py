import numpy as np
import pandas as pd
import sys

def process_n_k_files(glass_file, silver_file, sun_file,  output_file):
    wl = np.linspace(0.2, 20, 1000)  # Generate 1000 points between 0.2 and 20 (in um)
    
    # Load the files, ignoring the first line if it contains units
    glass_data = pd.read_csv(glass_file, delim_whitespace=True, names=["lambda_um", "n_glass", "k_glass"], comment='#', skiprows=1, dtype={"lambda_um": float})
    silver_data = pd.read_csv(silver_file, delim_whitespace=True, names=["lambda_um", "n_ag", "k_ag"], comment='#', skiprows=1, dtype={"lambda_um": float})
    sun_data = pd.read_csv(sun_file, delim_whitespace=True, names=["lambda_nm", "I"], comment='#', skiprows=2, dtype={"lambda_nm": float, "I": float})
    
    # Convert data to float
    glass_data = glass_data.astype(float)
    silver_data = silver_data.astype(float)
    sun_data = sun_data.astype(float)

    # Filter out values below 0.2 um and above 20 um for glass and silver
    glass_data = glass_data[(glass_data["lambda_um"] > 0.2) & (glass_data["lambda_um"] < 20)]
    silver_data = silver_data[(silver_data["lambda_um"] > 0.2) & (silver_data["lambda_um"] < 20)]
    
    # Filter out values below 200 nm and above 20000 nm for sun data
    sun_data = sun_data[(sun_data["lambda_nm"] > 200) & (sun_data["lambda_nm"] < 20000)]

    # Replace NaN values with 0
    glass_data = glass_data.fillna(0)
    silver_data = silver_data.fillna(0)
    sun_data = sun_data.fillna(0)

    # Interpolate the data for the specified wavelengths
    n_glass_interp = np.interp(wl, glass_data["lambda_um"], glass_data["n_glass"])
    k_glass_interp = np.interp(wl, glass_data["lambda_um"], glass_data["k_glass"])
    n_ag_interp = np.interp(wl, silver_data["lambda_um"], silver_data["n_ag"])
    k_ag_interp = np.interp(wl, silver_data["lambda_um"], silver_data["k_ag"])
    I_sun_interp = np.interp(wl, sun_data["lambda_nm"] / 1000, sun_data["I"] * 1000)  # Convert nm to um and scale intensity

    # Create a new DataFrame with interpolated values
    interpolated_data = pd.DataFrame({
        "lambda_um": wl,
        "n_glass": n_glass_interp,
        "k_glass": k_glass_interp,
        "n_ag": n_ag_interp,
        "k_ag": k_ag_interp,
        "I": I_sun_interp
    })

    # Replace the original glass_data with the interpolated data
    glass_data = interpolated_data

    
    # Sauvegarder le fichier
    glass_data.to_csv(output_file, sep=' ', index=False, header=True)
    print(f"Fichier sauvegardé sous {output_file}")

def process_solar_spectra(sun_file, output_file):
    sun_data = pd.read_excel(sun_file, usecols=[0, 2], names=["lambda_nm", "I"], skiprows=1, dtype={"lambda_nm": float, "I": float})
    sun_data = sun_data.astype(float)
    # write to in the output file, the output file is a txt file with two columns
    sun_data.to_csv(output_file, sep=' ', index=False, header=True)
    print(f"Fichier sauvegardé sous {output_file}")





path_glass = "Data/n_k_glass.txt"
path_silver = "Data/n_k_ag2.txt"
path_csv_sun = "Data/AM0AM1_5.xls"
path_sun = "Data/ASTM1.5Global.txt"


process_solar_spectra(path_csv_sun, path_sun)
process_n_k_files(path_glass, path_silver, path_sun, "Data/n_k_combined.txt")
