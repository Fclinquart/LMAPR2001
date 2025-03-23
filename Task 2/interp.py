import numpy as np
import pandas as pd
import sys

def process_n_k_files(glass_file, silver_file, sun_file,  output_file):
    # Charger les fichiers en ignorant la première ligne si elle contient des unités
    glass_data = pd.read_csv(glass_file, delim_whitespace=True, names=["lambda_um", "n_glass", "k_glass"], comment='#', skiprows=1, dtype={"lambda_um": float})
    silver_data = pd.read_csv(silver_file, delim_whitespace=True, names=["lambda_um", "n_ag", "k_ag"], comment='#', skiprows=1, dtype={"lambda_um": float})
    sun_data = pd.read_csv(sun_file, delim_whitespace=True, names=["lambda_nm", "I"], comment='#', skiprows=2, dtype={"lambda_nm": float, "I": float})
    glass_data = glass_data.astype(float)
    silver_data = silver_data.astype(float)
    sun_data = sun_data.astype(float)

    # ignore les valeur en dessous de lambda = 0.2
    glass_data = glass_data[glass_data["lambda_um"] > 0.2]
    silver_data = silver_data[silver_data["lambda_um"] > 0.2]
    sun_data = sun_data[sun_data["lambda_nm"] > 200]

    # filtre the 

    # ignore les valeurs en dessus de lambda = 20
    glass_data = glass_data[glass_data["lambda_um"] < 20]
    silver_data = silver_data[silver_data["lambda_um"] < 20]
    sun_data = sun_data[sun_data["lambda_nm"] < 20000]

    
    #remplace nan values with 0
    glass_data = glass_data.fillna(0)
    silver_data = silver_data.fillna(0)
    sun_data = sun_data.fillna(0)

    

    n_ag_interp = np.interp(glass_data["lambda_um"], silver_data["lambda_um"], silver_data["n_ag"])
    k_ag_interp = np.interp(glass_data["lambda_um"], silver_data["lambda_um"], silver_data["k_ag"])

    I_sun = np.interp(glass_data["lambda_um"], sun_data["lambda_nm"] / 1000, sun_data["I"]) # Convert nm to um
    

    glass_data["n_ag"] = n_ag_interp
    glass_data["k_ag"] = k_ag_interp
    glass_data["I"] = I_sun

    
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
path_silver = "Data/n_k_Cu.txt"
path_csv_sun = "Data/AM0AM1_5.xls"
path_sun = "Data/ASTM1.5Global.txt"


process_solar_spectra(path_csv_sun, path_sun)
process_n_k_files(path_glass, path_silver, path_sun, "Data/n_k_combined.txt")
