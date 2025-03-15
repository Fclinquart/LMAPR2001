import numpy as np
import pandas as pd
import sys

def process_n_k_files(glass_file, silver_file, output_file):
    # Charger les fichiers en ignorant la première ligne si elle contient des unités
    glass_data = pd.read_csv(glass_file, delim_whitespace=True, names=["lambda_um", "n_glass", "k_glass"], comment='#', skiprows=1, dtype={"lambda_um": float})
    silver_data = pd.read_csv(silver_file, delim_whitespace=True, names=["lambda_um", "n_ag", "k_ag"], comment='#', skiprows=1, dtype={"lambda_um": float})
    glass_data = glass_data.astype(float)
    silver_data = silver_data.astype(float)

    # ignore les valeur en dessous de lambda = 0.2
    glass_data = glass_data[glass_data["lambda_um"] > 0.4]
    silver_data = silver_data[silver_data["lambda_um"] > 0.4]

    # filtre the 

    # ignore les valeurs en dessus de lambda = 20
    glass_data = glass_data[glass_data["lambda_um"] < 0.7]
    silver_data = silver_data[silver_data["lambda_um"] < 0.7]

    
    #remplace nan values with 0
    glass_data = glass_data.fillna(0)
    silver_data = silver_data.fillna(0)

    

    n_ag_interp = np.interp(glass_data["lambda_um"], silver_data["lambda_um"], silver_data["n_ag"])
    k_ag_interp = np.interp(glass_data["lambda_um"], silver_data["lambda_um"], silver_data["k_ag"])
    

    glass_data["n_ag"] = n_ag_interp
    glass_data["k_ag"] = k_ag_interp

    
    # Sauvegarder le fichier
    glass_data.to_csv(output_file, sep=' ', index=False, header=True)
    print(f"Fichier sauvegardé sous {output_file}")

        

path_glass = "Data/n_k_glass.txt"
path_silver = "Data/n_k_gold.txt"

process_n_k_files(path_glass, path_silver, "Data/n_k_combined_gold.txt")
