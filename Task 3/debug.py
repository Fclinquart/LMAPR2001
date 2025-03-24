
    
import numpy as np 
import Extraction
import matplotlib.pyplot as plt
debug = False

lambda_um, n_air, n_zns, n_cu, n_zns, n_glass = Extraction.n_k_wl_trilayer("Data/ZnS_Querry.txt", "Data/Cu_Querry.txt", "Data/ZnS_Querry.txt", "Data/Glass_Palik.txt", 0.2, 20)

print (lambda_um)

config =[ ("ZnS", 20e-3), ("Cu", 30e-3), ("ZnS", 20e-3), ("glass", 0.5)]

layers = [ (np.real(n_zns), -np.imag(n_zns),0.02), (np.real(n_cu), -np.imag(n_cu),0.03), (np.real(n_zns), -np.imag(n_zns),0.02), (np.real(n_glass), -np.imag(n_glass),0.5)]


phi0 = 0.0
n0 = 1.0

phi0 = np.radians(phi0)


# Initialize the scattering matrix S as the identity matrix for each wavelength
S_p = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, len(lambda_um)))
S_s = np.tile(np.eye(2, dtype=complex)[:, :, np.newaxis], (1, 1, len(lambda_um)))


# Iterate over each layer
for i, (n, kappa, d) in enumerate(layers):
    N_layer = n - 1j * kappa
    
  
    # Calculate the angle of propagation in the current layer
    sin_theta_layer = n0 * np.sin(phi0) / N_layer
    cos_theta_layer = np.sqrt(1 - sin_theta_layer**2)
    
   
    # Calculate the phase shift beta for all wavelengths
    beta = 2 * np.pi * d * N_layer * cos_theta_layer / lambda_um
    
   
    # Create the layer matrix L as a 3D array
    L = np.zeros((2, 2, 126), dtype=complex)
    L[0, 0, :] = np.exp(1j * beta)  # Forward propagation
    L[1, 1, :] = np.exp(-1j * beta)  # Backward propagation
    
    
    
    # Calculate the Fresnel coefficients for p and s polarizations
    if i == 0:
        N_prev = n0
    else:
        N_prev = layers[i-1][0] - 1j * layers[i-1][1]
    
    r_p = (N_layer * np.cos(phi0) - N_prev * cos_theta_layer) / (N_layer * np.cos(phi0) + N_prev * cos_theta_layer)
    r_s = (N_prev * np.cos(phi0) - N_layer * cos_theta_layer) / (N_prev * np.cos(phi0) + N_layer * cos_theta_layer)
    
    t_p = (2 * N_prev * np.cos(phi0)) / (N_layer * np.cos(phi0) + N_prev * cos_theta_layer)
    t_s = (2 * N_prev * np.cos(phi0)) / (N_prev * np.cos(phi0) + N_layer * cos_theta_layer)
    
    
    
    # Interface matrix I for each wavelength
    I_p = np.zeros((2, 2, 126), dtype=complex)
    I_p[0, 0, :] = 1 / t_p
    I_p[0, 1, :] = r_p / t_p
    I_p[1, 0, :] = r_p / t_p
    I_p[1, 1, :] = 1 / t_p
    
    I_s = np.zeros((2, 2, 126), dtype=complex)
    I_s[0, 0, :] = 1 / t_s
    I_s[0, 1, :] = r_s / t_s
    I_s[1, 0, :] = r_s / t_s
    I_s[1, 1, :] = 1 / t_s
    
    
    
    # Update the scattering matrix S for each wavelength
    for wl in range(126):
        S_p[:, :, wl] = np.dot(S_p[:, :, wl], np.dot(I_p[:, :, wl], L[:, :, wl]))
        S_s[:, :, wl] = np.dot(S_s[:, :, wl], np.dot(I_s[:, :, wl], L[:, :, wl]))
    
    

# Calculate the reflection and transmission coefficients for each wavelength
R_p = np.abs(S_p[1, 0, :] / S_p[0, 0, :])**2
R_s = np.abs(S_s[1, 0, :] / S_s[0, 0, :])**2

T_p = np.abs(1 / S_p[0, 0, :])**2
T_s = np.abs(1 / S_s[0, 0, :])**2


# Correction factor for transmissivity
n_substrate = layers[-1][0]
correction_factor = (n_substrate * np.cos(phi0)) / (n0 * np.cos(phi0))
T_p_corrected = T_p * correction_factor
T_s_corrected = T_s * correction_factor



# Average R and T for unpolarized light
R = (R_p + R_s) / 2
T = (T_p_corrected + T_s_corrected) / 2
A = 1 - R - T

plt.figure()
plt.plot(lambda_um, R, label="R")
plt.xscale("log")
plt.xlabel("Wavelength (um)")
plt.ylabel("Reflectance")
plt.show()





