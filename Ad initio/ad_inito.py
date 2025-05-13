import numpy as np
import matplotlib.pyplot as plt
import Extraction
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

print("Start...")

# Function to read and format data from a file
def read_data(file_r, file_i):
    with open(file_r, 'r') as f:
        lines = f.readlines()
    data_lines = [line.replace('-', ' -') for line in lines[1:]]  # Fix formatting
    data_r = np.loadtxt(data_lines)
    data_i = np.loadtxt(file_i)
    energy_r, epsilon_r = data_r[:, 0], data_r[:, 2]
    energy_i, epsilon_i = data_i[:, 0], data_i[:, 2]
    return energy_r, epsilon_r, energy_i, epsilon_i


# Function to compute optical properties
def compute_optical_properties(epsilon1, epsilon2):
    magnitude = np.sqrt(epsilon1**2 + epsilon2**2)
    n = np.sqrt(0.5 * (magnitude + epsilon1))
    kappa = np.sqrt(0.5 * (magnitude - epsilon1))
    return n, kappa

# Reading data
energy_r, epsilon_r, energy_i, epsilon_i = read_data('data/espr_aiida.txt', 'data/espi_aiida.txt')
energy_r_24, epsilon_r_24, energy_i_24, epsilon_i_24 = read_data('data/espr_aiida_24.txt', 'data/espi_aiida_24.txt')
energy_r_18, epsilon_r_18, energy_i_18, epsilon_i_18 = read_data('data/espr_aiida_18.txt', 'data/espi_aiida_18.txt')


def compute_dielectric_function(n, kappa):
    epsilon_r = n**2 - kappa**2
    epsilon_i = 2 * n * kappa
    return epsilon_r.tolist(), epsilon_i.tolist()


wl_interp = np.linspace(0.2, 20, 1000)
n_yang, kappa_yang = Extraction.extract_nk('data/Ag_Yang.txt', wl_interp)
n_hm, kappa_hm = Extraction.extract_nk('data/Ag_Hagemann.txt', wl_interp)

epsilon_r_yang, epsilon_i_yang = compute_dielectric_function(n_yang, kappa_yang)
epsilon_r_hm, epsilon_i_hm = compute_dielectric_function(n_hm, kappa_hm)
print(wl_interp)
Energy_interp = 1.239 / wl_interp
# retourne toute les listes (inverse l'ordre)
Energy_interp = Energy_interp[::-1]
print("Energy_interp", Energy_interp)
epsilon_r_yang = np.array(epsilon_r_yang[::-1])
epsilon_i_yang = np.array(epsilon_i_yang[::-1])
epsilon_r_hm = np.array(epsilon_r_hm[::-1])
epsilon_i_hm = np.array(epsilon_i_hm[::-1])

# Plot epsilon values
def plot_epsilon(X_min=1, X_max=6, Y_min_real=0.1, Y_max_real=100, Y_min_imag=0.1, Y_max_imag=10):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Left subplot: Real part
    ax_real = axes[0]
    ax_real.plot(energy_r, -epsilon_r, color='black', linewidth=0.75, marker='o', markersize=8, markerfacecolor='white', markevery=20)
    ax_real.plot(energy_r_18, -epsilon_r_18, color='black', linewidth=0.75, marker='*', markersize=8, markerfacecolor='white', markevery=20)
    ax_real.plot(energy_r_24, -epsilon_r_24, color='black', linewidth=0.75, marker='D', markersize=8, markerfacecolor='white', markevery=20)
    ax_real.plot(Energy_interp, -epsilon_r_yang, color='blue', linewidth=0.75)
    ax_real.plot(Energy_interp, -epsilon_r_hm, color='red', linewidth=0.75)
    
    # Fill area between the curves
    ax_real.fill_between(energy_r, -epsilon_r, -epsilon_r_18, color='grey', alpha=0.3)
    ax_real.fill_between(energy_r_18, -epsilon_r_18, -epsilon_r_24, color='grey', alpha=0.3)
    
    ax_real.set_xlabel('Energy (eV)')
    ax_real.set_ylabel('$-\epsilon_1$')

    ax_real.set_xlim(X_min, X_max)
    ax_real.set_ylim(Y_min_real, Y_max_real)
    ax_real.set_yscale('log')
    ax_real.set_xscale('log')
    # Reduce number of x-ticks for clarity
    ax_real.set_xticks([X_min, 2, 3, 4, 5, X_max])
    ax_real.set_xticklabels([X_min, 2, 3, 4, 5, X_max])
    
    ax_real.set_yticks([Y_min_real, 1, 10, Y_max_real])
    ax_real.set_yticklabels([Y_min_real, 1, 10, Y_max_real])

    # Add secondary x-axis for wavelength (µm)
    ax1_real = ax_real.twiny()
    new_ticks_labels = np.array([0.7, 0.4, 0.2])
    new_ticks_locations = [1239 / tick for tick in new_ticks_labels]
    ax1_real.set_xlim(ax_real.get_xlim())
    ax1_real.set_xticks(new_ticks_locations)
    ax1_real.set_xticklabels(new_ticks_labels)
    ax1_real.xaxis.set_minor_locator(AutoMinorLocator())
    ax1_real.set_xlabel('Wavelength (µm)')

    # Right subplot: Imaginary part
    ax_imag = axes[1]
    ax_imag.plot(energy_i, epsilon_i, label='K = 12', linewidth=0.75, marker='o', markersize=8, markerfacecolor='white', markevery=20, color='black')
    ax_imag.plot(energy_i_18, epsilon_i_18, label='K = 18', linewidth=0.75, marker='*', markersize=8, markerfacecolor='white', markevery=20, color='black')
    ax_imag.plot(energy_i_24, epsilon_i_24, label='K = 24', linewidth=0.75, marker='D', markersize=8, markerfacecolor='white', markevery=20, color='black')
    ax_imag.plot(Energy_interp, epsilon_i_yang, label='Yang & al.,2015 ', linewidth=0.75, color='blue')
    ax_imag.plot(Energy_interp, epsilon_i_hm, label='Hagemann & al.,1972', linewidth=0.75, color='red')

    # Fill area between the curves
    ax_imag.fill_between(energy_i, epsilon_i, epsilon_i_18, color='grey', alpha=0.3)
    ax_imag.fill_between(energy_i_18, epsilon_i_18, epsilon_i_24, color='grey', alpha=0.3)
    ax_imag.set_xlabel('Energy (eV)')
    ax_imag.set_ylabel('$\epsilon_2$')
    ax_imag.set_xscale('log')
    ax_imag.set_yticks([Y_min_imag, 1, Y_max_imag])
    ax_imag.set_yticklabels([Y_min_imag, 1, Y_max_imag])
    ax_imag.set_xticks([X_min, 2, 3, 4, 5, X_max])
    ax_imag.set_xticklabels([X_min, 2, 3, 4, 5, X_max])

    ax_imag.set_xlim(X_min, X_max)
    ax_imag.set_ylim(Y_min_imag, Y_max_imag)
    ax_imag.set_yscale('log')

    # Add secondary x-axis for wavelength (µm)
    ax1_imag = ax_imag.twiny()
    ax1_imag.set_xlim(ax_imag.get_xlim())
    ax1_imag.set_xticks(new_ticks_locations)
    ax1_imag.set_xticklabels(new_ticks_labels)
    ax1_imag.xaxis.set_minor_locator(AutoMinorLocator())
    ax1_imag.set_xlabel('Wavelength (µm)')

    # Combine legends from both subplots
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, frameon=False, bbox_transform=fig.transFigure)

    plt.tight_layout()
    filename = f"Output/epsi_log_X{X_min}-{X_max}_Yreal{Y_min_real}-{Y_max_real}_Yimag{Y_min_imag}-{Y_max_imag}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')


n_12, kappa_12 = compute_optical_properties(epsilon_r, epsilon_i[3:])
n_18, kappa_18 = compute_optical_properties(epsilon_r_18, epsilon_i_18[3:])
n_24, kappa_24 = compute_optical_properties(epsilon_r_24, epsilon_i_24[3:])

wl_12 = 1.239 / energy_r
wl_18 = 1.239 / energy_r_18
wl_24 = 1.239 / energy_r_24

# Plotting the refractive index and extinction coefficient
def n_k():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Left subplot: Refractive index
    ax_n = axes[0]
    ax_n.plot(wl_12, n_12, label='K = 12', linewidth=0.75, marker='o', markersize=8, markerfacecolor='white', markevery=20, color='black')
    ax_n.plot(wl_18, n_18, label='K = 18', linewidth=0.75, marker='*', markersize=8, markerfacecolor='white', markevery=20, color='black')
    ax_n.plot(wl_24, n_24, label='K = 24', linewidth=0.75, marker='D', markersize=8, markerfacecolor='white', markevery=20, color='black')
    ax_n.fill_between(wl_12, n_12, n_18, color='grey', alpha=0.3)
    ax_n.fill_between(wl_18, n_18, n_24, color='grey', alpha=0.3)
    ax_n.plot(wl_interp, n_yang, label='Yang & al.,2015 ', linewidth=0.75, color='blue')
    ax_n.plot(wl_interp, n_hm, label='Hagemann & al.,1972', linewidth=0.75, color='red')
    ax_n.set_xlabel('Wavelength (µm)')
    ax_n.set_ylabel('Refractive index (n)')
    ax_n.set_xlim(0.2, 10)
    ax_n.set_xscale('log')
    ax_n.set_yscale('log')

    ax_k = axes[1]
    # Right subplot: Extinction coefficient no label 
    ax_k.plot(wl_12, kappa_12, linewidth=0.75, marker='o', markersize=8, markerfacecolor='white', markevery=20, color='black')
    ax_k.plot(wl_18, kappa_18, linewidth=0.75, marker='*', markersize=8, markerfacecolor='white', markevery=20, color='black')
    ax_k.plot(wl_24, kappa_24, linewidth=0.75, marker='D', markersize=8, markerfacecolor='white', markevery=20, color='black')

    # Fill area between the curves
    ax_k.fill_between(wl_12, kappa_12, kappa_18, color='grey', alpha=0.3)
    ax_k.fill_between(wl_18, kappa_18, kappa_24, color='grey', alpha=0.3)
    ax_k.plot(wl_interp, kappa_yang, linewidth=0.75, color='blue')
    ax_k.plot(wl_interp, kappa_hm, linewidth=0.75, color='red')
    ax_k.set_xlabel('Wavelength (µm)')
    ax_k.set_ylabel('Extinction coefficient ($\kappa$)')
    ax_k.set_xlim(0.2, 10)
    ax_k.set_xscale('log')
    ax_k.set_yscale('log')

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, frameon=False, bbox_transform=fig.transFigure)

    plt.tight_layout()
    filename = f"Output/nk_log_X.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def snells(n0, n1, phi0):
    """
    Calculate the angle of refraction using Snell's law.

    Parameters:
    n0 (float): Refractive index of the first medium.
    n1 (float): Refractive index of the second medium.
    phi0 (float): Angle of incidence in degrees.

    Returns:
    float: Angle of refraction in degrees.
    """
    print("Task 1 : Calculating angle of refraction...")
    phi0 = np.radians(phi0)
    phi1 = np.arcsin(n0 * np.sin(phi0) / n1)
    return phi1

def reflectivity_semi_infinite_layer(n0, n1, phi0):
    """
    Calculate the reflectivity of a semi-infinite layer.

    Parameters:
    n0 (float): Refractive index of the first medium.
    n1 (float): Refractive index of the second medium.
    phi0 (float): Angle of incidence in degrees.

    Returns:
    float: Reflectivity of the material.
    """
    print("Task 1 : Calculating reflectivity for semi-infinite layer...")
    phi0 = np.radians(phi0)
    phi1 = snells(n0, n1, phi0)
    
    # Calculate the reflection coefficients for s- and p-polarized light
    r_s = (n0 * np.cos(phi0) - n1 * np.cos(phi1)) / (n0 * np.cos(phi0) + n1 * np.cos(phi1))
    r_p = (n1 * np.cos(phi0) - n0 * np.cos(phi1)) / (n1 * np.cos(phi0) + n0 * np.cos(phi1))
    
    # Calculate the reflectivity as the average of |r_s|^2 and |r_p|^2
    R = (np.abs(r_s)**2 + np.abs(r_p)**2) / 2
        
    return R

fig, ax = plt.subplots(figsize=(14, 6))
n0 = 1.0  # Refractive index of air
n1 = n_12 - 1j * kappa_12
n2 = n_18 - 1j * kappa_18
n3 = n_24 - 1j * kappa_24
n4 = n_yang - 1j * kappa_yang
n5 = n_hm - 1j * kappa_hm

phi0 = 0.0  # Angle of incidence in degrees
R1 = reflectivity_semi_infinite_layer(n0, n1, phi0)
R2 = reflectivity_semi_infinite_layer(n0, n2, phi0)
R3 = reflectivity_semi_infinite_layer(n0, n3, phi0)
R4 = reflectivity_semi_infinite_layer(n0, n4, phi0)
R5 = reflectivity_semi_infinite_layer(n0, n5, phi0)

ax.plot(wl_12, R1, label='K = 12', linewidth=0.75, marker='o', markersize=8, markerfacecolor='white', markevery=20, color='black')
ax.plot(wl_18, R2, label='K = 18', linewidth=0.75, marker='*', markersize=8, markerfacecolor='white', markevery=20, color='black')
ax.plot(wl_24, R3, label='K = 24', linewidth=0.75, marker='D', markersize=8, markerfacecolor='white', markevery=20, color='black')
# Use a logarithmic spacing for markevery to show more points at the beginning, including the first point
log_markevery = np.unique(np.logspace(0, np.log10(len(wl_interp)-1), num=20, dtype=int))
ax.plot(wl_interp, R4, label='Yang & al.,2015 ', linewidth=0.75, color='blue', marker='d', markersize=8, markerfacecolor='white', markevery=log_markevery, markeredgecolor='blue')
ax.plot(wl_interp, R5, label='Hagemann & al.,1972', linewidth=0.75, color='red', marker='d', markersize=8, markerfacecolor='white', markevery=log_markevery, markeredgecolor='red')
ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Reflectivity')
ax.set_xlim(0.2, 10)
ax.set_xscale('log')
ax.legend()
plt.tight_layout()
plt.savefig("Output/reflectivity.png", dpi=300, bbox_inches='tight')