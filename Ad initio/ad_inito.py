import numpy as np
import matplotlib.pyplot as plt
print("Start...")
# Read the file as text, fix formatting, then load into numpy
with open('espr_aiida.txt', 'r') as f:
    lines = f.readlines()

# Skip the header line and replace '-' with ' -' to ensure proper separation
data_lines = [line.replace('-', ' -') for line in lines[1:]]

# Load the data using numpy
data_r = np.loadtxt(data_lines)

data_i = np.loadtxt("espi_aiida.txt")
energy_r, epsilon_r = data_r[:, 0], data_r[:, 2]
energy_i, epsilon_i = data_i[:, 0], data_i[:, 2]


with open('espr_aiida_24.txt', 'r') as f:
    lines = f.readlines()

# Skip the header line and replace '-' with ' -' to ensure proper separation
data_lines = [line.replace('-', ' -') for line in lines[1:]]

# Load the data using numpy
data_r_24 = np.loadtxt(data_lines)

data_i_24 = np.loadtxt("espi_aiida_24.txt")
energy_r_24, epsilon_r_24 = data_r_24[:, 0], data_r_24[:, 2]
energy_i_24, epsilon_i_24 = data_i_24[:, 0], data_i_24[:, 2]

with open('espr_aiida_18.txt', 'r') as f:
    lines = f.readlines()

# Skip the header line and replace '-' with ' -' to ensure proper separation
data_lines = [line.replace('-', ' -') for line in lines[1:]]

# Load the data using numpy
data_r_18 = np.loadtxt(data_lines)

data_i_18 = np.loadtxt("espi_aiida_18.txt")
energy_r_18, epsilon_r_18 = data_r_18[:, 0], data_r_18[:, 2]
energy_i_18, epsilon_i_18 = data_i_18[:, 0], data_i_18[:, 2]

plt.plot(energy_r, epsilon_r, lw=1, label="$\\epsilon_1$")
plt.plot(energy_i, epsilon_i, lw=1, label="$\\epsilon_2$")

plt.plot(energy_r_24, epsilon_r_24, lw=1, label="$\\epsilon_1$ 24", color='blue', linestyle='--')
plt.plot(energy_i_24, epsilon_i_24, lw=1, label="$\\epsilon_2$ 24", color='red', linestyle='--')
plt.plot(energy_r_18, epsilon_r_18, lw=1, label="$\\epsilon_1$ 18", color='blue', linestyle='--')
plt.plot(energy_i_18, epsilon_i_18, lw=1, label="$\\epsilon_2$ 18", color='red', linestyle='-.')
plt.xlim(0, 15)
plt.ylim(-40,40)
plt.xlabel("Energy (eV)")
plt.ylabel("$\\epsilon_1~/~\\epsilon_2$")
plt.legend(frameon=False)
plt.show()

epsilon1 = epsilon_r
epsilon2 = epsilon_i[3:]
# Compute n and kappa (k)
magnitude = np.sqrt(epsilon1**2 + epsilon2**2)  # sqrt(ε₁² + ε₂²)
n = np.sqrt(0.5 * (magnitude + epsilon1))  # Refractive index (n)
kappa = np.sqrt(0.5 * (magnitude - epsilon1))  # Extinction coefficient (κ)

# Plot n and kappa
plt.figure(figsize=(10, 6))
plt.plot(1.240/energy_r, n, label='Refractive index (n)', color='blue')
plt.plot(1.240/energy_r, kappa, label='Extinction coefficient (κ)', color='red', linestyle='--')

plt.xlabel('Energy (eV)')
plt.ylabel('Optical Properties')
plt.title('Refractive Index (n) and Extinction Coefficient (κ)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()