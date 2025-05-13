import numpy as np
import matplotlib.pyplot as plt

# Read the file as text, fix formatting, then load into numpy
with open('MESH_K18.txt', 'r') as f:
    lines = f.readlines()

# Skip the header line and replace '-' with ' -' to ensure proper separation
data_lines = [line.replace('-', ' -') for line in lines[1:]]

# Load the data using numpy
data = np.loadtxt(data_lines)

# Extract columns
energy = data[:, 0]  # Energy (eV)
epsilonX = data[:, 1]  # epsilonX
epsilonY = data[:, 2]  # epsilonY

# Plot
plt.figure(figsize=(10, 6))
plt.plot(energy, epsilonX, label='epsilonX', color='blue')
plt.plot(energy, epsilonY, label='epsilonY', color='red', linestyle='--')

plt.xlabel('Energy (eV)')
plt.ylabel('Epsilon')
plt.title('Comparison of epsilonX and epsilonY')
plt.legend()
plt.grid(True)
plt.show()

epsilon1 = epsilonX
epsilon2 = epsilonY
# Compute n and kappa (k)
magnitude = np.sqrt(epsilon1**2 + epsilon2**2)  # sqrt(ε₁² + ε₂²)
n = np.sqrt(0.5 * (magnitude + epsilon1))  # Refractive index (n)
kappa = np.sqrt(0.5 * (magnitude - epsilon1))  # Extinction coefficient (κ)

# Plot n and kappa
plt.figure(figsize=(10, 6))
plt.plot(energy, n, label='Refractive index (n)', color='blue')
plt.plot(energy, kappa, label='Extinction coefficient (κ)', color='red', linestyle='--')

plt.xlabel('Energy (eV)')
plt.ylabel('Optical Properties')
plt.title('Refractive Index (n) and Extinction Coefficient (κ)')
plt.legend()
plt.ylim(0,1)
plt.grid(True)
plt.show()
