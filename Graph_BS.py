import numpy as np
import matplotlib.pyplot as plt

NBANDS = 100
EF_LINE = 0.0

energy_file = "band_energies_shifted_eV-20.dat"
slope_file = "band_slopes_eVA-20.dat"
velocity_file = "band_velocities_m_per_s-20.dat"

E_raw = np.loadtxt(energy_file)
S_raw = np.loadtxt(slope_file)
V_raw = np.loadtxt(velocity_file)

# column 0 = band index, column 1 = value
energies = E_raw[:, 1].reshape(-1, NBANDS)
slopes = S_raw[:, 1].reshape(-1, NBANDS)
velocities = V_raw[:, 1].reshape(-1, NBANDS)

nk = energies.shape[0]
k_index = np.arange(nk)

# Approximate high-symmetry labels
tick_pos = np.linspace(0, nk - 1, 6)
tick_lab = [r"$\Gamma$", "X", "W", "K", r"$\Gamma$", "L"]

# -------- Band structure --------
plt.figure(figsize=(8, 6))
for b in range(NBANDS):
    plt.plot(k_index, energies[:, b], linewidth=0.8)

plt.axhline(EF_LINE, linestyle="--", linewidth=1)
plt.xticks(tick_pos, tick_lab)
plt.ylabel(r"$E_{n k}-E_F$ (eV)")
plt.xlabel("k-path")
plt.ylim(-10, 10)
plt.title("Bulk Au band structure")
plt.tight_layout()
plt.savefig("Au_band_structure_shifted-20.png", dpi=300)
plt.show()

# -------- Band slopes --------
plt.figure(figsize=(8, 6))
for b in range(NBANDS):
    plt.plot(k_index, slopes[:, b], linewidth=0.8)

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(tick_pos, tick_lab)
plt.ylabel(r"$dE/dk$ (eV Å)")
plt.xlabel("k-path")
plt.ylim(-20, 20)
plt.title("Bulk Au band slopes")
plt.tight_layout()
plt.savefig("Au_band_slopes-20.png", dpi=300)
plt.show()

# -------- Velocities --------
plt.figure(figsize=(8, 6))
for b in range(NBANDS):
    plt.plot(k_index, velocities[:, b], linewidth=0.8)

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(tick_pos, tick_lab)
plt.ylabel(r"$v_{n k}$ (m/s)")
plt.xlabel("k-path")
plt.ylim(-3e6, 3e6)
plt.title("Bulk Au band velocities")
plt.tight_layout()
plt.savefig("Au_band_velocities-20.png", dpi=300)
plt.show()
