import numpy as np
import re

HA_TO_EV = 27.211386245988
SLOPE_TO_VELOCITY = 1.5192674e5  # (eV Angstrom) -> m/s

OUTFILE = "bulk_au_band.out"
BSFILE = "bulk_au_band.bs"

def read_fermi_energy_cp2k(outfile):
    """
    Reads CP2K Fermi energy from bulk_au_band.out.
    Assumes CP2K prints value in Hartree.
    """
    with open(outfile, "r") as f:
        text = f.read()

    matches = re.findall(r"Fermi energy:\s*([-+]?\d*\.\d+|\d+)", text, re.IGNORECASE)

    if not matches:
        raise ValueError("Could not find Fermi energy in output file.")

    ef_ha = float(matches[-1])
    ef_ev = ef_ha * HA_TO_EV

    return ef_ha, ef_ev


def read_cp2k_bs_simple(bsfile):
    """
    Generic reader for numeric lines in CP2K .bs file.
    This assumes each useful line contains:
    k_distance  E_band1  E_band2  E_band3 ...
    
    If your .bs format is different, send me the first 50 lines.
    """
    rows = []

    with open(bsfile, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            parts = line.split()
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                continue

            if len(nums) >= 2:
                rows.append(nums)

    data = np.array(rows)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Could not read band data from .bs file.")

    k = data[:, 0]
    bands = data[:, 1:]

    return k, bands


ef_ha, ef_ev = read_fermi_energy_cp2k(OUTFILE)

print("CP2K Fermi energy:")
print(f"  {ef_ha:.12f} Ha")
print(f"  {ef_ev:.12f} eV")

print("\nCorrected Fermi energy for band plot:")
print("  EF = 0.000000 eV")

k, bands_ev = read_cp2k_bs_simple(BSFILE)

# Shift bands so Fermi level is zero
bands_shifted = bands_ev - ef_ev

# Calculate band slopes dE/dk
# units: eV Angstrom if k is in Angstrom^-1
slopes = np.gradient(bands_shifted, k, axis=0)

# Convert slopes to velocities
velocities = slopes * SLOPE_TO_VELOCITY

np.savetxt("band_energies_shifted_eV.dat", np.column_stack([k, bands_shifted]))
np.savetxt("band_slopes_eVA.dat", np.column_stack([k, slopes]))
np.savetxt("band_velocities_m_per_s.dat", np.column_stack([k, velocities]))

print("\nWrote outputs:")
print("  band_energies_shifted_eV.dat")
print("  band_slopes_eVA.dat")
print("  band_velocities_m_per_s.dat")
