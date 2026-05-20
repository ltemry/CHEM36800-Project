import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

HA_TO_EV = 27.211386245988

# ============================================================
# 1. Read Fermi energy from CP2K output
# ============================================================

out_file = "bulk_au_band_20_GB.out"

EF_HA = None

with open(out_file, "r") as f:
    for line in f:
        if "Fermi energy:" in line:
            EF_HA = float(line.split()[-1])

if EF_HA is None:
    raise RuntimeError("Could not find Fermi energy")

EF_EV = EF_HA * HA_TO_EV

print("Fermi energy:")
print("  ", EF_HA, "Ha")
print("  ", EF_EV, "eV")

# ============================================================
# 2. Read DOS file
# ============================================================

dos_file = "bulk_au_rel_20_GB-bulk_au_dos_20_GB-1.dos"

df = pd.read_csv(
    dos_file,
    comment="#",
    sep=r"\s+",
    header=None,
    names=["E_Ha", "DOS", "Occupation"]
)
# ============================================================
# 3. Convert energies
# ============================================================

df["E_eV"] = df["E_Ha"] * HA_TO_EV

# IMPORTANT:
# Shift by EF
df["E_minus_EF"] = df["E_eV"] - EF_EV

# ============================================================
# 4. Plot DOS relative to EF
# ============================================================
plt.figure(figsize=(8,5))

plt.plot(
    df["E_minus_EF"],
    df["DOS"],
    linewidth=1.5
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(r"$E - E_F$ (eV)")
plt.ylabel("DOS")
plt.title("Bulk Au DOS")

plt.xlim(-10, 10)

plt.tight_layout()

plt.savefig(
    "Au_DOS_E_minus_EF-20.png",
    dpi=300
)

df.to_csv(
    "Au_DOS_shifted-20.csv",
    index=False
)

plt.show()
