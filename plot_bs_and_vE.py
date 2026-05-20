#!/usr/bin/env python3
"""
Plot CP2K .bs band structure and velocity-vs-energy v(E).

Usage:
    python plot_bs_and_vE.py bulk_au_band.bs

Output PNG names:
    {bs_file_stem}_band_structure.png
    {bs_file_stem}_velocity_vs_energy.png

For Au fcc path in this file, the default labels are Gamma-X-W-L-Gamma.
The script shifts energies by EF. By default EF is estimated from occupations in the
.bs file as the highest occupied energy. For a metal, it is better to pass your corrected
Fermi energy explicitly:

    python plot_bs_and_vE.py bulk_au_band.bs --fermi-ev 11.68454551

Velocity:
    v = (1/hbar) dE/dk.
Here k is computed from fractional reciprocal coordinates assuming conventional
reciprocal units of 2*pi/a. Default a = 4.078 Angstrom for Au. Change it if your
CELL uses a different lattice constant:

    python plot_bs_and_vE.py bulk_au_band.bs --lattice-a 4.080
"""

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# Conversion: (1/hbar) * (1 eV Angstrom) in m/s
EV_ANG_OVER_HBAR_TO_M_PER_S = 1.519267447e5


@dataclass
class BandSet:
    start: np.ndarray
    end: np.ndarray
    k_frac: np.ndarray      # shape: nk, 3
    dist_raw: np.ndarray    # CP2K printed distance-like coordinate, shape: nk
    energies: np.ndarray    # shape: nk, nbands, eV
    occupations: np.ndarray # shape: nk, nbands


def parse_cp2k_bs(filename):
    """Parse CP2K .bs file with one or more Set blocks."""
    sets = []

    re_set = re.compile(r"^#\s*Set\s+(\d+):.*?(\d+)\s+k-points,\s+(\d+)\s+bands")
    re_sp = re.compile(
        r"^#\s*Special point\s+\d+\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
    )
    re_point = re.compile(
        r"^#\s*Point\s+(\d+)\s+Spin\s+\d+:\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)"
    )

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        m = re_set.match(lines[i])
        if not m:
            i += 1
            continue

        nk = int(m.group(2))
        nb = int(m.group(3))
        i += 1

        special = []
        while i < len(lines) and len(special) < 2:
            sm = re_sp.match(lines[i])
            if sm:
                special.append(np.array([float(sm.group(1)), float(sm.group(2)), float(sm.group(3))]))
            i += 1
        if len(special) != 2:
            raise ValueError("Could not read two special points for a Set block.")

        k_frac = []
        dist_raw = []
        energies = []
        occupations = []

        for _ in range(nk):
            while i < len(lines):
                pm = re_point.match(lines[i])
                i += 1
                if pm:
                    k_frac.append([float(pm.group(2)), float(pm.group(3)), float(pm.group(4))])
                    dist_raw.append(float(pm.group(5)))
                    break
            else:
                raise ValueError("Unexpected end of file while reading k-points.")

            # Skip '# Band Energy [eV] Occupation' header
            while i < len(lines) and lines[i].strip().startswith("#"):
                i += 1

            e_list, occ_list = [], []
            for _band in range(nb):
                parts = lines[i].split()
                if len(parts) < 3:
                    raise ValueError(f"Bad band line near line {i+1}: {lines[i]}")
                e_list.append(float(parts[1]))
                occ_list.append(float(parts[2]))
                i += 1

            energies.append(e_list)
            occupations.append(occ_list)

        sets.append(
            BandSet(
                start=special[0],
                end=special[1],
                k_frac=np.array(k_frac, dtype=float),
                dist_raw=np.array(dist_raw, dtype=float),
                energies=np.array(energies, dtype=float),
                occupations=np.array(occupations, dtype=float),
            )
        )

    if not sets:
        raise ValueError("No CP2K band Set blocks found. Is this a valid .bs file?")
    return sets


def estimate_fermi_energy(sets):
    """Estimate EF as highest occupied energy from occupation > 0."""
    all_e = np.concatenate([s.energies.reshape(-1) for s in sets])
    all_occ = np.concatenate([s.occupations.reshape(-1) for s in sets])
    occupied = all_e[all_occ > 1e-6]
    unoccupied = all_e[all_occ <= 1e-6]
    if occupied.size == 0:
        raise ValueError("No occupied states found; pass --fermi-ev manually.")
    homo = float(np.max(occupied))
    if unoccupied.size:
        lumo = float(np.min(unoccupied))
        print(f"Occupation-based HOMO/highest occupied energy = {homo:.8f} eV")
        print(f"Occupation-based LUMO/lowest unoccupied energy = {lumo:.8f} eV")
        if abs(lumo - homo) < 1e-3:
            print("Looks metallic or nearly gapless from the .bs occupations.")
    return homo


def high_symmetry_labels(sets):
    """Infer common fcc labels for this Au path; otherwise use coordinates."""
    label_map = {
        (0.0, 0.0, 0.0): r"$\Gamma$",
        (0.0, 0.5, 0.5): "X",
        (0.25, 0.5, 0.75): "W",
        (0.5, 0.5, 0.5): "L",
    }

    points = [sets[0].start] + [s.end for s in sets]
    labels = []
    for p in points:
        key = tuple(np.round(p, 8))
        labels.append(label_map.get(key, f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"))
    return labels


def cumulative_k_axis_fractional(sets):
    """Cumulative x-axis in fractional reciprocal-coordinate distance."""
    xs = []
    ticks = [0.0]
    offset = 0.0
    last_end = None

    for si, s in enumerate(sets):
        d = np.zeros(len(s.k_frac))
        for j in range(1, len(s.k_frac)):
            d[j] = d[j-1] + np.linalg.norm(s.k_frac[j] - s.k_frac[j-1])

        if si > 0:
            # Do not add a gap if path is continuous.
            jump = np.linalg.norm(s.k_frac[0] - last_end)
            offset += jump

        xs.append(d + offset)
        offset = xs[-1][-1]
        ticks.append(offset)
        last_end = s.k_frac[-1]

    return xs, ticks


def k_axis_cartesian_ang_inv(sets, lattice_a_angstrom):
    """Cumulative k distance in Angstrom^-1 using k_cart = frac * 2*pi/a."""
    factor = 2.0 * np.pi / lattice_a_angstrom
    xs_frac, ticks_frac = cumulative_k_axis_fractional(sets)
    xs_cart = [x * factor for x in xs_frac]
    ticks_cart = [x * factor for x in ticks_frac]
    return xs_cart, ticks_cart


def plot_band_structure(sets, fermi_ev, out_png, energy_window=None):
    xs, ticks = cumulative_k_axis_fractional(sets)
    labels = high_symmetry_labels(sets)

    plt.figure(figsize=(7.5, 5.5))
    for s, x in zip(sets, xs):
        shifted = s.energies - fermi_ev
        for b in range(shifted.shape[1]):
            plt.plot(x, shifted[:, b], linewidth=0.8)

    for t in ticks:
        plt.axvline(t, linewidth=0.8, linestyle="--")
    plt.axhline(0.0, linewidth=1.0, linestyle="--")

    plt.xticks(ticks, labels)
    plt.ylabel(r"Energy $E - E_F$ (eV)")
    plt.xlabel("k-path")
    plt.title("Band structure")
    if energy_window is not None:
        plt.ylim(-abs(energy_window), abs(energy_window))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def compute_velocities(sets, fermi_ev, lattice_a_angstrom):
    """
    Compute |v| along each path segment.
    Uses v = (1/hbar) dE/dk, with E in eV and k in Angstrom^-1.
    Returns energies shifted by EF and velocities in m/s.
    """
    xs_cart, _ = k_axis_cartesian_ang_inv(sets, lattice_a_angstrom)
    all_e_shifted = []
    all_v = []

    for s, kdist in zip(sets, xs_cart):
        e = s.energies
        for b in range(e.shape[1]):
            # dE/dk in eV Angstrom
            dE_dk = np.gradient(e[:, b], kdist, edge_order=2)
            v = np.abs(dE_dk) * EV_ANG_OVER_HBAR_TO_M_PER_S
            all_e_shifted.append(e[:, b] - fermi_ev)
            all_v.append(v)

    return np.concatenate(all_e_shifted), np.concatenate(all_v)


def plot_velocity_vs_energy(sets, fermi_ev, lattice_a_angstrom, out_png, energy_window=5.0):
    e_shifted, v = compute_velocities(sets, fermi_ev, lattice_a_angstrom)

    mask = np.isfinite(e_shifted) & np.isfinite(v)
    if energy_window is not None:
        mask &= np.abs(e_shifted) <= abs(energy_window)

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(e_shifted[mask], v[mask], s=8, alpha=0.6)
    plt.axvline(0.0, linewidth=1.0, linestyle="--")
    plt.xlabel(r"Energy $E - E_F$ (eV)")
    plt.ylabel(r"Band velocity $|v|$ (m/s)")
    plt.title(r"Velocity vs energy, $v(E)$")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot CP2K .bs band structure and v(E).")
    parser.add_argument("bs_file", help="Input CP2K .bs file, e.g. bulk_au_band.bs")
    parser.add_argument("--fermi-ev", type=float, default=None,
                        help="Corrected Fermi energy in eV. If omitted, estimated from occupations.")
    parser.add_argument("--lattice-a", type=float, default=4.078,
                        help="Conventional Au lattice constant in Angstrom. Default: 4.078")
    parser.add_argument("--energy-window", type=float, default=5.0,
                        help="Plot +/- this many eV around EF. Use a large value for all bands.")
    args = parser.parse_args()

    sets = parse_cp2k_bs(args.bs_file)
    print(f"Read {len(sets)} path segments from {args.bs_file}")
    print(f"Each segment has {sets[0].energies.shape[0]} k-points and {sets[0].energies.shape[1]} bands")
    print("Path labels:", " -> ".join(high_symmetry_labels(sets)))

    fermi_ev = args.fermi_ev
    if fermi_ev is None:
        fermi_ev = estimate_fermi_energy(sets)
        print(f"Using estimated EF = {fermi_ev:.8f} eV")
    else:
        print(f"Using user-supplied corrected EF = {fermi_ev:.8f} eV")

    stem = os.path.splitext(os.path.basename(args.bs_file))[0]
    band_png = f"{stem}_band_structure.png"
    ve_png = f"{stem}_velocity_vs_energy.png"

    plot_band_structure(sets, fermi_ev, band_png, energy_window=args.energy_window)
    plot_velocity_vs_energy(sets, fermi_ev, args.lattice_a, ve_png, energy_window=args.energy_window)

    print(f"Wrote {band_png}")
    print(f"Wrote {ve_png}")
    print("Interpretation:")
    print("  band_structure: E - EF vs k-path; crossings through 0 eV show metallic bands at EF.")
    print("  velocity_vs_energy: |v| = (1/hbar)|dE/dk| vs E - EF; values near 0 eV are hot/electron Fermi-level velocities.")
    print("  For absolute velocity, make sure --lattice-a matches the CELL used in CP2K.")


if __name__ == "__main__":
    main()
