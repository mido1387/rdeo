#!/usr/bin/env python
"""
xyz2smiles_v3.py -- XYZ -> SMILES, with support for deprotonated (anionic) sites.
Usage
-----
    python xyz2smiles_v3.py                 # all *.xyz in the cwd
    python xyz2smiles_v3.py -d ./anions -o anions.smi
    python xyz2smiles_v3.py --clean         # strip punctuation from names
"""

import argparse
import glob
import itertools
import math
import os
import re
import string

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog("rdApp.*")

# Covalent radii (Angstrom, Cordero et al.) used only for the connectivity guess
# that finds candidate anionic sites. RDKit does its own perception afterwards.
COVALENT_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}
BOND_TOLERANCE = 1.30  # bonded if d < tol * (r_i + r_j)

# Neutral valence used to spot under-coordinated atoms.
NEUTRAL_VALENCE = {"B": 3, "C": 4, "N": 3, "O": 2, "Si": 4, "P": 3, "S": 2}

# X-H bond lengths for the temporary cap.
XH_LENGTH = {"C": 1.09, "N": 1.01, "O": 0.97, "S": 1.34, "P": 1.42, "B": 1.19}

# Brønsted-style preference used only to break ties with --first-match.
SITE_PRIORITY = {"O": 0, "S": 1, "N": 2, "C": 3, "P": 4, "B": 5}

CHARGE_RE = re.compile(r"(?:chrg|charge|q)\s*[=:]\s*([+-]?\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Plain-text XYZ handling (no RDKit, so the geometry logic is easy to test)
# --------------------------------------------------------------------------- #

def read_xyz(path):
    """Return (symbols, coords, comment_line)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    n_atoms = int(lines[0].split()[0])
    comment = lines[1] if len(lines) > 1 else ""
    symbols, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0].capitalize())
        coords.append(tuple(float(x) for x in parts[1:4]))
    if len(symbols) != n_atoms:
        raise ValueError(f"{path}: header says {n_atoms} atoms, found {len(symbols)}")
    return symbols, coords, comment


def charge_from_comment(comment):
    """Pull an explicit charge out of the XYZ comment line, or None."""
    match = CHARGE_RE.search(comment)
    return int(match.group(1)) if match else None


def xyz_block(symbols, coords, comment="from xyz2smiles_v3"):
    lines = [str(len(symbols)), comment]
    lines += [f"{s:<2} {x:>14.8f} {y:>14.8f} {z:>14.8f}" for s, (x, y, z) in zip(symbols, coords)]
    return "\n".join(lines) + "\n"


def neighbour_lists(symbols, coords):
    """Distance-based connectivity. Returns a list of neighbour-index lists."""
    n = len(symbols)
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        ri = COVALENT_RADII.get(symbols[i])
        if ri is None:
            continue
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(symbols[j])
            if rj is None:
                continue
            dist = math.dist(coords[i], coords[j])
            if dist < BOND_TOLERANCE * (ri + rj):
                neighbours[i].append(j)
                neighbours[j].append(i)
    return neighbours


def candidate_sites(symbols, coords):
    """
    Heavy atoms with fewer connections than their neutral valence allows, i.e.
    atoms that could carry a negative charge once bond orders are assigned.
    Over-generating is fine -- every candidate is tested and only ones that give
    a sane molecule survive.
    """
    neighbours = neighbour_lists(symbols, coords)
    sites = []
    for idx, sym in enumerate(symbols):
        if sym == "H":
            continue
        valence = NEUTRAL_VALENCE.get(sym)
        if valence is None or sym not in XH_LENGTH:
            continue
        if len(neighbours[idx]) < valence:
            sites.append(idx)
    sites.sort(key=lambda i: (SITE_PRIORITY.get(symbols[i], 9), i))
    return sites, neighbours


def cap_position(idx, symbols, coords, neighbours):
    """
    Coordinates for a hydrogen placed on atom idx, pointing away from the mean
    direction of its existing neighbours. Only needs to be roughly right: it is
    thrown away again once bond orders have been perceived.
    """
    origin = coords[idx]
    direction = [0.0, 0.0, 0.0]
    for j in neighbours[idx]:
        vec = [coords[j][k] - origin[k] for k in range(3)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        for k in range(3):
            direction[k] -= vec[k] / norm
    norm = math.sqrt(sum(v * v for v in direction))
    if norm < 1e-6:                      # neighbours cancelled out (e.g. linear)
        direction, norm = [0.0, 0.0, 1.0], 1.0
    length = XH_LENGTH.get(symbols[idx], 1.05)
    return tuple(origin[k] + length * direction[k] / norm for k in range(3))


# --------------------------------------------------------------------------- #
# Perception
# --------------------------------------------------------------------------- #

def perceive(block, charge):
    """DetermineBonds on an XYZ block. Returns a sanitized Mol or None."""
    mol = Chem.MolFromXYZBlock(block)
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def to_smiles(mol):
    """Canonical SMILES with the explicit hydrogens folded back in."""
    try:
        trimmed = Chem.RemoveHs(mol)
    except Exception:
        trimmed = mol
    smiles = Chem.MolToSmiles(trimmed)
    if "[H]" in smiles:                   # RemoveHs keeps Hs on some charged atoms
        smiles = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
    return smiles


def strip_caps(mol, cap_indices):
    """
    Delete the temporary hydrogens (always the highest indices) and put the
    negative charge on the atoms they were attached to.
    """
    editable = Chem.RWMol(mol)
    parents = []
    for h_idx in cap_indices:
        neighbours = [nb.GetIdx() for nb in editable.GetAtomWithIdx(h_idx).GetNeighbors()]
        if len(neighbours) != 1:
            return None                  # cap migrated or ended up bridging
        parents.append(neighbours[0])
    for h_idx in sorted(cap_indices, reverse=True):
        editable.RemoveAtom(h_idx)
    for parent_idx in parents:
        atom = editable.GetAtomWithIdx(parent_idx)
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    out = editable.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def mol_from_xyz(path, charge_override=None, first_match=False):
    """
    Returns (mol, note). mol is None on failure and note explains what happened.
    """
    symbols, coords, comment = read_xyz(path)
    charge = charge_override
    if charge is None:
        charge = charge_from_comment(comment)
    if charge is None:
        charge = 0
        note_charge = "charge assumed 0 (none in comment line)"
    else:
        note_charge = f"charge {charge:+d}"

    # 1. Straight perception -- works for neutrals, carboxylates, alkoxides.
    mol = perceive(xyz_block(symbols, coords, comment), charge)
    if mol is not None:
        return mol, f"direct, {note_charge}"

    if charge >= 0:
        return None, f"direct perception failed, {note_charge} (capping only handles anions)"

    # 2. Cap the candidate anionic sites, perceive the neutral, strip the caps.
    n_caps = -charge
    sites, neighbours = candidate_sites(symbols, coords)
    if len(sites) < n_caps:
        return None, f"direct perception failed and no capping sites found, {note_charge}"

    combos = list(itertools.combinations(sites, n_caps))
    if len(combos) > 2000:
        combos = combos[:2000]

    results = {}                          # canonical SMILES -> (mol, site tuple)
    for combo in combos:
        capped_symbols = list(symbols)
        capped_coords = list(coords)
        cap_indices = []
        for site in combo:
            capped_coords.append(cap_position(site, symbols, coords, neighbours))
            capped_symbols.append("H")
            cap_indices.append(len(capped_symbols) - 1)

        neutral = perceive(xyz_block(capped_symbols, capped_coords), charge + n_caps)
        if neutral is None:
            continue
        anion = strip_caps(neutral, cap_indices)
        if anion is None:
            continue
        if Chem.GetFormalCharge(anion) != charge:
            continue
        smiles = to_smiles(anion)
        results.setdefault(smiles, (anion, combo))

    if not results:
        return None, f"direct perception and H-capping both failed, {note_charge}"

    if len(results) == 1:
        smiles, (mol, combo) = next(iter(results.items()))
        sites_txt = ",".join(f"{symbols[i]}{i + 1}" for i in combo)
        return mol, f"capped at {sites_txt}, {note_charge}"

    ranked = sorted(
        results.items(),
        key=lambda kv: tuple(SITE_PRIORITY.get(symbols[i], 9) for i in kv[1][1]),
    )
    alternatives = " | ".join(smi for smi, _ in ranked)
    if first_match:
        smiles, (mol, combo) = ranked[0]
        sites_txt = ",".join(f"{symbols[i]}{i + 1}" for i in combo)
        return mol, (f"AMBIGUOUS ({len(results)} options), took {sites_txt}, "
                     f"{note_charge}; others: {alternatives}")
    return None, f"AMBIGUOUS: {len(results)} valid cappings -> {alternatives}"


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Convert XYZ files to SMILES (anion-aware).")
    parser.add_argument("-d", "--directory", default=".", help="Directory of .xyz files")
    parser.add_argument("-o", "--output", default="molecules.smi", help="Output .smi file")
    parser.add_argument("-f", "--failures", default="xyz2smiles_failures.txt")
    parser.add_argument("-c", "--charge", type=int, default=None,
                        help="Force this charge for every file (default: read the comment line)")
    parser.add_argument("--first-match", action="store_true",
                        help="Resolve ambiguous cappings by acidity preference (O > S > N > C) "
                             "instead of logging them as failures")
    parser.add_argument("--clean", action="store_true",
                        help="Remove punctuation (except _ and -) from chemical names.")
    args = parser.parse_args()

    punctuation_to_remove = string.punctuation.replace("_", "").replace("-", "")
    clean_translator = str.maketrans("", "", punctuation_to_remove)

    xyz_files = sorted(glob.glob(os.path.join(args.directory, "*.xyz")))
    if not xyz_files:
        print(f"No .xyz files found in {args.directory}")
        return

    n_ok = n_fail = 0
    with open(args.output, "w") as out_file, open(args.failures, "w") as fail_file:
        for filepath in xyz_files:
            name = os.path.splitext(os.path.basename(filepath))[0]
            if args.clean:
                name = name.translate(clean_translator)
            try:
                mol, note = mol_from_xyz(filepath, charge_override=args.charge,
                                         first_match=args.first_match)
            except Exception as exc:
                mol, note = None, f"exception: {exc}"

            if mol is None:
                n_fail += 1
                fail_file.write(f"{name}\t{note}\n")
                print(f"[FAIL] {name}: {note}")
                continue

            smiles = to_smiles(mol)
            out_file.write(f"{smiles}\t{name}\n")
            n_ok += 1
            print(f"[ OK ] {name}: {smiles}   ({note})")

    print(f"\n{n_ok} written to {args.output}, {n_fail} logged in {args.failures}")


if __name__ == "__main__":
    main()
