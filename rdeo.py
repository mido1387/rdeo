#!/usr/bin/env python3
"""
RDEO - RDKit External Optimization

Batch conformer generation and refinement:

    SMILES -> RDKit ETKDGv3 embedding -> (MMFF94 pre-optimization)
           -> ORCA geometry optimization (g-xTB, an MLIP, or any ORCA method)
           -> CREST cregen deduplication and energy sorting
           -> top-N unique conformers written to the results directory

Usage:
    python rdeo.py molecules.smi
    python rdeo.py molecules.smi --engine uma --threads 16 --rename

Before the first run, edit the USER CONFIGURATION block below.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

__version__ = "0.1.9"


# =============================================================================
# USER CONFIGURATION
# -----------------------------------------------------------------------------
# Everything that depends on your machine lives here. Nothing below this block
# needs to be edited for normal use. Paths may be bare command names (resolved
# through $PATH) or absolute paths. Anything required but unset produces an
# explicit error before any calculation starts.
# =============================================================================

CONFIG = {
    # --- External programs -------------------------------------------------
    # Both are required. "orca" / "crest" work if they are already on $PATH.
    "orca_binary": "orca",
    "crest_binary": "crest",

    # --- Default optimization engine ---------------------------------------
    # One of: gxtb, gfn2, uma, aimnet2, custom.  Override per run with --engine.
    "engine": "gxtb",

    # --- MLIP external-tool clients ----------------------------------------
    # ORCA's EXTOPT interface calls an external executable that returns the
    # energy and gradient. Set the path for whichever MLIP you intend to use;
    # the other may be left empty.
    "uma_client": "",                 # e.g. "/home/you/uma_env/bin/oet_client"
    "uma_params": "-d cuda",          # extra flags passed via Ext_Params

    "aimnet2_client": "",             # e.g. "/home/you/aimnet2_env/bin/oet_client"
    "aimnet2_params": "-d cuda",

    # --- Custom ORCA method ("--engine custom") -----------------------------
    # Any ORCA optimization works here; the rest of the workflow is unchanged.
    "custom_keywords": "! r2SCAN-3c Opt",
    "custom_blocks": "",              # optional extra %blocks, as a raw string

    # --- Resource control ---------------------------------------------------
    # Threads are ORCA jobs run concurrently. Each job is given this many
    # OpenMP threads; leave at 1 unless you have cores to spare, otherwise
    # parallel xtb/MLIP jobs will oversubscribe the machine.
    # NOTE: g-XTB performs much better with 1 omp per job and many threads
    "omp_threads_per_job": 1,
    "orca_nprocs": 1,                 # %pal nprocs; >1 only for real ORCA methods

    # --- Naming -------------------------------------------------------------
    # Long chemical names produce long paths, which ORCA and some filesystems
    # will not accept. Names longer than this fall back to mol_0001-style tags
    # (--rename applies that style to everything).
    "max_name_length": 60,
}

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================


class ConfigError(Exception):
    """Raised when the CONFIG block is incomplete or points at something missing."""


@dataclass
class Engine:
    """A resolved ORCA optimization engine: simple-input line plus input blocks."""
    name: str
    keywords: str
    blocks: str = ""


# Engine definitions. "requires" names CONFIG keys that must point at an
# existing executable. Templates are formatted against CONFIG.
ENGINE_SPECS = {
    "gxtb": {
        "description": "g-xTB through ORCA's xtb interface",
        "keywords": "! XTB Opt",
        "blocks": '%xtb\n  XTBINPUTSTRING "--gxtb"\nend',
        "requires": [],
    },
    "gfn2": {
        "description": "GFN2-xTB (cheap baseline / sanity check)",
        "keywords": "! XTB2 Opt",
        "blocks": "",
        "requires": [],
    },
    "uma": {
        "description": "Meta UMA MLIP through ORCA's EXTOPT interface",
        "keywords": "! EXTOPT Opt",
        "blocks": '%method\n  ProgExt "{uma_client}"\n  Ext_Params "{uma_params}"\nend',
        "requires": ["uma_client"],
    },
    "aimnet2": {
        "description": "AIMNet2 MLIP through ORCA's EXTOPT interface",
        "keywords": "! EXTOPT Opt",
        "blocks": '%method\n  ProgExt "{aimnet2_client}"\n  Ext_Params "{aimnet2_params}"\nend',
        "requires": ["aimnet2_client"],
    },
    "custom": {
        "description": "Any ORCA method, defined by custom_keywords in CONFIG",
        "keywords": "{custom_keywords}",
        "blocks": "{custom_blocks}",
        "requires": [],
    },
}


# -----------------------------------------------------------------------------
# Configuration handling
# -----------------------------------------------------------------------------

def resolve_executable(config_key, value, purpose):
    """Turn a command name or path into a usable executable path, or explain why not."""
    if not value:
        raise ConfigError(
            f'CONFIG["{config_key}"] is empty, but {purpose} needs it.\n'
            f"    Edit the USER CONFIGURATION block at the top of this file, e.g.:\n"
            f'        "{config_key}": "/full/path/to/executable",'
        )

    resolved = shutil.which(value)
    if resolved is None:
        raise ConfigError(
            f'CONFIG["{config_key}"] is set to "{value}", which is not an '
            f"executable on this machine ({purpose}).\n"
            f"    Either put it on your $PATH or give the full path in the "
            f"USER CONFIGURATION block."
        )
    return resolved


def build_engine(name, config, dry_run=False):
    """Validate the requested engine against CONFIG and return a ready-to-use Engine."""
    if name not in ENGINE_SPECS:
        raise ConfigError(
            f'Unknown engine "{name}". Available: {", ".join(sorted(ENGINE_SPECS))}.'
        )

    spec = ENGINE_SPECS[name]

    for key in spec["requires"]:
        purpose = f'engine "{name}"'
        if not config.get(key):
            raise ConfigError(
                f'Engine "{name}" needs CONFIG["{key}"] to be set.\n'
                f"    Edit the USER CONFIGURATION block at the top of this file, e.g.:\n"
                f'        "{key}": "/home/you/env/bin/oet_client",\n'
                f"    Or override it for this run with --client."
            )
        # In a dry run the client is never called, so existence is a warning only.
        if dry_run:
            if shutil.which(config[key]) is None:
                print(f'  [WARNING] CONFIG["{key}"] = "{config[key]}" is not executable '
                      f"(ignored for --dry-run).")
        else:
            resolve_executable(key, config[key], purpose)

    keywords = spec["keywords"].format(**config).strip()
    blocks = spec["blocks"].format(**config).strip()

    if name == "custom" and not keywords:
        raise ConfigError(
            'Engine "custom" needs CONFIG["custom_keywords"] to be set, '
            'e.g. "! r2SCAN-3c Opt".'
        )

    nprocs = int(config.get("orca_nprocs", 1))
    if nprocs > 1:
        pal = f"%pal\n  nprocs {nprocs}\nend"
        blocks = f"{pal}\n\n{blocks}".strip()

    return Engine(name=name, keywords=keywords, blocks=blocks)


# -----------------------------------------------------------------------------
# Conformer generation
# -----------------------------------------------------------------------------

def get_heuristic_conformer_count(smiles):
    """Scale the conformer count with rotatable-bond count."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 200

    try:
        temp = Chem.RemoveHs(mol)
    except Exception:
        temp = mol

    try:
        n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(temp)
    except Exception:
        n_rotatable = 8

    if n_rotatable <= 7:
        return 50
    elif n_rotatable <= 12:
        return 200
    return 300


def _conformer_worker(smiles, num_confs, pre_optimize, out_queue):
    """Embedding runs in a subprocess so a hung ETKDG call can be killed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out_queue.put(None)
        return

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    print(f"  Generated {len(cids)} conformers.")

    if len(cids) == 0:
        out_queue.put(None)
        return

    if pre_optimize:
        print("  Pre-optimizing conformers with MMFF94...")
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol)
        except Exception:
            pass

    out_queue.put(mol.ToBinary())


def generate_conformers_with_timeout(smiles, num_confs=200, pre_optimize=True, timeout=120):
    ctx = multiprocessing.get_context("spawn")
    out_queue = ctx.Queue()

    p = ctx.Process(target=_conformer_worker,
                    args=(smiles, num_confs, pre_optimize, out_queue))
    p.start()

    try:
        binary_mol = out_queue.get(timeout=timeout)
        p.join()
        return Chem.Mol(binary_mol) if binary_mol is not None else None
    except queue.Empty:
        print(f"  [TIMEOUT] RDKit did not finish within {timeout} s. Terminating task.")
        p.terminate()
        p.join()
        return None


def get_charge_and_multiplicity(mol):
    """
    Derive total charge and spin multiplicity from the RDKit mol rather than
    assuming a neutral closed-shell species.
    """
    charge = Chem.GetFormalCharge(mol)
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = n_radical + 1

    # An even electron count needs an odd multiplicity and vice versa. If this
    # trips, the SMILES is wrong and every energy from this molecule would be
    # meaningless -- better to see it now than after the batch finishes.
    n_electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    if (n_electrons % 2) == (multiplicity % 2):
        print(f"  [WARNING] {n_electrons} electrons is inconsistent with "
              f"charge {charge:+d} / multiplicity {multiplicity}. Check the SMILES.")

    return charge, multiplicity


# -----------------------------------------------------------------------------
# ORCA input / execution
# -----------------------------------------------------------------------------

def write_orca_input(mol, conf_id, filename, engine, charge=0, multiplicity=1):
    """Write one ORCA optimization input directly from an RDKit conformer."""
    conf = mol.GetConformer(conf_id)

    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append(f"  {atom.GetSymbol():<2} {pos.x:>10.5f} {pos.y:>10.5f} {pos.z:>10.5f}")

    geometry = "\n".join([f"* xyz {charge} {multiplicity}", *coords, "*"])
    sections = [engine.keywords, engine.blocks, geometry]
    input_text = "\n\n".join(s for s in sections if s) + "\n"

    with open(filename, "w") as f:
        f.write(input_text)


def run_orca(input_file, orca_binary, omp_threads=1):
    """
    Run a single ORCA job. The job runs inside its own directory with a bare
    basename, which keeps paths short and scratch files local.
    """
    job_dir = os.path.dirname(os.path.abspath(input_file))
    basename = os.path.basename(input_file)
    output_file = os.path.splitext(input_file)[0] + ".out"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["MKL_NUM_THREADS"] = str(omp_threads)

    with open(output_file, "w") as out:
        result = subprocess.run([orca_binary, basename], cwd=job_dir,
                                stdout=out, stderr=subprocess.STDOUT, env=env)

    return input_file, result.returncode == 0


def run_orca_batch(input_files, orca_binary, max_workers, omp_threads=1):
    """Run ORCA jobs concurrently and report how many failed."""
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_orca, f, orca_binary, omp_threads)
                   for f in input_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                path, ok = future.result()
            except Exception as exc:
                failures += 1
                print(f"  [ERROR] ORCA job raised an exception: {exc}")
                continue
            if not ok:
                failures += 1

    if failures:
        print(f"  [WARNING] {failures}/{len(input_files)} ORCA jobs exited with an error.")
    return failures


# -----------------------------------------------------------------------------
# Ensemble assembly
# -----------------------------------------------------------------------------

def collect_optimized_conformers(job_dir):
    """
    Parse the finished ORCA jobs and return [(energy, xyz_block)] sorted by
    energy. The xyz comment line is replaced by the energy, which is what
    cregen reads.
    """
    entries = []

    out_files = sorted(glob.glob(os.path.join(job_dir, "*.out")))
    if not out_files:
        print("  Error: no ORCA output files found.")
        return entries

    for out_file in out_files:
        xyz_file = os.path.splitext(out_file)[0] + ".xyz"

        energy = None
        with open(out_file, "r", errors="replace") as f:
            for line in f:
                if "FINAL SINGLE POINT ENERGY" in line:
                    try:
                        energy = float(line.split()[-1])
                    except ValueError:
                        energy = None

        if energy is None or not os.path.exists(xyz_file):
            continue

        with open(xyz_file, "r") as f:
            lines = f.readlines()
        if len(lines) < 3:
            continue

        lines[1] = f"{energy:.8f}\n"
        entries.append((energy, "".join(lines)))

    entries.sort(key=lambda item: item[0])
    print(f"  Collated {len(entries)}/{len(out_files)} valid conformers.")
    return entries


def write_ensemble(entries, path):
    with open(path, "w") as f:
        for _, block in entries:
            f.write(block)


def run_cregen(reference_xyz, ensemble_xyz, output_xyz, crest_binary):
    """Deduplicate and sort the ensemble with CREST's cregen."""
    print("  Running CREST cregen to filter duplicates...")

    work_dir = os.path.dirname(os.path.abspath(ensemble_xyz))
    cmd = [crest_binary, os.path.abspath(reference_xyz),
           "--cregen", os.path.abspath(ensemble_xyz)]

    result = subprocess.run(cmd, cwd=work_dir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        print("  Error: CREST cregen failed. Last lines of its output:")
        for line in result.stdout.strip().splitlines()[-5:]:
            print(f"    {line}")
        return False

    candidates = [
        os.path.join(work_dir, f"{os.path.basename(ensemble_xyz)}.sorted"),
        os.path.join(work_dir, "crest_ensemble.xyz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            shutil.move(candidate, output_xyz)
            return True

    print("  Error: cregen finished but produced no sorted ensemble.")
    return False


def extract_top_conformers(unique_xyz_path, results_dir, base_name, top_n=5):
    """Copy the full unique ensemble and split out the lowest-energy structures."""
    print(f"  Extracting top {top_n} conformers...")

    shutil.copy(unique_xyz_path, os.path.join(results_dir, f"{base_name}_unique_conformers.xyz"))

    extracted = []
    with open(unique_xyz_path, "r") as f:
        for i in range(top_n):
            first_line = f.readline()
            if not first_line.strip():
                break

            num_atoms = int(first_line.strip())
            block = [first_line] + [f.readline() for _ in range(num_atoms + 1)]

            out_path = os.path.join(results_dir, f"{base_name}_c{i}.xyz")
            with open(out_path, "w") as out_f:
                out_f.writelines(block)
            extracted.append(out_path)

    return extracted


# -----------------------------------------------------------------------------
# Per-molecule driver
# -----------------------------------------------------------------------------

@dataclass
class Settings:
    engine: Engine
    orca_binary: str
    crest_binary: str
    workspace: str
    results_dir: str
    n_conformers: int
    top_n: int
    threads: int
    omp_threads: int
    timeout: int
    mmff: bool
    keep_intermediates: bool
    dry_run: bool


def process_molecule(smiles, job_name, original_name, settings):
    """Run the full workflow for one molecule. Returns 'done', 'skipped', or 'failed'."""
    final_file = os.path.join(settings.results_dir, f"{job_name}_unique_conformers.xyz")
    label = original_name if job_name == original_name else f"{original_name} (as {job_name})"

    if os.path.exists(final_file):
        print(f"\n--- Skipping: {label} - already processed ---")
        return "skipped"

    print(f"\n--- Processing: {label} ---")
    chem_dir = os.path.join(settings.workspace, job_name)
    job_dir = os.path.join(chem_dir, "orca_jobs")
    os.makedirs(job_dir, exist_ok=True)

    try:
        target_confs = (settings.n_conformers if settings.n_conformers > 0
                        else get_heuristic_conformer_count(smiles))
        print(f"  Targeting {target_confs} initial conformers.")

        mol = generate_conformers_with_timeout(smiles, num_confs=target_confs,
                                               pre_optimize=settings.mmff,
                                               timeout=settings.timeout)
        if mol is None:
            print("  Error: conformer generation produced nothing.")
            return "failed"

        charge, multiplicity = get_charge_and_multiplicity(mol)
        print(f"  Charge {charge:+d}, multiplicity {multiplicity}.")

        print(f"  Writing ORCA inputs ({settings.engine.name})...")
        input_files = []
        for conf_id in range(mol.GetNumConformers()):
            inp_file = os.path.join(job_dir, f"conf_{conf_id}.inp")
            write_orca_input(mol, conf_id, inp_file, settings.engine,
                             charge=charge, multiplicity=multiplicity)
            input_files.append(inp_file)

        if settings.dry_run:
            print(f"  [DRY RUN] Wrote {len(input_files)} inputs to {job_dir}; "
                  f"stopping before ORCA.")
            return "done"

        print(f"  Running {len(input_files)} ORCA jobs on {settings.threads} workers...")
        run_orca_batch(input_files, settings.orca_binary, settings.threads,
                       settings.omp_threads)

        entries = collect_optimized_conformers(job_dir)
        if not entries:
            print("  Error: no conformer survived optimization.")
            return "failed"

        ensemble_xyz = os.path.join(chem_dir, "all_optimized_conformers.xyz")
        write_ensemble(entries, ensemble_xyz)

        # The lowest-energy structure is the topology reference for cregen;
        # picking an arbitrary file risks referencing a broken geometry.
        reference_xyz = os.path.join(chem_dir, "reference.xyz")
        write_ensemble(entries[:1], reference_xyz)

        unique_xyz = os.path.join(chem_dir, f"{job_name}_unique_conformers.xyz")
        if not run_cregen(reference_xyz, ensemble_xyz, unique_xyz, settings.crest_binary):
            return "failed"

        extract_top_conformers(unique_xyz, settings.results_dir, job_name,
                               top_n=settings.top_n)
        return "done"

    except Exception as exc:
        print(f"  CRITICAL ERROR: processing failed ({exc})")
        traceback.print_exc()
        return "failed"

    finally:
        if not settings.keep_intermediates and not settings.dry_run:
            shutil.rmtree(job_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Input parsing and naming
# -----------------------------------------------------------------------------

def sanitize_name(name):
    """Filesystem-safe version of a name, kept as close to the original as possible."""
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))
    return cleaned.strip("._") or "unnamed"


def parse_smi_file(filepath):
    """
    Read a whitespace-separated .smi file. The first field is the SMILES; any
    remaining fields are joined into the name. Lines with only a SMILES get an
    automatic name.
    """
    chemicals = []
    for line_num, raw in enumerate(open(filepath, "r"), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        smiles = parts[0]
        name = "_".join(parts[1:]) if len(parts) > 1 else f"mol_{len(chemicals) + 1:04d}"

        if Chem.MolFromSmiles(smiles) is None:
            print(f"  [WARNING] Line {line_num}: RDKit cannot parse '{smiles}'. Skipping.")
            continue

        chemicals.append((smiles, name))

    return chemicals


def assign_job_names(chemicals, rename=False, max_length=60):
    """
    Map each molecule to a filesystem-safe, unique job name.

    Long names are the practical problem here: they become long directory and
    scratch-file paths, which ORCA and some filesystems reject. Anything over
    max_length falls back to a mol_NNNN tag, and --rename applies that style to
    every entry.
    """
    jobs = []
    seen = {}

    for index, (smiles, original_name) in enumerate(chemicals, 1):
        if rename:
            job_name = f"mol_{index:04d}"
        else:
            job_name = sanitize_name(original_name)
            if len(job_name) > max_length:
                job_name = f"mol_{index:04d}"
                print(f"  [INFO] '{original_name}' exceeds {max_length} characters "
                      f"-> using '{job_name}'.")

        # A repeated name would collide on disk and silently trip the
        # "already processed" skip, so disambiguate and say so loudly.
        if job_name in seen:
            seen[job_name] += 1
            job_name = f"{job_name}__dup{seen[job_name]}"
            print(f"  [WARNING] Duplicate name '{original_name}' -> using '{job_name}'.")
        else:
            seen[job_name] = 0

        jobs.append((smiles, original_name, job_name))

    return jobs


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def build_parser():
    engine_list = ", ".join(f"{k} ({v['description']})" for k, v in ENGINE_SPECS.items())
    parser = argparse.ArgumentParser(
        description="RDEO - batch conformer generation with external optimization.",
        epilog=f"Engines: {engine_list}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="Path to the .smi file (SMILES [name] per line)")
    parser.add_argument("-e", "--engine", default=None,
                        choices=sorted(ENGINE_SPECS),
                        help=f'Optimization engine (default: CONFIG["engine"] = {CONFIG["engine"]})')
    parser.add_argument("--client", default=None,
                        help="Override the MLIP client path for this run")
    parser.add_argument("--client-params", default=None,
                        help="Override the MLIP Ext_Params string for this run")
    parser.add_argument("-n", "--conformers", type=int, default=0,
                        help="Conformers to generate (0 = rotatable-bond heuristic)")
    parser.add_argument("-c", "--top", type=int, default=5,
                        help="Unique conformers written per molecule (default: 5)")
    parser.add_argument("-t", "--threads", type=int, default=8,
                        help="Parallel ORCA jobs (default: 8)")
    parser.add_argument("-w", "--workspace", default="batch_results",
                        help="Scratch directory (default: batch_results)")
    parser.add_argument("-r", "--results-dir", default="Results",
                        help="Output directory (default: Results)")
    parser.add_argument("-k", "--keep-intermediates", action="store_true",
                        help="Keep per-conformer ORCA jobs instead of deleting them")
    parser.add_argument("--rename", action="store_true",
                        help="Use mol_0001-style job names instead of the .smi names")
    parser.add_argument("--no-mmff", action="store_true",
                        help="Skip the MMFF94 pre-optimization")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Seconds allowed for RDKit embedding per molecule (default: 120)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write ORCA inputs and stop, without running anything")
    parser.add_argument("--version", action="version", version=f"RDEO {__version__}")
    return parser


def main():
    args = build_parser().parse_args()

    config = dict(CONFIG)
    engine_name = args.engine or config["engine"]

    # CLI overrides for the MLIP client, so a path can be tested without editing
    # the file.
    if args.client:
        config[f"{engine_name}_client"] = args.client
    if args.client_params:
        config[f"{engine_name}_params"] = args.client_params

    if not os.path.isfile(args.input_file):
        sys.exit(f"[ERROR] Input file not found: {args.input_file}")

    # Fail on a broken configuration before generating a single conformer.
    try:
        engine = build_engine(engine_name, config, dry_run=args.dry_run)
        if args.dry_run:
            orca_binary = config["orca_binary"]
            crest_binary = config["crest_binary"]
            for key in ("orca_binary", "crest_binary"):
                if shutil.which(config[key]) is None:
                    print(f'  [WARNING] CONFIG["{key}"] = "{config[key]}" not found '
                          f"(ignored for --dry-run).")
        else:
            orca_binary = resolve_executable("orca_binary", config["orca_binary"],
                                             "the optimization step")
            crest_binary = resolve_executable("crest_binary", config["crest_binary"],
                                              "the cregen deduplication step")
    except ConfigError as exc:
        sys.exit(f"\n[CONFIG ERROR] {exc}\n")

    chemicals = parse_smi_file(args.input_file)
    if not chemicals:
        sys.exit(f"[ERROR] No valid molecules found in {args.input_file}")

    os.makedirs(args.workspace, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    jobs = assign_job_names(chemicals, rename=args.rename,
                            max_length=int(config["max_name_length"]))

    mapping_path = os.path.join(args.workspace, "name_mapping.txt")
    with open(mapping_path, "w") as f:
        f.write("Job_Name\tOriginal_Name\tSMILES\n")
        for smiles, original_name, job_name in jobs:
            f.write(f"{job_name}\t{original_name}\t{smiles}\n")

    settings = Settings(
        engine=engine,
        orca_binary=orca_binary,
        crest_binary=crest_binary,
        workspace=args.workspace,
        results_dir=args.results_dir,
        n_conformers=args.conformers,
        top_n=args.top,
        threads=args.threads,
        omp_threads=int(config["omp_threads_per_job"]),
        timeout=args.timeout,
        mmff=not args.no_mmff,
        keep_intermediates=args.keep_intermediates,
        dry_run=args.dry_run,
    )

    print(f"RDEO {__version__}")                                               
    print(f"  Engine     : {engine.name} ({ENGINE_SPECS[engine.name]['description']})")
    print(f"  Molecules  : {len(jobs)} from {args.input_file}")
    print(f"  Workspace  : {os.path.abspath(args.workspace)}")
    print(f"  Results    : {os.path.abspath(args.results_dir)}")
    print(f"  Name map   : {mapping_path}")
    if args.dry_run:
        print("  Mode       : dry run (inputs only)")

    tally = {"done": 0, "skipped": 0, "failed": 0}
    for smiles, original_name, job_name in jobs:
        tally[process_molecule(smiles, job_name, original_name, settings)] += 1

    print(f"\n=== Finished: {tally['done']} completed, {tally['skipped']} skipped, "
          f"{tally['failed']} failed ===")

    return 1 if tally["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
