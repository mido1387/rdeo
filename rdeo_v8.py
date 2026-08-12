import os
import glob
import subprocess
import argparse
import concurrent.futures
import shutil
import traceback
import multiprocessing
import queue
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

def get_heuristic_conformer_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 200 
    
    try:
        temp = Chem.RemoveHs(mol)
    except Exception:
        temp = mol
        
    try:
        number_of_rotable_bonds = rdMolDescriptors.CalcNumRotatableBonds(temp)
    except Exception:
        number_of_rotable_bonds = 8 

    if number_of_rotable_bonds <= 7:
        return 50
    elif 8 <= number_of_rotable_bonds <= 12:
        return 200
    else:
        return 300

def _conformer_worker(smiles, num_confs, pre_optimize, out_queue):
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

def generate_conformers_with_timeout(smiles, num_confs=500, pre_optimize=True, timeout=120):
    ctx = multiprocessing.get_context('spawn')
    out_queue = ctx.Queue()
    
    p = ctx.Process(target=_conformer_worker, args=(smiles, num_confs, pre_optimize, out_queue))
    p.start()
    
    try:
        binary_mol = out_queue.get(timeout=timeout)
        p.join() 
        
        if binary_mol is not None:
            return Chem.Mol(binary_mol)
        return None
        
    except queue.Empty:
        print(f"\n  [TIMEOUT] RDKit hung for over {timeout} seconds. Terminating task...")
        p.terminate()
        p.join()
        return None

def get_charge_and_multiplicity(mol):
    """
    Derive the total charge and spin multiplicity from the RDKit mol rather
    than assuming a neutral closed-shell species.
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

def write_orca_extopt_input(mol, conf_id, filename, charge=0, multiplicity=1):
    """Phase 1: Bare optimization input directly from RDKit"""
    conf = mol.GetConformer(conf_id)
    
    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        coords.append(f"  {symbol:<2} {pos.x:>8.5f} {pos.y:>8.5f} {pos.z:>8.5f}")
        
    input_text = f"""! XTB Opt

%xtb
  XTBINPUTSTRING  "--gxtb" 
end

* xyz {charge} {multiplicity}
{chr(10).join(coords)}
*
"""
    with open(filename, 'w') as f:
        f.write(input_text)

def run_orca(input_file, orca_path="orca"):
    """Runs standard bare optimization jobs."""
    output_file = input_file.replace(".inp", ".out")
    with open(output_file, 'w') as out:
        subprocess.run([orca_path, input_file], stdout=out, stderr=subprocess.STDOUT)
    return output_file

def collate_and_sort_ensemble(job_dir, combined_xyz_path):
    print(f"  Parsing ORCA outputs in {job_dir}...")
    ensemble_data = []
    
    out_files = glob.glob(os.path.join(job_dir, "*.out"))
    if not out_files:
        print("  Error: No output files found to collate.")
        return False
        
    for out_file in out_files:
        # Skip solvator intermediate outputs if they exist
        if out_file.endswith("_solv.out"):
            continue
            
        base_path = os.path.splitext(out_file)[0]
        xyz_file = f"{base_path}.xyz"
        
        energy = None
        with open(out_file, 'r') as f:
            for line in f:
                if "FINAL SINGLE POINT ENERGY" in line:
                    energy = float(line.split()[-1])
                    
        if energy is None or not os.path.exists(xyz_file):
            continue
            
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
            
        lines[1] = f"{energy:.8f}\n"
        ensemble_data.append((energy, "".join(lines)))
        
    if not ensemble_data:
        print("  Error: No successful conformers found to collate.")
        return False
        
    ensemble_data.sort(key=lambda x: x[0])
    
    with open(combined_xyz_path, 'w') as f:
        for _, xyz_block in ensemble_data:
            f.write(xyz_block)
            
    print(f"  Collated {len(ensemble_data)} valid conformers.")
    return True

def run_cregen(job_dir, ensemble_xyz, output_xyz):
    print("  Running CREST cregen to filter duplicates...")
    
    reference_xyz = None
    # Find any standard or opt xyz file as a topological reference
    for out_file in glob.glob(os.path.join(job_dir, "*.xyz")):
        if "solvator" not in out_file and "ensemble" not in out_file:
            reference_xyz = out_file
            break
            
    if not reference_xyz:
        print("  Error: Could not find reference topology for CREST.")
        return False
        
    abs_reference = os.path.abspath(reference_xyz)
    abs_ensemble = os.path.abspath(ensemble_xyz)
        
    cmd = ["crest", abs_reference, "--cregen", abs_ensemble]
    work_dir = os.path.dirname(ensemble_xyz)
    
    result = subprocess.run(cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    if result.returncode != 0:
        print("  CREST cregen failed!")
        return False
    else:
        crest_sorted_file = os.path.join(work_dir, f"{os.path.basename(ensemble_xyz)}.sorted")
        if os.path.exists(crest_sorted_file):
            os.rename(crest_sorted_file, output_xyz)
            return True
        return False

def extract_top_conformers(unique_xyz_path, results_dir, base_name, top_n=10):
    """Original, clean extraction for the bare pipeline."""
    print(f"  Extracting top {top_n} bare conformers...")
    
    final_full_path = os.path.join(results_dir, f"{base_name}_unique_conformers.xyz")
    shutil.copy(unique_xyz_path, final_full_path)
    
    extracted_files = []
    with open(unique_xyz_path, 'r') as f:
        for i in range(top_n):
            first_line = f.readline()
            if not first_line: break 
            
            num_atoms = int(first_line.strip())
            conformer_lines = [first_line]
            for _ in range(num_atoms + 1):
                conformer_lines.append(f.readline())
                
            out_filename = f"{base_name}_c{i}.xyz"
            out_filepath = os.path.join(results_dir, out_filename)
            
            with open(out_filepath, 'w') as out_f:
                out_f.writelines(conformer_lines)
            extracted_files.append(out_filepath)
            
    return extracted_files

def run_solvation_post_process(extracted_xyz_files, base_dir, results_dir, short_name, solvent, nsolv, max_workers, keep_intermediates, charge=0, multiplicity=1):
    """Phase 2: Takes the top bare conformers, adds solvent, and re-optimizes."""
    print(f"\n  --- Commencing Solvation Post-Process for {short_name} ---")
    
    solv_dir = os.path.join(base_dir, short_name, "solvated_jobs")
    os.makedirs(solv_dir, exist_ok=True)
    
    job_infos = []
    
    # 1. Write the Solvator Inputs
    for idx, xyz_path in enumerate(extracted_xyz_files):
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
            num_solute_atoms = int(lines[0].strip())
            coords = lines[2:] # Skip atom count and comment
            
        base_filename = os.path.join(solv_dir, f"solv_c{idx}")
        solv_inp = f"{base_filename}_solv.inp"
        solv_xyz = f"{base_filename}_solv.solvator.xyz"
        opt_inp = f"{base_filename}_opt.inp"
        
        # Solvator Step (Requires ALPB for docking calculation)
        solv_text = f"""! XTB ALPB({solvent})

%solvator
  nsolv {nsolv}
end

* xyz {charge} {multiplicity}
{''.join(coords)}*
"""
        with open(solv_inp, 'w') as f:
            f.write(solv_text)
            
        # Optimization Step (Bare gxtb on the complex, NO ALPB)
        opt_text = f"""! XTB Opt

%xtb
  XTBINPUTSTRING  "--gxtb"
end

* xyzfile {charge} {multiplicity} {os.path.basename(solv_xyz)}
"""
        with open(opt_inp, 'w') as f:
            f.write(opt_text)
            
        job_infos.append({"solv_inp": solv_inp, "opt_inp": opt_inp, "solv_xyz": solv_xyz})

    # 2. Run the two-step ORCA pipelines
    def run_solv_pipeline(job):
        solv_inp = job["solv_inp"]
        job_dir = os.path.dirname(solv_inp)
        
        # Step A: Solvator
        with open(solv_inp.replace(".inp", ".out"), 'w') as out:
            subprocess.run(["orca", os.path.basename(solv_inp)], cwd=job_dir, stdout=out, stderr=subprocess.STDOUT)
            
        if not os.path.exists(job["solv_xyz"]): return None
        
        # Step B: Opt
        opt_inp = job["opt_inp"]
        with open(opt_inp.replace(".inp", ".out"), 'w') as out:
            subprocess.run(["orca", os.path.basename(opt_inp)], cwd=job_dir, stdout=out, stderr=subprocess.STDOUT)
        return opt_inp

    print(f"  Running {len(job_infos)} explicit solvation jobs...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_solv_pipeline, job_infos)
        
    # 3. Collate, Cregen, and Extract Solvated Complex
    combined_xyz = os.path.join(solv_dir, "all_solvated_conformers.xyz")
    if collate_and_sort_ensemble(solv_dir, combined_xyz):
        final_solv_xyz = os.path.join(solv_dir, f"{short_name}_solvated_unique.xyz")
        if run_cregen(solv_dir, combined_xyz, final_solv_xyz):
            
            print(f"  Extracting final solvated and stripped conformers...")
            shutil.copy(final_solv_xyz, os.path.join(results_dir, f"{short_name}_solvated_ensemble.xyz"))
            
            with open(final_solv_xyz, 'r') as f:
                for i in range(len(extracted_xyz_files)): # Extract up to original amount
                    first_line = f.readline()
                    if not first_line: break
                    
                    num_atoms_total = int(first_line.strip())
                    comment_line = f.readline()
                    atom_lines = [f.readline() for _ in range(num_atoms_total)]
                    
                    # Save Full Solvated Complex
                    solv_out = os.path.join(results_dir, f"{short_name}_solvated_c{i}.xyz")
                    with open(solv_out, 'w') as out_f:
                        out_f.write(f"{num_atoms_total}\n{comment_line}")
                        out_f.writelines(atom_lines)
                        
                    # Save Stripped Version
                    strip_out = os.path.join(results_dir, f"{short_name}_stripped_c{i}.xyz")
                    with open(strip_out, 'w') as out_f:
                        out_f.write(f"{num_solute_atoms}\n{comment_line.strip()} (Stripped)\n")
                        out_f.writelines(atom_lines[:num_solute_atoms])
                        
    if not keep_intermediates:
        shutil.rmtree(solv_dir)

def process_chemical(smiles, short_name, original_name, base_dir, results_dir, num_conformers, max_workers, keep_intermediates, solvent, nsolv):
    final_expected_file = os.path.join(results_dir, f"{short_name}_unique_conformers.xyz")
    label = original_name if short_name == original_name else f"{original_name} (as {short_name})"

    if os.path.exists(final_expected_file):
        print(f"\n--- Skipping: {label} - Already processed! ---")
        return

    print(f"\n--- Processing: {label} ---")
    chem_dir = os.path.join(base_dir, short_name)
    job_dir = os.path.join(chem_dir, "orca_extopt_jobs")
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # Phase 1: Standard Bare Workflow
        target_confs = num_conformers if num_conformers > 0 else get_heuristic_conformer_count(smiles)
        print(f"  Targeting {target_confs} initial bare conformers.")

        mol = generate_conformers_with_timeout(smiles, num_confs=target_confs)
        if mol is None: return

        charge, multiplicity = get_charge_and_multiplicity(mol)
        print(f"  Charge {charge:+d}, multiplicity {multiplicity}.")

        print("  Writing ORCA inputs...")
        input_files = []
        for conf_id in range(mol.GetNumConformers()):
            inp_file = os.path.join(job_dir, f"conf_{conf_id}.inp")
            write_orca_extopt_input(mol, conf_id, inp_file,
                                    charge=charge, multiplicity=multiplicity)
            input_files.append(inp_file)
            
        print(f"  Running {len(input_files)} bare ORCA jobs...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_orca, input_files)
            
        combined_xyz = os.path.join(chem_dir, "all_optimized_conformers.xyz")
        if collate_and_sort_ensemble(job_dir, combined_xyz):
            final_xyz = os.path.join(chem_dir, f"{short_name}_unique_conformers.xyz")
            if run_cregen(job_dir, combined_xyz, final_xyz):
                # Extract Top N (Default 5)
                top_bare_files = extract_top_conformers(final_xyz, results_dir, short_name, top_n=5)
                
                # Phase 2: Solvation Post-Process (Triggered only if nsolv > 0)
                if nsolv > 0 and top_bare_files:
                    run_solvation_post_process(top_bare_files, base_dir, results_dir, short_name, solvent, nsolv, max_workers, keep_intermediates, charge=charge, multiplicity=multiplicity)

    except Exception as e:
        print(f"  CRITICAL ERROR: Processing failed. Details: {e}")
        
    finally:
        if not keep_intermediates and os.path.exists(job_dir):
            shutil.rmtree(job_dir)

def sanitize_name(name):
    """
    Filesystem-safe version of the name column from the .smi file.
    Keeps the original name intact wherever possible so that neutral and anion
    jobs line up by name later; only characters that would break a path are
    replaced.
    """
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"

def parse_smi_file(filepath):
    chemicals = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 2:
                chemicals.append((parts[0], "_".join(parts[1:])))
    return chemicals

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Conformer & Solvation Workflow")
    parser.add_argument("input_file", help="Path to the .smi file")
    parser.add_argument("-n", "--conformers", type=int, default=0, help="Conformers to generate (0 = auto)")
    parser.add_argument("-t", "--threads", type=int, default=8, help="Parallel ORCA jobs")
    parser.add_argument("-w", "--workspace", type=str, default="batch_results")
    parser.add_argument("-r", "--results-dir", type=str, default="Results")
    parser.add_argument("-k", "--keep-intermediates", action="store_true")
    
    # Solvator Args
    parser.add_argument("-s", "--solvent", type=str, default="WATER", help="Solvent for ORCA SOLVATOR")
    parser.add_argument("-ns", "--nsolv", type=int, default=0, help="Explicit solvent molecules (0 = Bare Workflow Only)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file): return
        
    chemicals = parse_smi_file(args.input_file)
    os.makedirs(args.workspace, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    with open(os.path.join(args.workspace, "name_mapping.txt"), "w") as f:
        f.write("Job_Name\tOriginal_Name\tSMILES\n")
        seen = {}
        for smiles, original_name in chemicals:
            job_name = sanitize_name(original_name)

            # a repeated name would collide on disk and silently trip the
            # "already processed" skip, so disambiguate and say so loudly
            if job_name in seen:
                seen[job_name] += 1
                collided = job_name
                job_name = f"{job_name}__dup{seen[collided]}"
                print(f"  [WARNING] Duplicate name '{original_name}' -> using '{job_name}'")
            else:
                seen[job_name] = 0

            f.write(f"{job_name}\t{original_name}\t{smiles}\n")
            f.flush()
            process_chemical(smiles, job_name, original_name, args.workspace, args.results_dir, 
                             args.conformers, args.threads, args.keep_intermediates, args.solvent, args.nsolv)

if __name__ == "__main__":
    main()
