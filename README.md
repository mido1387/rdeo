# rdeo
RDEO - (RDKit External Optimization)

RDEO takes an input list of chemical smiles and iterates through the following conformer sampling workflow. This workflow was designed to leverage the new g-xtb on release and achieve a balance of speed and accuracy. The results of this workflow resemble wB97X-v optimized geometries. 

Usage: python rdeo.py molecules.smi

Arguments:
-n, "--conformers", type=int, default=0, help="Conformers to generate (0 = auto)")
-t, "--threads", type=int, default=8, help="Parallel ORCA jobs")
-w, "--workspace", type=str, default="batch_results")
-r, "--results-dir", type=str, default="Results")
-k, "--keep-intermediates", action="store_true")
-s, "--solvent", type=str, default="WATER", help="Solvent for ORCA SOLVATOR")
-ns, "--nsolv", type=int, default=0, help="Explicit solvent molecules (0 = Bare Workflow Only)")

Note: 
The number of conformers automatically generated varies from 50 to 200 depending on the number of rotatable bonds based on this heuristic: add doi link here before release

Steps:
1. RDKit ETKDGv3 builds the SMILES and generates conformers
2. Conformers are optimized using ORCA's external optimizer. This is set to g-xtb right now, but other good options include MLIPs like UMA or AIMNet2
3. CREST's cregen function remvoes duplicate conformers and sorts conformers by the energies from step 2. Extracts the top 5 (by defualt, can be changed)

If explicit solvent is added, it is added before step 2, and the solvent is then removed after step 3 for the conformational chaperone workflow. The bare and solvated geometries are saved. This is skipped by default or with -ns 0
