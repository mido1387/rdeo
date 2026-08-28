RDEO — RDKit External Optimization

                                                    
        ║│▓▓▓▓▓   ║│▓▓▓    ║│▓▓▓▓▓   ║│▓▓▓          
        ║│     ▓  ║│   ▓   ║│       ║│    ▓         
        ║│     ▓  ║│    ▓  ║│       ║│    ▓         
        ║│▓▓▓▓    ║│    ▓  ║│▓▓     ║│    ▓         
        ║│    ▓   ║│   ▓   ║│       ║│    ▓         
        ║│     ▓  ║│▓▓▓    ║│▓▓▓▓▓   ║│▓▓▓          
                                                    
                     "YEEHAW" 
                        °o                       
            ()            {)        ()            
              \        c==//\      /              
               \      /   |_|     /               
                \   ,/,   //'    /                
                 ()══════||════()                 
                /        ~'      \                
               /                  \               
              /    ╔ MOLECULE ╗    \              
    ()------()     ╚ WRANGLER ╝     ()            

                                                    

A batch conformer workflow that takes a list of SMILES and returns a small set of
unique, energy-sorted conformers. Structure generation is handled by RDKit, the
geometry optimization is handed to ORCA, and duplicate removal is handled by
CREST's `cregen`.
The workflow was designed around g-xTB to balance speed and accuracy: the
resulting geometries closely resemble ωB97X-V optimized structures at a small
fraction of the cost. Any ORCA-accessible method can be used instead, including
machine-learned interatomic potentials (MLIPs) driven through ORCA's external
optimizer interface.
> **Status:** preliminary release / work in progress. Interfaces and defaults may
> still change.
---
Workflow
Embed — RDKit builds the molecule from SMILES and generates conformers
with ETKDGv3, followed by an optional MMFF94 pre-optimization.
Optimize — every conformer is optimized by ORCA using the selected engine
(g-xTB by default). Jobs run concurrently.
Deduplicate — the optimized ensemble is passed to CREST `cregen`, which
removes duplicates and sorts by energy.
Extract — the top N unique conformers (5 by default) are written to the
results directory, along with the full unique ensemble.
Charge and spin multiplicity are read from the SMILES, so anions, cations and
radicals are handled without extra flags. An inconsistent electron count is
flagged before the calculation runs.
The number of conformers generated per molecule scales with the rotatable-bond
count:
Rotatable bonds	Conformers
≤ 7	50
8–12	200
≥ 13	300
Heuristic based on (DOI to be added before release). Override with `-n`.
---
Requirements
Python 3.9+ with RDKit (`pip install rdkit`)
ORCA with the `XTB` and `EXTOPT` interfaces
CREST for `cregen`
For `gxtb`: an `xtb` build that accepts `--gxtb`, plus the g-xTB parameter files
For `uma` / `aimnet2`: the corresponding external-tool client executable
Tested with ORCA 6.x, CREST 3.x, RDKit 2024.03+.
---
Configuration
All machine-specific settings live in a single `CONFIG` block at the top of
`rdeo.py`. Nothing below that block needs to be edited for normal use, and any
required value that is missing produces an explicit error before a single
conformer is generated:
```
[CONFIG ERROR] Engine "uma" needs CONFIG["uma_client"] to be set.
    Edit the USER CONFIGURATION block at the top of this file, e.g.:
        "uma_client": "/home/you/env/bin/oet_client",
    Or override it for this run with --client.
```
Key	Purpose
`orca_binary`	ORCA executable — a bare name works if it is on `$PATH`
`crest_binary`	CREST executable, used for `cregen`
`engine`	Default optimization engine (see below)
`uma_client`	Path to the UMA external-tool client
`uma_params`	Flags passed through `Ext_Params`, e.g. `-d cuda`
`aimnet2_client`	Path to the AIMNet2 external-tool client
`aimnet2_params`	Flags passed through `Ext_Params`
`custom_keywords`	ORCA simple-input line for `--engine custom`
`custom_blocks`	Optional extra `%blocks` for `--engine custom`
`omp_threads_per_job`	OpenMP threads given to each ORCA job
`orca_nprocs`	`%pal nprocs` value; leave at 1 for xTB and MLIP engines
`max_name_length`	Names longer than this fall back to `mol_NNNN` tags
Use `--dry-run` to write the ORCA inputs and stop, which is the quickest way to
confirm a new configuration before committing a long batch.
---
Optimization engines
Select an engine with `-e/--engine`, or set a default in `CONFIG["engine"]`.
Engine	ORCA input generated	Notes
`gxtb`	`! XTB Opt` + `%xtb XTBINPUTSTRING "--gxtb"`	Default. Best speed/accuracy balance
`gfn2`	`! XTB2 Opt`	Cheap baseline, useful as a sanity check
`uma`	`! EXTOPT Opt` + `%method ProgExt ...`	Meta UMA MLIP; needs `uma_client`
`aimnet2`	`! EXTOPT Opt` + `%method ProgExt ...`	AIMNet2 MLIP; needs `aimnet2_client`
`custom`	Whatever `custom_keywords` defines	Any ORCA method, e.g. `! r2SCAN-3c Opt`
Because the optimization step is just an ORCA input, the workflow is not limited
to these five. Anything ORCA can optimize — a composite method, a DFT functional
with a solvation model, another external tool — works by setting
`custom_keywords` and `custom_blocks`, or by adding an entry to the
`ENGINE_SPECS` dictionary in the script.
A new MLIP normally needs only its client path:
```bash
python rdeo.py molecules.smi --engine uma --client /home/you/uma_env/bin/oet_client
```
---
Usage
```bash
python rdeo.py molecules.smi
python rdeo.py molecules.smi --engine uma --threads 16 --rename
python rdeo.py molecules.smi --engine custom --top 10 --keep-intermediates
```
Input format
A whitespace-separated `.smi` file — SMILES first, optional name after. Blank
lines and `#` comments are ignored, unparseable SMILES are reported and skipped,
and unnamed entries are given automatic `mol_NNNN` tags.
```
CCO                                   ethanol
OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F   PFPeA
[O-]C(=O)C(F)(F)C(F)(F)F              PFPrA_anion
```
Options
Flag	Default	Description
`-e`, `--engine`	from `CONFIG`	`gxtb`, `gfn2`, `uma`, `aimnet2`, `custom`
`--client`	from `CONFIG`	Override the MLIP client path for this run
`--client-params`	from `CONFIG`	Override the `Ext_Params` string for this run
`-n`, `--conformers`	`0` (heuristic)	Conformers to generate per molecule
`-c`, `--top`	`5`	Unique conformers written per molecule
`-t`, `--threads`	`8`	ORCA jobs run concurrently
`-w`, `--workspace`	`batch_results`	Scratch directory
`-r`, `--results-dir`	`Results`	Output directory
`-k`, `--keep-intermediates`	off	Keep per-conformer ORCA jobs
`--rename`	off	Use `mol_NNNN` job names instead of the `.smi` names
`--no-mmff`	off	Skip the MMFF94 pre-optimization
`--timeout`	`120`	Seconds allowed for RDKit embedding per molecule
`--dry-run`	off	Write ORCA inputs and stop
Naming and `--rename`
By default the names from the `.smi` file are reused as job names, with unsafe
characters replaced, so results stay readable. Long chemical names become long
directory and scratch-file paths, which ORCA and some filesystems reject; any
name over `max_name_length` characters automatically falls back to a `mol_NNNN`
tag, and `--rename` applies that style to everything.
Either way, `workspace/name_mapping.txt` records the job name, the original name
and the SMILES for every entry, so results can always be mapped back:
```
Job_Name    Original_Name    SMILES
ethanol     ethanol          CCO
mol_0006    perfluoro...     C(F)(F)...
```
---
Output
```
Results/
  <name>_unique_conformers.xyz   full deduplicated, energy-sorted ensemble
  <name>_c0.xyz                  lowest-energy conformer
  <name>_c1.xyz ...              subsequent conformers, up to --top

batch_results/
  name_mapping.txt               job name <-> original name <-> SMILES
  <name>/                        per-molecule scratch (ensembles, reference)
```
Energies (Hartree) are written to the comment line of each xyz block.
A molecule whose `<name>_unique_conformers.xyz` already exists in the results
directory is skipped, so an interrupted batch can simply be rerun.
---
Practical notes
Thread oversubscription. `--threads` controls how many ORCA jobs run at
once; each job gets `omp_threads_per_job` OpenMP threads. Leave the latter at
1 unless cores are spare, otherwise parallel xTB or MLIP jobs will fight over
the machine.
GPU MLIPs. Concurrent UMA/AIMNet2 jobs share one GPU. Reduce `--threads`
if memory becomes a limit.
Failed jobs. A conformer whose ORCA job fails is reported and dropped from
the ensemble rather than aborting the batch. The run exits non-zero if any
molecule failed outright.
---
Roadmap
Explicit solvation ("conformational chaperone") using ORCA's `%solvator`:
solvent molecules are docked before optimization and stripped afterwards, with
both bare and solvated geometries retained. Removed from this preliminary
release; to be reinstated once the workflow is documented.
Restart/resume across engines for direct method comparisons.
Packaging and a proper test suite.
---
Citation
To be added.
License
To be added.
