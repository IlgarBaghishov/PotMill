# PotMill

Automated active-design pipeline for machine-learned interatomic potentials (MLIPs).

PotMill iteratively generates training data for MLIPs by maximizing information entropy in the
descriptor space, then labels, featurizes, fits, and Pareto-ranks candidate potentials — all
orchestrated on HPC clusters with [Flux](https://flux-framework.org/) and
[executorlib](https://github.com/pyiron/executorlib). The stages overlap via a futures-based
dynamic load balancer (see `CLAUDE.md` for the architecture).

## Pipeline stages

1. **Structure generation** — entropy maximization over SNAP bispectrum descriptors (`structuregen/`)
2. **Labeling** — energies/forces from a configurable backend (`labeling/`): UMA (fairchem), VASP, or LAMMPS
3. **Featurization** — ACE/SNAP descriptors via FitSNAP (`featurization/`)
4. **Fitting** — least-squares MLIP coefficients across a hyperparameter grid (`fitting/`)
5. **Pareto front & uncertainty** — accuracy-vs-cost ranking and POPSRegression intervals (`analysis/`, `fitting/`)

## Installation

PotMill depends on the Flux scheduler, LAMMPS (with MLIAP/SNAP/ML-PACE), and FitSNAP, which are
conda-only or built from source — so installation is conda-based, not `pip install potmill`.

```bash
# 1. Create the conda environment (Perlmutter example; adjust the CUDA override for your system)
CONDA_OVERRIDE_CUDA="12.9" mamba create -n potmill -c conda-forge \
    python=3.12 flux-core flux-sched executorlib cxx-compiler mpi4py "libhwloc=*=cuda*" \
    ase numpy scipy pandas spglib jax scikit-learn Cython
mamba activate potmill

# 2. Verify Flux works
srun -N 2 -n 2 flux start flux resource list

# 3. Labeling / structure-gen / uncertainty extras (pip)
pip install fairchem-core ase-ga POPSRegression mendeleev

# 4. Build LAMMPS (MLIAP/SNAP/ML-PACE) and FitSNAP per the FitSNAP installation guide.

# 5. Install PotMill itself into the environment
pip install -e .
```

## Running

From a working directory containing a `config.ini` and a `FitSNAP.in`:

```bash
srun -N $SLURM_NNODES -n $SLURM_NNODES flux start python -u -m potmill
```

**Always run on `$SCRATCH` (Lustre), not `$WORK` (CFS)** — see `CLAUDE.md` "Run directory placement".
After a run, plot the resource/stage monitor with:

```bash
python -m potmill.analysis.plot_monitor pipeline_monitor.csv
```

## Configuration

The pipeline reads `config.ini` (parsed by `potmill.config.ConfigManager`). Sections are of two kinds:

- **"our" sections** — PotMill's own parameters with documented defaults in `ConfigManager.DEFAULTS`
  (e.g. `[MAIN]`, `[RCUT]`, `[NMAX]`, `[EWEIGHT]`, `[STRUCTUREGEN]`, `[FitSNAP]`, `[ourLabeling]`).
  Unknown keys are warned about.
- **passthrough sections** — keyword arguments forwarded verbatim to external calculator classes
  (`[FAIRChemCalculator]`, `[Vasp]`, `[LAMMPS]`); omitted keys fall back to that library's defaults.

The labeling backend is selected by `[ourLabeling] calculator` (`FAIRChemCalculator`, `Vasp`, or
`LAMMPS`). Both labeling and fitting devices are configurable (`device` / `fit_device` = `cpu` or `cuda`).

See `examples/` for complete, runnable configs (`HBeW/ACE` is the multi-element UMA reference).

## Examples

| Example | Method | Labeling | Notes |
|---|---|---|---|
| `examples/HBeW/ACE` | multi_element | UMA | Ternary H-Be-W, the proven 100k reference run |
| `examples/WRe/ACE`, `WRe/SNAP` | binary | VASP | W-Re |
| `examples/Be/ACE`, `Be/SNAP` | binary | VASP | Single-element |

## License

BSD-3-Clause (see `LICENSE`).
