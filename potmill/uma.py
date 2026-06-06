import os
from ase import Atoms
from ase.io import read, write

from potmill.bfile import write_b


def init_uma_calculator():
    """executorlib init_function: pre-load UMA calculator once per GPU worker."""
    from fairchem.core import FAIRChemCalculator
    calc = FAIRChemCalculator.from_model_checkpoint("uma-m-1p1", task_name="omat", device="cuda")
    return {"calc": calc}


def init_uma_predictor():
    """executorlib init_function: load UMA model once per GPU worker (for batched inference
    via uma_batch -- avoids ASE calculator's per-config overhead by reusing the predict_unit
    across many configs in one .predict(batch) call)."""
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    predictor = get_predict_unit("uma-m-1p1", device="cuda")
    return {"predictor": predictor}


def uma(start_path, input_file, job_id, first_index, dirpath, calc):
    os.chdir(dirpath)
    if isinstance(input_file, Atoms):
        atoms = input_file
    else:
        atoms = read(start_path+input_file, index=0, format='vasp')
    atoms.pbc = True
    atoms.calc = calc

    ener = atoms.get_potential_energy()
    forces = atoms.get_forces()

    write_b("b", job_id, ener, len(atoms), forces)

    write(f"atoms_{job_id}.traj", images=atoms, format='traj')

    atoms.calc = None
    return {"job_ID":job_id, "atoms":atoms}


def uma_batch(start_path, atoms_list, job_ids, labeling_dir, predictor):
    """Batch inference: process N structures in a single GPU forward pass. UMA's forward
    has a large fixed per-call overhead (~160 ms on A100) and only ~1 ms / atom of compute,
    so batches of 16-32 amortize the overhead 10x+. Returns a LIST of N result dicts."""
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    # If items are tagged dicts ({"atoms":..., "job_id":...}) and job_ids is None, extract.
    if job_ids is None:
        job_ids = [item["job_id"] if isinstance(item, dict) else None for item in atoms_list]

    resolved = []
    data_list = []
    for item in atoms_list:
        atoms = item if isinstance(item, Atoms) else item["atoms"]
        atoms.pbc = True
        atoms.calc = None
        data_list.append(AtomicData.from_ase(atoms, task_name="omat"))
        resolved.append(atoms)

    batch = atomicdata_list_to_batch(data_list)
    preds = predictor.predict(batch)

    energies = preds["energy"].detach().cpu().numpy()
    forces = preds["forces"].detach().cpu().numpy()
    natoms_list = [len(a) for a in resolved]

    results = []
    force_offset = 0
    for i, (atoms, job_id) in enumerate(zip(resolved, job_ids)):
        n_atoms = natoms_list[i]
        ener = float(energies[i])
        f = forces[force_offset:force_offset + n_atoms].ravel()
        force_offset += n_atoms

        dirpath = f"{labeling_dir}/{job_id}/"
        os.makedirs(dirpath, exist_ok=True)
        write_b(f"{dirpath}/b", job_id, ener, n_atoms, f)
        write(f"{dirpath}/atoms_{job_id}.traj", images=atoms, format='traj')
        atoms.calc = None
        results.append({"job_ID": job_id, "atoms": atoms})
    return results
