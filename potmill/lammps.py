import os, traceback
from ase.io import read, write
from ase.calculators.lammpsrun import LAMMPS

from potmill.bfile import write_b


def lammps(start_path, input_file, job_id, first_index):

    os.environ['ASE_LAMMPSRUN_COMMAND'] = start_path+"run_lammps_ase.sh"

    atom_type_mapping = ["Be"]
    ace_file = start_path + "pot.yace"
    pair_coeff = ['* * ' + ace_file + ' ' + ' '.join(atom_type_mapping)]
    files = [ace_file]
    parameters = {'pair_style': 'pace', 'pair_coeff': pair_coeff}
    calc = LAMMPS(files=files, **parameters, keep_tmp_files=True, tmp_dir="lammps_temp", log_file="log.lammps")

    atoms = read(input_file, index=0, format='vasp')
    atoms.pbc = True
    atoms.calc = calc

    print("RUN DIRECTORY: ", os.getcwd(), " INPUT FILE: ", input_file, flush=True)

    #execute the calculation
    try:
        ener = atoms.get_potential_energy()
        forces = atoms.get_forces()

        write_b("b", job_id, ener, len(atoms), forces)

        #write the output in ASE traj format
        write("atoms_%i.traj" % job_id,images=atoms,format='traj')

        # #look into using Custodian here to do error detection/validation
    except Exception:
        print("Error while running LAMMPS or writing the output files", flush=True)
        traceback.print_exc()

    return job_id