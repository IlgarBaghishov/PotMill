def max_entropy_atoms_iterator(structuregen_config):

    import os

    # Set threading environment BEFORE importing LAMMPS/JAX/numpy.
    # LAMMPS SNAP bispectrum computation uses OpenMP, and JAX/MKL/OpenBLAS
    # also respect these variables. Must be set before library import.
    n_threads = str(structuregen_config.get('n_threads', 1))
    os.environ['OMP_NUM_THREADS'] = n_threads
    os.environ['MKL_NUM_THREADS'] = n_threads
    os.environ['OPENBLAS_NUM_THREADS'] = n_threads

    # Configure JAX for CPU with 64-bit precision
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    from autopiad.structuregen.renorm import RandomEntropyInitializer
    from autopiad.structuregen.optimizer import EntropyMaximizer

    os.makedirs("renorm_configs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    rand_entropy = RandomEntropyInitializer(structuregen_config)
    rand_entropy.looping()

    entropy_maximizer = EntropyMaximizer(structuregen_config)
    first_index = [0]
    for entropy_atoms in entropy_maximizer.looping():
        n_atoms = len(entropy_atoms)
        first_index.append(first_index[-1] + 1 + 3 * n_atoms)
        yield entropy_atoms
