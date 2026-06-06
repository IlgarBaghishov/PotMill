"""The ``b`` file: the labeling -> fitting interchange format.

One CSV row per target value (no header), with columns ``(local_index, job_id, value)``:

* ``local_index`` -- 0 for the per-atom energy, then 1..3*n_atoms for the flattened forces.
  The fitting code identifies energy rows by ``local_index == 0``, so this must reset per
  configuration (it is NOT a global running index).
* ``job_id`` -- configuration id; keeps a config's energy and all its forces together so a
  configuration always lands on the same side of a k-fold split.
* ``value`` -- energy-per-atom (row 0) or a single force component (rows 1..).
"""

import numpy as np
import pandas as pd


def write_b(path, job_id, energy, n_atoms, forces):
    """Write one configuration's targets to ``path`` (see module docstring)."""
    forces = np.asarray(forces).ravel()
    rows = np.vstack([
        np.arange(0, 1 + forces.size),
        np.full(1 + forces.size, job_id),
        np.concatenate([[energy / n_atoms], forces]),
    ]).T
    np.savetxt(path, rows, delimiter=",", fmt=["%i", "%i", "%.10f"])


def read_b(path):
    """Read a ``b`` file into ``(local_index, job_id, value)`` numpy arrays."""
    df = pd.read_csv(path, header=None)
    return df[0].values, df[1].values, df[2].values
