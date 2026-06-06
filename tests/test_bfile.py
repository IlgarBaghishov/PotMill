import os
import tempfile
import unittest

import numpy as np

from potmill.bfile import write_b, read_b


class TestBFile(unittest.TestCase):
    def test_roundtrip_and_format(self):
        job_id = 7
        n_atoms = 2
        energy = 12.0
        forces = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b")
            write_b(path, job_id, energy, n_atoms, forces)
            local_idx, jid, val = read_b(path)

        # 1 energy row + 3*n_atoms force rows
        self.assertEqual(len(local_idx), 1 + 3 * n_atoms)
        # local index resets per config: energy is row 0
        np.testing.assert_array_equal(local_idx, np.arange(0, 1 + 3 * n_atoms))
        # job_id constant
        np.testing.assert_array_equal(jid, np.full(1 + 3 * n_atoms, job_id))
        # energy stored per-atom
        self.assertAlmostEqual(val[0], energy / n_atoms)
        # forces flattened in row-major order
        np.testing.assert_allclose(val[1:], forces.ravel())

    def test_energy_row_is_index_zero(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b")
            write_b(path, 0, 5.0, 1, np.zeros((1, 3)))
            local_idx, _, _ = read_b(path)
        self.assertEqual(int(local_idx[0]), 0)


if __name__ == "__main__":
    unittest.main()
