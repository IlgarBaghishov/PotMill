import unittest

from potmill.tools import (
    seq_to_string, rcuts_to_string, nmaxes_to_string, twojmaxes_to_string,
    create_rcut_range, create_nmax_range, create_lmax_range, create_twojmax_range,
    create_eweight_range, combined_ace_hyperparameters, combined_snap_hyperparameters,
    interpret_string, hyperparameters_to_string,
)


class TestStringHelpers(unittest.TestCase):
    def test_scalar_and_list(self):
        self.assertEqual(seq_to_string(5), "5")
        self.assertEqual(seq_to_string(5.5), "5.5")
        self.assertEqual(seq_to_string([1, 2, 3]), "1 2 3")
        self.assertEqual(seq_to_string([1, 2], delimiter="_"), "1_2")

    def test_aliases_are_seq_to_string(self):
        self.assertIs(rcuts_to_string, seq_to_string)
        self.assertIs(nmaxes_to_string, seq_to_string)
        self.assertIs(twojmaxes_to_string, seq_to_string)


class TestRanges(unittest.TestCase):
    def test_int_ranges(self):
        self.assertEqual(create_nmax_range(5, 7), [[5], [6], [7]])
        self.assertEqual(create_lmax_range(0, 2), [[0], [1], [2]])
        self.assertEqual(create_twojmax_range(4, 6), [[4], [5], [6]])

    def test_nmax_uses_cartesian_product(self):
        # two ranks: [5,6] x [2,3] -> 4 combinations
        self.assertEqual(
            create_nmax_range([5, 2], [6, 3]),
            [[5, 2], [5, 3], [6, 2], [6, 3]],
        )

    def test_twojmax_uses_elementwise_zip(self):
        # two ranks combined element-wise (vstack().T), NOT cartesian
        self.assertEqual(
            create_twojmax_range([4, 6], [5, 7]),
            [[4, 6], [5, 7]],
        )

    def test_rcut_range_scalar(self):
        r = create_rcut_range(5.0, 6.0, 3)
        self.assertEqual(r, [[5.0], [5.5], [6.0]])

    def test_rcut_range_list_is_product(self):
        r = create_rcut_range([5.0, 5.0], [6.0, 6.0], [2, 2])
        self.assertEqual(len(r), 4)
        self.assertIn([5.0, 5.0], r)
        self.assertIn([6.0, 6.0], r)

    def test_eweight_range(self):
        self.assertEqual(create_eweight_range(10, 5), [2.5, 5.0, 10.0, 20.0, 40.0])


class TestCombinedHyperparameters(unittest.TestCase):
    def _ace_config(self):
        return {
            "RCUT": {"min_rcut": 5.0, "max_rcut": 6.0, "num_rcut": 2},
            "NMAX": {"min_nmax": 5, "max_nmax": 6},
            "LMAX": {"min_lmax": 0, "max_lmax": 1},
            "EWEIGHT": {"middle_eweight": 10, "num_eweights": 3},
        }

    def test_ace_counts(self):
        cfg = self._ace_config()
        with_e = combined_ace_hyperparameters(cfg)
        no_e = combined_ace_hyperparameters(cfg, w_eweight=False)
        # 2 rcut * 2 nmax * 2 lmax = 8 subsets; *3 eweights = 24
        self.assertEqual(len(no_e), 8)
        self.assertEqual(len(with_e), 24)
        self.assertEqual(len(with_e[0]), 4)
        self.assertEqual(len(no_e[0]), 3)

    def test_snap_counts(self):
        cfg = {
            "RCUT": {"min_rcut": 5.0, "max_rcut": 6.0, "num_rcut": 2},
            "TWOJMAX": {"min_twojmax": 4, "max_twojmax": 6},
            "EWEIGHT": {"middle_eweight": 10, "num_eweights": 2},
        }
        with_e = combined_snap_hyperparameters(cfg)
        no_e = combined_snap_hyperparameters(cfg, w_eweight=False)
        # 2 rcut * 3 twojmax = 6 subsets; *2 eweights = 12
        self.assertEqual(len(no_e), 6)
        self.assertEqual(len(with_e), 12)


class TestInterpretString(unittest.TestCase):
    def test_types(self):
        self.assertEqual(interpret_string("5"), 5)
        self.assertEqual(interpret_string("5.5"), 5.5)
        self.assertEqual(interpret_string("5 6 7"), [5, 6, 7])
        self.assertEqual(interpret_string("ACE"), "ACE")


class TestHyperparametersToString(unittest.TestCase):
    def test_ace_and_snap(self):
        ace = hyperparameters_to_string("ACE", [[5.0], [5, 2], [0, 1], 10.0], delimiter="_")
        self.assertEqual(ace, "5.0_5_2_0_1_10.0")
        snap = hyperparameters_to_string("SNAP", [[5.0], [6], 10.0], delimiter="_")
        self.assertEqual(snap, "5.0_6_10.0")


if __name__ == "__main__":
    unittest.main()
