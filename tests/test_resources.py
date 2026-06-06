import os
import tempfile
import unittest

from potmill.config import ConfigManager
from potmill.resources import worker_layout


def _cfg(text=""):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.ini")
        with open(path, "w") as f:
            f.write(text)
        return ConfigManager(path)


class TestWorkerLayout(unittest.TestCase):
    def test_default_split(self):
        # 2 nodes, 128 cores, 8 GPUs -> 4 GPUs/node; defaults: fit_gpus_per_node=2, 1 entropy/featurize per node
        res = worker_layout(_cfg(), nnodes=2, ncores=128, ngpus=8)
        self.assertEqual(res.gpus_per_node, 4)
        self.assertEqual(res.n_fit_workers, 4)        # 2/node * 2 nodes
        self.assertEqual(res.n_label_workers, 4)      # (4-2)/node * 2 nodes
        self.assertEqual(res.n_entropy_workers, 2)    # 1/node * 2 nodes
        self.assertEqual(res.threads_per_worker, 16)  # 32 // 1
        self.assertEqual(res.n_featurize_workers, 2)

    def test_entropy_and_featurize_knobs(self):
        cfg = _cfg("[MAIN]\nfeaturize_workers_per_node = 5\n[STRUCTUREGEN]\nn_entropy_workers = 32\n")
        res = worker_layout(cfg, nnodes=4, ncores=128, ngpus=16)
        self.assertEqual(res.n_entropy_workers, 128)        # 32 * 4
        self.assertEqual(res.threads_per_worker, 1)         # max(1, 32 // 128)
        self.assertEqual(res.n_featurize_workers, 20)       # 5 * 4

    def test_too_many_fit_gpus_raises(self):
        cfg = _cfg("[MAIN]\nfit_gpus_per_node = 4\n")
        with self.assertRaises(AssertionError):
            worker_layout(cfg, nnodes=1, ncores=64, ngpus=4)  # fit_gpus == gpus_per_node


if __name__ == "__main__":
    unittest.main()
