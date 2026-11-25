import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import kvrank


class T(unittest.TestCase):
    def test_gram_matches_svd(self):
        rng = np.random.default_rng(1)
        W = rng.standard_normal((300, 128)) @ np.diag(np.linspace(3, 0.01, 128))
        s = np.linalg.svd(W, compute_uv=False) ** 2
        c = np.cumsum(s) / s.sum()
        st = kvrank.energy_stats(kvrank.gram(W))
        self.assertEqual(st["rank95"], int(np.searchsorted(c, 0.95) + 1))
        self.assertAlmostEqual(st["kept32"], c[31], places=8)

    def test_exact_low_rank(self):
        rng = np.random.default_rng(2)
        Wk, Wv = (rng.standard_normal((64, 8)) @ rng.standard_normal((8, 64)) for _ in range(2))
        st = kvrank.layer_stats(Wk, Wv)
        self.assertEqual(st["kv"]["rank99"], 16)  # joint rank <= 8 + 8 (sum of two rank-8 row spaces)
        self.assertLessEqual(st["k"]["rank99"], 8)


if __name__ == "__main__":
    unittest.main()
