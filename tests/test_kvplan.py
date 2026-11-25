import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import kvplan as k

class T(unittest.TestCase):
    def test_baseline_bytes_per_frame(self):
        # 32 layers x (K+V) x 32 heads x 128 dim x 2 B
        self.assertEqual(k.metrics(k.Design(), 24, 0, 0)["stored_pf"], 524288)
    def test_deepseek_mla_ratio(self):
        c = k.load("deepseek_v3_attn.json")
        mha, mla = k.kv_elems("mha", c), k.kv_elems("mla:512+64", c)
        self.assertEqual((mha, mla), (32768, 576))
        self.assertAlmostEqual(mha / mla, 56.9, delta=0.05)
    def test_window_seconds(self):
        self.assertAlmostEqual(2048 / k.CFG["frame_hz"], 163.84)
        self.assertAlmostEqual(k.frames_to_hours(2048) * 3600, 163.84)
    def test_hours_roundtrip(self):
        for h in (0.5, 1, 10, 100):
            self.assertAlmostEqual(k.frames_to_hours(k.hours_to_frames(h)), h)
        m = k.metrics(k.Design("i", indexer=True, window_frames=512), 24, 64, 0)
        self.assertAlmostEqual(m["fit"]((24 + 64) * 1e9) * k.FRAME_S / 3600, m["hours_dram"])
    def test_ladder_runs_and_ordered(self):
        hs = [k.metrics(d, 24, 64, 0)["stored_pf"] for d in k.ladder()]
        self.assertEqual(hs[0], 524288); self.assertTrue(hs[1] < hs[0] and hs[2] < hs[1])

if __name__ == "__main__": unittest.main()
