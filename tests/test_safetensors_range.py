import json, os, struct, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import safetensors_range as sr


def write_st(path, tensors):
    hdr, data, off = {}, b"", 0
    for name, (dt, arr, raw) in tensors.items():
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        data += raw; off += len(raw)
    h = json.dumps(hdr).encode()
    open(path, "wb").write(struct.pack("<Q", len(h)) + h + data)


class T(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((6, 5)).astype(np.float32)
        a_bf = (a.view(np.uint32) & 0xFFFF0000).view(np.float32)  # bf16-representable values
        raw_bf = (a_bf.view(np.uint32) >> 16).astype("<u2").tobytes()
        b = rng.standard_normal((4, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            sr.CACHE = os.path.join(d, "cache")
            p = os.path.join(d, "t.safetensors")
            write_st(p, {"a": ("BF16", a, raw_bf), "b": ("F32", b, b.tobytes())})
            r = sr.Remote("file://" + p)
            np.testing.assert_array_equal(r.tensor("a"), a_bf)
            np.testing.assert_array_equal(r.tensor("a", rows=(2, 5)), a_bf[2:5])
            np.testing.assert_array_equal(r.tensor("b", rows=(1, 3)), b[1:3])
            np.testing.assert_array_equal(r.tensor("b", rows=(1, 3)), b[1:3])  # cache hit


if __name__ == "__main__":
    unittest.main()
