"""Read single tensors (or row slices) out of a remote safetensors file via HTTP range requests."""
import hashlib, json, os, struct, subprocess, time
import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch(url, start, end, max_time=900, retries=6):
    """Bytes [start, end) of url, cached on disk. curl follows redirects (HF -> CDN); file:// works too."""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(f"{url}:{start}:{end}".encode()).hexdigest()
    path = os.path.join(CACHE, key + ".bin")
    if os.path.exists(path) and os.path.getsize(path) == end - start:
        return open(path, "rb").read()
    for i in range(retries):
        r = subprocess.run(["curl", "-sfL", "--max-time", str(max_time), "-r", f"{start}-{end - 1}", "-o", path + ".part", url])
        if r.returncode == 0 and os.path.getsize(path + ".part") == end - start:
            os.replace(path + ".part", path)
            return open(path, "rb").read()
        time.sleep(10 * (i + 1))  # HF CDN drops ranges now and then
    raise IOError(f"range fetch failed: {url} {start}-{end}")


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


DTYPES = {"F32": lambda b: np.frombuffer(b, dtype="<f4"), "F16": lambda b: np.frombuffer(b, dtype="<f2").astype(np.float32), "BF16": bf16_to_f32}
SIZES = {"F32": 4, "F16": 2, "BF16": 2}


class Remote:
    def __init__(self, url):
        self.url = url
        n = struct.unpack("<Q", fetch(url, 0, 8))[0]
        self.header = json.loads(fetch(url, 8, 8 + n))
        self.base = 8 + n  # data offsets are relative to the byte after the header
        self.header.pop("__metadata__", None)

    def tensor(self, name, rows=None):
        """float32 array; rows=(a, b) reads only rows a..b of a 2-D tensor."""
        h = self.header[name]
        dt, shape, (s, e) = h["dtype"], h["shape"], h["data_offsets"]
        if rows is None:
            a, b = 0, shape[0]
        else:
            a, b = rows
        rowbytes = (e - s) // shape[0]
        raw = fetch(self.url, self.base + s + a * rowbytes, self.base + s + b * rowbytes)
        return DTYPES[dt](raw).reshape([b - a] + shape[1:])
