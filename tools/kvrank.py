"""How low-rank are a transformer's K/V projections? Reads weights by HTTP range request.

usage: kvrank.py moshi|smollm [--layers N]   -> results/kvrank_<model>.json
Energy = cumulative sum of squared singular values (from eigvalsh of the Gram matrix W^T W, float64).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from safetensors_range import Remote

HF = "https://huggingface.co/"
KEEP = (32, 64, 128, 256, 512, 1024)
LEVELS = (0.90, 0.95, 0.99)
ROPE = 64  # DeepSeek-V2 decoupled RoPE dims per layer, cached on top of the latent


def energy_stats(G):
    """G = W^T W (n x n). Rank for each energy level and energy kept at fixed ranks."""
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    c = np.cumsum(ev) / ev.sum()
    out = {f"rank{int(p * 100)}": int(np.searchsorted(c, p - 1e-12) + 1) for p in LEVELS}
    out.update({f"kept{r}": float(c[r - 1]) for r in KEEP if r <= len(c)})
    return out


def gram(W):
    W = W.astype(np.float64)
    return W.T @ W


class Moshi:
    name, layers, kv_dim = "moshi", 32, 4096  # K (or V) output width; heads 32 x 128
    url = HF + "kyutai/moshiko-pytorch-bf16/resolve/main/model.safetensors"
    repo = None

    def __init__(self):
        self.r = Remote(self.url)

    def kv(self, i):
        # moshi transformer.py: rearrange(proj, "b t (p h d) -> p ...", p=3) => rows [Q; K; V]
        n = f"transformer.layers.{i}.self_attn.in_proj_weight"
        d = self.kv_dim
        return self.r.tensor(n, rows=(d, 2 * d)), self.r.tensor(n, rows=(2 * d, 3 * d))


class SmolLM:
    name, layers, kv_dim = "smollm-1.7b", 24, 2048  # MHA: 32 heads x 64, no GQA; paper's 1B7 model
    repo = "HuggingFaceTB/SmolLM-1.7B"

    def __init__(self):
        idx = json.loads(_get(HF + self.repo + "/resolve/main/model.safetensors.index.json"))
        self.map, self.remotes = idx["weight_map"], {}

    def _t(self, n):
        f = self.map[n]
        if f not in self.remotes:
            self.remotes[f] = Remote(HF + self.repo + "/resolve/main/" + f)
        return self.remotes[f].tensor(n)

    def kv(self, i):
        return self._t(f"model.layers.{i}.self_attn.k_proj.weight"), self._t(f"model.layers.{i}.self_attn.v_proj.weight")


def _get(url):
    import subprocess
    return subprocess.run(["curl", "-sfL", "--max-time", "60", url], check=True, capture_output=True).stdout


def layer_stats(Wk, Wv):
    gk, gv = gram(Wk), gram(Wv)
    return {"k": energy_stats(gk), "v": energy_stats(gv), "kv": energy_stats(gk + gv)}  # [Wk;Wv]^T[Wk;Wv] = gk+gv


def summarize(layers, kv_dim, n_layers):
    s = {}
    base = n_layers * 2 * kv_dim * 2  # bytes/frame today, bf16
    for p in (95, 99):
        rs = [l["kv"][f"rank{p}"] for l in layers]
        s[f"joint_rank{p}"] = {"min": min(rs), "median": float(np.median(rs)), "max": max(rs)}
        for w in ("k", "v"):
            r = [l[w][f"rank{p}"] for l in layers]
            s[f"{w}_rank{p}"] = {"min": min(r), "median": float(np.median(r)), "max": max(r)}
        s[f"kv_bytes_joint{p}"] = sum(r + ROPE for r in rs) * 2  # latent + decoupled RoPE, bf16
    s["kv_bytes_today"] = base
    s["kv_bytes_deepseek512"] = n_layers * (512 + ROPE) * 2
    return s


def main():
    name = sys.argv[1]
    lim = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else None
    m = {"moshi": Moshi, "smollm": SmolLM}[name]()
    n = lim or m.layers
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", f"kvrank_{m.name}.json")
    layers = []
    for i in range(n):
        layers.append(layer_stats(*m.kv(i)))
        print(i, layers[-1]["kv"]["rank95"], layers[-1]["kv"]["rank99"], flush=True)
    res = {"model": m.name, "n_layers": n, "kv_dim": m.kv_dim, "layers": layers, "summary": summarize(layers, m.kv_dim, n),
           "caveat": "weight energy is a proxy; activation-weighted error is the real test and is not measured. "
                     "Moshi applies RoPE to all of K, so a real conversion must keep some RoPE dims (MHA2MLA partial RoPE); K energy here is for the whole K."}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
