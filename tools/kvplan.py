#!/usr/bin/env python3
"""kvplan: how many hours can a full-duplex speech LM remember, and can it still emit a frame per 80 ms?

Memory-bound estimate only (bytes read per frame / GPU bandwidth). Stdlib only.
Units: GB = 1e9 B, GiB not used. 1 frame = 80 ms (12.5 Hz).
"""
import argparse, json, os, sys
from dataclasses import dataclass, replace

HERE = os.path.dirname(os.path.abspath(__file__))
def load(name): return json.load(open(os.path.join(HERE, "..", "configs", name)))
CFG, GPUS = load("personaplex.json"), load("gpus.json")
FRAME_S = 1 / CFG["frame_hz"]            # 0.08 s
# Tier link bandwidths (GB/s). dram: PCIe Gen4 x16 = 64 GB/s bidirectional on nvidia.com L40S page -> 32 each way.
# ssd: ASSUMPTION, NVMe Gen4 sequential read. Both overridable on the CLI.
DRAM_GBS, SSD_GBS = 32.0, 7.0

def frames_to_hours(n): return n * FRAME_S / 3600
def hours_to_frames(h): return h * 3600 / FRAME_S

# ---- parameter count, from loaders.py kwargs (Moshi/PersonaPlex structure) ----------------------
def count_params(c=CFG):
    d, L, dd = c["dim"], c["layers"], c["depformer_dim"]
    # gated-SiLU FFN: hidden = 2*int(hidden_scale*dim)//3, three matrices (in-gate, in-up, out)
    ffn = lambda dim, dff: 3 * dim * (2 * dff // 3)
    temporal = L * (4 * d * d + ffn(d, int(c["hidden_scale"] * d)))        # qkv+out = 4d^2
    text_out = d * c["text_card"]                                            # text_linear
    emb = (c["text_card"] + 1) * d + c["n_q"] * (c["card"] + 1) * d          # lookups, not streamed per frame
    q = c["dep_q"]                                                           # weights_per_step: each codebook step has its own weights
    dep_layers = q * c["depformer_layers"] * (4 * dd * dd + ffn(dd, c["depformer_dim_feedforward"]))
    dep_in = q * d * dd
    dep_out = q * dd * c["card"]
    dep_emb = (q - 1) * (c["card"] + 1) * dd + (c["text_card"] + 1) * dd
    read = temporal + text_out + dep_layers + dep_in + dep_out   # every one of these is touched once per 80 ms frame
    return {"total": read + emb + dep_emb, "read_per_frame": read}

# ---- design -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Design:
    name: str = "baseline"
    kv_bits: int = 16
    kv_form: str = "mha"          # mha | gqa:<groups> | mla:<latent>+<rope>
    window_frames: int = 0        # exact recent tier; 0 = whole trained context
    linear_ratio: float = 0.0     # fraction of layers with constant recurrent state (3:1 hybrid = 0.75)
    indexer: bool = False         # DSA-style lightning indexer over full history in lower tiers
    indexer_dim: int = 128        # key dim per frame (DSA/V4 index_head_dim 128; research/deepseek.md:61,70)
    indexer_bytes: float = 1.0    # FP8 (research/deepseek.md:61)
    indexer_per_layer: bool = False  # ASSUMPTION: one shared indexer; DSA proper is per layer
    refresh_every: int = 12       # re-run top-k selection every N frames, prefetch selection to HBM
    topk_frames: int = 2048       # research/deepseek.md:61
    text_pin_tokens: int = 0      # pinned summary tokens, always attended, one KV slot each

def kv_elems(form, c=CFG):
    """KV elements per frame per attention layer."""
    if form == "mha": return 2 * c["heads"] * c["head_dim"]                  # K and V, all heads
    if form.startswith("gqa:"): return 2 * int(form[4:]) * c["head_dim"]     # K and V, g kv-heads
    if form.startswith("mla:"):
        lat, rope = form[4:].split("+"); return int(lat) + int(rope)         # compressed latent + decoupled RoPE key
    raise ValueError(form)

def metrics(d, gpu_gb, dram_gb, ssd_gb, c=CFG, weight_bits=None):
    wb = (weight_bits or c["weight_bits"]) / 8
    P = count_params(c)
    L = c["layers"]
    attn_layers = L * (1 - d.linear_ratio)
    kvpl = kv_elems(d.kv_form, c) * d.kv_bits / 8          # B/frame/attention layer
    idx_layers = attn_layers if d.indexer_per_layer else 1
    idx_pf = d.indexer_dim * d.indexer_bytes * idx_layers if d.indexer else 0
    stored_pf = attn_layers * kvpl + idx_pf                # bytes per frame of history
    # linear layers: state = heads x head_dim x head_dim (Kimi Linear KDA 128x128/head, research/kimi-qwen-minimax.md:51), kept in bf16
    state = L * d.linear_ratio * c["heads"] * c["head_dim"] ** 2 * 2
    pinned = d.text_pin_tokens * attn_layers * kvpl
    ctx_win = c["context_frames"] - c["prompt_frames"]
    win = d.window_frames or ctx_win
    window_kv = win * attn_layers * kvpl
    weights = P["total"] * wb
    fixed = weights + state + pinned + (window_kv if d.indexer else 0)
    # Without an indexer old frames are dropped (ring buffer): memory horizon = context minus pinned prompt.
    ring_cap = None if d.indexer else (d.window_frames or ctx_win)
    def fit_frames(tier_bytes):
        n = max(0.0, tier_bytes - fixed) / stored_pf
        return min(n, ring_cap) if ring_cap else n
    return dict(d=d, P=P, wb=wb, attn_layers=attn_layers, kvpl=kvpl, stored_pf=stored_pf, idx_pf=idx_pf, state=state,
                pinned=pinned, weights=weights, fixed=fixed, ring_cap=ring_cap, win=win, fit=fit_frames,
                hours_gpu=frames_to_hours(fit_frames(gpu_gb * 1e9)),
                hours_dram=frames_to_hours(fit_frames((gpu_gb + dram_gb) * 1e9)),
                hours_all=frames_to_hours(fit_frames((gpu_gb + dram_gb + ssd_gb) * 1e9)))

def ms_per_frame(m, hist_frames, gpu, gpu_gb, dram_gb, ssd_gb, dram_gbs=DRAM_GBS, ssd_gbs=SSD_GBS):
    """Memory-bound time for one frame at a given history length. Returns (ms, note); ms None if history does not fit."""
    d = m["d"]
    H = min(hist_frames, m["ring_cap"]) if m["ring_cap"] else hist_frames
    cap = gpu_gb * 1e9 + dram_gb * 1e9 + ssd_gb * 1e9
    if m["fixed"] + H * m["stored_pf"] > cap: return None, "no fit"
    bw = gpu["bandwidth_gbs"] * 1e9
    if not d.indexer:
        hbm = m["weights"] + m["state"] * 2 + m["pinned"] + min(H, m["win"]) * m["attn_layers"] * m["kvpl"]  # state read+write
        return hbm / bw * 1e3, ""
    recent = min(H, m["win"])
    old = max(0, H - recent)
    k = min(old, d.topk_frames)
    topk_kv = k * m["attn_layers"] * m["kvpl"]
    hbm = m["weights"] + m["state"] * 2 + m["pinned"] + recent * m["attn_layers"] * m["kvpl"] + topk_kv
    t = hbm / bw
    # history lives in the fastest tier that holds it; every `refresh_every` frames the indexer scans all old keys
    # and the selected top-k KV is copied up to HBM, so tier traffic is amortised over the refresh interval.
    hist_bytes = old * m["stored_pf"]
    gpu_free = gpu_gb * 1e9 - m["fixed"]
    if hist_bytes <= gpu_free: tier_bw = bw
    elif hist_bytes <= gpu_free + dram_gb * 1e9: tier_bw = dram_gbs * 1e9
    else: tier_bw = ssd_gbs * 1e9
    t += (old * m["idx_pf"] + topk_kv) / d.refresh_every / tier_bw
    return t * 1e3, ""

def ladder(mla="512+64"):
    b = Design("baseline PersonaPlex")
    s = [b]
    s.append(replace(s[-1], name="+4-bit KV", kv_bits=4))
    s.append(replace(s[-1], name=f"+MLA {mla}", kv_form=f"mla:{mla}"))
    s.append(replace(s[-1], name="+indexer top-2048 (DRAM)", indexer=True, window_frames=512))
    s.append(replace(s[-1], name="+3:1 hybrid linear", linear_ratio=0.75))
    s.append(replace(s[-1], name="+text pin (512 tok)", text_pin_tokens=512))
    return s

def fmt_h(h, m): return f"{h:.2f}" + ("*" if m["ring_cap"] else "")

def ladder_table(mla="512+64", dram_gb=64, ssd_gb=0):
    rows = ["| design | bytes/frame | hours in 24 GB GPU (after weights) | hours with +%d GB DRAM | ms/frame at 10 h, L4 | ms/frame at 10 h, H100 | pass (L4 / H100) |" % dram_gb,
            "|---|---|---|---|---|---|---|"]
    for d in ladder(mla):
        m = metrics(d, 24, dram_gb, ssd_gb)
        cells, ok = [], []
        for g in ("l4", "h100"):
            ms, note = ms_per_frame(m, hours_to_frames(10), GPUS[g], GPUS[g]["mem_gb"], dram_gb, ssd_gb)
            cells.append("n/a (%s)" % note if ms is None else f"{ms:.1f}")
            ok.append("Y" if ms is not None and ms <= 80 else "N")
        rows.append(f"| {d.name} | {m['stored_pf']:,.0f} | {fmt_h(m['hours_gpu'], m)} | {fmt_h(m['hours_dram'], m)} | {cells[0]} | {cells[1]} | {'/'.join(ok)} |")
    return "\n".join(rows) + "\n\n`*` = bounded by the trained context window (ring buffer), not by memory."

def detail(d, gpu_key, dram_gb, ssd_gb):
    g = GPUS[gpu_key]
    m = metrics(d, g["mem_gb"], dram_gb, ssd_gb)
    print(f"{d.name} on {g['name']} ({g['bandwidth_gbs']} GB/s, {g['mem_gb']} GB) + {dram_gb} GB DRAM + {ssd_gb} GB SSD")
    print(f"  params total {m['P']['total']/1e9:.2f} B, read per frame {m['P']['read_per_frame']/1e9:.2f} B -> weights {m['weights']/1e9:.2f} GB")
    print(f"  stored bytes/frame {m['stored_pf']:,.0f}; hours in GPU {fmt_h(m['hours_gpu'], m)}, +DRAM {fmt_h(m['hours_dram'], m)}, +SSD {fmt_h(m['hours_all'], m)}")
    for h in (1, 10, 100):
        ms, note = ms_per_frame(m, hours_to_frames(h), g, g["mem_gb"], dram_gb, ssd_gb)
        print(f"  history {h:>3} h: " + (note if ms is None else f"{ms:.1f} ms/frame {'PASS' if ms <= 80 else 'FAIL'} (budget 80)"))
    if d.indexer and m["win"] + d.topk_frames + d.text_pin_tokens > CFG["context_frames"]:
        print(f"  warning: attended frames {m['win'] + d.topk_frames} exceed trained context {CFG['context_frames']}; retrieved frames need re-positioning")

def main(a=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ladder", action="store_true"); p.add_argument("--mla-latent", default="512+64")
    p.add_argument("--gpu", default="l4", choices=GPUS); p.add_argument("--dram-gb", type=float, default=64); p.add_argument("--ssd-gb", type=float, default=0)
    p.add_argument("--kv-bits", type=int, default=16); p.add_argument("--kv-form", default="mha")
    p.add_argument("--window-frames", type=int, default=0); p.add_argument("--linear-ratio", type=float, default=0.0)
    p.add_argument("--indexer", action="store_true"); p.add_argument("--indexer-dim", type=int, default=128)
    p.add_argument("--indexer-bytes", type=float, default=1.0); p.add_argument("--indexer-per-layer", action="store_true")
    p.add_argument("--refresh-every", type=int, default=12); p.add_argument("--topk-frames", type=int, default=2048)
    p.add_argument("--text-pin-tokens", type=int, default=0)
    p.add_argument("--list-gpus", action="store_true")
    o = p.parse_args(a)
    if o.list_gpus:
        for k, g in GPUS.items(): print(f"{k}: {g['name']} {g['mem_gb']} GB {g['bandwidth_gbs']} GB/s  {g['src']}")
    elif o.ladder: print(ladder_table(o.mla_latent, o.dram_gb, o.ssd_gb))
    else:
        d = Design("custom", o.kv_bits, o.kv_form, o.window_frames, o.linear_ratio, o.indexer, o.indexer_dim, o.indexer_bytes,
                   o.indexer_per_layer, o.refresh_every, o.topk_frames, o.text_pin_tokens)
        detail(d, o.gpu, o.dram_gb, o.ssd_gb)

if __name__ == "__main__": main()
