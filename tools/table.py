"""Print the README markdown tables from results/kvrank_*.json."""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "results")
ms = [json.load(open(os.path.join(R, f"kvrank_{n}.json"))) for n in ("moshi", "smollm-1.7b")]
print("Rank (min / median / max over layers) for the energy level, joint [Wk;Wv]:\n")
print("| model | layers | K+V width | K 95% | V 95% | joint 95% | joint 99% | joint energy @256 / @512 / @1024 (median) |\n|---|---|---|---|---|---|---|---|")
f = lambda d: f"{d['min']} / {d['median']:.0f} / {d['max']}"
med = lambda m, k: sorted(l["kv"][k] for l in m["layers"])[len(m["layers"]) // 2]
for m in ms:
    s = m["summary"]
    e = " / ".join(f"{100 * med(m, f'kept{r}'):.1f}%" for r in (256, 512, 1024))
    print(f"| {m['model']} | {m['n_layers']} | {2 * m['kv_dim']} | {f(s['k_rank95'])} | {f(s['v_rank95'])} | {f(s['joint_rank95'])} | {f(s['joint_rank99'])} | {e} |")
m = ms[0]; s = m["summary"]
print("\nImplied KV cache for Moshi (bf16, bytes per frame, latent + 64 RoPE dims per layer):\n")
print("| scheme | bytes/frame | vs today |\n|---|---|---|")
for k, n in (("kv_bytes_today", "today (full K,V)"), ("kv_bytes_joint99", "per-layer joint rank for 99% energy"), ("kv_bytes_joint95", "per-layer joint rank for 95% energy"), ("kv_bytes_deepseek512", "DeepSeek-V2: 512 latent + 64 RoPE")):
    print(f"| {n} | {s[k]/1024:.0f} KiB | {s[k]/s['kv_bytes_today']*100:.0f}% |")
