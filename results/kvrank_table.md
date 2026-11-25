Rank (min / median / max over layers) for the energy level, joint [Wk;Wv]:

| model | layers | K+V width | K 95% | V 95% | joint 95% | joint 99% | joint energy @256 / @512 / @1024 (median) |
|---|---|---|---|---|---|---|---|
| moshi | 32 | 8192 | 131 / 1884 / 2103 | 1649 / 2287 / 2392 | 766 / 2476 / 2874 | 1868 / 3432 / 3653 | 41.7% / 56.9% / 75.2% |
| smollm-1.7b | 24 | 4096 | 156 / 1020 / 1116 | 1027 / 1142 / 1225 | 547 / 1394 / 1485 | 1184 / 1797 / 1846 | 46.1% / 66.2% / 87.6% |

Implied KV cache for Moshi (bf16, bytes per frame, latent + 64 RoPE dims per layer):

| scheme | bytes/frame | vs today |
|---|---|---|
| today (full K,V) | 512 KiB | 100% |
| per-layer joint rank for 99% energy | 215 KiB | 42% |
| per-layer joint rank for 95% energy | 156 KiB | 30% |
| DeepSeek-V2: 512 latent + 64 RoPE | 36 KiB | 7% |
