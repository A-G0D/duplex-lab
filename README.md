# duplex-lab

Tools for working out how a full-duplex speech model (NVIDIA PersonaPlex / Kyutai Moshi) could remember hours of conversation instead of about 2.5 minutes.

## The problem

- The model attends over a 2048-frame window at 12.5 Hz (163.84 s); the voice and role prompt is pinned inside it (up to 512 frames), leaving about 2 to 2.5 minutes of conversation.
- Each frame stores 512 KiB of K/V (524,288 bytes), so memory is not what ends the conversation: the trained window is.
- Source: github.com/NVIDIA/personaplex.

## kvrank

`tools/kvrank.py` measures how low-rank the K and V projection weights are, layer by layer, for Moshi and SmolLM-1.7B. It reads single tensors from the remote safetensors files by HTTP range request (`tools/safetensors_range.py`), so no full checkpoint is downloaded. Results are in `results/kvrank_*.json`.

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

Caveat: this measures weight energy only. Whether activations survive the compression is untested. The calibration point is SmolLM-1.7B in MHA2MLA (arXiv 2502.14837): d_kv=32 cuts KV by 68.75% for a 1.05-point score drop, after fine-tuning.

## kvplan

`tools/kvplan.py` is a stdlib-only memory-bound calculator. It estimates how many hours of conversation a full-duplex speech model (PersonaPlex, Moshi architecture) can keep, and whether it can still emit a frame every 80 ms.

```
python3 tools/kvplan.py --ladder                # table below (--mla-latent 512+64 by default)
python3 tools/kvplan.py --kv-form mla:512+64 --indexer --window-frames 512 --gpu a10g
python3 tools/kvplan.py --list-gpus             # specs with source URLs (configs/gpus.json)
```

Model shape comes from `configs/personaplex.json` (read from NVIDIA/personaplex `loaders.py`, dep_q=16). `configs/deepseek_v3_attn.json` is used only to check MLA 576 vs MHA 32,768 elements per token per layer.

| design | bytes/frame | hours in 24 GB GPU (after weights) | hours with +64 GB DRAM | ms/frame at 10 h, L4 | ms/frame at 10 h, H100 | pass (L4 / H100) |
|---|---|---|---|---|---|---|
| baseline PersonaPlex | 524,288 | 0.03* | 0.03* | 58.5 | 5.2 | Y/Y |
| +4-bit KV | 131,072 | 0.03* | 0.03* | 56.5 | 5.1 | Y/Y |
| +MLA 512+64 | 9,216 | 0.03* | 0.03* | 55.9 | 5.0 | Y/Y |
| +indexer top-2048 (DRAM) | 9,344 | 17.25 | 169.46 | 55.9 | 5.0 | Y/Y |
| +3:1 hybrid linear | 2,432 | 66.08 | 650.87 | 56.0 | 5.0 | Y/Y |
| +text pin (512 tok) | 2,432 | 66.07 | 650.86 | 56.0 | 5.0 | Y/Y |

`*` = bounded by the trained context window (ring buffer), not by memory.

Columns: bytes/frame is stored history per 80 ms frame. Hours use 24 GB GPU after ~16.7 GB of bf16 weights, then +64 GB DRAM. ms/frame is weights + window KV + top-k KV + indexer scan, divided by vendor bandwidth (L4 300 GB/s, H100 3.35 TB/s), at 10 h of history. Baseline is capped by the 2048-frame trained context minus the 512-frame prompt: 1536 frames = 122.9 s (2048 frames = 163.84 s).

Assumptions (flagged in code): indexer dim 128 FP8 and shared across layers; DRAM link 32 GB/s, SSD 7 GB/s; A10G uses A10 datasheet numbers; embedding tables are excluded from the per-frame weight read (8.37 B total params, 8.04 B read).

**Limits.** This is a memory-bound estimate only. It ignores compute, output quality, and whether the model can attend past its trained positions: the ladder rows with the indexer attend 512 + 2048 frames, beyond the 2048 trained context, so retrieved frames would need re-positioning inside the trained window (InfLLM-style). Nothing here was measured on hardware.

## What this suggests

- The limit is the trained window, not memory bandwidth.
- Next test needs a GPU: attention sparsity over a real conversation.
- Next test needs a GPU: re-pinning facts as text instead of keeping their frames.

## Quickstart

```
pip install numpy
python -m unittest discover -s tests   # offline
python3 tools/kvplan.py --ladder
python3 tools/kvrank.py smollm         # network: HTTP range reads from huggingface.co
./run_all.sh                           # re-measure both models, rebuild the table, run tests
```

No test touches the network.

## License

MIT, see LICENSE.
