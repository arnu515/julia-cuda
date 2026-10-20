# julia-gpu

A GPU inference engine for Julia-1 (the 22-layer ModernBERT decision model), with kernels in
Rust ([cuda-oxide](https://github.com/NVIDIA/cuda-rust)) and a host on
[cudarc](https://github.com/coreylowman/cudarc). [julia-rs](https://github.com/arnu515/julia-rs)
(the CPU engine) is the reference: checkpoint loading, tokenizer, request format and f32 outputs.

This is the **unfused reference**: every step of the forward pass is its own kernel launch,
with the fusions that fit in a GEMM epilogue. A persistent kernel per layer, then one for the
whole model, are to be measured against it.

## Results

The 2000 typed-decisions questions (`--max-length 1024 --head-length 512`, 293 tokens on
average), RTX 2080 Ti, one request at a time:

| engine | correct | same answer as julia-rs f32 | mean latency per request |
| --- | ---: | ---: | ---: |
| julia-cuda | 1451 | 2000 / 2000 | 2.16 ms (1.95 ms on the GPU) |
| original runtime (PyTorch fp16, `tools/torch_baseline.py`) | 1452 | 1998 / 2000 | 23.2 ms |
| julia-rs f32 (CPU) | 1451 | | 68.1 ms (Core Ultra 9 185H, 6 cores) |

## What runs on the GPU

Per request: `embed`, then per layer five launches, then the head, 124 launches in all.

| step | kernel | notes |
| --- | --- | --- |
| embedding + LayerNorm | `embed` | one warp per token |
| qkv | `gemm_*_rope` / `_lnrope` | attention LayerNorm folded in (from layer 1), RoPE in the epilogue |
| attention | `attention` | FlashAttention-2 style on tensor cores; global or ±64-token window |
| wo | `gemm_*_resstats` | + residual in place, and row Σ / Σ² for the next LayerNorm |
| wi | `gemm_*_lngeglu` | MLP LayerNorm folded in, GeGLU in the epilogue |
| wo_mlp | `gemm_*_resstats` | as wo |
| final norm + type embedding | `norm_add` | |
| head layers (2) | `gemm_*_lnbias`, `attention`, `gemm_*_biasresstats`, `gemm_*_lnbiasrelu`, `gemm_*_biasresstats` | LayerNorms with bias folded too |
| scorer | `gemm_*_lnbiasgelu`, `score` | `score` is the last 384 → 1 linear, on the marker rows |

The GEMMs are tensor-core `mma.sync` (two `m16n8k8` on Turing, `m16n8k16` from sm_80) with
`ldmatrix`, swizzled shared tiles and register prefetch. Tiles
are 32/64/128 × 64 (4 or 8 warps), with split-K; `gemm::choose` picks the configuration per
GEMM and token count. Storage is f16, accumulation f32.

Weights are prepared once at load (`weights.rs`, `prep.rs`): f16 conversion; LayerNorm γ (and β,
in the head) folded into the next linear with per-output column sums
(`LN(x)·Wᵀ = rstd·(x·W'ᵀ − μ·colsum)`); Wqkv's q/k rows permuted so RoPE pairs sit in one thread;
Wi's rows interleaved so GeGLU's input and gate sit in one thread; RoPE cos/sin tables.

`gemm_tiled`, a shared-memory GEMM on CUDA cores, is kept as a baseline (`--time`).

## Layout

- `kernels/`: the device crate (its own workspace, built by `cargo oxide`): `main.rs` (every
  kernel), `gemm.rs` and `attn.rs` (their building blocks), `helpers.rs`.
- `src/main.rs`: PTX selection and loading, the `embed` and `gemm_tiled` launches, the CLI.
- `src/checkpoint.rs`, `src/weights.rs`, `src/prep.rs`: reading the checkpoint, the weight
  layout and its preparation.
- `src/gemm.rs`, `src/attn.rs`: launching the GEMMs (and their checks) and attention.
- `src/forward.rs`: the forward pass, `GpuEncoder` (the layers) and `GpuHead`.
- `src/serve.rs`: the request mode.
- `src/check.rs`: `--check`: julia-rs references for every step, host emulations of the
  kernels' arithmetic, and the comparisons. `src/timing.rs`: `--time` (CUDA events).
- `tools/torch_baseline.py`: the original PyTorch runtime on the GPU, for baselines.

## Build

Everything builds inside the nix shell (`nix develop`, or direnv), which provides the CUDA
toolkit and the Rust nightly that cuda-oxide needs.

```sh
make                 # PTX for every tier + the release binary (tier ARCH, default 75)
make dist ARCH=75    # dist/julia-cuda-sm_75: stripped, runs outside nix (needs glibc 2.34+)
make dist ARCH=89    # the same for the 4070
make -B ptx VERBOSE=1    # ptxas's register / spill / shared memory report per kernel
make CHECKED=1       # PTX with slice bounds checks (an out-of-bounds index traps)
cargo test --release --features sm_89    # host-side tests (checkpoint in ./model or JULIA_MODEL)
```

Tiers: `sm_75` (Turing: 2080 Ti), `sm_80` (Ampere), `sm_89` (Ada: 4070), `sm_120` (Blackwell).
A binary embeds one tier's PTX and refuses older GPUs; it names the better tier for newer ones.
Tested on the 2080 Ti and the 4070; sm_80 and sm_120 build but haven't run.

## Run

```sh
julia-cuda MODEL_DIR --max-length 1024 --head-length 512 < requests.jsonl > results.jsonl
```

One request per line, julia-rs's format and result fields (`logits`, `probabilities`,
`truncated`, `choice` / `score` / `noul`); latency percentiles on stderr. `MODEL_DIR` is the
Julia-1 checkpoint. The 2000 questions: `tools/typed_requests.py` gives `rows.jsonl` and
`labels.jsonl` from the typed-decisions test split; julia-rs gives the f32 results. Then

```sh
python3 tools/score.py labels.jsonl results.jsonl julia-rs-f32.jsonl
```

Diagnostics, on a 72- and a 363-token sample request:

- `julia-cuda MODEL_DIR --check`: the weight upload (bit-exact), then every step against julia-rs:
  embed, each layer's GEMMs and attention (from julia-rs's stream, then chained through the
  encoder), all 16 tile × split-K configurations, the head, the whole forward pass. Each check
  has a limit derived from a host emulation of the kernels' arithmetic (`cargo test`).
- `julia-cuda MODEL_DIR --time`: GPU time per kernel (CUDA events after a warm-up), a sweep of
  every GEMM configuration, attention, the encoder and the forward pass.

## Baselines: the original runtime

`tools/torch_baseline.py` runs the Hugging Face repo's own PyTorch runtime (`FastEngine`, cuBLAS +
SDPA) on the same requests and writes the same result format. It needs `torch`,
`transformers` and `safetensors` (see its docstring).

```sh
python3 tools/torch_baseline.py MODEL_DIR --dtype fp16 < rows.jsonl > torch.jsonl               # eager
python3 tools/torch_baseline.py MODEL_DIR --dtype fp16 --cuda-graphs < rows.jsonl > graphs.jsonl # CUDA graphs
python3 tools/torch_baseline.py MODEL_DIR --dtype fp16 --batch < rows.jsonl > /dev/null         # throughput
```

`--dtype fp16` on Turing (no BF16 hardware; the runtime's own is BF16). With transformers newer
than the pinned 5.0.0, add `--upstream-encoder`.

## Limits

- One request at a time, buffers allocated per request; no batching across requests.
- The head's last layer runs on every row (julia-rs: marker rows only; the results are the same).
- Attention uses ~170 registers per thread (the GEMMs 56-154).
- Laptop GPUs vary in clock speed between and within runs; compare timings on the 2080 Ti.
