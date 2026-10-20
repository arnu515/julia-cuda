"""The original Julia-1 runtime (PyTorch, from the Hugging Face repo) on the GPU, measured like
julia-cuda: the same requests (julia-rs JSONL, e.g. data/rows.jsonl), one at a time, wall time per
request; results in julia-rs's JSONL format so tools/score.py can compare them.

    python tools/torch_baseline.py MODEL_DIR < data/rows.jsonl > data/torch-4070.jsonl
    python tools/torch_baseline.py MODEL_DIR --batch < data/rows.jsonl > /dev/null

MODEL_DIR is the Hugging Face snapshot (SupersonicLabs/Julia-1, with its julia/ package).
The runtime is FastEngine with the settings of scripts/reproduce_typed.py (strict encoding,
max_length 1024, head_length 512, marker_only_head off), on CUDA. Its autocast is BF16; --dtype
fp16 or fp32 swaps it (Turing has no BF16 hardware, so on a 2080 Ti BF16 is emulated).
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', type=Path, help='Julia-1 repository snapshot')
    ap.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16',
                    help='autocast dtype (the runtime\'s own is bf16)')
    ap.add_argument('--batch', action='store_true',
                    help='one call over all requests (the runtime batches 16 by length) for throughput, '
                         'instead of one request at a time')
    ap.add_argument('--compile', action='store_true', help='FastEngine(compile_model=True)')
    ap.add_argument('--warmup', type=int, default=10, help='requests run first and not timed')
    ap.add_argument('--upstream-encoder', action='store_true',
                    help="skip the runtime's ModernBERT forward replacement (julia/router/encoder.py), "
                         "which needs transformers 5.0.0's private _update_attention_mask, and run "
                         "transformers' own forward")
    args = ap.parse_args()

    sys.path.insert(0, str(args.model.resolve()))
    import torch
    from julia.router import engine as engine_module
    from julia.router.engine import FastEngine

    if args.dtype != 'bf16':
        # The runtime hard-codes torch.autocast(dtype=torch.bfloat16) on CUDA (engine.py and
        # inference.py). Swap the dtype it asks for, nothing else.
        dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
        real = torch.autocast

        def autocast(device_type, dtype_=None, enabled=True, **kw):
            kw.pop('dtype', None)
            return real(device_type, dtype=dtype, enabled=enabled and args.dtype != 'fp32', **kw)

        class _Torch:
            def __getattr__(self, name):
                return autocast if name == 'autocast' else getattr(torch, name)
        engine_module.torch = _Torch()
        # julia/cuda.py refuses GPUs without BF16 unless asked for another precision; its check is
        # all that precision changes. (It also enables TF32, so fp32 matmuls run as TF32.)
        import julia.cuda
        import julia.inference
        julia.inference.configure = lambda device: julia.cuda.configure(device, precision='fp32')

    if args.upstream_encoder:
        # engine.py imports this at construction time, so replacing it here is enough.
        import julia.router.encoder
        julia.router.encoder.specialize_decision_encoder = lambda model: False
    import transformers
    engine = FastEngine(args.model.resolve(), device='cuda', transformer_backend='torch',
                        strict_encoding=True, max_length=1024, head_length=512,
                        marker_only_head=False, encoding_cache=0, token_cache=0,
                        compile_model=args.compile)
    gpu = torch.cuda.get_device_name()
    rows = [json.loads(line) for line in sys.stdin if line.strip()]
    print(f'{gpu}, torch {torch.__version__}, transformers {transformers.__version__}, {args.dtype}'
          f'{", compiled" if args.compile else ""}'
          f'{", upstream encoder forward" if args.upstream_encoder else ""}: '
          f'{len(rows)} requests', file=sys.stderr)

    for row in rows[:args.warmup]:
        engine.logits([row])
    torch.cuda.synchronize()

    if args.batch:
        start = time.perf_counter()
        all_logits = engine.logits(rows)
        elapsed = time.perf_counter() - start
        print(f'batched: {elapsed:.2f} s for {len(rows)} requests, {len(rows) / elapsed:.1f} requests/s',
              file=sys.stderr)
    else:
        all_logits, times = [], []
        for row in rows:
            start = time.perf_counter()
            # logits() tokenizes, uploads, runs and copies the logits back (a synchronizing copy).
            all_logits.append(engine.logits([row])[0])
            times.append((time.perf_counter() - start) * 1e3)
        times_sorted = sorted(times)
        at = lambda p: times_sorted[round((len(times_sorted) - 1) * p)]
        print(f'per request, wall (tokenize, upload, run, download): mean {statistics.fmean(times):.3f}, '
              f'p50 {at(0.5):.3f}, p90 {at(0.9):.3f}, p99 {at(0.99):.3f}, max {at(1.0):.3f} ms; '
              f'{sum(times) / 1e3:.2f} s in all', file=sys.stderr)

    out = sys.stdout
    for row, logits in zip(rows, all_logits):
        m = max(logits)
        e = [math.exp(z - m) for z in logits]
        p = [v / sum(e) for v in e]
        result = dict(logits=logits, probabilities=p, truncated=False)
        kind = row.get('type', 'choice')
        if kind == 'choice':
            result['choice'] = max(range(len(p)), key=p.__getitem__)
        elif kind == 'score':
            result['score'] = sum(i * q for i, q in enumerate(p))
        else:
            result['noul'] = p[1]
        out.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
