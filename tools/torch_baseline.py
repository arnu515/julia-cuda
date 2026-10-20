"""The original Julia-1 runtime (PyTorch, from the Hugging Face repo) on the GPU, measured like
julia-cuda: the same requests (julia-rs JSONL, e.g. data/rows.jsonl), one at a time, wall time per
request; results in julia-rs's JSONL format so tools/score.py can compare them.

    python tools/torch_baseline.py MODEL_DIR < data/rows.jsonl > data/torch-4070.jsonl
    python tools/torch_baseline.py MODEL_DIR --batch < data/rows.jsonl > /dev/null
    python tools/torch_baseline.py MODEL_DIR --cuda-graphs < data/rows.jsonl > data/torch-graphs.jsonl

MODEL_DIR is the Hugging Face snapshot (SupersonicLabs/Julia-1, with its julia/ package).
The runtime is FastEngine with the settings of scripts/reproduce_typed.py (strict encoding,
max_length 1024, head_length 512, marker_only_head off), on CUDA. Its autocast is BF16; --dtype
fp16 or fp32 swaps it (Turing has no BF16 hardware, so on a 2080 Ti BF16 is emulated).

--cuda-graphs: the same model and kernels, with each forward pass
captured once as a CUDA graph and replayed, which removes the per-kernel launch cost from
Python. A graph has fixed shapes, so requests are padded to a length bucket (a multiple of
--graph-bucket tokens; padding is masked out, as in the runtime's own batches) and one graph is
captured per bucket before timing starts.
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


class Graphs:
    """One CUDA graph of the runtime's forward pass per padded length, replayed per request."""

    # A request has 2-20 options; graphs take all 20 marker slots, unused ones masked.
    MARKERS = 20

    def __init__(self, engine, torch, dtype, bucket):
        self.engine, self.torch, self.bucket = engine, torch, bucket
        self.dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=None)[dtype]
        self.graphs = {}
        # The nn.TransformerEncoderLayer fast path (the head's layers, when autocast is off) uses
        # nested tensors of data-dependent shape, which a graph can't capture.
        torch.backends.mha.set_fastpath_enabled(False)

    def _encode(self, row):
        e = self.engine._encode([row])[0]
        n = len(e['ids'])
        length = min(self.engine.max_length, -(-n // self.bucket) * self.bucket)
        return e, n, length

    def _fill(self, inputs, e, n):
        """The request's tensors, padded as FastEngine._pack does, into the graph's inputs."""
        torch = self.torch
        k = len(e['markers'])
        length = inputs['input_ids'].shape[1]
        ids = torch.full((1, length), self.engine.tokenizer.pad_token_id, dtype=torch.long)
        ids[0, :n] = torch.tensor(e['ids'])
        attention = torch.zeros((1, length), dtype=torch.long)
        attention[0, :n] = 1
        markers = torch.zeros((1, self.MARKERS), dtype=torch.long)
        markers[0, :k] = torch.tensor(e['markers'])
        marker_mask = torch.zeros((1, self.MARKERS), dtype=torch.bool)
        marker_mask[0, :k] = True
        qtype = torch.tensor([e['qtype']], dtype=torch.long)
        for name, value in [('input_ids', ids), ('attention_mask', attention), ('marker_pos', markers),
                            ('marker_mask', marker_mask), ('qtype', qtype)]:
            inputs[name].copy_(value, non_blocking=True)
        return k

    def _forward(self, inputs):
        torch = self.torch
        # Autocast's weight cache must be off during capture (PyTorch's CUDA graph notes).
        with torch.autocast('cuda', dtype=self.dtype or torch.float32, enabled=self.dtype is not None,
                            cache_enabled=False):
            return self.engine.model(**inputs)

    def prepare(self, row):
        """Captures the graph for `row`'s length bucket, once."""
        torch = self.torch
        e, n, length = self._encode(row)
        if length in self.graphs:
            return
        device = self.engine.device
        inputs = dict(input_ids=torch.zeros((1, length), dtype=torch.long, device=device),
                      attention_mask=torch.zeros((1, length), dtype=torch.long, device=device),
                      marker_pos=torch.zeros((1, self.MARKERS), dtype=torch.long, device=device),
                      marker_mask=torch.zeros((1, self.MARKERS), dtype=torch.bool, device=device),
                      qtype=torch.zeros((1,), dtype=torch.long, device=device))
        self._fill(inputs, e, n)
        with torch.inference_mode():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._forward(inputs)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    out = self._forward(inputs)
            except RuntimeError as err:
                raise RuntimeError(f'capturing the {length}-token graph failed (something in the '
                                   f'forward pass syncs with the host or has data-dependent '
                                   f'shapes): {err}') from err
        self.graphs[length] = (graph, inputs, out)

    def logits(self, row):
        e, n, length = self._encode(row)
        graph, inputs, out = self.graphs[length]
        k = self._fill(inputs, e, n)
        graph.replay()
        return out[0, :k].float().cpu().tolist()


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
    ap.add_argument('--cuda-graphs', action='store_true',
                    help='replay one captured CUDA graph per length bucket')
    ap.add_argument('--graph-bucket', type=int, default=64,
                    help='with --cuda-graphs, pad lengths to a multiple of this many tokens')
    ap.add_argument('--upstream-encoder', action='store_true',
                    help="skip the runtime's ModernBERT forward replacement (julia/router/encoder.py), "
                         "which needs transformers 5.0.0's private _update_attention_mask, and run "
                         "transformers' own forward")
    args = ap.parse_args()
    if args.cuda_graphs and (args.batch or args.compile):
        ap.error('--cuda-graphs runs one request at a time, without torch.compile')

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
          f'{", upstream encoder forward" if args.upstream_encoder else ""}'
          f'{f", CUDA graphs per {args.graph_bucket}-token bucket" if args.cuda_graphs else ""}: '
          f'{len(rows)} requests', file=sys.stderr)

    if args.cuda_graphs:
        graphs = Graphs(engine, torch, args.dtype, args.graph_bucket)
        start = time.perf_counter()
        for row in rows:
            graphs.prepare(row)
        torch.cuda.synchronize()
        print(f'captured {len(graphs.graphs)} graphs (lengths {sorted(graphs.graphs)}) in '
              f'{time.perf_counter() - start:.1f} s', file=sys.stderr)
        run_one = graphs.logits
    else:
        run_one = lambda row: engine.logits([row])[0]

    for row in rows[:args.warmup]:
        run_one(row)
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
            # Tokenize, upload, run, copy the logits back (a synchronizing copy).
            all_logits.append(run_one(row))
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
