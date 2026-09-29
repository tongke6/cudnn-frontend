#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Measure SM100 indexer-backward GEMM JIT churn and warm GPU latency.

Run this script in separate processes against unmodified develop and the
candidate branch, using the same GPU, CuTe DSL version, and command line.
The reported source paths make an accidental editable-install mix-up visible.

Example:
    python benchmark/dsa/benchmark_indexer_backward_dynamic.py \
        --shapes 128x2048,192x2304,256x2560,320x2816 --repeat 50
"""

import argparse
import importlib
import json
import math
import statistics
import time
from pathlib import Path

import torch


def parse_shapes(value):
    shapes = []
    for item in value.split(","):
        try:
            sq, sk = (int(extent) for extent in item.lower().split("x"))
        except (ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(f"invalid shape {item!r}; expected SqxSk") from exc
        if sq < 1 or sk < 1:
            raise argparse.ArgumentTypeError(f"shape extents must be positive: {item!r}")
        shapes.append((sq, sk))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return shapes


def make_case(module, sq, sk, topk, generator):
    device = torch.device("cuda")
    q = torch.randn((1, sq, 64, 128), device=device, dtype=torch.bfloat16, generator=generator)
    w = torch.randn((1, sq, 64), device=device, dtype=torch.bfloat16, generator=generator).abs().mul_(0.1)
    k = torch.randn((1, sk, 128), device=device, dtype=torch.bfloat16, generator=generator)
    grad = torch.randn((1, sq, topk), device=device, dtype=torch.float32, generator=generator)
    indices = torch.randint(0, sk, (1, sq, topk), device=device, dtype=torch.int32, generator=generator)
    dq = torch.empty_like(q)
    dw = torch.empty_like(w)
    dk = torch.empty_like(k, dtype=torch.float32)
    kernel = module.indexer_backward_sm100(1, sq, sk, 64, 128, topk)

    def run():
        dk.zero_()
        kernel.gemm_only(q, w, k, dq, dw, dk, grad, indices)

    def run_gemm_only():
        kernel.gemm_only(q, w, k, dq, dw, dk, grad, indices)

    return run, run_gemm_only, dk


def bench_gpu_us(run_gemm_only, dk, warmup, repeat):
    for _ in range(warmup):
        dk.zero_()
        run_gemm_only()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for start, stop in zip(starts, stops):
        dk.zero_()
        start.record()
        run_gemm_only()
        stop.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(stop) * 1000.0 for start, stop in zip(starts, stops))
    return {"p50_us": statistics.median(times), "p95_us": times[math.ceil(0.95 * len(times)) - 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=parse_shapes, default=parse_shapes("128x2048,192x2304,256x2560,320x2816"))
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--expected-source", type=Path, help="fail if the imported cudnn module is outside this checkout")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup must be nonnegative")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        parser.error("this benchmark requires an SM100+ CUDA device")

    import cudnn

    module = importlib.import_module("cudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_sm100")
    module_path = Path(module.__file__).resolve()
    if args.expected_source and not module_path.is_relative_to(args.expected_source.resolve()):
        parser.error(f"imported {module_path}, not a module under {args.expected_source.resolve()}")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    module._compile_cache.clear()
    real_compile = module.cute.compile
    compile_durations = []

    def timed_compile(*compile_args, **compile_kwargs):
        started = time.perf_counter()
        try:
            return real_compile(*compile_args, **compile_kwargs)
        finally:
            compile_durations.append(time.perf_counter() - started)

    module.cute.compile = timed_compile
    rows = []
    try:
        for sq, sk in args.shapes:
            run, run_gemm_only, dk = make_case(module, sq, sk, args.topk, generator)
            torch.cuda.synchronize()
            compile_start = len(compile_durations)
            started = time.perf_counter()
            run()
            torch.cuda.synchronize()
            first_call_s = time.perf_counter() - started
            compile_times = compile_durations[compile_start:]
            gpu = bench_gpu_us(run_gemm_only, dk, args.warmup, args.repeat)
            rows.append(
                {
                    "sq": sq,
                    "sk": sk,
                    "first_call_s": first_call_s,
                    "compile_calls": len(compile_times),
                    "compile_s": sum(compile_times),
                    "gemm_p50_us": gpu["p50_us"],
                    "gemm_p95_us": gpu["p95_us"],
                    "cache_variants": len(module._compile_cache),
                }
            )
    finally:
        module.cute.compile = real_compile

    print(
        json.dumps(
            {
                "module_path": str(module_path),
                "cudnn_frontend_version": cudnn.__version__,
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "sm_capability": torch.cuda.get_device_capability(),
                "topk": args.topk,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "total_compile_calls": len(compile_durations),
                "total_compile_s": sum(compile_durations),
                "cases": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
