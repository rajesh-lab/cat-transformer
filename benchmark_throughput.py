"""Training throughput benchmark: CAT_Transformer (parallel) vs CAT_Transformer_Looped."""

import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F

from cat_transformer import CAT_Config, CAT_Transformer
from cat_transformer_looped import CAT_Transformer_Looped


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt_num(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def build_models(dim, num_layers, n_head, block_size, chunk_size, device):
    decoder_dim = 2 * dim
    dim_fx = decoder_dim
    n_head_decoder = 2 * n_head

    compressor_config = CAT_Config(
        dim=dim, n_head=n_head, dim_fx=dim_fx,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=max(1, num_layers // 4),
    )
    decoder_config = CAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=block_size, chunk_size=chunk_size,
        n_layer=num_layers,
    )

    parallel = CAT_Transformer(decoder_config, compressor_config).to(device)
    parallel.setup_cache(device=device)

    looped = CAT_Transformer_Looped(decoder_config, compressor_config).to(device)
    looped.setup_cache(device=device)

    return parallel, looped, decoder_config


def benchmark_step(model, input_ids, labels, optimizer, amp_ctx, grad_scaler, n_steps):
    """Run n_steps of fwd + bwd + optimizer and return per-step times in ms."""
    torch.cuda.synchronize()

    timings = []
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        with amp_ctx:
            loss = model(input_ids, labels=labels)
        loss.backward()
        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        timings.append((t1 - t0) * 1000)  # ms

    return timings


def main():
    parser = argparse.ArgumentParser(description="CAT throughput benchmark")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", help="torch.compile both models")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required for throughput benchmark"

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    use_amp = args.dtype != "fp32"
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()
    grad_scaler = torch.amp.GradScaler("cuda") if args.dtype == "fp16" else None

    print("=" * 70)
    print("CAT Training Throughput Benchmark")
    print("=" * 70)
    print(f"  dim={args.dim}  layers={args.num_layers}  heads={args.n_head}")
    print(f"  block_size={args.block_size}  chunk_size={args.chunk_size}")
    print(f"  batch_size={args.batch_size}  dtype={args.dtype}")
    print(f"  warmup={args.warmup}  steps={args.steps}  compile={args.compile}")
    print("=" * 70)

    # ---- build models ----
    parallel, looped, decoder_config = build_models(
        args.dim, args.num_layers, args.n_head,
        args.block_size, args.chunk_size, device,
    )

    print(f"\nParallel params : {fmt_num(count_parameters(parallel))} "
          f"(trainable: {fmt_num(count_trainable_parameters(parallel))})")
    print(f"Looped   params : {fmt_num(count_parameters(looped))} "
          f"(trainable: {fmt_num(count_trainable_parameters(looped))})")

    # ---- optional compile (per-model, with fallback) ----
    compiled_flags = {"Parallel": False, "Looped": False}
    if args.compile:
        print("\nCompiling models (this may take a minute) ...")
        dry_ids = torch.randint(0, decoder_config.vocab_size, (1, args.block_size), device=device)
        models_to_compile = {"Parallel": parallel, "Looped": looped}
        for tag, model in models_to_compile.items():
            try:
                compiled = torch.compile(model, mode="default")
                with torch.no_grad(), (torch.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()):
                    _ = compiled(dry_ids)
                models_to_compile[tag] = compiled
                compiled_flags[tag] = True
                print(f"  {tag}: compiled OK")
            except Exception as e:
                print(f"  {tag}: compile failed ({type(e).__name__}), falling back to eager")
        parallel, looped = models_to_compile["Parallel"], models_to_compile["Looped"]
        del dry_ids
        torch.cuda.empty_cache()

    # ---- synthetic data ----
    input_ids = torch.randint(
        0, decoder_config.vocab_size,
        (args.batch_size, args.block_size), device=device,
    )
    labels = torch.randint(
        0, decoder_config.vocab_size,
        (args.batch_size, args.block_size), device=device,
    )
    tokens_per_step = args.batch_size * args.block_size

    # ---- benchmark each model ----
    results = {}
    for name, model in [("Parallel", parallel), ("Looped", looped)]:
        suffix = " [compiled]" if compiled_flags.get(name) else ""
        print(f"\n--- {name}{suffix} ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # warmup
        print(f"\n[{name}] Warming up ({args.warmup} steps) ...")
        benchmark_step(model, input_ids, labels, optimizer, amp_ctx, grad_scaler, args.warmup)

        # timed steps
        print(f"[{name}] Benchmarking ({args.steps} steps) ...")
        timings = benchmark_step(model, input_ids, labels, optimizer, amp_ctx, grad_scaler, args.steps)

        avg_ms = sum(timings) / len(timings)
        min_ms = min(timings)
        max_ms = max(timings)
        throughput = tokens_per_step / (avg_ms / 1000)

        results[name] = {
            "avg_ms": avg_ms, "min_ms": min_ms, "max_ms": max_ms,
            "throughput": throughput, "timings": timings,
        }

        print(f"[{name}] step time: {avg_ms:.1f} ms  "
              f"(min={min_ms:.1f}, max={max_ms:.1f})  "
              f"throughput: {throughput:.0f} tok/s")

        # free memory before next model
        del optimizer
        torch.cuda.empty_cache()

    # ---- summary ----
    p, l = results["Parallel"], results["Looped"]
    speedup = l["avg_ms"] / p["avg_ms"]

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    pc = " [compiled]" if compiled_flags.get("Parallel") else " [eager]"
    lc = " [compiled]" if compiled_flags.get("Looped") else " [eager]"
    print(f"  Parallel{pc:>11s} : {p['avg_ms']:8.1f} ms/step  |  {p['throughput']:10.0f} tok/s")
    print(f"  Looped  {lc:>11s} : {l['avg_ms']:8.1f} ms/step  |  {l['throughput']:10.0f} tok/s")
    print(f"  Ratio    : Looped is {speedup:.2f}x {'slower' if speedup > 1 else 'faster'} than Parallel")
    print()

    target_tokens = 5e9
    for tag in ["Parallel", "Looped"]:
        r = results[tag]
        secs = target_tokens / r["throughput"]
        hrs = secs / 3600
        days = hrs / 24
        print(f"  {tag:>8s} time to 5B tokens: {hrs:8.1f} hrs  ({days:.1f} days)")

    print("=" * 70)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  Peak GPU memory: {peak_mem:.2f} GiB")


if __name__ == "__main__":
    main()
