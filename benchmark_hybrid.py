"""Training throughput benchmark: CAT_Transformer (parallel) vs CAT_Transformer_Hybrid."""

import time
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn

from cat_transformer import CAT_Config, CAT_Transformer
from cat_transformer_hybrid import HybridCAT_Config, CAT_Transformer_Hybrid


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fmt_num(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def benchmark_step(model, input_ids, labels, optimizer, amp_ctx, n_steps):
    """Run n_steps of fwd + bwd + optimizer and return per-step times in ms."""
    torch.cuda.synchronize()
    timings = []
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        with amp_ctx:
            loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)
    return timings


def main():
    parser = argparse.ArgumentParser(description="Parallel vs Hybrid CAT benchmark")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--linear_frac", type=float, default=0.5,
                        help="Fraction of layers that use linear attention (0.0 = all standard, 1.0 = all linear)")
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA required"

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if args.dtype != "fp32" else nullcontext()

    # decide which layers are linear
    n_linear = max(0, min(args.num_layers, round(args.num_layers * args.linear_frac)))
    # spread them evenly: pick every k-th layer
    if n_linear == 0:
        linear_layers = []
    elif n_linear == args.num_layers:
        linear_layers = list(range(args.num_layers))
    else:
        step = args.num_layers / n_linear
        linear_layers = [int(i * step) for i in range(n_linear)]

    print("=" * 70)
    print("Parallel vs Hybrid CAT — Training Throughput")
    print("=" * 70)
    print(f"  dim={args.dim}  layers={args.num_layers}  heads={args.n_head}")
    print(f"  block_size={args.block_size}  chunk_size={args.chunk_size}")
    print(f"  batch_size={args.batch_size}  dtype={args.dtype}")
    print(f"  warmup={args.warmup}  steps={args.steps}")
    print(f"  linear_frac={args.linear_frac}  → linear layers: {linear_layers}")
    print("=" * 70)

    decoder_dim = 2 * args.dim
    dim_fx = decoder_dim
    n_head_decoder = 2 * args.n_head

    compressor_config = CAT_Config(
        dim=args.dim, n_head=args.n_head, dim_fx=dim_fx,
        block_size=args.block_size, chunk_size=args.chunk_size,
        n_layer=max(1, args.num_layers // 4),
    )

    parallel_config = CAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=args.block_size, chunk_size=args.chunk_size,
        n_layer=args.num_layers,
    )
    hybrid_config = HybridCAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=args.block_size, chunk_size=args.chunk_size,
        n_layer=args.num_layers,
        linear_attn_layers=linear_layers,
        use_naive_linear_attn=False,
    )
    naive_config = HybridCAT_Config(
        dim=decoder_dim, n_head=n_head_decoder,
        block_size=args.block_size, chunk_size=args.chunk_size,
        n_layer=args.num_layers,
        linear_attn_layers=linear_layers,
        use_naive_linear_attn=True,
    )

    # synthetic data (kept on GPU across runs)
    input_ids = torch.randint(
        0, parallel_config.vocab_size,
        (args.batch_size, args.block_size), device=device,
    )
    labels = torch.randint(
        0, parallel_config.vocab_size,
        (args.batch_size, args.block_size), device=device,
    )
    tokens_per_step = args.batch_size * args.block_size

    def build_model(name, config):
        if name == "Parallel":
            m = CAT_Transformer(config, compressor_config).to(device)
        else:
            m = CAT_Transformer_Hybrid(config, compressor_config).to(device)
        m.setup_cache(device=device)
        return m

    model_specs = [
        ("Parallel",     parallel_config),
        ("Hybrid",       hybrid_config),
        ("Hybrid-Naive", naive_config),
    ]

    # --- benchmark each model independently for accurate memory ---
    results = {}
    for name, config in model_specs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = build_model(name, config)
        n_params = count_parameters(model)
        print(f"\n[{name}] params: {fmt_num(n_params)}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # record memory after optimizer init (weights + optimizer states)
        torch.cuda.synchronize()
        mem_model = torch.cuda.memory_allocated() / (1024 ** 3)

        print(f"[{name}] Warming up ({args.warmup} steps) ...")
        benchmark_step(model, input_ids, labels, optimizer, amp_ctx, args.warmup)

        torch.cuda.reset_peak_memory_stats()

        print(f"[{name}] Benchmarking ({args.steps} steps) ...")
        timings = benchmark_step(model, input_ids, labels, optimizer, amp_ctx, args.steps)

        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        avg = sum(timings) / len(timings)
        mn, mx = min(timings), max(timings)
        tput = tokens_per_step / (avg / 1000)
        results[name] = {
            "avg_ms": avg, "min_ms": mn, "max_ms": mx,
            "throughput": tput, "n_params": n_params,
            "mem_model_gib": mem_model, "peak_mem_gib": peak_mem,
        }

        print(f"[{name}] step: {avg:.1f} ms  (min={mn:.1f}, max={mx:.1f})  "
              f"throughput: {tput:.0f} tok/s")
        print(f"[{name}] model+optim: {mem_model:.2f} GiB  |  peak (train): {peak_mem:.2f} GiB")

        del optimizer, model
        torch.cuda.empty_cache()

    # --- summary ---
    tags = ["Parallel", "Hybrid", "Hybrid-Naive"]
    p = results["Parallel"]

    print("\n" + "=" * 70)
    print("Summary — Throughput")
    print("=" * 70)
    for tag in tags:
        r = results[tag]
        ratio_str = ""
        if tag != "Parallel":
            ratio = r["avg_ms"] / p["avg_ms"]
            ratio_str = f"  ({ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than Parallel)"
        print(f"  {tag:>12s} : {r['avg_ms']:8.1f} ms/step  |  {r['throughput']:10.0f} tok/s{ratio_str}")

    print()
    target_tokens = 5e9
    for tag in tags:
        r = results[tag]
        hrs = target_tokens / r["throughput"] / 3600
        print(f"  {tag:>12s} time to 5B tokens: {hrs:8.1f} hrs  ({hrs/24:.1f} days)")

    print("\n" + "=" * 70)
    print("Summary — Memory")
    print("=" * 70)
    print(f"  {'Model':>12s}   {'Params':>8s}   {'Weights+Optim':>13s}   {'Peak Train':>10s}   {'Activations':>11s}")
    print(f"  {'-'*12}   {'-'*8}   {'-'*13}   {'-'*10}   {'-'*11}")
    for tag in tags:
        r = results[tag]
        act_mem = r["peak_mem_gib"] - r["mem_model_gib"]
        print(f"  {tag:>12s}   {fmt_num(r['n_params']):>8s}   {r['mem_model_gib']:10.2f} GiB"
              f"   {r['peak_mem_gib']:7.2f} GiB   {act_mem:8.2f} GiB")

    print("=" * 70)


if __name__ == "__main__":
    main()
