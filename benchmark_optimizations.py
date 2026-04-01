"""
Benchmark script for CogVideo optimizations.
Measures before/after execution times for key operations.
"""
import torch
import time
import torch.nn.functional as F
from einops import rearrange

def benchmark(func, *args, iterations=100, warmup=10, **kwargs):
    """Benchmark a function with warmup."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'times': times
    }

def benchmark_einsum_vs_matmul():
    """Benchmark einsum vs matmul for attention."""
    print("=" * 60)
    print("BENCHMARK: einsum vs matmul")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, h, n, d = 2, 8, 4096, 64
    dtype = torch.float16 if device == 'cuda' else torch.float32

    q = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, h, n, d, device=device, dtype=dtype, requires_grad=True)

    # einsum approach
    def einsum_attention():
        scale = d ** -0.5
        sim = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        attn = F.softmax(sim, dim=-1)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        return out

    # matmul approach
    def matmul_attention():
        scale = d ** -0.5
        sim = torch.bmm(q.view(b*h, n, d), k.view(b*h, n, d).transpose(-2, -1)) * scale
        attn = F.softmax(sim, dim=-1)
        out = torch.bmm(attn, v.view(b*h, n, d)).view(b, h, n, d)
        return out

    # scaled_dot_product_attention
    def sdp_attention():
        out = F.scaled_dot_product_attention(q, k, v)
        return out

    print("\n[BEFORE] einsum attention:")
    result_einsum = benchmark(einsum_attention, iterations=50)
    print(f"  Mean: {result_einsum['mean']*1000:.4f} ms")

    print("\n[AFTER] matmul attention:")
    result_matmul = benchmark(matmul_attention, iterations=50)
    print(f"  Mean: {result_matmul['mean']*1000:.4f} ms")

    print("\n[OPTIMAL] scaled_dot_product_attention:")
    result_sdp = benchmark(sdp_attention, iterations=50)
    print(f"  Mean: {result_sdp['mean']*1000:.4f} ms")

    speedup = result_einsum['mean'] / result_matmul['mean']
    print(f"\nSpeedup (einsum -> matmul): {speedup:.2f}x")

    return result_einsum, result_matmul, result_sdp


def benchmark_split_interpolate_cat():
    """Benchmark the split+interpolate+cat pattern vs batched."""
    print("\n" + "=" * 60)
    print("BENCHMARK: split+loop+cat vs batched interpolate")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 2, 128, 32, 32
    target_size = (64, 64)
    chunk_size = 32

    x = torch.randn(b, c, h, w, device=device)

    # BEFORE: split + loop + cat (inefficient)
    def old_interpolate(x):
        splits = torch.split(x, chunk_size, dim=1)
        interpolated_splits = [
            F.interpolate(split, size=target_size, mode="nearest")
            for split in splits
        ]
        result = torch.cat(interpolated_splits, dim=1)
        return result

    # AFTER: batch the channels
    def new_interpolate_batched(x):
        # Interpolate all at once
        result = F.interpolate(x, size=target_size, mode="nearest")
        return result

    # AFTER: with adaptive pool (even faster for downsampling)
    def new_interpolate_adaptive(x):
        result = F.adaptive_avg_pool2d(x, target_size)
        return result

    print("\n[BEFORE] split + loop + cat:")
    result_old = benchmark(old_interpolate, x, iterations=50)
    print(f"  Mean: {result_old['mean']*1000:.4f} ms")

    print("\n[AFTER] batched interpolate:")
    result_batched = benchmark(new_interpolate_batched, x, iterations=50)
    print(f"  Mean: {result_batched['mean']*1000:.4f} ms")

    print("\n[OPTIMAL] adaptive_avg_pool2d:")
    result_adaptive = benchmark(new_interpolate_adaptive, x, iterations=50)
    print(f"  Mean: {result_adaptive['mean']*1000:.4f} ms")

    speedup = result_old['mean'] / result_batched['mean']
    print(f"\nSpeedup (old -> batched): {speedup:.2f}x")

    return result_old, result_batched, result_adaptive


def benchmark_contiguous_calls():
    """Benchmark unnecessary contiguous calls."""
    print("\n" + "=" * 60)
    print("BENCHMARK: redundant contiguous() calls")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 2, 256, 64, 64

    x = torch.randn(b, c, h, w, device=device)

    # BEFORE: multiple unnecessary contiguous()
    def old_pattern(x):
        x = x.transpose(1, 2).contiguous()
        x = x.transpose(2, 3).contiguous()
        x = x.view(b, h, w, c).contiguous()
        return x

    # AFTER: only call contiguous when needed
    def new_pattern(x):
        x = x.transpose(1, 2)
        x = x.transpose(2, 3)
        x = x.view(b, h, w, c)
        return x

    # AFTER 2: fuse transpose + contiguous
    def fused_pattern(x):
        x = x.transpose(1, 2).transpose(2, 3).contiguous()
        return x.view(b, h, w, c)

    print("\n[BEFORE] multiple contiguous calls:")
    result_old = benchmark(old_pattern, x, iterations=100)
    print(f"  Mean: {result_old['mean']*1000:.4f} ms")

    print("\n[AFTER] minimal contiguous:")
    result_new = benchmark(new_pattern, x, iterations=100)
    print(f"  Mean: {result_new['mean']*1000:.4f} ms")

    print("\n[OPTIMAL] fused transpose + single contiguous:")
    result_fused = benchmark(fused_pattern, x, iterations=100)
    print(f"  Mean: {result_fused['mean']*1000:.4f} ms")

    speedup = result_old['mean'] / result_fused['mean']
    print(f"\nSpeedup (old -> fused): {speedup:.2f}x")

    return result_old, result_new, result_fused


def benchmark_torch_einsum_vs_aten():
    """Benchmark torch.einsum vs aten operators."""
    print("\n" + "=" * 60)
    print("BENCHMARK: torch.einsum vs aten operators")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, c, h, w = 4, 64, 128, 128
    dtype = torch.float16 if device == 'cuda' else torch.float32

    q = torch.randn(b, c, h, w, device=device, dtype=dtype)
    k = torch.randn(b, c, h, w, device=device, dtype=dtype)

    # einsum: bcij,bjk->bik
    def einsum_pattern():
        b_cij, b_cjk = q, k
        # reshape to 2D
        q_flat = q.view(b, c, h*w)
        k_flat = k.view(b, c, h*w)
        result = torch.einsum('bdn,bde->bne', q_flat, k_flat)
        return result

    # aten: bmm
    def bmm_pattern():
        q_flat = q.view(b, c, h*w).transpose(-2, -1)  # b, n, c
        k_flat = k.view(b, c, h*w)  # b, c, n
        result = torch.bmm(q_flat, k_flat)  # b, n, n
        return result

    # matmul operator
    def matmul_pattern():
        q_flat = q.view(b, c, h*w).transpose(-2, -1)  # b, n, c
        k_flat = k.view(b, c, h*w)  # b, c, n
        result = torch.matmul(q_flat, k_flat)  # b, n, n
        return result

    print("\n[BEFORE] torch.einsum:")
    result_einsum = benchmark(einsum_pattern, iterations=50)
    print(f"  Mean: {result_einsum['mean']*1000:.4f} ms")

    print("\n[AFTER] torch.bmm:")
    result_bmm = benchmark(bmm_pattern, iterations=50)
    print(f"  Mean: {result_bmm['mean']*1000:.4f} ms")

    print("\n[OPTIMAL] torch.matmul:")
    result_matmul = benchmark(matmul_pattern, iterations=50)
    print(f"  Mean: {result_matmul['mean']*1000:.4f} ms")

    speedup = result_einsum['mean'] / result_matmul['mean']
    print(f"\nSpeedup (einsum -> matmul): {speedup:.2f}x")

    return result_einsum, result_bmm, result_matmul


def benchmark_rearrange_vs_reshape():
    """Benchmark einops rearrange vs native reshape."""
    print("\n" + "=" * 60)
    print("BENCHMARK: einops rearrange vs native reshape")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b, h, n, d = 2, 8, 4096, 64
    dtype = torch.float16 if device == 'cuda' else torch.float32

    x = torch.randn(b, h, n, d, device=device, dtype=dtype)

    # einops rearrange
    def rearrange_pattern():
        out = rearrange(x, 'b h n d -> b n (h d)')
        return out

    # native reshape
    def reshape_pattern():
        b, h, n, d = x.shape
        out = x.permute(0, 2, 1, 3).reshape(b, n, h * d)
        return out

    print("\n[BEFORE] einops rearrange:")
    result_rearr = benchmark(rearrange_pattern, iterations=100)
    print(f"  Mean: {result_rearr['mean']*1000:.4f} ms")

    print("\n[AFTER] native reshape:")
    result_reshape = benchmark(reshape_pattern, iterations=100)
    print(f"  Mean: {result_reshape['mean']*1000:.4f} ms")

    speedup = result_rearr['mean'] / result_reshape['mean']
    print(f"\nSpeedup (rearrange -> reshape): {speedup:.2f}x")

    return result_rearr, result_reshape


def benchmark_cat_vs_stack():
    """Benchmark torch.cat vs torch.stack for combining tensors."""
    print("\n" + "=" * 60)
    print("BENCHMARK: torch.cat vs torch.stack")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_tensors = 8
    shape = (4, 256, 32, 32)

    tensors = [torch.randn(shape, device=device) for _ in range(num_tensors)]

    def cat_pattern():
        return torch.cat(tensors, dim=0)

    def stack_pattern():
        return torch.stack(tensors, dim=0).view(-1, *shape[1:])

    print(f"\n[BEFORE] torch.cat ({num_tensors} tensors):")
    result_cat = benchmark(cat_pattern, iterations=100)
    print(f"  Mean: {result_cat['mean']*1000:.4f} ms")

    print(f"\n[AFTER] torch.stack + view:")
    result_stack = benchmark(stack_pattern, iterations=100)
    print(f"  Mean: {result_stack['mean']*1000:.4f} ms")

    return result_cat, result_stack


def main():
    print("=" * 60)
    print("CogVideo Optimization Benchmarks")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("Running on CPU")

    results = {}

    # Run benchmarks
    results['attention'] = benchmark_einsum_vs_matmul()
    results['interpolate'] = benchmark_split_interpolate_cat()
    results['contiguous'] = benchmark_contiguous_calls()
    results['einsum_aten'] = benchmark_torch_einsum_vs_aten()
    results['rearrange'] = benchmark_rearrange_vs_reshape()
    results['cat_stack'] = benchmark_cat_vs_stack()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = [
        ("Attention (einsum->matmul)", results['attention'][0]['mean'], results['attention'][1]['mean']),
        ("Interpolate split/loop/cat->batched", results['interpolate'][0]['mean'], results['interpolate'][1]['mean']),
        ("Contiguous calls", results['contiguous'][0]['mean'], results['contiguous'][2]['mean']),
        ("Einsum -> matmul", results['einsum_aten'][0]['mean'], results['einsum_aten'][2]['mean']),
        ("Rearrange -> reshape", results['rearrange'][0]['mean'], results['rearrange'][1]['mean']),
    ]

    print(f"\n{'Optimization':<40} {'Before (ms)':<15} {'After (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    for name, before, after in summary:
        speedup = before / after
        print(f"{name:<40} {before*1000:<15.4f} {after*1000:<15.4f} {speedup:<10.2f}x")

if __name__ == "__main__":
    main()
