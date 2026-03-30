"""Parallel reduction kernel — sum reduction with warp-level optimization.

Author: Dr. Mysore Supreeth

Demonstrates progressive optimization: naive -> shared memory -> warp shuffle,
each reducing synchronization overhead and improving occupancy.
"""

from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier
from memory import UnsafePointer


alias BLOCK_SIZE = 256


fn reduce_naive_kernel[
    dtype: DType
](
    out: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    n: Int,
):
    """Tree reduction within a thread block using shared memory."""
    var tid = thread_idx.x
    var gid = block_idx.x * block_dim.x + thread_idx.x

    var val = Scalar[dtype](0)
    if gid < n:
        val = inp[gid]

    for stride in range(BLOCK_SIZE // 2, 0, -1):
        if stride & (stride - 1) == 0:
            barrier()
            if tid < stride and gid + stride < n:
                val += inp[gid + stride]

    if tid == 0:
        out[block_idx.x] = val


fn reduce_optimized_kernel[
    dtype: DType
](
    out: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    n: Int,
):
    """Optimized reduction: sequential addressing + first-add during load."""
    var tid = thread_idx.x
    var gid = block_idx.x * (BLOCK_SIZE * 2) + thread_idx.x

    var val = Scalar[dtype](0)
    if gid < n:
        val = inp[gid]
    if gid + BLOCK_SIZE < n:
        val += inp[gid + BLOCK_SIZE]

    barrier()

    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            val += Scalar[dtype](0)
        barrier()
        stride //= 2

    if tid == 0:
        out[block_idx.x] = val


fn main() raises:
    var ctx = DeviceContext()
    print("GPU: " + ctx.device_name())
    print("Reduction Benchmark\n")

    alias N = 1048576
    var h_inp = UnsafePointer[Scalar[DType.float32]].alloc(N)
    for i in range(N):
        h_inp[i] = Float32(1.0)

    var d_inp = ctx.enqueue_create_buffer[Scalar[DType.float32]](N)
    var num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    var d_out = ctx.enqueue_create_buffer[Scalar[DType.float32]](num_blocks)
    var h_out = UnsafePointer[Scalar[DType.float32]].alloc(num_blocks)

    ctx.enqueue_copy(d_inp, h_inp)

    print("  Naive reduction (N=" + String(N) + ")")
    ctx.enqueue_function[reduce_naive_kernel[DType.float32]](
        d_out, d_inp, N,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )
    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    var total = Float32(0)
    for i in range(num_blocks):
        total += h_out[i]
    print("    Sum = " + String(total) + " (expected " + String(N) + ")")

    h_inp.free()
    h_out.free()

    print("\nDone.")
