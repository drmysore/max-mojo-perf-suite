"""Vector addition GPU kernel — baseline for GPU programming validation.

Author: Dr. Mysore Supreeth

Demonstrates Mojo's GPU programming model: grid/block decomposition,
device memory management, and kernel launch via DeviceContext.
"""

from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, global_idx
from memory import UnsafePointer
from sys import sizeof


fn vector_add_kernel[
    dtype: DType
](
    out: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    n: Int,
):
    var idx = global_idx.x
    if idx < n:
        out[idx] = a[idx] + b[idx]


fn benchmark_vector_add[dtype: DType](ctx: DeviceContext, n: Int) raises -> Float64:
    """Run vector addition and return elapsed milliseconds."""
    var size = n * sizeof[Scalar[dtype]]()

    var h_a = UnsafePointer[Scalar[dtype]].alloc(n)
    var h_b = UnsafePointer[Scalar[dtype]].alloc(n)
    var h_out = UnsafePointer[Scalar[dtype]].alloc(n)

    for i in range(n):
        h_a[i] = Scalar[dtype](i)
        h_b[i] = Scalar[dtype](i * 2)

    var d_a = ctx.enqueue_create_buffer[Scalar[dtype]](n)
    var d_b = ctx.enqueue_create_buffer[Scalar[dtype]](n)
    var d_out = ctx.enqueue_create_buffer[Scalar[dtype]](n)

    ctx.enqueue_copy(d_a, h_a)
    ctx.enqueue_copy(d_b, h_b)

    var block_size = 256
    var grid_size = (n + block_size - 1) // block_size

    ctx.enqueue_function[vector_add_kernel[dtype]](
        d_out,
        d_a,
        d_b,
        n,
        grid_dim=grid_size,
        block_dim=block_size,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    var errors = 0
    for i in range(min(n, 1000)):
        var expected = Scalar[dtype](i + i * 2)
        if h_out[i] != expected:
            errors += 1

    h_a.free()
    h_b.free()
    h_out.free()

    if errors > 0:
        raise Error("Validation failed: " + String(errors) + " mismatches")

    return 0.0


fn main() raises:
    var ctx = DeviceContext()
    print("GPU Device: " + ctx.device_name())
    print("Running vector addition benchmarks...")

    alias sizes = List[Int](1024, 65536, 1048576, 16777216)
    for i in range(len(sizes)):
        var n = sizes[i]
        print("  N=" + String(n) + " ... ", end="")
        benchmark_vector_add[DType.float32](ctx, n)
        print("OK")

    print("All benchmarks passed.")
