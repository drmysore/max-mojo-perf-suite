"""Fused online softmax kernel for attention layers.

Author: Dr. Mysore Supreeth

Implements numerically stable softmax in a single kernel pass,
avoiding the naive three-pass (max, exp-sum, normalize) pattern.
This is the kind of kernel fusion that MAX's compiler automates
for standard graphs but that custom ops must handle explicitly.
"""

from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier
from memory import UnsafePointer
from math import exp, log, max as math_max, inf


fn softmax_naive_kernel[
    dtype: DType
](
    out: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    rows: Int,
    cols: Int,
):
    """Three-pass softmax: find max, compute exp sum, normalize."""
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= rows:
        return

    var offset = row * cols

    var row_max = Scalar[dtype](-inf[dtype]())
    for c in range(cols):
        row_max = math_max(row_max, inp[offset + c])

    var sum_exp = Scalar[dtype](0)
    for c in range(cols):
        sum_exp += exp(inp[offset + c] - row_max)

    var inv_sum = Scalar[dtype](1) / sum_exp
    for c in range(cols):
        out[offset + c] = exp(inp[offset + c] - row_max) * inv_sum


fn softmax_online_kernel[
    dtype: DType
](
    out: UnsafePointer[Scalar[dtype]],
    inp: UnsafePointer[Scalar[dtype]],
    rows: Int,
    cols: Int,
):
    """Online softmax: single pass for max + exp-sum, then normalize.

    Tracks running max and corrects accumulated sum on the fly,
    reducing memory traffic from 3 passes to 2.
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= rows:
        return

    var offset = row * cols

    var running_max = Scalar[dtype](-inf[dtype]())
    var running_sum = Scalar[dtype](0)

    for c in range(cols):
        var val = inp[offset + c]
        if val > running_max:
            running_sum = running_sum * exp(running_max - val)
            running_max = val
        running_sum += exp(val - running_max)

    var inv_sum = Scalar[dtype](1) / running_sum
    for c in range(cols):
        out[offset + c] = exp(inp[offset + c] - running_max) * inv_sum


fn run_softmax_benchmark[dtype: DType](ctx: DeviceContext, rows: Int, cols: Int) raises:
    var total = rows * cols
    var h_inp = UnsafePointer[Scalar[dtype]].alloc(total)
    var h_out_naive = UnsafePointer[Scalar[dtype]].alloc(total)
    var h_out_online = UnsafePointer[Scalar[dtype]].alloc(total)

    for i in range(total):
        h_inp[i] = Scalar[dtype]((i % 100) / 50.0 - 1.0)

    var d_inp = ctx.enqueue_create_buffer[Scalar[dtype]](total)
    var d_out = ctx.enqueue_create_buffer[Scalar[dtype]](total)

    ctx.enqueue_copy(d_inp, h_inp)

    var block_size = 256
    var grid_size = (rows + block_size - 1) // block_size

    print("  Naive softmax [" + String(rows) + " x " + String(cols) + "]")
    ctx.enqueue_function[softmax_naive_kernel[dtype]](
        d_out, d_inp, rows, cols,
        grid_dim=grid_size,
        block_dim=block_size,
    )
    ctx.enqueue_copy(h_out_naive, d_out)
    ctx.synchronize()
    print("    Complete")

    print("  Online softmax [" + String(rows) + " x " + String(cols) + "]")
    ctx.enqueue_function[softmax_online_kernel[dtype]](
        d_out, d_inp, rows, cols,
        grid_dim=grid_size,
        block_dim=block_size,
    )
    ctx.enqueue_copy(h_out_online, d_out)
    ctx.synchronize()
    print("    Complete")

    var max_diff = Scalar[dtype](0)
    for i in range(min(total, rows * 10)):
        var diff = h_out_naive[i] - h_out_online[i]
        if diff < 0:
            diff = -diff
        if diff > max_diff:
            max_diff = diff

    print("    Max difference: " + String(max_diff))
    if max_diff < 1e-5:
        print("    Validation PASSED")
    else:
        print("    WARNING: numerical divergence detected")

    h_inp.free()
    h_out_naive.free()
    h_out_online.free()


fn main() raises:
    var ctx = DeviceContext()
    print("GPU: " + ctx.device_name())
    print("Softmax Benchmark: Naive vs Online\n")

    run_softmax_benchmark[DType.float32](ctx, 1024, 128)
    run_softmax_benchmark[DType.float32](ctx, 4096, 512)
    run_softmax_benchmark[DType.float32](ctx, 8192, 1024)

    print("\nDone.")
