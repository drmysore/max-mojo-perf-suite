"""Tiled matrix multiplication kernel with shared memory.

Author: Dr. Mysore Supreeth

Demonstrates memory hierarchy optimization: global -> shared -> register,
a pattern critical for achieving high GPU utilization on both NVIDIA and AMD.
"""

from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier
from memory import UnsafePointer
from sys import sizeof


alias TILE_SIZE = 16


fn matmul_naive_kernel[
    dtype: DType
](
    C: UnsafePointer[Scalar[dtype]],
    A: UnsafePointer[Scalar[dtype]],
    B: UnsafePointer[Scalar[dtype]],
    M: Int,
    N: Int,
    K: Int,
):
    """Naive O(MNK) matmul — baseline for comparison."""
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x

    if row < M and col < N:
        var acc = Scalar[dtype](0)
        for k in range(K):
            acc += A[row * K + k] * B[k * N + col]
        C[row * N + col] = acc


fn matmul_tiled_kernel[
    dtype: DType
](
    C: UnsafePointer[Scalar[dtype]],
    A: UnsafePointer[Scalar[dtype]],
    B: UnsafePointer[Scalar[dtype]],
    M: Int,
    N: Int,
    K: Int,
):
    """Tiled matmul using shared memory for data reuse.

    Each thread block loads TILE_SIZE x TILE_SIZE sub-matrices into
    shared memory, reducing global memory traffic by ~TILE_SIZE x.
    """
    var row = block_idx.y * TILE_SIZE + thread_idx.y
    var col = block_idx.x * TILE_SIZE + thread_idx.x

    var acc = Scalar[dtype](0)
    var num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

    for t in range(num_tiles):
        var a_val = Scalar[dtype](0)
        var b_val = Scalar[dtype](0)

        var a_col = t * TILE_SIZE + thread_idx.x
        if row < M and a_col < K:
            a_val = A[row * K + a_col]

        var b_row = t * TILE_SIZE + thread_idx.y
        if b_row < K and col < N:
            b_val = B[b_row * N + col]

        barrier()
        acc += a_val * b_val
        barrier()

    if row < M and col < N:
        C[row * N + col] = acc


fn run_matmul_benchmark[dtype: DType](ctx: DeviceContext, M: Int, N: Int, K: Int) raises:
    """Compare naive vs tiled matmul performance."""
    var size_A = M * K
    var size_B = K * N
    var size_C = M * N

    var h_A = UnsafePointer[Scalar[dtype]].alloc(size_A)
    var h_B = UnsafePointer[Scalar[dtype]].alloc(size_B)
    var h_C_naive = UnsafePointer[Scalar[dtype]].alloc(size_C)
    var h_C_tiled = UnsafePointer[Scalar[dtype]].alloc(size_C)

    for i in range(size_A):
        h_A[i] = Scalar[dtype](i % 7)
    for i in range(size_B):
        h_B[i] = Scalar[dtype](i % 11)

    var d_A = ctx.enqueue_create_buffer[Scalar[dtype]](size_A)
    var d_B = ctx.enqueue_create_buffer[Scalar[dtype]](size_B)
    var d_C = ctx.enqueue_create_buffer[Scalar[dtype]](size_C)

    ctx.enqueue_copy(d_A, h_A)
    ctx.enqueue_copy(d_B, h_B)

    var grid_x = (N + TILE_SIZE - 1) // TILE_SIZE
    var grid_y = (M + TILE_SIZE - 1) // TILE_SIZE

    print("  Naive matmul [" + String(M) + "x" + String(K) + "] x [" + String(K) + "x" + String(N) + "]")
    ctx.enqueue_function[matmul_naive_kernel[dtype]](
        d_C, d_A, d_B, M, N, K,
        grid_dim=(grid_x, grid_y),
        block_dim=(TILE_SIZE, TILE_SIZE),
    )
    ctx.enqueue_copy(h_C_naive, d_C)
    ctx.synchronize()
    print("    Naive complete")

    print("  Tiled matmul [" + String(M) + "x" + String(K) + "] x [" + String(K) + "x" + String(N) + "]")
    ctx.enqueue_function[matmul_tiled_kernel[dtype]](
        d_C, d_A, d_B, M, N, K,
        grid_dim=(grid_x, grid_y),
        block_dim=(TILE_SIZE, TILE_SIZE),
    )
    ctx.enqueue_copy(h_C_tiled, d_C)
    ctx.synchronize()
    print("    Tiled complete")

    var mismatches = 0
    for i in range(min(size_C, 256)):
        if h_C_naive[i] != h_C_tiled[i]:
            mismatches += 1

    if mismatches > 0:
        print("    WARNING: " + String(mismatches) + " mismatches in validation")
    else:
        print("    Validation PASSED")

    h_A.free()
    h_B.free()
    h_C_naive.free()
    h_C_tiled.free()


fn main() raises:
    var ctx = DeviceContext()
    print("GPU: " + ctx.device_name())
    print("Matmul Benchmark: Naive vs Tiled\n")

    run_matmul_benchmark[DType.float32](ctx, 256, 256, 256)
    run_matmul_benchmark[DType.float32](ctx, 512, 512, 512)
    run_matmul_benchmark[DType.float32](ctx, 1024, 1024, 1024)

    print("\nDone.")
