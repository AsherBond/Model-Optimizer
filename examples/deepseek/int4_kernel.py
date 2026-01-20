import torch
import triton
import triton.language as tl


@triton.jit
def int4_dequant_kernel(
    x_ptr,  # pointer to int32 input [M, N]
    s_ptr,  # pointer to bf16 scale [M, 8*N//BLOCK_SIZE]
    y_ptr,  # pointer to bf16 output [M, 8N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    m = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)

    # Load int32 values, unroll to int4.
    NUM_INT32_PER_BLOCK = BLOCK_SIZE // 8
    int32_vals = tl.load(
        x_ptr + m * N + n_block * NUM_INT32_PER_BLOCK + tl.arange(0, BLOCK_SIZE) // 8
    )

    offset = (tl.arange(0, BLOCK_SIZE) % 8) * 4
    vals = ((int32_vals >> offset) & 0xF) - 8

    # # Compute scale per block
    # # Each scale covers block_size contiguous y
    scale = tl.load(s_ptr + m * 8 * N // BLOCK_SIZE + n_block)

    vals = vals.to(tl.float32) * scale.to(tl.float32)
    tl.store(y_ptr + m * N * 8 + n_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), vals)


def int4_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """
    Dequantizes a packed int4 tensor to bf16.

    Args:
        x: int32 tensor of shape [M, N] with packed int4.
        s: bf16 tensor of shape [M, 8 * N // BLOCK_SIZE].
        block_size: number of output columns per block.

    Returns: bf16 tensor of shape [M, 8 * N]
    """
    m, n = x.shape
    y = torch.empty((m, 8 * n), dtype=torch.get_default_dtype(), device=x.device)

    grid = (m, 8 * n // block_size)
    int4_dequant_kernel[grid](x, s, y, m, n, BLOCK_SIZE=block_size)
    return y
