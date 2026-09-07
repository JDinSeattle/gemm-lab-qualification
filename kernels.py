from __future__ import annotations

import torch
import triton
import triton.language as tl


def _autotune_configs() -> list[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]


@triton.autotune(configs=_autotune_configs(), key=["M", "N", "K", "dtype_id"])
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dtype_id: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + ((program_id % programs_per_group) % group_size_m)
    program_n = (program_id % programs_per_group) // group_size_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_pointers = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_pointers = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for inner_tile in range(0, tl.cdiv(K, BLOCK_K)):
        remaining_k = K - inner_tile * BLOCK_K
        a = tl.load(
            a_pointers,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        b = tl.load(
            b_pointers,
            mask=(offsets_k[:, None] < remaining_k) & (offsets_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_pointers += BLOCK_K * stride_ak
        b_pointers += BLOCK_K * stride_bk

    output_pointers = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(output_pointers, accumulator, mask=output_mask)


def _validate_inputs(a: torch.Tensor, b: torch.Tensor, operation: str) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"{operation} expects two rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible dimensions: {tuple(a.shape)} and {tuple(b.shape)}")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError(f"{operation} requires CUDA tensors")
    if a.device != b.device:
        raise ValueError("input tensors must be on the same CUDA device")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("inputs must share dtype float16 or bfloat16")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError(f"{operation} requires contiguous row-major inputs")
    return int(a.shape[0]), int(b.shape[1]), int(a.shape[1])


def triton_matmul_into(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    m, n, k = _validate_inputs(a, b, "triton_matmul")
    if output.shape != (m, n) or output.device != a.device or output.dtype != a.dtype:
        raise ValueError("output tensor has the wrong shape, device, or dtype")
    if not output.is_contiguous():
        raise ValueError("output tensor must be contiguous")

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a,
        b,
        output,
        M=m,
        N=n,
        K=k,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        dtype_id=0 if a.dtype == torch.float16 else 1,
    )
    return output


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n, _ = _validate_inputs(a, b, "triton_matmul")
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    return triton_matmul_into(a, b, output)


class TritonMatmulPlan:
    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        m, n, _ = _validate_inputs(a, b, "triton_matmul")
        self.a = a
        self.b = b
        self.output = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def __call__(self) -> torch.Tensor:
        return triton_matmul_into(self.a, self.b, self.output)


@triton.autotune(configs=_autotune_configs(), key=["M", "N", "K", "dtype_id"])
@triton.jit
def _matmul_bias_silu_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dtype_id: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    program_id = tl.program_id(axis=0)
    programs_m = tl.cdiv(M, BLOCK_M)
    programs_n = tl.cdiv(N, BLOCK_N)
    programs_per_group = GROUP_M * programs_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * GROUP_M
    group_size_m = tl.minimum(programs_m - first_program_m, GROUP_M)
    program_m = first_program_m + ((program_id % programs_per_group) % group_size_m)
    program_n = (program_id % programs_per_group) // group_size_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_pointers = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_pointers = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for inner_tile in range(0, tl.cdiv(K, BLOCK_K)):
        remaining_k = K - inner_tile * BLOCK_K
        a = tl.load(
            a_pointers,
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < remaining_k),
            other=0.0,
        )
        b = tl.load(
            b_pointers,
            mask=(offsets_k[:, None] < remaining_k) & (offsets_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        a_pointers += BLOCK_K * stride_ak
        b_pointers += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < N, other=0.0)
    activated = (accumulator + bias[None, :]) * tl.sigmoid(accumulator + bias[None, :])
    output_pointers = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(output_pointers, activated, mask=output_mask)


def triton_bias_silu_into(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    m, n, k = _validate_inputs(a, b, "triton_bias_silu")
    if bias.ndim != 1 or bias.shape[0] != n or bias.device != a.device or bias.dtype != a.dtype:
        raise ValueError("bias tensor has the wrong shape, device, or dtype")
    if output.shape != (m, n) or output.device != a.device or output.dtype != a.dtype:
        raise ValueError("output tensor has the wrong shape, device, or dtype")
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    _matmul_bias_silu_kernel[grid](
        a,
        b,
        bias,
        output,
        M=m,
        N=n,
        K=k,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        dtype_id=0 if a.dtype == torch.float16 else 1,
    )
    return output


def triton_bias_silu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, n, _ = _validate_inputs(a, b, "triton_bias_silu")
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    return triton_bias_silu_into(a, b, bias, output)


def adaptive_bias_silu_path(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype | None = None,
) -> str:
    measured_mlp_shape = (k, n) in ((4096, 11008), (11008, 4096))
    if dtype == torch.float16 and m <= 8 and measured_mlp_shape:
        return "triton_fused"
    return "torch_unfused"


def adaptive_bias_silu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    m, n, k = _validate_inputs(a, b, "adaptive_bias_silu")
    if adaptive_bias_silu_path(m, n, k, a.dtype) == "triton_fused":
        return triton_bias_silu(a, b, bias)
    return torch.nn.functional.silu(torch.matmul(a, b) + bias)


class TritonBiasSiluPlan:
    def __init__(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> None:
        m, n, _ = _validate_inputs(a, b, "triton_bias_silu")
        self.a = a
        self.b = b
        self.bias = bias
        self.output = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def __call__(self) -> torch.Tensor:
        return triton_bias_silu_into(self.a, self.b, self.bias, self.output)


@triton.jit
def _splitk_matmul_kernel(
    a_ptr,
    b_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_program = tl.program_id(axis=0)
    split_id = tl.program_id(axis=1)
    programs_n = tl.cdiv(N, BLOCK_N)
    program_m = output_program // programs_n
    program_n = output_program % programs_n

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_offsets_k = tl.arange(0, BLOCK_K)
    k_per_split = tl.cdiv(K, SPLIT_K)
    split_start = split_id * k_per_split
    split_end = tl.minimum(split_start + k_per_split, K)
    global_offsets_k = split_start + local_offsets_k

    a_pointers = a_ptr + offsets_m[:, None] * stride_am + global_offsets_k[None, :] * stride_ak
    b_pointers = b_ptr + global_offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(k_per_split, BLOCK_K)):
        k_mask = global_offsets_k < split_end
        a = tl.load(
            a_pointers,
            mask=(offsets_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_pointers,
            mask=k_mask[:, None] & (offsets_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)
        global_offsets_k += BLOCK_K
        a_pointers += BLOCK_K * stride_ak
        b_pointers += BLOCK_K * stride_bk

    partial_offsets = split_id * M * N + offsets_m[:, None] * N + offsets_n[None, :]
    output_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(partial_ptr + partial_offsets, accumulator, mask=output_mask)


@triton.jit
def _splitk_reduce_kernel(
    partial_ptr,
    output_ptr,
    ELEMENTS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for split_id in range(0, SPLIT_K):
        accumulator += tl.load(partial_ptr + split_id * ELEMENTS + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, accumulator, mask=mask)


def triton_splitk_matmul_into(
    a: torch.Tensor,
    b: torch.Tensor,
    partial: torch.Tensor,
    output: torch.Tensor,
    split_k: int = 4,
) -> torch.Tensor:
    if split_k not in (2, 4, 8):
        raise ValueError("split_k must be 2, 4, or 8")
    m, n, k = _validate_inputs(a, b, "triton_splitk_matmul")
    if partial.shape != (split_k, m, n) or partial.device != a.device or partial.dtype != torch.float32:
        raise ValueError("partial tensor has the wrong shape, device, or dtype")
    if output.shape != (m, n) or output.device != a.device or output.dtype != a.dtype:
        raise ValueError("output tensor has the wrong shape, device, or dtype")
    block_m = 16 if m <= 16 else 32
    block_n = 128
    block_k = 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), split_k)
    _splitk_matmul_kernel[grid](
        a,
        b,
        partial,
        M=m,
        N=n,
        K=k,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=4,
        num_warps=4,
    )
    elements = m * n
    _splitk_reduce_kernel[(triton.cdiv(elements, 256),)](
        partial,
        output,
        ELEMENTS=elements,
        SPLIT_K=split_k,
        BLOCK=256,
        num_warps=4,
    )
    return output


def triton_splitk_matmul(a: torch.Tensor, b: torch.Tensor, split_k: int = 4) -> torch.Tensor:
    m, n, _ = _validate_inputs(a, b, "triton_splitk_matmul")
    partial = torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    return triton_splitk_matmul_into(a, b, partial, output, split_k)


class TritonSplitKPlan:
    def __init__(self, a: torch.Tensor, b: torch.Tensor, split_k: int = 4) -> None:
        if split_k not in (2, 4, 8):
            raise ValueError("split_k must be 2, 4, or 8")
        m, n, _ = _validate_inputs(a, b, "triton_splitk_matmul")
        self.a = a
        self.b = b
        self.split_k = split_k
        self.partial = torch.empty((split_k, m, n), device=a.device, dtype=torch.float32)
        self.output = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def workspace_bytes(self) -> int:
        """FP32 partial-sum storage held outside the output tensor."""

        return self.partial.numel() * self.partial.element_size()

    def __call__(self) -> torch.Tensor:
        return triton_splitk_matmul_into(
            self.a, self.b, self.partial, self.output, self.split_k
        )


class AdaptiveMatmulPlan:
    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        m, n, k = _validate_inputs(a, b, "triton_adaptive_matmul")
        self.selected_path = adaptive_path(m, n, k, a.dtype)
        if self.selected_path == "splitk8":
            self.plan = TritonSplitKPlan(a, b, 8)
        else:
            output = torch.empty((m, n), device=a.device, dtype=a.dtype)
            self.plan = lambda: torch.matmul(a, b, out=output)

    def workspace_bytes(self) -> int:
        """Zero on the vendor path; Split-K partials otherwise."""

        inner = getattr(self.plan, "workspace_bytes", None)
        return inner() if callable(inner) else 0

    def __call__(self) -> torch.Tensor:
        return self.plan()


def adaptive_path(m: int, n: int, k: int, dtype: torch.dtype | None = None) -> str:
    """Return the evidence-backed custom/vendor path for a linear layer."""

    max_custom_m = 8 if dtype == torch.bfloat16 else 32
    if m <= max_custom_m and n == 4096 and k == 4096:
        return "splitk8"
    return "torch_vendor"


def triton_adaptive_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible rank-2 matrix shapes")
    path = adaptive_path(int(a.shape[0]), int(b.shape[1]), int(a.shape[1]), a.dtype)
    if path == "splitk8":
        return triton_splitk_matmul(a, b, split_k=8)
    return torch.matmul(a, b)


# Backward-compatible name for early artifacts; new code should use the
# implementation-neutral AdaptiveMatmulPlan.
TritonAdaptivePlan = AdaptiveMatmulPlan


@torch.library.custom_op("gemm_lab::triton_matmul", mutates_args=())
def triton_matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inference-oriented PyTorch custom op backed by the Triton kernel."""

    return triton_matmul(a, b)


@triton_matmul_op.register_fake
def _triton_matmul_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matrix shapes")
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)


@torch.library.custom_op("gemm_lab::triton_splitk_matmul", mutates_args=())
def triton_splitk_matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Two-pass split-K custom op for small-M inference shapes."""

    return triton_splitk_matmul(a, b, split_k=4)


@triton_splitk_matmul_op.register_fake
def _triton_splitk_matmul_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matrix shapes")
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)


@torch.library.custom_op("gemm_lab::triton_adaptive_matmul", mutates_args=())
def triton_adaptive_matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inference custom op with a shape-aware small-M Split-K dispatch."""

    return triton_adaptive_matmul(a, b)


@triton_adaptive_matmul_op.register_fake
def _triton_adaptive_matmul_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matrix shapes")
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)


@torch.library.custom_op("gemm_lab::adaptive_matmul", mutates_args=())
def adaptive_matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Evidence-bounded Split-K dispatch with a vendor fallback."""

    return triton_adaptive_matmul(a, b)


@adaptive_matmul_op.register_fake
def _adaptive_matmul_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matrix shapes")
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)


@torch.library.custom_op("gemm_lab::triton_bias_silu", mutates_args=())
def triton_bias_silu_op(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Fused inference epilogue: output = SiLU(A @ B + bias)."""

    return triton_bias_silu(a, b, bias)


@triton_bias_silu_op.register_fake
def _triton_bias_silu_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or bias.ndim != 1 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matrix/bias shapes")
    if b.shape[1] != bias.shape[0]:
        raise ValueError("bias length must match output columns")
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)


@torch.library.custom_op("gemm_lab::adaptive_bias_silu", mutates_args=())
def adaptive_bias_silu_op(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Evidence-bounded fused epilogue with a PyTorch sequence fallback."""

    return adaptive_bias_silu(a, b, bias)


@adaptive_bias_silu_op.register_fake
def _adaptive_bias_silu_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or bias.ndim != 1 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matrix/bias shapes")
    if b.shape[1] != bias.shape[0]:
        raise ValueError("bias length must match output columns")
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
