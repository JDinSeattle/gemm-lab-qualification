from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


CUTLASS_VERSION = "4.6.1"
CUTLASS_COMMIT = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
UPSTREAM_EXAMPLE = Path(
    "examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py"
)


def cutlass_source_root() -> Path:
    configured = os.environ.get("GEMM_LAB_CUTLASS_SOURCE")
    if configured:
        return Path(configured).expanduser().resolve()
    repository_root = Path(__file__).resolve().parent
    return repository_root / ".deps" / "cutlass-v4.6.1"


@lru_cache(maxsize=1)
def load_upstream_example() -> ModuleType:
    source = cutlass_source_root() / UPSTREAM_EXAMPLE
    if not source.is_file():
        raise RuntimeError(
            f"CUTLASS {CUTLASS_VERSION} Ampere example not found at {source}; "
            "run scripts/setup_cute.sh or set GEMM_LAB_CUTLASS_SOURCE"
        )
    module_name = "gemm_lab_cutlass_v461_tensorop_gemm"
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load CUTLASS example module from {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def validate_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("CuTe GEMM expects compatible rank-2 matrices")
    if a.device.type != "cuda" or a.device != b.device:
        raise ValueError("CuTe GEMM inputs must be on the same CUDA device")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CuTe GEMM inputs must share dtype float16 or bfloat16")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("CuTe GEMM adapter requires contiguous row-major inputs")
    _, k = a.shape
    _, n = b.shape
    if k % 8 or n % 8:
        raise ValueError(
            "CuTe Ampere skeleton requires 16-byte alignment: K and N must be multiples of 8"
        )


def atom_layout_for_shape(m: int) -> tuple[int, int, int]:
    # The upstream kernel keeps a 128x128 CTA tile.  For decode-like shapes,
    # map the four warps across N to avoid assigning two warp rows to padded M;
    # larger shapes use the balanced upstream 2x2 layout.
    return (1, 4, 1) if m <= 32 else (2, 2, 1)


class CuteMatmulPlan:
    """Compiled, fixed-buffer adapter around CUTLASS's Ampere CuTe example."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        validate_inputs(a, b)
        try:
            import cutlass
            from cuda.bindings import driver as cuda
        except ImportError as exc:
            raise RuntimeError(
                "CuTe DSL is not installed; run scripts/setup_cute.sh"
            ) from exc

        upstream = load_upstream_example()
        m, k = a.shape
        _, n = b.shape
        self.output = torch.empty((m, n), device=a.device, dtype=a.dtype)
        self.atom_layout_mnk = atom_layout_for_shape(m)
        self._cuda = cuda
        self._a = a.unsqueeze(0)
        self._b = b.unsqueeze(0)
        self._c = self.output.unsqueeze(0)
        ab_dtype: Any = cutlass.Float16 if a.dtype == torch.float16 else cutlass.BFloat16
        self._cute_a, self._cute_b, self._cute_c = upstream.mark_dynamic_layout(
            self._a,
            self._b,
            self._c,
            2,
            2,
            2,
            ab_dtype,
            ab_dtype,
        )
        self._compiled = upstream.compile_bmm(
            (m, n, k, 1),
            self._cute_a,
            self._cute_b,
            self._cute_c,
            ab_dtype,
            ab_dtype,
            cutlass.Float32,
            self.atom_layout_mnk,
            epilogue_op=lambda x: x,
        )

    def __call__(self) -> torch.Tensor:
        stream = self._cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._compiled(self._cute_a, self._cute_b, self._cute_c, stream)
        return self.output


def cute_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compile and execute one CuTe GEMM; benchmark callers should reuse a plan."""

    return CuteMatmulPlan(a, b)()
