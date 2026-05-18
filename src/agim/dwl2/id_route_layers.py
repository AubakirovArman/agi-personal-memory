from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .triton_id_matmul import id_route_linear_matmul

class PackedIDRouteLinear(nn.Module):
    def __init__(
        self,
        ids: Tensor,              # int16[N,K] or int32[N,K]
        codebook_w: Tensor,       # fp16[M, L_max] with already-signed scales
        row_scale: Tensor,        # fp16[N,1]
        bias: Tensor | None = None,
        *,
        use_int16: bool = True,   # Triton kernels need int32
    ) -> None:
        super().__init__()
        # store ids as int16 if codebook fits to halve bandwidth vs int32
        if use_int16 and codebook_w.shape[0] <= 32767 and ids.dtype != torch.int16:
            ids = ids.to(torch.int16)
        self.register_buffer("ids", ids.contiguous())
        # codebook_sum is the per-route scalar: sum_l (digit[l] * ladder[l])
        # single fp16 per route → M*2 bytes total (tiny, L2-cacheable)
        self.register_buffer("codebook_sum", codebook_w.sum(dim=-1).contiguous().to(torch.float16))
        self.register_buffer("row_scale", row_scale.to(torch.float16).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16).contiguous())
        else:
            self.bias = None
        self.out_features = ids.shape[0]
        self.in_features = ids.shape[1]

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,           # int32[N,K]
        codebook_digits: Tensor,   # int8[M, L_max]
        ladder: Tensor,        # fp32[L_max]
        row_scale: Tensor,     # fp16 or fp32[N,1]
        bias: Tensor | None = None,
        **_: object,
    ) -> "PackedIDRouteLinear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias)

    def reconstruct_weight(self) -> Tensor:
        # gather per-position route sums, multiply by row scale → fp16 weight matrix
        w = self.codebook_sum[self.ids.long()]         # fp16[N,K]
        w = w * self.row_scale                         # broadcast over K
        return w

    def forward(self, x: Tensor) -> Tensor:
        w = self.reconstruct_weight()
        return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class FusedIDRouteLinear(PackedIDRouteLinear):
    def __init__(self, ids, codebook_w, row_scale, bias=None):
        # Triton kernel doesn't support int16 pointers; keep int32
        super().__init__(ids.to(torch.int32), codebook_w, row_scale, bias, use_int16=False)

    def forward(self, x: Tensor) -> Tensor:
        out = id_route_linear_matmul(x, self.ids, self.codebook_sum, self.row_scale)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class CachedPackedIDRouteLinear(PackedIDRouteLinear):
    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        max_cache_bytes: int = 128 * 2**20,
    ) -> None:
        super().__init__(ids, codebook_w, row_scale, bias)
        self.max_cache_bytes = max(int(max_cache_bytes), 0)
        self._cached_weight: Tensor | None = None
        self._cached_weight_device: torch.device | None = None
        self._cached_weight_dtype: torch.dtype | None = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.cache_skip_count = 0

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        max_cache_bytes: int = 128 * 2**20,
        **_: object,
    ) -> "CachedPackedIDRouteLinear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias, max_cache_bytes=max_cache_bytes)

    def clear_cache(self) -> None:
        self._cached_weight = None
        self._cached_weight_device = None
        self._cached_weight_dtype = None

    def _weight_for(self, x: Tensor) -> Tensor:
        weight_bytes = self.out_features * self.in_features * x.element_size()
        if self.max_cache_bytes > 0 and weight_bytes > self.max_cache_bytes:
            self.cache_skip_count += 1
            return self.reconstruct_weight().to(x.dtype).contiguous()
        if (
            self._cached_weight is None
            or self._cached_weight_device != x.device
            or self._cached_weight_dtype != x.dtype
        ):
            self._cached_weight = self.reconstruct_weight().to(x.dtype).contiguous()
            self._cached_weight_device = x.device
            self._cached_weight_dtype = x.dtype
            self.cache_miss_count += 1
        else:
            self.cache_hit_count += 1
        return self._cached_weight

    def forward(self, x: Tensor) -> Tensor:
        weight = self._weight_for(x)
        return F.linear(x, weight, self.bias.to(x.dtype) if self.bias is not None else None)

class AdaptiveFusedIDRouteLinear(PackedIDRouteLinear):
    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        validate_calls: int = 1,
        disable_primary_after_nonfinite: bool = True,
        shadow_validate: bool = False,
        shadow_rel_mse_tol: float = 1e-4,
    ) -> None:
        super().__init__(ids, codebook_w, row_scale, bias)
        self.validate_calls = max(int(validate_calls), 0)
        self.disable_primary_after_nonfinite = disable_primary_after_nonfinite
        self.shadow_validate = shadow_validate
        self.shadow_rel_mse_tol = float(shadow_rel_mse_tol)
        self._validation_calls_left = self.validate_calls
        self._primary_enabled = True
        self.fallback_count = 0
        self.shadow_mismatch_count = 0
        self.last_validation_rel_mse: float | None = None

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        validate_calls: int = 1,
        disable_primary_after_nonfinite: bool = True,
        shadow_validate: bool = False,
        shadow_rel_mse_tol: float = 1e-4,
        **_: object,
    ) -> "AdaptiveFusedIDRouteLinear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(
            ids,
            codebook_w,
            row_scale.to(torch.float16),
            bias,
            validate_calls=validate_calls,
            disable_primary_after_nonfinite=disable_primary_after_nonfinite,
            shadow_validate=shadow_validate,
            shadow_rel_mse_tol=shadow_rel_mse_tol,
        )

    def _primary_forward(self, x: Tensor) -> Tensor:
        out = id_route_linear_matmul(x, self.ids, self.codebook_sum, self.row_scale)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def forward(self, x: Tensor) -> Tensor:
        if not self._primary_enabled:
            return super().forward(x)
        try:
            out = self._primary_forward(x)
        except RuntimeError:
            self.fallback_count += 1
            if self.disable_primary_after_nonfinite:
                self._primary_enabled = False
            return super().forward(x)
        if self._validation_calls_left > 0:
            self._validation_calls_left -= 1
            if not bool(torch.isfinite(out).all().item()):
                self.fallback_count += 1
                if self.disable_primary_after_nonfinite:
                    self._primary_enabled = False
                return super().forward(x)
            if self.shadow_validate:
                reference = super().forward(x)
                denom = reference.float().square().mean().clamp_min(1e-12)
                rel_mse = float(((out.float() - reference.float()).square().mean() / denom).item())
                self.last_validation_rel_mse = rel_mse
                if rel_mse > self.shadow_rel_mse_tol:
                    self.shadow_mismatch_count += 1
                    if self.disable_primary_after_nonfinite:
                        self._primary_enabled = False
                    return reference
        return out
