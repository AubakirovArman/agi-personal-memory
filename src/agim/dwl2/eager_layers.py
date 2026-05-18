from __future__ import annotations

import torch
from torch import Tensor, nn

class EagerBf16Linear(nn.Linear):
    """Materialize the route-decoded weight once and reuse the standard nn.Linear path.

    Inherits from nn.Linear so HuggingFace accelerate / torch.compile / fused
    attention dispatchers see an identical layer signature to the baseline.
    Storage on disk stays 3-bit (codebook), VRAM holds a single bf16 weight
    exactly like the baseline nn.Linear.
    """

    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        out_features = int(ids.shape[0])
        in_features = int(ids.shape[1])
        device = ids.device
        # Skip the default Parameter init (we overwrite it); use empty meta-style.
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        # Build weight directly in target dtype to avoid an fp32 transient.
        codebook_sum = codebook_w.sum(dim=-1).to(target_dtype)
        weight = (codebook_sum[ids.long()] * row_scale.to(target_dtype)).contiguous()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(target_dtype).contiguous(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        target_dtype: torch.dtype = torch.bfloat16,
        **_: object,
    ) -> "EagerBf16Linear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias, target_dtype=target_dtype)

    def reconstruct_weight(self) -> Tensor:
        return self.weight.detach().float()


class EagerBlockRVQLinear(nn.Linear):
    """Materialize a block-RVQ approximation once and reuse the standard nn.Linear path."""

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        out_features = int(weight.shape[0])
        in_features = int(weight.shape[1])
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(weight.to(target_dtype).contiguous(), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(target_dtype).contiguous(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def reconstruct_weight(self) -> Tensor:
        return self.weight.detach().float()

class EagerFp8Linear(nn.Module):
    """Materialize route weight to FP8 (e4m3) with per-row scale, run via torch._scaled_mm.

    H200-native FP8 storage path: ~50% VRAM versus ``EagerBf16Linear``.

    Numerical recipe (best-quality FP8 inference combo on H200):
      * Weight: e4m3 (more precision, less range) with per-output-row scale.
      * Activation: e5m2 (more range, less precision) with per-token scale.
      * Output: bf16 to preserve residual stream precision.

    The route row_scale already captures per-row dynamic range optimally; we
    re-derive it here from the decoded amax to honor the actual fp8 codomain.
    """

    FP8_W_MAX = 448.0    # max |x| representable in float8_e4m3fn
    FP8_A_MAX = 57344.0  # max |x| representable in float8_e5m2

    def __init__(
        self,
        ids: Tensor,
        codebook_w: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        x_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        super().__init__()
        out_features = int(ids.shape[0])
        in_features = int(ids.shape[1])
        self.in_features = in_features
        self.out_features = out_features
        self.x_dtype = x_dtype
        self._x_max = self.FP8_A_MAX if x_dtype is torch.float8_e5m2 else self.FP8_W_MAX

        codebook_sum = codebook_w.sum(dim=-1).to(torch.float32)
        weight_fp32 = codebook_sum[ids.long()] * row_scale.to(torch.float32)
        amax = weight_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        weight_scale = (amax / self.FP8_W_MAX).to(torch.float32)
        w_q = (weight_fp32 / weight_scale).clamp(-self.FP8_W_MAX, self.FP8_W_MAX).to(torch.float8_e4m3fn)
        del weight_fp32

        self.register_buffer("weight_fp8", w_q.contiguous())
        self.register_buffer("weight_scale", weight_scale.view(1, -1).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.bfloat16).contiguous())
        else:
            self.bias = None

    @classmethod
    def from_encoded(
        cls,
        ids: Tensor,
        codebook_digits: Tensor,
        ladder: Tensor,
        row_scale: Tensor,
        bias: Tensor | None = None,
        *,
        x_dtype: torch.dtype = torch.float8_e4m3fn,
        **_: object,
    ) -> "EagerFp8Linear":
        codebook_w = (codebook_digits.to(torch.float32) * ladder.to(torch.float32)).to(torch.float16)
        return cls(ids, codebook_w, row_scale.to(torch.float16), bias, x_dtype=x_dtype)

    def reconstruct_weight(self) -> Tensor:
        return (self.weight_fp8.float() * self.weight_scale.view(-1, 1).float())

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        x_amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        x_scale = (x_amax.float() / self._x_max)
        x_q = (x_flat / x_amax * self._x_max).clamp(-self._x_max, self._x_max).to(self.x_dtype)
        out = torch._scaled_mm(
            x_q,
            self.weight_fp8.t(),
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)
