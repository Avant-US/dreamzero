"""Static FP8 Linear layer with per-channel weight scaling.

Pre-quantizes weights to FP8 E4M3 at construction time with per-output-channel
scales (rowwise). At runtime, activations are dynamically quantized per-row.
No TE dependency — uses torch._scaled_mm directly.

Compatible with torch.compile(mode="reduce-overhead") because all state is
tensor-based (no Python bookkeeping that triggers guard failures).

Usage:
    replace_linear_with_fp8(model)  # post-load, converts nn.Linear → StaticFP8Linear
"""
from __future__ import annotations

import torch
import torch.nn as nn

_FP8_MAX = 448.0  # max representable value for E4M3


# Register as a custom op so torch.compile can trace through it
@torch.library.custom_op("dreamzero::fp8_linear", mutates_args=())
def _fp8_linear_op(x: torch.Tensor, weight_fp8: torch.Tensor,
                    weight_scale_inv: torch.Tensor) -> torch.Tensor:
    """Rowwise FP8 linear: per-row activation scale + per-channel weight scale."""
    orig_shape = x.shape
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])

    M = x.shape[0]
    # Per-row activation quantization
    x_absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    x_scale = x_absmax / _FP8_MAX
    x_fp8 = (x / x_scale).to(torch.float8_e4m3fn)
    x_scale_inv = x_scale.float().reciprocal()  # (M, 1)

    # weight_scale_inv is (1, out_features) for rowwise _scaled_mm
    out = torch._scaled_mm(x_fp8, weight_fp8.t(),
                            scale_a=x_scale_inv, scale_b=weight_scale_inv,
                            out_dtype=torch.bfloat16)
    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], weight_fp8.shape[0])
    return out

@_fp8_linear_op.register_fake
def _fp8_linear_fake(x: torch.Tensor, weight_fp8: torch.Tensor,
                      weight_scale_inv: torch.Tensor) -> torch.Tensor:
    out_features = weight_fp8.shape[0]
    return torch.empty(*x.shape[:-1], out_features, dtype=torch.bfloat16, device=x.device)


class StaticFP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with per-channel FP8 weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Placeholder — actual FP8 weight set by from_linear()
        self.register_buffer("weight_fp8", torch.empty(out_features, in_features,
                             device=device, dtype=torch.float8_e4m3fn))
        # Per-channel scale: (1, out_features) for rowwise _scaled_mm
        self.register_buffer("weight_scale_inv", torch.ones(1, out_features,
                             device=device, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device,
                                                  dtype=dtype or torch.bfloat16))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "StaticFP8Linear":
        """Convert a pretrained nn.Linear to per-channel FP8."""
        m = cls(linear.in_features, linear.out_features,
                bias=linear.bias is not None,
                device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            w = linear.weight.data.float()
            # Per-output-channel (row-wise) quantization
            w_absmax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            w_scale = w_absmax / _FP8_MAX  # (out_features, 1)
            m.weight_fp8 = (w / w_scale).to(torch.float8_e4m3fn)
            # _scaled_mm wants scale_b as (1, out_features) for rowwise
            m.weight_scale_inv = (1.0 / w_scale).t().to(torch.float32)  # (1, out_features)
            if linear.bias is not None:
                m.bias = nn.Parameter(linear.bias.data.clone())
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.ops.dreamzero.fp8_linear(x, self.weight_fp8, self.weight_scale_inv)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_linear_with_fp8(module: nn.Module, target_modules: set[str] | None = None,
                             _prefix: str = "") -> int:
    """Replace nn.Linear with StaticFP8Linear in-place. Returns count of replaced modules.

    Args:
        module: root module to transform
        target_modules: if set, only replace modules whose full name contains one of these strings.
                       e.g. {"self_attn", "ffn"} to only replace attention and FFN linears.
    """
    count = 0
    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name
        if isinstance(child, nn.Linear):
            if target_modules is None or any(t in full_name for t in target_modules):
                setattr(module, name, StaticFP8Linear.from_linear(child))
                count += 1
        elif isinstance(child, nn.Sequential):
            for i, subchild in enumerate(child):
                sub_name = f"{full_name}.{i}"
                if isinstance(subchild, nn.Linear):
                    if target_modules is None or any(t in sub_name for t in target_modules):
                        child[i] = StaticFP8Linear.from_linear(subchild)
                        count += 1
                else:
                    count += replace_linear_with_fp8(subchild, target_modules, sub_name)
        else:
            count += replace_linear_with_fp8(child, target_modules, full_name)
    return count
