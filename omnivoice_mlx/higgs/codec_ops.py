"""Snake activation and weight-normalised Conv1d used by the Higgs acoustic codec."""
# Vendored from mlx-audio (MIT, github.com/Blaizzy/mlx-audio) so the codec needs
# neither mlx-audio nor transformers at runtime. Local edits are marked "# vendored:".
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def normalize_weight(x: mx.array, except_dim: int = 0) -> mx.array:
    """Compute weight normalization factor."""
    if x.ndim != 3:
        raise ValueError("Input tensor must have 3 dimensions")
    axes = tuple(i for i in range(x.ndim) if i != except_dim)
    return mx.sqrt(mx.sum(x * x, axis=axes, keepdims=True))


# =============================================================================
# Basic Layers
# =============================================================================


def snake(x: mx.array, alpha: mx.array) -> mx.array:
    """Snake activation function.

    Computed in float32 to avoid inf/NaN when x or alpha is float16:
    alpha near zero in float16 makes 1/(alpha+eps) = inf, and inf*sin²(0) = NaN.
    """
    dtype = x.dtype
    x32 = x.astype(mx.float32)
    a32 = alpha.astype(mx.float32)
    recip = 1.0 / (a32 + 1e-9)
    return (x32 + recip * mx.power(mx.sin(a32 * x32), 2)).astype(dtype)


class Snake1d(nn.Module):
    """Snake activation for 1D signals."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, 1, channels))

    def __call__(self, x: mx.array) -> mx.array:
        return snake(x, self.alpha)


class WNConv1d(nn.Module):
    """Weight-normalized 1D convolution with optional causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
        pad_mode: str = "none",
        norm: str = "weight_norm",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.causal = causal
        self.pad_mode = pad_mode
        self.use_weight_norm = norm == "weight_norm"

        # Calculate padding for pad_mode="none"
        if pad_mode == "none":
            self.padding = (kernel_size - stride) * dilation // 2
        else:
            self.padding = 0

        if self.use_weight_norm:
            scale = math.sqrt(1 / (in_channels * kernel_size))
            weight_init = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels, kernel_size, in_channels),
            )
            self.weight_g = normalize_weight(weight_init)
            self.weight_v = weight_init / (self.weight_g + 1e-12)
        else:
            scale = math.sqrt(1 / (in_channels * kernel_size))
            self.weight = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(out_channels, kernel_size, in_channels),
            )

        self.bias = mx.zeros((out_channels,)) if bias else None

    def _get_weight(self):
        if self.use_weight_norm:
            return self.weight_g * self.weight_v / normalize_weight(self.weight_v)
        return self.weight

    def _auto_pad(self, x: mx.array) -> mx.array:
        """Apply automatic padding for causal/non-causal convolutions."""
        if self.pad_mode == "none":
            return x

        length = x.shape[1]
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        padding_total = effective_kernel_size - self.stride
        n_frames = (length - effective_kernel_size + padding_total) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (
            self.kernel_size - padding_total
        )
        extra_padding = max(0, ideal_length - length)

        if self.causal:
            # Causal: all padding on left
            pad_left = padding_total
            pad_right = extra_padding
        else:
            # Non-causal: symmetric padding
            pad_right = extra_padding // 2
            pad_left = padding_total - pad_right + extra_padding - pad_right

        if pad_left > 0 or pad_right > 0:
            x = mx.pad(x, [(0, 0), (pad_left, pad_right), (0, 0)])

        return x

    def __call__(self, x: mx.array) -> mx.array:
        x = self._auto_pad(x)
        weight = self._get_weight()
        y = mx.conv1d(x, weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            y = y + self.bias
        return y
