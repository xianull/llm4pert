"""Shared MLP builder for configurable layer stacks.

Used by MLPCellEncoder, GeneEncoder adapter, Contextualizer, and DecoderV2
to construct variable-depth nn.Sequential from a hidden_dims list.
"""

from typing import List

import torch.nn as nn


_ACT = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: List[int],
    activation: str = "gelu",
    norm: str = "none",
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    """Build a variable-depth MLP from a list of hidden dimensions.

    Produces::

        in -> h1 -> act [-> norm] [-> drop] -> h2 -> ... -> out [-> act -> norm -> drop]

    When ``final_activation=False``, the **last element** in the returned
    Sequential is guaranteed to be ``nn.Linear``.  This invariant is relied
    upon by callers that zero-init the output layer.

    Args:
        in_dim:  Input feature dimension.
        out_dim: Output feature dimension.
        hidden_dims: List of intermediate widths.  ``[]`` produces a single
            Linear(in_dim, out_dim).
        activation: ``"gelu"`` or ``"relu"``.
        norm: ``"layernorm"`` after each block, or ``"none"``.
        dropout: Dropout rate (0 disables).
        final_activation: Whether to apply activation (+ norm + dropout)
            after the output Linear.
    """
    act_cls = _ACT[activation]
    layers: list = []
    prev = in_dim

    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        if norm == "layernorm":
            layers.append(nn.LayerNorm(h))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    # Output layer
    layers.append(nn.Linear(prev, out_dim))

    if final_activation:
        layers.append(act_cls())
        if norm == "layernorm":
            layers.append(nn.LayerNorm(out_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)
