"""Ablation cell encoders for testing scGPT contribution.

Three variants to replace scGPT CellEncoder:
  - MLPCellEncoder:      ctrl_expression -> MLP -> cell_query
  - ConstantCellEncoder: learnable constant (no cell-specific info)

Usage: set cfg.cell_encoder.model_name to "mlp" or "constant".
"""

import torch
import torch.nn as nn


class MLPCellEncoder(nn.Module):
    """Encodes cell state from raw control expression via MLP.

    Tests whether scGPT pretraining adds value beyond what a
    simple learned projection of the expression vector provides.
    """

    def __init__(self, num_genes: int, target_dim: int = 768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, target_dim),
            nn.GELU(),
            nn.LayerNorm(target_dim),
        )

        trainable = sum(p.numel() for p in self.parameters())
        print(
            f"[MLPCellEncoder] num_genes={num_genes}, target_dim={target_dim}, "
            f"trainable={trainable:,}"
        )

    def forward(self, ctrl_expression: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            ctrl_expression: (B, num_genes) raw control expression.

        Returns:
            cell_query: (B, target_dim)
        """
        return self.encoder(ctrl_expression)


class ConstantCellEncoder(nn.Module):
    """Learnable constant shared across all cells.

    Tests whether cell-specific information matters at all
    for cross-attention query. If this performs similarly to
    scGPT, then cell state is not contributing meaningful signal.
    """

    def __init__(self, target_dim: int = 768):
        super().__init__()
        self.constant = nn.Parameter(torch.randn(1, target_dim) * 0.02)

        print(f"[ConstantCellEncoder] target_dim={target_dim}, trainable={target_dim}")

    def forward(self, ctrl_expression: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            ctrl_expression: (B, num_genes) only used for batch size.

        Returns:
            cell_query: (B, target_dim) same vector for every cell.
        """
        B = ctrl_expression.size(0)
        return self.constant.expand(B, -1)
