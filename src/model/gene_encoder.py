"""Module 1: Multifaceted Gene Encoder.

Serves precomputed gene facet embeddings. Loads the static tensor
(num_genes, K, D) and provides lookup by gene index.
"""

import torch
import torch.nn as nn
from typing import Dict


class GeneEncoder(nn.Module):
    """Loads and serves precomputed gene facet embeddings.

    The tensor is frozen by default (no gradient). Each gene is
    represented by K facet vectors of dimension D.
    """

    def __init__(
        self,
        facet_embeddings_path: str,
        num_facets: int = 8,
        embed_dim: int = 768,
        freeze: bool = True,
    ):
        super().__init__()
        self.num_facets = num_facets
        self.embed_dim = embed_dim

        # Load saved data: {"tensor": ..., "gene_to_idx": ..., "facet_names": ...}
        saved = torch.load(facet_embeddings_path, map_location="cpu", weights_only=False)
        self.gene_to_idx: Dict[str, int] = saved["gene_to_idx"]
        self.facet_names = saved["facet_names"]

        # Register as buffer (moves with model to device) or parameter
        if freeze:
            self.register_buffer("facet_tensor", saved["tensor"])  # (G, K, D)
        else:
            self.facet_tensor = nn.Parameter(saved["tensor"])

        # Learnable null embedding for unknown genes (index = -1)
        self.null_embedding = nn.Parameter(
            torch.randn(1, num_facets, embed_dim) * 0.01
        )

        print(
            f"[GeneEncoder] Loaded {self.facet_tensor.shape[0]} genes, "
            f"{num_facets} facets, dim={embed_dim}, freeze={freeze}"
        )

    def gene_symbols_to_indices(self, gene_symbols: list) -> torch.LongTensor:
        """Convert gene symbol strings to integer indices.

        Unknown genes get index -1 (mapped to null_embedding in forward).
        """
        return torch.LongTensor(
            [self.gene_to_idx.get(g, -1) for g in gene_symbols]
        )

    def forward(self, gene_indices: torch.LongTensor) -> torch.Tensor:
        """Look up facet embeddings for a batch of gene indices.

        Args:
            gene_indices: (batch, num_perts) integer indices.
                          -1 means unknown gene -> null embedding.

        Returns:
            (batch, num_perts, K, D) facet embedding tensor.
        """
        # Create mask for unknown genes
        unknown_mask = gene_indices == -1  # (B, P)

        # Clamp to valid range for indexing
        safe_indices = gene_indices.clamp(min=0)  # (B, P)

        # Index into facet tensor: (B, P, K, D)
        embeddings = self.facet_tensor[safe_indices]

        # Replace unknown gene embeddings with null embedding
        if unknown_mask.any():
            null = self.null_embedding.expand(
                -1, self.num_facets, self.embed_dim
            )  # (1, K, D)
            embeddings[unknown_mask] = null.squeeze(0)

        return embeddings
