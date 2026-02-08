"""Module 1: Multifaceted Gene Encoder.

Serves precomputed gene facet embeddings. Loads the static tensor
(num_genes, K, D) and provides lookup by gene index. Optionally
loads a confidence mask from KG-based facet imputation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .mlp_utils import build_mlp


class GeneEncoder(nn.Module):
    """Loads and serves precomputed gene facet embeddings.

    The tensor is frozen by default (no gradient). Each gene is
    represented by K facet vectors of dimension D.

    If the tensor file contains a "confidence" key (from KG imputation),
    it is loaded as a buffer and returned alongside embeddings.
    """

    def __init__(
        self,
        facet_embeddings_path: str,
        num_facets: int = 8,
        embed_dim: int = 768,
        freeze: bool = True,
        adapter_hidden_dims: list = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Load saved data: {"tensor": ..., "gene_to_idx": ..., "facet_names": ..., "confidence": ...}
        saved = torch.load(facet_embeddings_path, map_location="cpu", weights_only=False)
        self.gene_to_idx: Dict[str, int] = saved["gene_to_idx"]
        self.facet_names = saved["facet_names"]

        # Register as buffer (moves with model to device) or parameter
        if freeze:
            self.register_buffer("facet_tensor", saved["tensor"])  # (G, K, D)
        else:
            self.facet_tensor = nn.Parameter(saved["tensor"])

        # Derive actual num_facets from loaded tensor (may differ from config
        # if precompute was run with a different facet count)
        actual_k = self.facet_tensor.shape[1]
        if actual_k != num_facets:
            print(
                f"[GeneEncoder] WARNING: config num_facets={num_facets} but "
                f"loaded tensor has K={actual_k}. Using K={actual_k} from tensor."
            )
        self.num_facets = actual_k

        # Load confidence mask if present (from KG imputation)
        if "confidence" in saved:
            self.register_buffer("confidence_mask", saved["confidence"])  # (G, K)
            print(
                f"[GeneEncoder] Loaded confidence mask: "
                f"native={int((saved['confidence'] == 1.0).sum())}, "
                f"imputed={int(((saved['confidence'] > 0) & (saved['confidence'] < 1.0)).sum())}, "
                f"null={int((saved['confidence'] == 0).sum())}"
            )
        else:
            self.confidence_mask = None

        # Learnable null embedding for unknown genes (index = -1)
        self.null_embedding = nn.Parameter(
            torch.randn(1, self.num_facets, embed_dim) * 0.01
        )

        # Learnable adapter to fine-tune frozen embeddings for downstream task
        if adapter_hidden_dims is not None and len(adapter_hidden_dims) > 0:
            self.adapter = build_mlp(
                in_dim=embed_dim,
                out_dim=embed_dim,
                hidden_dims=adapter_hidden_dims,
                activation="gelu",
                norm="layernorm",
                final_activation=True,
            )
        else:
            # Default: single linear layer (backward compatible)
            self.adapter = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )

        print(
            f"[GeneEncoder] Loaded {self.facet_tensor.shape[0]} genes, "
            f"{self.num_facets} facets, dim={embed_dim}, freeze={freeze}"
        )

    def gene_symbols_to_indices(self, gene_symbols: list) -> torch.LongTensor:
        """Convert gene symbol strings to integer indices.

        Unknown genes get index -1 (mapped to null_embedding in forward).
        """
        return torch.LongTensor(
            [self.gene_to_idx.get(g, -1) for g in gene_symbols]
        )

    def forward(
        self, gene_indices: torch.LongTensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Look up facet embeddings for a batch of gene indices.

        Args:
            gene_indices: (batch, num_perts) integer indices.
                          -1 means unknown gene -> null embedding.

        Returns:
            embeddings: (batch, num_perts, K, D) facet embedding tensor.
            confidence: (batch, num_perts, K) confidence scores, or None
                        if no confidence mask was loaded.
        """
        # Create mask for unknown genes
        unknown_mask = gene_indices == -1  # (B, P)

        # Clamp to valid range for indexing
        safe_indices = gene_indices.clamp(min=0)  # (B, P)

        # Index into facet tensor: (B, P, K, D)
        embeddings = self.facet_tensor[safe_indices]

        # Apply learnable adapter to adapt frozen embeddings
        embeddings = self.adapter(embeddings)

        # Replace unknown gene embeddings with null embedding
        if unknown_mask.any():
            null = self.null_embedding.expand(
                -1, self.num_facets, self.embed_dim
            )  # (1, K, D)
            embeddings[unknown_mask] = null.squeeze(0)

        # Look up confidence if available
        confidence = None
        if self.confidence_mask is not None:
            confidence = self.confidence_mask[safe_indices]  # (B, P, K)
            if unknown_mask.any():
                confidence[unknown_mask] = 0.5  # moderate confidence for unknown genes

        return embeddings, confidence
