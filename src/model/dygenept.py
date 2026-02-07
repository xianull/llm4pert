"""DyGenePT: Full model composing all four modules.

Data flow:
  1. GeneEncoder:      pert_gene_indices -> (B, P, K, D) facets
  2. CellEncoder:      scGPT tokens      -> (B, D) cell_query
  3. CrossAttention:   cell_query × facets -> (B, P, D) dynamic_emb
  4. Decoder:          dynamic_emb + cell_query + ctrl_expr -> (B, G) prediction
"""

import torch
import torch.nn as nn

from .gene_encoder import GeneEncoder
from .cell_encoder import CellEncoder
from .cross_attention import FacetCrossAttention
from .decoder import PerturbationDecoder


class DyGenePT(nn.Module):
    """Complete DyGenePT model for perturbation response prediction."""

    def __init__(self, cfg, num_genes: int):
        super().__init__()

        # Module 1: Gene facet embeddings (static, frozen)
        self.gene_encoder = GeneEncoder(
            facet_embeddings_path=cfg.paths.facet_embeddings,
            num_facets=cfg.facets.num_facets,
            embed_dim=cfg.embedding.embed_dim,
            freeze=True,
        )

        # Module 2: Cell state encoder (scGPT backbone)
        self.cell_encoder = CellEncoder(cfg)

        # Module 3: Cross-attention (Sparsemax)
        self.cross_attention = FacetCrossAttention(cfg)

        # Module 4: Perturbation decoder (latent arithmetic)
        self.decoder = PerturbationDecoder(cfg, num_genes)

        # Print parameter summary
        self._print_param_summary()

    def _print_param_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[DyGenePT] Parameters: trainable={trainable:,} / "
            f"total={total:,} ({100 * trainable / total:.1f}%)"
        )

    def forward(
        self,
        gene_ids: torch.LongTensor,         # (B, L) scGPT token IDs
        values: torch.Tensor,                # (B, L) expression values
        padding_mask: torch.BoolTensor,      # (B, L) True=masked
        pert_gene_indices: torch.LongTensor, # (B, P) indices into facet tensor
        ctrl_expression: torch.Tensor,       # (B, G) raw control expression
    ) -> dict:
        """
        Returns:
            dict with:
                'pred_expression': (B, G) predicted post-perturbation expression
                'attention_weights': (B, P, H, K) sparse facet attention weights
        """
        # Module 2: Cell state query
        cell_query = self.cell_encoder(
            gene_ids, values, padding_mask
        )  # (B, D)

        # Module 1: Gene facet lookup
        gene_facets = self.gene_encoder(
            pert_gene_indices
        )  # (B, P, K, D)

        # Module 3: Cross-attention
        dynamic_emb, attn_weights = self.cross_attention(
            cell_query, gene_facets
        )  # (B, P, D), (B, P, H, K)

        # Module 4: Decode prediction
        pred_expression = self.decoder(
            dynamic_emb, cell_query, ctrl_expression
        )  # (B, G)

        return {
            "pred_expression": pred_expression,
            "attention_weights": attn_weights,
        }
