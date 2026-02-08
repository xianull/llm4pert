"""DyGenePT: Full model composing all four modules.

Data flow:
  1. GeneEncoder:      pert_gene_indices -> (B, P, K, D) facets
  2. CellEncoder:      scGPT tokens      -> (B, D) cell_query
  3. CrossAttention:   cell_query × facets -> (B, P, D) dynamic_emb
  3.5 PertInteraction: cross-pert self-attn for combinatorial perturbations
  4. Decoder:          dynamic_emb + cell_query + ctrl_expr -> (B, G) prediction
"""

import torch
import torch.nn as nn

from .gene_encoder import GeneEncoder
from .cell_encoder import CellEncoder
from .cross_attention import FacetCrossAttention
from .decoder import PerturbationDecoder


class PerturbationInteraction(nn.Module):
    """Cross-perturbation self-attention for combinatorial perturbations.

    When multiple genes are perturbed simultaneously (P>1), each gene's
    dynamic embedding should be aware of the other perturbed genes.
    Self-attention over the P dimension lets gene A's representation
    be influenced by gene B and vice versa, before the decoder computes
    shift vectors.

    For P=1 (single perturbation), this reduces to a learned linear
    transformation with residual (self-attention on one token).

    Key property: permutation-equivariant, so A+B == B+A automatically.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, D) dynamic embeddings for each perturbed gene.

        Returns:
            (B, P, D) interaction-enriched embeddings.
        """
        residual = x
        out, _ = self.self_attn(x, x, x)
        return self.norm(residual + self.dropout(out))


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

        # Module 3.5: Cross-perturbation interaction (for combinatorial perts)
        self.pert_interaction = PerturbationInteraction(
            embed_dim=cfg.cross_attention.hidden_dim,
            num_heads=4,
            dropout=cfg.cross_attention.dropout,
        )

        # Perturbation type embedding (activation / repression / knockout)
        pt_cfg = getattr(cfg, 'perturbation_type', None)
        if pt_cfg is not None and getattr(pt_cfg, 'enabled', False):
            self.pert_type_embedding = nn.Embedding(
                num_embeddings=pt_cfg.num_types,           # 3
                embedding_dim=cfg.cross_attention.hidden_dim,  # 768
            )
            nn.init.normal_(self.pert_type_embedding.weight, std=0.02)
        else:
            self.pert_type_embedding = None

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
        pert_type_ids: torch.LongTensor = None,  # (B, P) perturbation type indices
        pert_flags: torch.LongTensor = None,     # (B, L) per-token pert flags (PACE)
    ) -> dict:
        """
        Returns:
            dict with:
                'pred_expression': (B, G) predicted post-perturbation expression
                'attention_weights': (B, P, H, K) sparse facet attention weights
        """
        # Module 2: Cell state query (with optional PACE pert_flags)
        cell_query = self.cell_encoder(
            gene_ids, values, padding_mask, pert_flags=pert_flags
        )  # (B, D)

        # Module 1: Gene facet lookup (returns confidence if available)
        gene_facets, confidence = self.gene_encoder(
            pert_gene_indices
        )  # (B, P, K, D), optional (B, P, K)

        # Perturbation type embedding
        pert_type_emb = None
        if self.pert_type_embedding is not None and pert_type_ids is not None:
            pert_type_emb = self.pert_type_embedding(pert_type_ids)  # (B, P, D)

        # Module 3: Cross-attention (with confidence bias and pert type conditioning)
        dynamic_emb, attn_weights = self.cross_attention(
            cell_query, gene_facets, confidence, pert_type_emb=pert_type_emb
        )  # (B, P, D), (B, P, H, K)

        # Add pert_type_emb as residual to dynamic_emb (belt-and-suspenders)
        if pert_type_emb is not None:
            dynamic_emb = dynamic_emb + pert_type_emb

        # Module 3.5: Cross-perturbation interaction
        # Lets co-perturbed genes exchange information before decoding.
        # For P=1 this is a lightweight pass-through with residual.
        dynamic_emb = self.pert_interaction(dynamic_emb)  # (B, P, D)

        # Module 4: Decode prediction
        pred_expression = self.decoder(
            dynamic_emb, cell_query, ctrl_expression
        )  # (B, G)

        return {
            "pred_expression": pred_expression,
            "attention_weights": attn_weights,
        }
