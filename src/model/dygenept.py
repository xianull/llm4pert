"""DyGenePT: Full model composing all modules.

V1 data flow (legacy):
  1. GeneEncoder:      pert_gene_indices -> (B, P, K, D) facets
  2. CellEncoder:      scGPT/MLP         -> (B, D) cell_query
  3. CrossAttention:   cell_query × facets -> (B, P, D) dynamic_emb
  3.5 PertInteraction: cross-pert self-attn for combinatorial perturbations
  4. Decoder:          dynamic_emb + cell_query + ctrl_expr -> (B, G) prediction

V2 data flow (context-aware semantic cross-attention):
  1. GeneEncoder:       pert_gene_indices -> (B, P, K, D) facets
  2. MLPCellEncoder:    ctrl_expression   -> (B, D) cell_emb
  2.5 Contextualizer:   FiLM(cell_emb, facets) -> (B, P, K, D) context_Q
  3. SemanticCrossAttn: context_Q × genome_facets -> (B, P, G) impact_scores
  4. DecoderV2:         gate(ctrl) × impact × magnitude -> (B, G) prediction
"""

import torch
import torch.nn as nn

from .gene_encoder import GeneEncoder
from .cell_encoder import CellEncoder
from .ablation_cell_encoder import MLPCellEncoder, ConstantCellEncoder
from .cross_attention import FacetCrossAttention, SemanticCrossAttention
from .decoder import PerturbationDecoder, PerturbationDecoderV2
from .mlp_utils import build_mlp


# =====================================================================
# V1 helper: PerturbationInteraction (self-attention over P dimension)
# =====================================================================
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


# =====================================================================
# V2 helper: Contextualizer (FiLM conditioning)
# =====================================================================
class Contextualizer(nn.Module):
    """Per-facet FiLM conditioning: injects cell state into perturbed gene facets.

    Makes Q environment-specific AND facet-specific: different facets get
    different modulation in different cell types.

        gamma_k, beta_k = MLP(cell_emb)  for each facet k
        context_Q[:, :, k, :] = LayerNorm(gamma_k * facets[:, :, k, :] + beta_k)

    In K562 (p53-null): might amplify "Essentiality" facet, suppress "Regulatory Network"
    In RPE1 (p53-wt):  might amplify "Regulatory Network" facet, different essentiality pattern
    """

    def __init__(self, embed_dim: int, num_facets: int = 4, dropout: float = 0.1,
                 hidden_dims: list = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_facets = num_facets

        if hidden_dims is None:
            hidden_dims = [embed_dim]

        # Per-facet FiLM: cell_emb → K separate (gamma, beta) pairs
        # Output: K * 2 * D = num_facets * 2 * embed_dim
        self.film_net = build_mlp(
            in_dim=embed_dim,
            out_dim=num_facets * embed_dim * 2,  # K * (gamma + beta)
            hidden_dims=hidden_dims,
            activation="gelu",
            final_activation=False,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize gamma close to 1, beta close to 0 for each facet
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)
        # Set gamma bias to 1 for all K facets
        for k in range(num_facets):
            start = k * embed_dim * 2
            self.film_net[-1].bias.data[start:start + embed_dim] = 1.0

        print(f"[Contextualizer] embed_dim={embed_dim}, num_facets={num_facets}, "
              f"per-facet FiLM, hidden_dims={hidden_dims}")

    def forward(
        self,
        cell_emb: torch.Tensor,       # (B, D)
        pert_facets: torch.Tensor,     # (B, P, K, D)
    ) -> torch.Tensor:
        """
        Returns:
            context_Q: (B, P, K, D) cell-conditioned facet representations.
        """
        B = cell_emb.size(0)
        D = self.embed_dim
        K = self.num_facets

        film_out = self.film_net(cell_emb)                     # (B, K*2*D)
        film_out = film_out.view(B, K, 2, D)                   # (B, K, 2, D)
        gamma = film_out[:, :, 0, :].unsqueeze(1)              # (B, 1, K, D)
        beta = film_out[:, :, 1, :].unsqueeze(1)               # (B, 1, K, D)

        # Per-facet modulation: different gamma/beta for each facet
        out = self.layer_norm(gamma * pert_facets + beta)       # (B, P, K, D)
        return self.dropout(out)


# =====================================================================
# Main model
# =====================================================================
class DyGenePT(nn.Module):
    """Complete DyGenePT model for perturbation response prediction.

    Supports two architecture versions controlled by cfg.architecture_version:
      - "v1": Original architecture (cell_query attends over pert gene facets)
      - "v2": Context-aware semantic cross-attention (pert gene attends over genome)
    """

    def __init__(self, cfg, num_genes: int, gene_names: list = None):
        super().__init__()
        self.arch_version = getattr(cfg, 'architecture_version', 'v1')
        target_dim = cfg.cross_attention.hidden_dim
        num_facets = cfg.facets.num_facets
        embed_dim = cfg.embedding.embed_dim

        # Module 1: Gene facet embeddings (shared, frozen)
        ge_cfg = getattr(cfg, 'gene_encoder', None)
        adapter_hidden_raw = getattr(ge_cfg, 'adapter_hidden_dims', None) if ge_cfg else None
        adapter_hidden = list(adapter_hidden_raw) if adapter_hidden_raw is not None else None

        self.gene_encoder = GeneEncoder(
            facet_embeddings_path=cfg.paths.facet_embeddings,
            num_facets=num_facets,
            embed_dim=embed_dim,
            freeze=True,
            adapter_hidden_dims=adapter_hidden,
        )

        if self.arch_version == 'v2':
            self._init_v2(cfg, num_genes, gene_names, target_dim, num_facets, embed_dim)
        else:
            self._init_v1(cfg, num_genes, gene_names, target_dim)

        # Print parameter summary
        self._print_param_summary()

    # -----------------------------------------------------------------
    # V1 initialization
    # -----------------------------------------------------------------
    def _init_v1(self, cfg, num_genes, gene_names, target_dim):
        """Initialize v1 architecture (legacy)."""
        # Cell encoder
        cell_encoder_type = getattr(cfg.cell_encoder, 'model_name', 'scGPT')
        self._cell_encoder_type = cell_encoder_type
        if cell_encoder_type == 'mlp':
            cell_hidden = list(getattr(cfg.cell_encoder, 'hidden_dims', None) or [1024])
            self.cell_encoder = MLPCellEncoder(num_genes, target_dim, hidden_dims=cell_hidden)
        elif cell_encoder_type == 'constant':
            self.cell_encoder = ConstantCellEncoder(target_dim)
        else:
            self.cell_encoder = CellEncoder(cfg)

        # Cross-attention (Sparsemax over K facets)
        self.cross_attention = FacetCrossAttention(cfg)

        # Build aligned facet tensor for gene_embed_aware decoder
        gene_embed_aware = getattr(cfg.decoder, 'gene_embed_aware', False)
        aligned_facet_tensor = None
        if gene_embed_aware:
            aligned_facet_tensor = self._build_aligned_facets(num_genes, gene_names)

        # Decoder
        self.decoder = PerturbationDecoder(
            cfg, num_genes, gene_facet_tensor=aligned_facet_tensor,
        )

        # Cross-perturbation interaction
        self.pert_interaction = PerturbationInteraction(
            embed_dim=cfg.cross_attention.hidden_dim,
            num_heads=4,
            dropout=cfg.cross_attention.dropout,
        )

        # Perturbation type embedding
        pt_cfg = getattr(cfg, 'perturbation_type', None)
        if pt_cfg is not None and getattr(pt_cfg, 'enabled', False):
            self.pert_type_embedding = nn.Embedding(
                num_embeddings=pt_cfg.num_types,
                embedding_dim=cfg.cross_attention.hidden_dim,
            )
            nn.init.normal_(self.pert_type_embedding.weight, std=0.02)
        else:
            self.pert_type_embedding = None

    # -----------------------------------------------------------------
    # V2 initialization
    # -----------------------------------------------------------------
    def _init_v2(self, cfg, num_genes, gene_names, target_dim, num_facets, embed_dim):
        """Initialize v2 architecture (context-aware semantic cross-attention)."""
        self._cell_encoder_type = 'mlp'

        # --- Read configurable hidden dims with backward-compatible defaults ---
        cell_hidden = list(getattr(cfg.cell_encoder, 'hidden_dims', None) or [1024])

        ctx_cfg = getattr(cfg, 'contextualizer', None)
        ctx_dropout = ctx_cfg.dropout if ctx_cfg else cfg.cross_attention.dropout
        ctx_hidden_raw = getattr(ctx_cfg, 'hidden_dims', None) if ctx_cfg else None
        ctx_hidden = list(ctx_hidden_raw) if ctx_hidden_raw is not None else None

        impact_hidden = list(getattr(cfg.decoder, 'v2_impact_hidden_dims', None) or [1024, 1024])
        gate_hidden = list(getattr(cfg.decoder, 'v2_gate_hidden_dims', None) or [1024])

        # Module 2: MLP cell encoder (replaces scGPT)
        self.cell_encoder = MLPCellEncoder(num_genes, target_dim, hidden_dims=cell_hidden)

        # Build aligned genome facets: (num_genes, K, D) — frozen, no adapter
        assert gene_names is not None, (
            "gene_names is required for v2 architecture"
        )
        aligned_facets = self._build_aligned_facets(num_genes, gene_names)

        # Use actual K from loaded tensor (may differ from config if
        # precompute was run with a different facet count)
        actual_k = self.gene_encoder.num_facets

        # Module 2.5: Contextualizer (per-facet FiLM)
        self.contextualizer = Contextualizer(
            target_dim, num_facets=actual_k, dropout=ctx_dropout, hidden_dims=ctx_hidden,
        )

        # Module 3: Semantic cross-attention (pert → genome)
        topk = getattr(cfg.cross_attention, 'topk', 200)
        self.cross_attention = SemanticCrossAttention(
            embed_dim=target_dim,
            num_facets=actual_k,
            dropout=cfg.cross_attention.dropout,
            topk=topk,
        )
        # Register genome facets as frozen buffer inside cross_attention
        self.cross_attention.register_buffer('_genome_facets', aligned_facets)

        # Module 4: Simplified decoder
        self.decoder = PerturbationDecoderV2(
            embed_dim=target_dim,
            num_genes=num_genes,
            dropout=getattr(cfg.decoder, 'dropout', 0.1),
            impact_hidden_dims=impact_hidden,
            gate_hidden_dims=gate_hidden,
        )

        # No perturbation type embedding in v2 (CRISPRi only)
        self.pert_type_embedding = None

    # -----------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------
    def _build_aligned_facets(self, num_genes, gene_names):
        """Build (num_genes, K, D) facet tensor aligned to gene_names order."""
        facet_tensor = self.gene_encoder.facet_tensor   # (G_vocab, K, D)
        gene_to_idx = self.gene_encoder.gene_to_idx     # {symbol: int}
        K, D = facet_tensor.shape[1], facet_tensor.shape[2]
        aligned = torch.zeros(num_genes, K, D)
        matched = 0
        for i, name in enumerate(gene_names):
            if name in gene_to_idx:
                aligned[i] = facet_tensor[gene_to_idx[name]]
                matched += 1
        print(
            f"[DyGenePT] Gene-embed alignment: {matched}/{num_genes} "
            f"genes matched to facet embeddings"
        )
        return aligned

    def _print_param_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100 * trainable / total if total > 0 else 0
        print(
            f"[DyGenePT-{self.arch_version}] Parameters: "
            f"trainable={trainable:,} / total={total:,} ({pct:.1f}%)"
        )

    # -----------------------------------------------------------------
    # Forward dispatch
    # -----------------------------------------------------------------
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
                'attention_weights': attention info for interpretability
        """
        if self.arch_version == 'v2':
            return self._forward_v2(
                pert_gene_indices, ctrl_expression,
            )
        return self._forward_v1(
            gene_ids, values, padding_mask,
            pert_gene_indices, ctrl_expression,
            pert_type_ids, pert_flags,
        )

    # -----------------------------------------------------------------
    # V1 forward (legacy, unchanged logic)
    # -----------------------------------------------------------------
    def _forward_v1(
        self,
        gene_ids, values, padding_mask,
        pert_gene_indices, ctrl_expression,
        pert_type_ids=None, pert_flags=None,
    ):
        # Module 2: Cell state query
        if self._cell_encoder_type == 'scGPT':
            cell_query = self.cell_encoder(
                gene_ids, values, padding_mask, pert_flags=pert_flags
            )
        else:
            cell_query = self.cell_encoder(ctrl_expression)

        # Module 1: Gene facet lookup
        gene_facets, confidence = self.gene_encoder(pert_gene_indices)

        # Perturbation type embedding
        pert_type_emb = None
        if self.pert_type_embedding is not None and pert_type_ids is not None:
            pert_type_emb = self.pert_type_embedding(pert_type_ids)

        # Module 3: Cross-attention
        dynamic_emb, attn_weights = self.cross_attention(
            cell_query, gene_facets, confidence, pert_type_emb=pert_type_emb
        )

        # Add pert_type_emb as residual
        if pert_type_emb is not None:
            dynamic_emb = dynamic_emb + pert_type_emb

        # Module 3.5: Cross-perturbation interaction
        dynamic_emb = self.pert_interaction(dynamic_emb)

        # Module 4: Decode prediction
        pred_expression = self.decoder(
            dynamic_emb, cell_query, ctrl_expression
        )

        return {
            "pred_expression": pred_expression,
            "attention_weights": attn_weights,
        }

    # -----------------------------------------------------------------
    # V2 forward (context-aware semantic cross-attention)
    # -----------------------------------------------------------------
    def _forward_v2(self, pert_gene_indices, ctrl_expression):
        # Module 2: Cell state via MLP
        cell_emb = self.cell_encoder(ctrl_expression)               # (B, D)

        # Module 1: Perturbed gene facet lookup
        pert_facets, confidence = self.gene_encoder(pert_gene_indices)  # (B,P,K,D), (B,P,K)

        # Module 2.5: Contextualizer — inject cell state into Q
        context_Q = self.contextualizer(cell_emb, pert_facets)      # (B, P, K, D)

        # Module 3: Semantic cross-attention (pert → genome)
        impact_scores, facet_w = self.cross_attention(
            context_Q,
            self.cross_attention._genome_facets,
            confidence,
        )  # (B, P, G), (K,)

        # Combine across perturbations: sum valid slots
        pert_mask = (pert_gene_indices >= 0).float()                # (B, P)
        impact_map = (impact_scores * pert_mask.unsqueeze(-1)).sum(dim=1)  # (B, G)

        # Module 4: Decode prediction
        pred_expression = self.decoder(
            impact_map, cell_emb, ctrl_expression
        )  # (B, G)

        return {
            "pred_expression": pred_expression,
            "attention_weights": facet_w,
        }
