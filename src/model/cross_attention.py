"""Module 3: Cross-Attention modules.

V1: FacetCrossAttention — cell query attends over K gene facets (sparsemax).
V2: SemanticCrossAttention — context-aware pert gene attends over entire genome.
"""

import math

import torch
import torch.nn as nn
from entmax import sparsemax


class FacetCrossAttention(nn.Module):
    """Multi-head cross-attention: cell query attends to gene facets.

    Q = project(cell_query)          ->  (B, H, 1, d_head)
    K = project(gene_facets)         ->  (B, P, H, K, d_head)
    V = project(gene_facets)         ->  (B, P, H, K, d_head)
    scores = Q·K^T / sqrt(d_head)    ->  (B, P, H, K)
    alpha  = sparsemax(scores, -1)   ->  (B, P, H, K)   [sparse!]
    out    = alpha · V               ->  (B, P, H, d_head)
    result = concat(heads) @ W_O     ->  (B, P, D)
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.cross_attention.hidden_dim   # 768
        self.num_heads = cfg.cross_attention.num_heads     # 8
        self.head_dim = self.hidden_dim // self.num_heads  # 96
        self.dropout_p = cfg.cross_attention.dropout

        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )

        # Projection matrices
        self.W_Q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_K = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_V = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_O = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.attn_dropout = nn.Dropout(self.dropout_p)

    def forward(
        self,
        cell_query: torch.Tensor,    # (B, D)
        gene_facets: torch.Tensor,   # (B, P, K, D)
        confidence: torch.Tensor = None,  # (B, P, K) optional
        pert_type_emb: torch.Tensor = None,  # (B, P, D) optional
    ) -> tuple:
        """
        Args:
            cell_query:  Cell state embedding from CellEncoder.  (B, D)
            gene_facets: Facet embeddings for perturbed gene(s).  (B, P, K, D)
            confidence:  Optional confidence scores from KG imputation. (B, P, K)
                         1.0=native, (0,0.8]=imputed, 0.0=still NULL.
                         Used as log-space bias on attention scores.
            pert_type_emb: Optional perturbation type embedding. (B, P, D)
                         When provided, conditions Q per-perturbation so the
                         model can select different facets for activation vs
                         repression vs knockout.

        Returns:
            dynamic_emb:       (B, P, D)          context-aware gene embeddings
            attention_weights: (B, P, H, K)       sparse weights for interpretability
        """
        B = gene_facets.size(0)
        P = gene_facets.size(1)
        n_facets = gene_facets.size(2)
        D = gene_facets.size(3)
        H = int(self.num_heads)
        d = int(self.head_dim)

        # --- Q from cell query, optionally conditioned on perturbation type ---
        if pert_type_emb is not None:
            # Per-perturbation query: cell state + perturbation type
            # cell_query: (B, D) -> (B, 1, D) -> broadcast to (B, P, D)
            conditioned_query = cell_query.unsqueeze(1) + pert_type_emb  # (B, P, D)
            Q_exp = self.W_Q(conditioned_query).view(B, P, H, 1, d)     # (B, P, H, 1, d)
        else:
            # Original behavior: single shared Q across all P
            Q = self.W_Q(cell_query).view(B, H, 1, d)
            Q_exp = Q.unsqueeze(1).expand(B, P, H, 1, d)

        # --- K, V from gene facets ---
        # (B, P, K, D) -> (B, P, K, D) -> (B, P, K, H, d) -> (B, P, H, K, d)
        K = self.W_K(gene_facets).view(B, P, n_facets, H, d).permute(0, 1, 3, 2, 4)
        V = self.W_V(gene_facets).view(B, P, n_facets, H, d).permute(0, 1, 3, 2, 4)

        # --- Attention scores ---
        # scores = Q @ K^T / sqrt(d)  ->  (B, P, H, 1, K) -> squeeze -> (B, P, H, K)
        scores = torch.matmul(Q_exp, K.transpose(-2, -1)).squeeze(-2)
        scores = scores / math.sqrt(d)

        # --- Confidence bias ---
        # Imputed facets get lower attention via log(confidence) additive bias.
        # Native facets (conf=1.0) get bias=0, imputed get negative bias,
        # NULL facets (conf≈0) get strongly negative bias.
        if confidence is not None:
            # (B, P, K) -> (B, P, 1, K) to broadcast over H heads
            conf_bias = torch.log(confidence.clamp(min=1e-6)).unsqueeze(2)
            scores = scores + conf_bias

        # --- Sparsemax attention ---
        # Apply sparsemax over the K (facet) dimension
        alpha = sparsemax(scores, dim=-1)   # (B, P, H, K)
        alpha = self.attn_dropout(alpha)

        # --- Weighted combination ---
        # alpha: (B, P, H, K) -> (B, P, H, 1, K)
        # V:     (B, P, H, K, d)
        # out:   (B, P, H, 1, d) -> squeeze -> (B, P, H, d)
        out = torch.matmul(alpha.unsqueeze(-2), V).squeeze(-2)

        # --- Concatenate heads and project ---
        # (B, P, H, d) -> (B, P, H*d) = (B, P, D)
        out = out.reshape(B, P, self.hidden_dim)
        out = self.W_O(out)
        out = self.layer_norm(out)

        return out, alpha


class SemanticCrossAttention(nn.Module):
    """V2: Context-aware perturbed gene facets attend over all genome genes.

    Q = W_Q(context_Q)           ->  (B, P, K, D)   context-aware pert facets
    K = W_K(genome_facets)       ->  (G, K, D)       all genes' static facets
    sim = einsum(Q, K)           ->  (B, P, K, G)    per-facet similarity
    impact = combine(sim, w)     ->  (B, P, G)       weighted facet aggregation
    impact = topk_sparse(impact) ->  (B, P, G)       sparse impact scores

    Unlike V1 which attends over 8 facets of the perturbed gene,
    V2 attends over the entire genome to compute a direct impact map.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_facets: int = 8,
        dropout: float = 0.1,
        topk: int = 200,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_facets = num_facets
        self.topk = topk

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)

        # Learnable facet channel weights (which facets matter more)
        self.facet_weights = nn.Parameter(
            torch.ones(num_facets) / num_facets
        )

        self.scale = embed_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        print(
            f"[SemanticCrossAttention] embed_dim={embed_dim}, "
            f"num_facets={num_facets}, topk={topk}"
        )

    def forward(
        self,
        context_Q: torch.Tensor,            # (B, P, K, D)
        genome_facets: torch.Tensor,         # (G, K, D)  frozen buffer
        confidence: torch.Tensor = None,     # (B, P, K)  optional
    ) -> tuple:
        """
        Args:
            context_Q:      FiLM-conditioned perturbed gene facets.
            genome_facets:  Static facet embeddings for ALL genes (frozen).
            confidence:     KG imputation confidence for pert gene facets.

        Returns:
            impact_scores:  (B, P, G) per-gene impact from each perturbation.
            facet_weights:  (K,) normalized facet channel weights.
        """
        # Project Q and K
        Q = self.W_Q(context_Q)      # (B, P, K, D)
        K = self.W_K(genome_facets)   # (G, K, D)

        # Per-facet channel similarity via einsum
        # (B, P, K, D) × (G, K, D) -> (B, P, K, G)
        sim = torch.einsum('bpkd,gkd->bpkg', Q, K) * self.scale

        # Confidence bias: imputed Q-side facets get lower attention
        if confidence is not None:
            conf_bias = torch.log(confidence.clamp(min=1e-6))  # (B, P, K)
            sim = sim + conf_bias.unsqueeze(-1)  # (B, P, K, 1) broadcast over G

        # Learned facet combination: weighted sum over K channels
        w = torch.softmax(self.facet_weights, dim=0)  # (K,)
        impact = torch.einsum('bpkg,k->bpg', sim, w)  # (B, P, G)

        # Top-k sparsification with straight-through gradient
        if self.topk > 0 and self.topk < impact.size(-1):
            _, topk_idx = impact.topk(self.topk, dim=-1)       # (B, P, topk)
            mask = torch.zeros_like(impact)
            mask.scatter_(-1, topk_idx, 1.0)
            # Straight-through: gradient flows through impact, mask is detached
            impact = impact * mask.detach()

        impact = self.dropout(impact)

        return impact, w
