"""Module 3: Context-Aware Cross-Attention with Sparsemax.

The cell query vector attends over K gene facets to produce a dynamic,
cell-context-specific gene embedding. Sparsemax replaces softmax for
interpretable, sparse attention weights.
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
    ) -> tuple:
        """
        Args:
            cell_query:  Cell state embedding from CellEncoder.  (B, D)
            gene_facets: Facet embeddings for perturbed gene(s).  (B, P, K, D)
            confidence:  Optional confidence scores from KG imputation. (B, P, K)
                         1.0=native, (0,0.8]=imputed, 0.0=still NULL.
                         Used as log-space bias on attention scores.

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

        # --- Q from cell query ---
        # (B, D) -> (B, D) -> (B, H, 1, d)
        Q = self.W_Q(cell_query).view(B, H, 1, d)

        # --- K, V from gene facets ---
        # (B, P, K, D) -> (B, P, K, D) -> (B, P, K, H, d) -> (B, P, H, K, d)
        K = self.W_K(gene_facets).view(B, P, n_facets, H, d).permute(0, 1, 3, 2, 4)
        V = self.W_V(gene_facets).view(B, P, n_facets, H, d).permute(0, 1, 3, 2, 4)

        # --- Attention scores ---
        # Q: (B, H, 1, d) -> expand to (B, P, H, 1, d)
        Q_exp = Q.unsqueeze(1).expand(B, P, H, 1, d)

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
