"""Module 4: Perturbation Decoder (scLAMBDA-style latent arithmetic).

Predicts post-perturbation expression via:
    z_pred = z_control + shift(dynamic_emb)
    pred_expression = decode(z_pred) + ctrl_expression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp_utils import build_mlp


class PerturbationDecoder(nn.Module):
    """MLP-based decoder using latent space arithmetic.

    Architecture:
      1. shift_encoder: maps dynamic_emb (D) -> shift vector (latent_dim)
      2. cell_projector: maps cell_query (D) -> latent representation (latent_dim)
      3. latent arithmetic: z_pred = z_control + sum(shifts)
      4. expression_decoder: maps z_pred (latent_dim) -> full transcriptome (num_genes)
      5. residual: pred = decoded + ctrl_expression

    For combinatorial perturbations (2 genes), shift vectors are summed.

    If gene_embed_aware=True, the last linear layer is replaced by a
    gene-embedding-aware output: effect @ gene_embeddings.T + gene_bias,
    enabling generalization to unseen genes.
    """

    def __init__(self, cfg, num_genes: int, gene_facet_tensor: torch.Tensor = None):
        super().__init__()
        self.embed_dim = cfg.cross_attention.hidden_dim  # 768
        self.num_genes = num_genes
        hidden_dims = list(cfg.decoder.hidden_dims)      # [512, 256]
        dropout = cfg.decoder.dropout
        self.gene_embed_aware = getattr(cfg.decoder, 'gene_embed_aware', False)

        # --- Shift encoder: dynamic_emb -> latent shift ---
        shift_layers = []
        in_dim = self.embed_dim
        for h_dim in hidden_dims:
            shift_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.shift_encoder = nn.Sequential(*shift_layers)
        self.latent_dim = hidden_dims[-1]  # 256

        # --- Cell state projector: cell_query -> latent ---
        self.cell_projector = nn.Sequential(
            nn.Linear(self.embed_dim, self.latent_dim),
            nn.GELU(),
            nn.LayerNorm(self.latent_dim),
        )

        # --- Expression decoder: latent (with cell skip) -> full transcriptome ---
        backbone_hidden = 1024  # intermediate dimension in backbone
        if self.gene_embed_aware:
            # Backbone: first two layers (non-linear feature extraction)
            self.expression_backbone = nn.Sequential(
                nn.Linear(self.latent_dim * 2, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, backbone_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Gene-embedding-aware output head
            d_out = getattr(cfg.decoder, 'gene_embed_dim', 512)
            self.effect_proj = nn.Linear(backbone_hidden, d_out)
            self.gene_bias = nn.Parameter(torch.zeros(num_genes))

            # Learnable temperature for scaled dot-product
            self.logit_scale = nn.Parameter(torch.tensor(1.0 / (d_out ** 0.5)))

            # Project gene facet embeddings -> gene_embeddings
            assert gene_facet_tensor is not None, (
                "gene_facet_tensor is required when gene_embed_aware=True"
            )
            # gene_facet_tensor: (G, K, 768) -> mean pool -> (G, 768)
            gene_mean = gene_facet_tensor.mean(dim=1)  # (G, 768)

            # Identify unmatched genes (all-zero rows from alignment)
            unmatched_mask = gene_mean.norm(dim=-1) == 0  # (G,)
            self.register_buffer("_unmatched_mask", unmatched_mask)

            # Learnable fallback embedding for unmatched genes
            self._fallback_raw = nn.Parameter(
                torch.randn(1, gene_facet_tensor.shape[-1]) * 0.02
            )

            self.register_buffer("_gene_mean", gene_mean)  # frozen raw embeddings

            self.gene_embed_proj = nn.Sequential(
                nn.Linear(gene_facet_tensor.shape[-1], d_out),
                nn.GELU(),
                nn.LayerNorm(d_out),
            )

            num_unmatched = int(unmatched_mask.sum().item())
            print(
                f"[PerturbationDecoder] gene_embed_aware=True, d_out={d_out}, "
                f"gene_mean={gene_mean.shape}, unmatched={num_unmatched}"
            )
        else:
            self.expression_decoder = nn.Sequential(
                nn.Linear(self.latent_dim * 2, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, backbone_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(backbone_hidden, num_genes),
            )

        # --- ctrl_expression encoder for gate conditioning ---
        self.ctrl_encoder = nn.Sequential(
            nn.Linear(num_genes, self.latent_dim),
            nn.GELU(),
        )

        # --- Residual gate: conditioned on both pred_latent and ctrl ---
        self.gate_layer = nn.Linear(self.latent_dim * 2, num_genes)

        print(
            f"[PerturbationDecoder] embed_dim={self.embed_dim}, "
            f"latent_dim={self.latent_dim}, num_genes={num_genes}, "
            f"gene_embed_aware={self.gene_embed_aware}"
        )

    def forward(
        self,
        dynamic_emb: torch.Tensor,     # (B, P, D)
        cell_query: torch.Tensor,       # (B, D)
        ctrl_expression: torch.Tensor,  # (B, num_genes)
    ) -> torch.Tensor:
        """
        Args:
            dynamic_emb:     Context-aware gene embeddings from cross-attention.
            cell_query:      Cell state from CellEncoder.
            ctrl_expression: Baseline (control) expression vector.

        Returns:
            pred_expression: (B, num_genes) predicted post-perturbation expression.
        """
        # 1. Compute shift vectors for each perturbation
        #    (B, P, D) -> (B, P, latent_dim)
        shifts = self.shift_encoder(dynamic_emb)

        # 2. Sum shifts across perturbations
        #    Cross-perturbation interaction is handled upstream (Module 3.5)
        #    (B, P, latent_dim) -> (B, latent_dim)
        combined_shift = shifts.sum(dim=1)

        # 3. Project cell query to latent space
        #    (B, D) -> (B, latent_dim)
        cell_latent = self.cell_projector(cell_query)

        # 4. Latent arithmetic: control + shift
        pred_latent = cell_latent + combined_shift  # (B, latent_dim)

        # 5. Decode to full transcriptome (delta prediction)
        #    cell_latent skip connection: provide unmixed cell state to decoder
        decoder_input = torch.cat([pred_latent, cell_latent], dim=-1)  # (B, latent_dim*2)
        if self.gene_embed_aware:
            h = self.expression_backbone(decoder_input)       # (B, 1024)
            effect = self.effect_proj(h)                       # (B, d_out)

            # Build gene embeddings: replace unmatched zeros with fallback
            gene_raw = self._gene_mean.clone()
            if self._unmatched_mask.any():
                gene_raw[self._unmatched_mask] = self._fallback_raw.expand(
                    int(self._unmatched_mask.sum()), -1
                )
            gene_emb = self.gene_embed_proj(gene_raw)          # (G, d_out)

            # L2-normalize + learnable temperature for stable dot-product
            effect_norm = F.normalize(effect, dim=-1)          # (B, d_out)
            gene_emb_norm = F.normalize(gene_emb, dim=-1)      # (G, d_out)
            delta = self.logit_scale * (effect_norm @ gene_emb_norm.t()) + self.gene_bias
        else:
            delta = self.expression_decoder(decoder_input)    # (B, G)

        # 6. Residual gate: conditioned on perturbation latent + baseline expression
        ctrl_encoded = self.ctrl_encoder(ctrl_expression)  # (B, latent_dim)
        gate_input = torch.cat([pred_latent, ctrl_encoded], dim=-1)  # (B, latent_dim*2)
        gate = torch.sigmoid(self.gate_layer(gate_input))  # (B, num_genes)
        pred_expression = ctrl_expression + gate * delta

        return pred_expression


class PerturbationDecoderV2(nn.Module):
    """V2 decoder: factored gene-embedding projection (like GEARS per-gene layers).

    Instead of MLP → (B, G) direct mapping, we use:
      effect = MLP(impact_summary + cell_emb)  → (B, d_effect)
      delta = effect @ gene_emb.T + bias       → (B, G)

    This is equivalent to having a per-gene linear layer (like GEARS w_u),
    where each gene's "projection" comes from its semantic embedding.
    Plus a direct residual from the cross-attention impact_map.

    Architecture:
      impact_summary = Linear(impact_map)             → (B, summary_dim)
      effect = MLP(impact_summary + cell_emb)          → (B, d_effect)
      factored_delta = effect @ gene_emb.T + gene_bias → (B, G)
      delta = impact_map * scale + factored_delta
      gate = sigmoid(MLP(ctrl_expression))             → (B, G)
      pred = ctrl + gate * delta
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_genes: int = 5000,
        dropout: float = 0.1,
        impact_hidden_dims: list = None,
        gate_hidden_dims: list = None,
        impact_summary_dim: int = 256,
        effect_dim: int = 512,
        gene_facet_tensor: torch.Tensor = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_genes = num_genes
        self.effect_dim = effect_dim

        if impact_hidden_dims is None:
            impact_hidden_dims = [1024]
        if gate_hidden_dims is None:
            gate_hidden_dims = [1024]

        # Direct residual: impact_map * learnable per-gene scale
        self.impact_scale = nn.Parameter(torch.ones(num_genes) * 0.01)

        # Compress impact_map → compact perturbation fingerprint
        self.impact_projector = nn.Sequential(
            nn.Linear(num_genes, impact_summary_dim),
            nn.GELU(),
        )

        # Effect network: (impact_summary + cell_emb) → compact effect vector
        correction_input_dim = impact_summary_dim + embed_dim
        self.effect_net = build_mlp(
            in_dim=correction_input_dim,
            out_dim=effect_dim,
            hidden_dims=impact_hidden_dims,
            activation="gelu",
            dropout=dropout,
            final_activation=False,
        )

        # Per-gene bias
        self.gene_bias = nn.Parameter(torch.zeros(num_genes))

        # Gene embeddings for factored output: effect @ gene_emb.T
        if gene_facet_tensor is not None:
            # Mean-pool facets → (G, facet_dim), then project → (G, effect_dim)
            gene_mean = gene_facet_tensor.mean(dim=1)  # (G, 768)
            self.register_buffer("_gene_mean", gene_mean)
            self.gene_embed_proj = nn.Sequential(
                nn.Linear(gene_facet_tensor.shape[-1], effect_dim),
                nn.GELU(),
                nn.LayerNorm(effect_dim),
            )
            self._has_gene_emb = True
        else:
            # Fallback: direct linear output (no factorization)
            self.direct_output = nn.Linear(effect_dim, num_genes)
            nn.init.zeros_(self.direct_output.weight)
            nn.init.zeros_(self.direct_output.bias)
            self._has_gene_emb = False

        # Learnable temperature for scaled dot-product
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / (effect_dim ** 0.5)))

        # Gate: ctrl expression → per-gene activation filter
        self.gate_net = build_mlp(
            in_dim=num_genes,
            out_dim=num_genes,
            hidden_dims=gate_hidden_dims,
            activation="gelu",
            dropout=dropout,
            final_activation=False,
        )

        trainable = sum(p.numel() for p in self.parameters())
        print(
            f"[PerturbationDecoderV2] embed_dim={embed_dim}, "
            f"num_genes={num_genes}, effect_dim={effect_dim}, "
            f"summary_dim={impact_summary_dim}, "
            f"gene_emb={'yes' if self._has_gene_emb else 'no'}, "
            f"trainable={trainable:,}"
        )

    def forward(
        self,
        impact_map: torch.Tensor,        # (B, G)
        cell_emb: torch.Tensor,           # (B, D)
        ctrl_expression: torch.Tensor,    # (B, G)
    ) -> torch.Tensor:
        """
        Returns:
            pred_expression: (B, G) predicted post-perturbation expression.
        """
        # Residual path: direct scaling (gradient to cross-attn)
        scaled_impact = impact_map * self.impact_scale                    # (B, G)

        # Perturbation fingerprint
        impact_summary = self.impact_projector(impact_map)                 # (B, summary_dim)

        # Effect vector: perturbation-aware + cell-state-aware
        effect_input = torch.cat([impact_summary, cell_emb], dim=-1)       # (B, summary+D)
        effect = self.effect_net(effect_input)                              # (B, effect_dim)

        # Factored gene-level prediction: effect @ gene_emb.T
        if self._has_gene_emb:
            gene_emb = self.gene_embed_proj(self._gene_mean)               # (G, effect_dim)
            # Normalized dot product with learnable temperature
            effect_norm = nn.functional.normalize(effect, dim=-1)           # (B, effect_dim)
            gene_emb_norm = nn.functional.normalize(gene_emb, dim=-1)      # (G, effect_dim)
            factored_delta = self.logit_scale * (effect_norm @ gene_emb_norm.t()) + self.gene_bias
        else:
            factored_delta = self.direct_output(effect)                     # (B, G)

        # Combined delta
        delta = scaled_impact + factored_delta                              # (B, G)

        # Gate conditioned on baseline expression
        gate = torch.sigmoid(self.gate_net(ctrl_expression))                # (B, G)

        pred_expression = ctrl_expression + gate * delta
        return pred_expression
