"""Module 4: Perturbation Decoder (scLAMBDA-style latent arithmetic).

Predicts post-perturbation expression via:
    z_pred = z_control + shift(dynamic_emb)
    pred_expression = decode(z_pred) + ctrl_expression
"""

import torch
import torch.nn as nn


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

            # Project gene facet embeddings -> gene_embeddings
            assert gene_facet_tensor is not None, (
                "gene_facet_tensor is required when gene_embed_aware=True"
            )
            # gene_facet_tensor: (G, K, 768) -> mean pool -> (G, 768)
            gene_mean = gene_facet_tensor.mean(dim=1)  # (G, 768)
            self.register_buffer("_gene_mean", gene_mean)  # frozen raw embeddings

            self.gene_embed_proj = nn.Sequential(
                nn.Linear(gene_facet_tensor.shape[-1], d_out),
                nn.GELU(),
                nn.LayerNorm(d_out),
            )

            print(
                f"[PerturbationDecoder] gene_embed_aware=True, d_out={d_out}, "
                f"gene_mean={gene_mean.shape}"
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
            gene_emb = self.gene_embed_proj(self._gene_mean)   # (G, d_out)
            delta = effect @ gene_emb.t() + self.gene_bias     # (B, G)
        else:
            delta = self.expression_decoder(decoder_input)    # (B, G)

        # 6. Residual gate: conditioned on perturbation latent + baseline expression
        ctrl_encoded = self.ctrl_encoder(ctrl_expression)  # (B, latent_dim)
        gate_input = torch.cat([pred_latent, ctrl_encoded], dim=-1)  # (B, latent_dim*2)
        gate = torch.sigmoid(self.gate_layer(gate_input))  # (B, num_genes)
        pred_expression = ctrl_expression + gate * delta

        return pred_expression
