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
    """

    def __init__(self, cfg, num_genes: int):
        super().__init__()
        self.embed_dim = cfg.cross_attention.hidden_dim  # 768
        self.num_genes = num_genes
        hidden_dims = list(cfg.decoder.hidden_dims)      # [512, 256]
        dropout = cfg.decoder.dropout

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
        self.expression_decoder = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_genes),
        )

        # --- Interaction MLP for combinatorial perturbations (P=2) ---
        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
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
            f"latent_dim={self.latent_dim}, num_genes={num_genes}"
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

        # 2. Combine shifts across perturbations
        #    (B, P, latent_dim) -> (B, latent_dim)
        P = shifts.size(1)
        if P > 1:
            # Interaction term for combinatorial perturbations (pairwise)
            pair_input = torch.cat([shifts[:, 0], shifts[:, 1]], dim=-1)
            interaction = self.interaction_mlp(pair_input)
            combined_shift = shifts.sum(dim=1) + interaction
        else:
            combined_shift = shifts.sum(dim=1)

        # 3. Project cell query to latent space
        #    (B, D) -> (B, latent_dim)
        cell_latent = self.cell_projector(cell_query)

        # 4. Latent arithmetic: control + shift
        pred_latent = cell_latent + combined_shift  # (B, latent_dim)

        # 5. Decode to full transcriptome (delta prediction)
        #    cell_latent skip connection: provide unmixed cell state to decoder
        decoder_input = torch.cat([pred_latent, cell_latent], dim=-1)  # (B, latent_dim*2)
        delta = self.expression_decoder(decoder_input)  # (B, num_genes)

        # 6. Residual gate: conditioned on perturbation latent + baseline expression
        ctrl_encoded = self.ctrl_encoder(ctrl_expression)  # (B, latent_dim)
        gate_input = torch.cat([pred_latent, ctrl_encoded], dim=-1)  # (B, latent_dim*2)
        gate = torch.sigmoid(self.gate_layer(gate_input))  # (B, num_genes)
        pred_expression = ctrl_expression + gate * delta

        return pred_expression
