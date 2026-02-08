"""Module 2: Cell State Querier.

Wraps a frozen scGPT encoder to convert scRNA-seq expression vectors
into dense cell-state query vectors aligned with the text embedding space.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


class AttentionPooling(nn.Module):
    """Learned attention pooling over sequence tokens.

    A single learnable query attends to all tokens in the sequence,
    producing a weighted summary that captures global patterns beyond
    what the CLS token alone encodes.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) sequence of token embeddings.
            padding_mask: (B, L) True = masked (padding) positions.

        Returns:
            pooled: (B, D) attention-weighted summary.
        """
        B = x.size(0)
        query = self.query.expand(B, -1, -1)                    # (B, 1, D)
        keys = self.key_proj(x)                                   # (B, L, D)
        scores = (query @ keys.transpose(-2, -1)) * self.scale   # (B, 1, L)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1), float('-inf'))
        weights = torch.softmax(scores, dim=-1)                   # (B, 1, L)
        pooled = (weights @ x).squeeze(1)                         # (B, D)
        return pooled


class CellEncoder(nn.Module):
    """Encodes cell expression profiles using frozen scGPT, then projects
    from scGPT's d_model (e.g. 512) to cross-attention dimension (768).

    Integration with scGPT:
      1. Load GeneVocab from vocab.json
      2. Build TransformerModel with args from args.json
      3. Load pretrained weights
      4. Freeze all but last N transformer layers
      5. Forward: tokenised expression -> model -> cell_emb -> projection
    """

    def __init__(self, cfg):
        super().__init__()
        ckpt_dir = Path(cfg.paths.scgpt_checkpoint)
        self.max_seq_len = cfg.cell_encoder.max_seq_len  # 3000
        target_dim = cfg.cross_attention.hidden_dim  # 768

        # ------------------------------------------------------------------
        # 1. Load vocabulary
        # ------------------------------------------------------------------
        from scgpt.tokenizer import GeneVocab

        vocab_path = ckpt_dir / "vocab.json"
        self.vocab = GeneVocab.from_file(vocab_path)
        self.pad_token_id = self.vocab["<pad>"]
        self.cls_token_id = self.vocab["<cls>"]

        # ------------------------------------------------------------------
        # 2. Load model config
        # ------------------------------------------------------------------
        args_path = ckpt_dir / "args.json"
        with open(args_path, "r") as f:
            model_args = json.load(f)

        self.scgpt_dim = model_args.get("embsize", 512)

        # ------------------------------------------------------------------
        # 3. Instantiate TransformerModel
        # ------------------------------------------------------------------
        from scgpt.model import TransformerModel

        self.scgpt = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.scgpt_dim,
            nhead=model_args.get("nheads", 8),
            d_hid=model_args.get("d_hid", 512),
            nlayers=model_args.get("nlayers", 12),
            vocab=self.vocab,
            dropout=0.0,  # no dropout at inference for frozen layers
            pad_token="<pad>",
            pad_value=-2,
            cell_emb_style="cls",
            input_emb_style="continuous",
            # Disable auxiliary heads we don't need
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
        )

        # ------------------------------------------------------------------
        # 4. Load pretrained weights
        # ------------------------------------------------------------------
        ckpt_path = ckpt_dir / "best_model.pt"
        pretrained = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Handle potential key mismatches (flash-attn renaming etc.)
        model_state = self.scgpt.state_dict()
        filtered = {}
        for k, v in pretrained.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
        missing = set(model_state.keys()) - set(filtered.keys())
        if missing:
            print(
                f"[CellEncoder] {len(missing)} keys not loaded from checkpoint "
                f"(will use random init). Examples: {list(missing)[:5]}"
            )
        self.scgpt.load_state_dict(filtered, strict=False)

        # ------------------------------------------------------------------
        # 5. Freeze layers
        # ------------------------------------------------------------------
        freeze_layers = cfg.cell_encoder.freeze_layers  # e.g. -2
        self._freeze_layers(freeze_layers)

        # ------------------------------------------------------------------
        # 6. Projection head: scGPT dim -> cross-attention dim
        # ------------------------------------------------------------------
        self.projection = nn.Sequential(
            nn.Linear(self.scgpt_dim, target_dim),
            nn.GELU(),
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, target_dim),
            nn.GELU(),
            nn.LayerNorm(target_dim),
        )

        # ------------------------------------------------------------------
        # 7. Attention pooling for richer cell representation
        # ------------------------------------------------------------------
        self.attn_pool = AttentionPooling(self.scgpt_dim)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[CellEncoder] scGPT loaded: d_model={self.scgpt_dim}, "
            f"trainable={trainable:,}/{total:,}"
        )

    # ------------------------------------------------------------------
    # Freezing logic
    # ------------------------------------------------------------------
    def _freeze_layers(self, freeze_layers: int):
        """Freeze all parameters except the last |freeze_layers| transformer layers."""
        # Freeze everything in scGPT first
        for param in self.scgpt.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        if freeze_layers != 0:
            encoder = self.scgpt.transformer_encoder
            if hasattr(encoder, "layers"):
                layers = encoder.layers
            elif hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
                layers = encoder.encoder.layers
            else:
                raise RuntimeError(
                    "[CellEncoder] Cannot locate transformer layers in scGPT model. "
                    "Check that the scGPT checkpoint matches the expected model architecture. "
                    "Expected attribute: model.transformer_encoder.layers"
                )

            n_total = len(layers)
            # freeze_layers=-2 means unfreeze last 2 layers
            start_unfreeze = max(0, n_total + freeze_layers) if freeze_layers < 0 else n_total - freeze_layers
            for layer in layers[start_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

    # ------------------------------------------------------------------
    # Tokenization helper
    # ------------------------------------------------------------------
    def tokenize_cell_batch(
        self,
        gene_names: list,
        expression_matrix: torch.Tensor,
    ) -> dict:
        """Convert a batch of cell expression vectors to scGPT input format.

        Args:
            gene_names: List of gene symbol strings (length G).
            expression_matrix: (batch, G) expression values.

        Returns:
            dict with 'gene_ids' (B, L), 'values' (B, L),
            'padding_mask' (B, L) — all padded to max_seq_len.
        """
        B, G = expression_matrix.shape
        device = expression_matrix.device

        # Map gene names to scGPT vocab IDs; -1 for unknown
        vocab_ids = []
        valid_mask = []
        for gn in gene_names:
            if gn in self.vocab:
                vocab_ids.append(self.vocab[gn])
                valid_mask.append(True)
            else:
                vocab_ids.append(-1)
                valid_mask.append(False)

        vocab_ids = np.array(vocab_ids)
        valid_mask = np.array(valid_mask)
        valid_indices = np.where(valid_mask)[0]

        # For each cell, select top-expressed genes (up to max_seq_len - 1 for CLS)
        max_genes = self.max_seq_len - 1  # reserve 1 for <cls>
        batch_gene_ids = []
        batch_values = []
        batch_lengths = []

        expr_np = expression_matrix.cpu().numpy()

        for b in range(B):
            expr = expr_np[b]
            # Get non-zero expressed genes that are in scGPT vocab
            cell_valid = valid_indices.copy()
            cell_expr = expr[cell_valid]

            # Sort by expression value (descending), take top max_genes
            if len(cell_valid) > max_genes:
                top_idx = np.argsort(-np.abs(cell_expr))[:max_genes]
                cell_valid = cell_valid[top_idx]
                cell_expr = cell_expr[top_idx]

            # Gene IDs (prepend CLS)
            gids = np.concatenate([[self.cls_token_id], vocab_ids[cell_valid]])
            vals = np.concatenate([[0.0], cell_expr])  # CLS has value 0

            batch_gene_ids.append(gids)
            batch_values.append(vals)
            batch_lengths.append(len(gids))

        # Pad to max length in batch
        max_len = max(batch_lengths)
        padded_ids = np.full((B, max_len), self.pad_token_id, dtype=np.int64)
        padded_vals = np.full((B, max_len), -2.0, dtype=np.float32)  # scGPT pad_value
        padding_mask = np.ones((B, max_len), dtype=bool)  # True = masked

        for b in range(B):
            L = batch_lengths[b]
            padded_ids[b, :L] = batch_gene_ids[b]
            padded_vals[b, :L] = batch_values[b]
            padding_mask[b, :L] = False  # False = not masked

        return {
            "gene_ids": torch.from_numpy(padded_ids).to(device),
            "values": torch.from_numpy(padded_vals).to(device),
            "padding_mask": torch.from_numpy(padding_mask).to(device),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        gene_ids: torch.LongTensor,  # (B, L)
        values: torch.Tensor,  # (B, L)
        padding_mask: torch.BoolTensor,  # (B, L) True=masked
    ) -> torch.Tensor:
        """Run scGPT encoder and project to cross-attention dimension.

        Uses the full transformer output (B, L, d_model) for both
        CLS token extraction and attention pooling, then combines
        them via residual addition for a richer cell representation.

        Returns:
            cell_query: (B, target_dim) projected cell embedding.
        """
        # Get full transformer output (not just CLS)
        transformer_output = self.scgpt._encode(
            gene_ids, values, padding_mask
        )  # (B, L, scgpt_dim)

        # CLS token at position 0
        cls_emb = transformer_output[:, 0, :]  # (B, scgpt_dim)

        # Attention pooling over all gene tokens
        pooled = self.attn_pool(transformer_output, padding_mask)  # (B, scgpt_dim)

        # Combine: CLS + pooled (residual)
        cell_emb = cls_emb + pooled

        cell_query = self.projection(cell_emb)  # (B, target_dim)
        return cell_query
