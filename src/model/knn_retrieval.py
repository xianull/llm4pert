"""Facet-based kNN retrieval for perturbation response prediction.

Implements the LangPert-inspired kNN component: for each unseen perturbed
gene, retrieve the most similar training genes by facet embedding cosine
similarity and average their known perturbation responses (deltas).

This provides a strong non-parametric prior that the learned decoder
only needs to correct, rather than predicting from scratch.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional


class FacetKNNRetriever:
    """Retrieves kNN-averaged perturbation deltas using facet similarity.

    Usage:
        retriever = FacetKNNRetriever(gene_facet_tensor, gene_names, k=10)
        retriever.build_training_index(train_dataset)
        knn_delta = retriever.retrieve(pert_gene_indices, device)
    """

    def __init__(
        self,
        gene_facet_tensor: torch.Tensor,
        gene_names: List[str],
        gene_to_facet_idx: Dict[str, int],
        k: int = 10,
    ):
        """
        Args:
            gene_facet_tensor: (G_vocab, K, D) facet embeddings for all genes.
            gene_names:        Ordered gene names in the expression space.
            gene_to_facet_idx: Maps gene_symbol -> row index in facet tensor.
            k:                 Number of nearest neighbors to retrieve.
        """
        self.k = k
        self.gene_names = gene_names
        self.gene_to_facet_idx = gene_to_facet_idx
        self.num_genes = len(gene_names)

        # Build per-gene representation: mean-pool facets, then normalize
        # gene_facet_tensor: (G_vocab, K, D) -> mean -> (G_vocab, D)
        gene_mean = gene_facet_tensor.mean(dim=1).float()  # (G_vocab, D)
        # L2 normalize for cosine similarity via dot product
        norms = gene_mean.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.gene_emb_normed = gene_mean / norms  # (G_vocab, D)

        # Training gene index: will be populated by build_training_index
        self.train_gene_deltas: Dict[int, np.ndarray] = {}  # facet_idx -> mean_delta
        self.train_facet_indices: List[int] = []  # list of training gene facet indices
        self.train_delta_matrix: Optional[torch.Tensor] = None  # (N_train, G)
        self.train_emb_normed: Optional[torch.Tensor] = None  # (N_train, D)

        print(
            f"[FacetKNNRetriever] k={k}, gene_vocab={gene_facet_tensor.shape[0]}, "
            f"expression_genes={len(gene_names)}"
        )

    def build_training_index(self, train_dataset) -> None:
        """Compute mean delta per training perturbation gene.

        Iterates through training dataset, collects (ctrl, target) pairs
        grouped by perturbation gene, and computes mean delta for each.
        """
        # Collect deltas grouped by perturbation gene name
        gene_deltas: Dict[str, List[np.ndarray]] = defaultdict(list)

        for i in range(len(train_dataset)):
            item = train_dataset[i]
            ctrl = item["ctrl_expression"]
            target = item["target_expression"]
            pert_name = item["pert_name"]

            delta = target - ctrl  # (G,)

            # Extract individual perturbation gene names
            for gene in pert_name.split("+"):
                if gene != "ctrl":
                    gene_deltas[gene].append(delta)

        # Compute mean delta per gene and store with facet index
        train_gene_deltas = {}
        train_facet_indices = []
        delta_list = []

        for gene_name, deltas in gene_deltas.items():
            facet_idx = self.gene_to_facet_idx.get(gene_name, -1)
            if facet_idx < 0:
                continue

            mean_delta = np.mean(np.stack(deltas), axis=0)  # (G,)
            train_gene_deltas[facet_idx] = mean_delta
            train_facet_indices.append(facet_idx)
            delta_list.append(mean_delta)

        self.train_gene_deltas = train_gene_deltas
        self.train_facet_indices = train_facet_indices

        if delta_list:
            # Pre-compute matrices for batch retrieval
            self.train_delta_matrix = torch.from_numpy(
                np.stack(delta_list)
            ).float()  # (N_train, G)

            # Extract and normalize training gene embeddings
            idx_tensor = torch.tensor(train_facet_indices, dtype=torch.long)
            train_emb = self.gene_emb_normed[idx_tensor]  # (N_train, D)
            self.train_emb_normed = train_emb

        print(
            f"[FacetKNNRetriever] Built training index: "
            f"{len(train_facet_indices)} perturbation genes, "
            f"avg {np.mean([len(v) for v in gene_deltas.values()]):.0f} cells/gene"
        )

    def retrieve(
        self,
        pert_gene_indices: torch.LongTensor,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Retrieve kNN-averaged delta for each perturbed gene in the batch.

        Args:
            pert_gene_indices: (B, P) facet tensor indices for perturbed genes.
                               -1 = padding/no perturbation.
            device:            Target device for output tensor.

        Returns:
            knn_delta: (B, G) averaged delta from top-k similar training genes.
        """
        if self.train_emb_normed is None or len(self.train_facet_indices) == 0:
            B = pert_gene_indices.shape[0]
            return torch.zeros(B, self.num_genes, device=device)

        B, P = pert_gene_indices.shape
        k = min(self.k, len(self.train_facet_indices))

        # Move pre-computed matrices to device (lazy, once)
        if device is not None and self.train_emb_normed.device != device:
            self.train_emb_normed = self.train_emb_normed.to(device)
            self.train_delta_matrix = self.train_delta_matrix.to(device)
            self.gene_emb_normed = self.gene_emb_normed.to(device)

        result = torch.zeros(B, self.num_genes, device=device)

        for p in range(P):
            gene_idx = pert_gene_indices[:, p]  # (B,)

            # Mask valid perturbations (not padding)
            valid_mask = gene_idx >= 0  # (B,)
            if not valid_mask.any():
                continue

            # Get query embeddings for valid genes
            safe_idx = gene_idx.clamp(min=0)
            query_emb = self.gene_emb_normed[safe_idx]  # (B, D)

            # Cosine similarity: query @ train_emb.T
            # (B, D) @ (D, N_train) -> (B, N_train)
            sim = torch.mm(query_emb, self.train_emb_normed.t())

            # For each query gene, exclude itself from neighbors if present
            for i in range(B):
                gi = gene_idx[i].item()
                if gi in self.train_gene_deltas:
                    try:
                        train_pos = self.train_facet_indices.index(gi)
                        sim[i, train_pos] = -1.0  # exclude self
                    except ValueError:
                        pass

            # Top-k selection
            topk_sim, topk_idx = sim.topk(k, dim=1)  # (B, k)

            # Softmax weights based on similarity (temperature=1)
            weights = F.softmax(topk_sim, dim=1)  # (B, k)

            # Gather deltas for top-k neighbors: (B, k, G)
            topk_deltas = self.train_delta_matrix[topk_idx.cpu()].to(device)  # (B, k, G)

            # Weighted average: (B, k, 1) * (B, k, G) -> sum -> (B, G)
            knn_delta = (weights.unsqueeze(-1) * topk_deltas).sum(dim=1)

            # Add to result (only for valid perturbations)
            result[valid_mask] += knn_delta[valid_mask]

        return result
