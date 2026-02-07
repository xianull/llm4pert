"""Encode facet texts with frozen BiomedBERT to produce the static gene facet tensor."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

NULL_TOKEN = "<NULL>"


class FacetEmbedder:
    """Encodes decomposed facet texts into a static tensor (num_genes, K, D).

    Uses frozen BiomedBERT to extract [CLS] token embeddings for each facet
    paragraph. NULL facets are mapped to zero vectors.
    """

    def __init__(self, cfg):
        self.model_name = cfg.embedding.model_name
        self.max_length = cfg.embedding.max_length
        self.batch_size = cfg.embedding.batch_size
        self.embed_dim = cfg.embedding.embed_dim  # 768
        self.num_facets = cfg.facets.num_facets
        self.facet_names: List[str] = list(cfg.facets.names)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load frozen BiomedBERT
        print(f"[FacetEmbedder] Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts, returning [CLS] token embeddings.

        Args:
            texts: List of strings (facet descriptions).

        Returns:
            Tensor of shape (len(texts), embed_dim).
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        # [CLS] token is at index 0 of last_hidden_state
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, D)
        return cls_embeddings.cpu()

    def build_tensor(
        self,
        facets_jsonl_path: str,
        gene_order: List[str],
        output_path: str,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Build the complete static facet tensor and save to disk.

        Args:
            facets_jsonl_path: Path to JSONL file from FacetDecomposer.
            gene_order: Ordered list of gene symbols (defines row ordering).
            output_path: Path to save the .pt tensor file.

        Returns:
            (tensor, gene_to_idx) where tensor has shape (num_genes, K, D)
            and gene_to_idx maps gene_symbol -> row index.
        """
        # Load facets from JSONL
        gene_facets: Dict[str, Dict[str, str]] = {}
        with open(facets_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    gene_facets[obj["gene"]] = obj["facets"]

        # Build gene_to_idx mapping
        gene_to_idx = {g: i for i, g in enumerate(gene_order)}
        num_genes = len(gene_order)

        # Initialize output tensor with zeros (NULL facets stay zero)
        tensor = torch.zeros(num_genes, self.num_facets, self.embed_dim)

        # Collect all non-NULL texts with their positions for batch encoding
        texts_to_encode: List[str] = []
        positions: List[Tuple[int, int]] = []  # (gene_idx, facet_idx)

        for gene in gene_order:
            gene_idx = gene_to_idx[gene]
            facets = gene_facets.get(gene, {})
            for k, facet_name in enumerate(self.facet_names):
                text = facets.get(facet_name, NULL_TOKEN)
                if text != NULL_TOKEN and text.strip():
                    texts_to_encode.append(text)
                    positions.append((gene_idx, k))

        print(
            f"[FacetEmbedder] Encoding {len(texts_to_encode)} non-NULL facets "
            f"for {num_genes} genes..."
        )

        # Batch encode
        for start in tqdm(
            range(0, len(texts_to_encode), self.batch_size),
            desc="Encoding facets",
        ):
            end = min(start + self.batch_size, len(texts_to_encode))
            batch_texts = texts_to_encode[start:end]
            batch_emb = self.encode_texts(batch_texts)  # (batch, D)

            for i, (gene_idx, facet_idx) in enumerate(positions[start:end]):
                tensor[gene_idx, facet_idx] = batch_emb[i]

        # Save tensor and metadata
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "tensor": tensor,  # (num_genes, K, D)
            "gene_to_idx": gene_to_idx,
            "facet_names": self.facet_names,
        }
        torch.save(save_data, output_path)
        print(
            f"[FacetEmbedder] Saved tensor {tuple(tensor.shape)} to {output_path}"
        )

        return tensor, gene_to_idx
