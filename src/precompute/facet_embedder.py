"""Encode facet texts to produce the static gene facet tensor.

Supports two backends:
  - local : HuggingFace model (BiomedBERT, BioLORD, etc.) with CLS / mean pooling
  - api   : OpenAI-compatible /v1/embeddings endpoint (SiliconFlow, OpenAI, etc.)
            with async concurrent requests for high throughput
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

NULL_TOKEN = "<NULL>"


class FacetEmbedder:
    """Encodes decomposed facet texts into a static tensor (num_genes, K, D).

    NULL facets are mapped to zero vectors.
    """

    def __init__(self, cfg):
        self.embed_dim = cfg.embedding.embed_dim  # target dimension
        self.batch_size = cfg.embedding.batch_size
        self.num_facets = cfg.facets.num_facets
        self.facet_names: List[str] = list(cfg.facets.names)

        # Determine backend: "api" or "local"
        self.backend = getattr(cfg.embedding, "backend", "local")

        if self.backend == "api":
            self._init_api(cfg)
        else:
            self._init_local(cfg)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_local(self, cfg):
        from transformers import AutoTokenizer, AutoModel

        self.model_name = cfg.embedding.model_name
        self.pooling = getattr(cfg.embedding, "pooling", "mean")
        self.max_length = cfg.embedding.max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[FacetEmbedder] Loading local model {self.model_name} "
              f"(pooling={self.pooling})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _init_api(self, cfg):
        from openai import AsyncOpenAI

        api_cfg = cfg.embedding.api
        self.api_model = api_cfg.model
        self.api_dimensions = getattr(api_cfg, "dimensions", None)
        self.api_batch_limit = min(getattr(api_cfg, "batch_limit", 32), 32)
        self.api_concurrency = getattr(api_cfg, "concurrency", 8)
        self.api_retry_max = getattr(api_cfg, "retry_max", 3)
        self.api_retry_delay = getattr(api_cfg, "retry_delay", 2)

        base_url = api_cfg.base_url
        api_key = api_cfg.api_key
        if isinstance(api_key, str) and api_key.startswith("${"):
            api_key = os.environ.get("EMBED_API_KEY", "")

        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        print(f"[FacetEmbedder] Using API backend: {base_url}  "
              f"model={self.api_model}  dim={self.api_dimensions}  "
              f"concurrency={self.api_concurrency}")

    # ------------------------------------------------------------------
    # Encoding — local
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_local(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)

        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)

        return embeddings.cpu()

    # ------------------------------------------------------------------
    # Encoding — API (async concurrent)
    # ------------------------------------------------------------------
    async def _call_embedding_api(
        self, chunk: List[str], semaphore: asyncio.Semaphore
    ) -> List[List[float]]:
        """Single API call for one chunk, with semaphore and retry."""
        async with semaphore:
            for attempt in range(1, self.api_retry_max + 1):
                try:
                    kwargs = {"model": self.api_model, "input": chunk}
                    if self.api_dimensions is not None:
                        kwargs["dimensions"] = self.api_dimensions
                    resp = await self.async_client.embeddings.create(**kwargs)
                    sorted_data = sorted(resp.data, key=lambda x: x.index)
                    return [item.embedding for item in sorted_data]
                except Exception as e:
                    if attempt == self.api_retry_max:
                        raise
                    wait = self.api_retry_delay * (2 ** (attempt - 1))
                    print(f"[FacetEmbedder] API error: {e}. "
                          f"Retry {attempt}/{self.api_retry_max} in {wait}s...")
                    await asyncio.sleep(wait)

    async def _encode_api_async(self, texts: List[str]) -> torch.Tensor:
        """Split texts into chunks and call API concurrently."""
        chunks = [
            texts[i: i + self.api_batch_limit]
            for i in range(0, len(texts), self.api_batch_limit)
        ]
        semaphore = asyncio.Semaphore(self.api_concurrency)

        results = await asyncio.gather(
            *(self._call_embedding_api(chunk, semaphore) for chunk in chunks)
        )

        all_embeddings = []
        for chunk_result in results:
            all_embeddings.extend(chunk_result)

        return torch.tensor(all_embeddings, dtype=torch.float32)

    def _encode_api(self, texts: List[str]) -> torch.Tensor:
        """Sync wrapper around the async API encoder."""
        return asyncio.run(self._encode_api_async(texts))

    # ------------------------------------------------------------------
    # Public encode interface
    # ------------------------------------------------------------------
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts into fixed-size embeddings.

        Returns:
            Tensor of shape (len(texts), embed_dim).
        """
        if self.backend == "api":
            return self._encode_api(texts)
        return self._encode_local(texts)

    # ------------------------------------------------------------------
    # Build tensor
    # ------------------------------------------------------------------
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
        gene_facets: Dict[str, Dict[str, str]] = {}
        with open(facets_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    gene_facets[obj["gene"]] = obj["facets"]

        gene_to_idx = {g: i for i, g in enumerate(gene_order)}
        num_genes = len(gene_order)

        tensor = torch.zeros(num_genes, self.num_facets, self.embed_dim)

        texts_to_encode: List[str] = []
        positions: List[Tuple[int, int]] = []

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

        if self.backend == "api":
            # API: encode all at once with async concurrency
            all_emb = self._encode_api(texts_to_encode)
            for i, (gene_idx, facet_idx) in enumerate(positions):
                tensor[gene_idx, facet_idx] = all_emb[i]
        else:
            # Local: encode in batches on GPU
            for start in tqdm(
                range(0, len(texts_to_encode), self.batch_size),
                desc="Encoding facets",
            ):
                end = min(start + self.batch_size, len(texts_to_encode))
                batch_texts = texts_to_encode[start:end]
                batch_emb = self._encode_local(batch_texts)

                for i, (gene_idx, facet_idx) in enumerate(positions[start:end]):
                    tensor[gene_idx, facet_idx] = batch_emb[i]

        # Save tensor and metadata
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "tensor": tensor,
            "gene_to_idx": gene_to_idx,
            "facet_names": self.facet_names,
        }
        torch.save(save_data, output_path)
        print(
            f"[FacetEmbedder] Saved tensor {tuple(tensor.shape)} to {output_path}"
        )

        return tensor, gene_to_idx
