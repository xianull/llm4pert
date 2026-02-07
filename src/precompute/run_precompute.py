"""Orchestrator for the offline precomputation pipeline.

Usage:
    python -m src.precompute.run_precompute --config configs/default.yaml [--step STEP]

Steps:
    1 (text)   - Build gene text corpus from TSV
    2 (facets) - Decompose gene texts into facets via LLM API
    3 (embed)  - Encode facet texts with BiomedBERT
    all        - Run all steps sequentially
"""

import argparse
import json
from pathlib import Path

import src.torchtext_shim  # noqa: F401 — must be before any scgpt import

from src.config import load_config
from src.precompute.gene_text_builder import GeneTextBuilder
from src.precompute.facet_decomposer import FacetDecomposer
from src.precompute.facet_embedder import FacetEmbedder


def get_gene_list_from_gears(data_dir: str, dataset_name: str):
    """Load the GEARS dataset to extract the gene vocabulary.

    Returns a list of gene symbol strings.
    """
    from gears import PertData

    pert_data = PertData(data_dir)
    pert_data.load(data_name=dataset_name)

    # GEARS stores gene names in adata.var
    if "gene_name" in pert_data.adata.var.columns:
        gene_list = list(pert_data.adata.var["gene_name"])
    else:
        gene_list = list(pert_data.adata.var_names)

    print(f"[run_precompute] Loaded {len(gene_list)} genes from {dataset_name} dataset.")
    return gene_list, pert_data


def step1_build_text(cfg, gene_list):
    """Step 1: Build gene text corpus from TSV."""
    print("\n=== Step 1: Building gene text corpus ===")
    builder = GeneTextBuilder(cfg.paths.gene_info)
    corpus = builder.build_corpus(gene_list)

    # Save corpus for inspection
    corpus_path = Path(cfg.paths.facet_output).parent / "gene_corpus.json"
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"[Step 1] Saved corpus ({len(corpus)} genes) to {corpus_path}")

    return corpus


def step2_decompose_facets(cfg, corpus):
    """Step 2: Decompose gene texts into K facets via LLM API."""
    print("\n=== Step 2: Decomposing gene facets via LLM ===")
    decomposer = FacetDecomposer(cfg)
    decomposer.run(corpus, cfg.paths.facet_output)
    print(f"[Step 2] Facets saved to {cfg.paths.facet_output}")


def step3_embed_facets(cfg, gene_list):
    """Step 3: Encode facet texts with BiomedBERT to produce the static tensor."""
    print("\n=== Step 3: Encoding facets with BiomedBERT ===")
    embedder = FacetEmbedder(cfg)
    tensor, gene_to_idx = embedder.build_tensor(
        facets_jsonl_path=cfg.paths.facet_output,
        gene_order=gene_list,
        output_path=cfg.paths.facet_embeddings,
    )
    print(f"[Step 3] Tensor shape: {tuple(tensor.shape)}")
    return tensor, gene_to_idx


def main():
    parser = argparse.ArgumentParser(description="DyGenePT Precomputation Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["1", "2", "3", "all", "text", "facets", "embed"],
        help="Which step to run. Default: all."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override dataset name (norman, adamson, dixit)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = args.dataset or cfg.training.dataset

    # Determine which steps to run
    step = args.step
    run_1 = step in ("all", "1", "text")
    run_2 = step in ("all", "2", "facets")
    run_3 = step in ("all", "3", "embed")

    # Load gene list from GEARS
    gene_list, _ = get_gene_list_from_gears(cfg.paths.perturb_data_dir, dataset_name)

    corpus = None
    if run_1:
        corpus = step1_build_text(cfg, gene_list)

    if run_2:
        if corpus is None:
            # Load previously saved corpus
            corpus_path = Path(cfg.paths.facet_output).parent / "gene_corpus.json"
            with open(corpus_path, "r") as f:
                corpus = json.load(f)
        step2_decompose_facets(cfg, corpus)

    if run_3:
        step3_embed_facets(cfg, gene_list)

    print("\n=== Precomputation complete ===")


if __name__ == "__main__":
    main()
