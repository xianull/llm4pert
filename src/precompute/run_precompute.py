"""Orchestrator for the offline precomputation pipeline.

Usage:
    python -m src.precompute.run_precompute --config configs/default.yaml [--step STEP]

Steps:
    1 (text)      - Build gene text corpus from TSV
    1.5 (enrich)  - Enrich gene texts with KG neighbor context
    2 (facets)    - Decompose gene texts into facets via LLM API
    retry         - Re-run LLM for genes that failed (all-NULL with rich text)
    2.5 (regen)   - Re-generate NULL facets with KG-informed LLM prompts
    3 (embed)     - Encode facet texts with BiomedBERT
    4 (impute)    - Impute remaining NULL facets using KG neighbor propagation
    all           - Run steps 1 → 1.5 → 2 → 2.5 → 3 → 4 sequentially
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

    # Ensure output directory exists
    corpus_path = Path(cfg.paths.facet_output).parent / "gene_corpus.json"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"[Step 1] Saved corpus ({len(corpus)} genes) to {corpus_path}")

    return corpus


def step1_5_enrich_with_kg(cfg, corpus):
    """Step 1.5: Enrich gene texts with KG neighbor context.

    Uses STRING PPI, GO, Reactome, and BioPlex to augment gene
    descriptions, especially for poorly-annotated genes.
    """
    from src.precompute.kg_text_enricher import KGTextEnricher

    kg_cfg = getattr(cfg, "kg_enrichment", None)
    if kg_cfg is not None and not kg_cfg.get("enabled", True):
        print("[Step 1.5] KG enrichment disabled, skipping.")
        return corpus

    kg_dir = cfg.paths.get("kg_dir", "data/kg")
    evidence_filter = None
    max_neighbors = 5
    if kg_cfg is not None:
        evidence_filter = list(kg_cfg.get("string_evidence_filter", []))
        max_neighbors = kg_cfg.get("max_neighbors", 5)

    print("\n=== Step 1.5: Enriching gene texts with KG context ===")
    enricher = KGTextEnricher(
        kg_dir=kg_dir,
        string_evidence_filter=evidence_filter if evidence_filter else None,
        max_neighbors=max_neighbors,
    )
    enriched_corpus = enricher.enrich_corpus(corpus)

    # Save enriched corpus
    enriched_path = Path(cfg.paths.facet_output).parent / "gene_corpus_enriched.json"
    with open(enriched_path, "w") as f:
        json.dump(enriched_corpus, f, indent=2, ensure_ascii=False)
    print(f"[Step 1.5] Saved enriched corpus to {enriched_path}")

    return enriched_corpus


def step2_decompose_facets(cfg, corpus):
    """Step 2: Decompose gene texts into K facets via LLM API."""
    print("\n=== Step 2: Decomposing gene facets via LLM ===")
    decomposer = FacetDecomposer(cfg)
    decomposer.run(corpus, cfg.paths.facet_output)
    print(f"[Step 2] Facets saved to {cfg.paths.facet_output}")


def step2_retry_failures(cfg, corpus):
    """Retry LLM decomposition for genes that failed (all-NULL with rich text).

    Identifies genes where all 8 facets are NULL but the corpus text is
    substantial (> 200 chars), indicating an API failure rather than
    genuinely unannotated genes. Re-runs decomposition for just those genes.
    """
    print("\n=== Step 2 Retry: Re-running LLM for failed genes ===")

    facet_path = Path(cfg.paths.facet_output)
    if not facet_path.exists():
        print("[Retry] No facets file found. Run step 2 first.")
        return

    # Load existing facets
    gene_facets = {}
    with open(facet_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                gene_facets[obj["gene"]] = obj["facets"]

    # Identify failures: all-NULL but text > 200 chars
    failures = {}
    for gene, facets in gene_facets.items():
        all_null = all(v == "<NULL>" for v in facets.values())
        if all_null and gene in corpus and len(corpus[gene]) > 200:
            failures[gene] = corpus[gene]

    if not failures:
        print("[Retry] No failed genes found. All good!")
        return

    print(f"[Retry] Found {len(failures)} failed genes to retry:")
    for g in sorted(failures.keys())[:10]:
        print(f"  - {g} ({len(failures[g])} chars)")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")

    # Remove old entries for these genes from JSONL
    # Read all lines, filter out failures, write back
    kept_lines = []
    with open(facet_path, "r") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped:
                obj = json.loads(line_stripped)
                if obj["gene"] not in failures:
                    kept_lines.append(line_stripped)

    with open(facet_path, "w") as f:
        for line in kept_lines:
            f.write(line + "\n")

    print(f"[Retry] Removed {len(failures)} old entries, re-running decomposition...")

    # Re-run decomposer on just the failed genes
    decomposer = FacetDecomposer(cfg)
    decomposer.run(failures, cfg.paths.facet_output)

    print(f"[Retry] Done. Re-decomposed {len(failures)} genes.")


def step2_5_regenerate_null_facets(cfg, corpus):
    """Step 2.5: Use LLM + KG neighbors to regenerate NULL facets.

    For genes that have partial NULLs (some facets known, some NULL),
    constructs a targeted prompt with KG neighbor context and asks the
    LLM to infer the gene's function in each NULL facet.

    This is higher quality than embedding-level imputation because
    the LLM reasons about biological relationships.
    """
    from src.precompute.null_facet_regenerator import NullFacetRegenerator
    from src.precompute.kg_text_enricher import KGTextEnricher

    print("\n=== Step 2.5: Regenerating NULL facets with KG-informed LLM ===")

    facet_path = Path(cfg.paths.facet_output)
    if not facet_path.exists():
        print("[Step 2.5] No facets file found. Run step 2 first.")
        return

    # Load all existing facets
    all_facets = {}
    with open(facet_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                all_facets[obj["gene"]] = obj["facets"]

    # Initialize regenerator
    regenerator = NullFacetRegenerator(cfg)

    # Find genes with partial NULLs
    genes_with_nulls = regenerator.find_null_facets(cfg.paths.facet_output)
    if not genes_with_nulls:
        print("[Step 2.5] No partial-NULL genes found.")
        return

    total_null_facets = sum(
        sum(1 for v in f.values() if v == "<NULL>")
        for f in genes_with_nulls.values()
    )
    print(
        f"[Step 2.5] Found {len(genes_with_nulls)} genes with "
        f"{total_null_facets} NULL facets to regenerate."
    )

    # Build neighbor map from KG
    kg_dir = cfg.paths.get("kg_dir", "data/kg")
    enricher = KGTextEnricher(kg_dir=kg_dir)

    # Build a simple neighbor map: gene -> [neighbor_genes]
    neighbor_map = {}
    for gene in genes_with_nulls:
        neighbors = set()
        if gene in enricher.string:
            for entry in enricher.string[gene]:
                neighbors.add(entry[0])
        if gene in enricher.bioplex:
            for entry in enricher.bioplex[gene]:
                neighbors.add(entry[0])
        neighbor_map[gene] = list(neighbors)[:20]

    # Run LLM regeneration
    results = regenerator.run(genes_with_nulls, corpus, all_facets, neighbor_map)

    # Merge results back into facets JSONL
    updated = 0
    facets_filled = 0
    for gene, new_facets in results.items():
        if gene not in all_facets:
            continue
        for facet_name, new_val in new_facets.items():
            if new_val != "<NULL>" and all_facets[gene].get(facet_name) == "<NULL>":
                all_facets[gene][facet_name] = new_val
                facets_filled += 1
                updated += 1

    # Rewrite JSONL with updated facets
    with open(facet_path, "w") as f:
        for gene, facets in all_facets.items():
            record = {"gene": gene, "facets": facets}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[Step 2.5] Done. Filled {facets_filled} NULL facets "
        f"across {len(results)} genes."
    )


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


def step4_impute_facets(cfg):
    """Step 4: Impute NULL facets using KG neighbor propagation.

    Loads the facet embedding tensor, uses STRING/BioPlex PPI networks
    to fill in NULL (zero-vector) facets, and saves the augmented tensor
    with a confidence mask.
    """
    import torch
    from src.precompute.kg_facet_imputer import KGFacetImputer

    print("\n=== Step 4: Imputing NULL facets with KG neighbors ===")

    imp_cfg = getattr(cfg, "imputation", None)
    if imp_cfg is not None and not imp_cfg.get("enabled", True):
        print("[Step 4] Imputation disabled, skipping.")
        return

    # Load existing tensor
    emb_path = cfg.paths.facet_embeddings
    saved = torch.load(emb_path, map_location="cpu", weights_only=False)
    tensor = saved["tensor"]
    gene_to_idx = saved["gene_to_idx"]
    facet_names = saved["facet_names"]

    # Parse imputation config
    kg_dir = cfg.paths.get("kg_dir", "data/kg")
    max_neighbors = 20
    min_informed = 2
    confidence_cap = 0.8
    num_rounds = 3
    evidence_weights = None
    if imp_cfg is not None:
        max_neighbors = imp_cfg.get("max_neighbors", 20)
        min_informed = imp_cfg.get("min_informed_neighbors", 2)
        confidence_cap = imp_cfg.get("confidence_cap", 0.8)
        num_rounds = imp_cfg.get("num_rounds", 3)
        if "evidence_weights" in imp_cfg:
            evidence_weights = dict(imp_cfg.evidence_weights)

    imputer = KGFacetImputer(
        kg_dir=kg_dir,
        max_neighbors=max_neighbors,
        min_informed=min_informed,
        confidence_cap=confidence_cap,
        num_rounds=num_rounds,
        evidence_weights=evidence_weights,
    )

    imputed, confidence = imputer.impute(tensor, gene_to_idx, facet_names)

    # Save augmented tensor with confidence
    save_data = {
        "tensor": imputed,
        "gene_to_idx": gene_to_idx,
        "facet_names": facet_names,
        "confidence": confidence,
    }
    torch.save(save_data, emb_path)
    print(f"[Step 4] Saved imputed tensor + confidence to {emb_path}")


def main():
    parser = argparse.ArgumentParser(description="DyGenePT Precomputation Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=[
            "1", "1.5", "2", "2.5", "3", "4",
            "all", "text", "enrich", "facets", "regen", "embed", "impute",
            "retry",
        ],
        help="Which step to run. Default: all."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override dataset name (norman, adamson, dixit)."
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["local", "api"],
        help="Override embedding backend for step 3 (local or api)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.backend is not None:
        cfg.embedding.backend = args.backend
    dataset_name = args.dataset or cfg.training.dataset

    # Determine which steps to run
    step = args.step
    run_1 = step in ("all", "1", "text")
    run_1_5 = step in ("all", "1.5", "enrich")
    run_2 = step in ("all", "2", "facets")
    run_retry = step == "retry"
    run_2_5 = step in ("all", "2.5", "regen")
    run_3 = step in ("all", "3", "embed")
    run_4 = step in ("all", "4", "impute")

    # Load gene list from GEARS
    gene_list, _ = get_gene_list_from_gears(cfg.paths.perturb_data_dir, dataset_name)

    corpus = None
    if run_1:
        corpus = step1_build_text(cfg, gene_list)

    if run_1_5:
        if corpus is None:
            corpus_path = Path(cfg.paths.facet_output).parent / "gene_corpus.json"
            with open(corpus_path, "r") as f:
                corpus = json.load(f)
        corpus = step1_5_enrich_with_kg(cfg, corpus)

    if run_2:
        if corpus is None:
            # Try enriched corpus first, fall back to original
            enriched_path = Path(cfg.paths.facet_output).parent / "gene_corpus_enriched.json"
            corpus_path = Path(cfg.paths.facet_output).parent / "gene_corpus.json"
            if enriched_path.exists():
                with open(enriched_path, "r") as f:
                    corpus = json.load(f)
                print(f"[Step 2] Using enriched corpus from {enriched_path}")
            else:
                with open(corpus_path, "r") as f:
                    corpus = json.load(f)
        step2_decompose_facets(cfg, corpus)

    if run_retry:
        if corpus is None:
            enriched_path = Path(cfg.paths.facet_output).parent / "gene_corpus_enriched.json"
            corpus_path = Path(cfg.paths.facet_output).parent / "gene_corpus.json"
            if enriched_path.exists():
                with open(enriched_path, "r") as f:
                    corpus = json.load(f)
            else:
                with open(corpus_path, "r") as f:
                    corpus = json.load(f)
        step2_retry_failures(cfg, corpus)

    if run_2_5:
        if corpus is None:
            enriched_path = Path(cfg.paths.facet_output).parent / "gene_corpus_enriched.json"
            corpus_path = Path(cfg.paths.facet_output).parent / "gene_corpus.json"
            if enriched_path.exists():
                with open(enriched_path, "r") as f:
                    corpus = json.load(f)
            else:
                with open(corpus_path, "r") as f:
                    corpus = json.load(f)
        step2_5_regenerate_null_facets(cfg, corpus)

    if run_3:
        step3_embed_facets(cfg, gene_list)

    if run_4:
        step4_impute_facets(cfg)

    print("\n=== Precomputation complete ===")


if __name__ == "__main__":
    main()
