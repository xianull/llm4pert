"""Merge facet embeddings from multiple datasets into a unified tensor.

Usage:
    python -m src.precompute.merge_facet_embeddings \
        --inputs data/replogle_k562/gene_facet_embeddings.pt \
                 data/replogle_rpe1/gene_facet_embeddings.pt \
        --output data/joint_k562_rpe1/gene_facet_embeddings.pt
"""

import argparse
from pathlib import Path

import torch


def merge_facet_embeddings(input_paths: list, output_path: str):
    """Merge multiple per-dataset facet embedding files into a union tensor.

    For genes present in multiple datasets, the first occurrence is used.
    """
    all_gene_to_idx = {}
    all_tensors = []
    all_confidences = []
    facet_names = None

    for path in input_paths:
        print(f"Loading {path}...")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        tensor = saved["tensor"]         # (G_i, K, D)
        g2i = saved["gene_to_idx"]       # {symbol: int}
        fn = saved["facet_names"]
        conf = saved.get("confidence")   # (G_i, K) or None

        if facet_names is None:
            facet_names = fn
        else:
            assert fn == facet_names, (
                f"Facet names mismatch: {fn} vs {facet_names}"
            )

        all_tensors.append((tensor, g2i, conf))
        print(f"  {len(g2i)} genes, tensor shape {tuple(tensor.shape)}")

    # Build union gene list (sorted for reproducibility)
    union_genes = set()
    for _, g2i, _ in all_tensors:
        union_genes.update(g2i.keys())
    union_genes = sorted(union_genes)
    union_g2i = {g: i for i, g in enumerate(union_genes)}

    K = all_tensors[0][0].shape[1]
    D = all_tensors[0][0].shape[2]
    G = len(union_genes)

    union_tensor = torch.zeros(G, K, D)
    union_conf = torch.zeros(G, K)
    matched = 0

    for tensor, g2i, conf in all_tensors:
        for gene, local_idx in g2i.items():
            unified_idx = union_g2i[gene]
            # Only fill if not already filled (first dataset wins)
            if union_tensor[unified_idx].abs().sum() == 0:
                union_tensor[unified_idx] = tensor[local_idx]
                if conf is not None:
                    union_conf[unified_idx] = conf[local_idx]
                else:
                    union_conf[unified_idx] = 1.0
                matched += 1

    print(f"\nUnion: {G} genes, {matched} filled, {G - matched} empty")
    print(f"Tensor shape: {tuple(union_tensor.shape)}")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tensor": union_tensor,
        "gene_to_idx": union_g2i,
        "facet_names": facet_names,
        "confidence": union_conf,
    }, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    merge_facet_embeddings(args.inputs, args.output)
