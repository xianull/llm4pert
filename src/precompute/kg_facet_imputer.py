"""KG-based facet imputation for NULL facet embeddings.

After BiomedBERT encoding, many gene facets are zero vectors (NULL).
This module uses multiple KG sources to impute NULL facet embeddings:
  1. STRING PPI + BioPlex PPI (direct protein interactions)
  2. GO co-annotation (genes sharing GO terms)
  3. Reactome co-pathway (genes in the same biological pathway)

Supports multi-round iterative propagation: round N uses values
imputed in round N-1 as sources for further imputation.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


class KGFacetImputer:
    """Imputes NULL facet embeddings using multi-source KG neighbor propagation.

    Confidence semantics:
        1.0      = native (non-NULL from LLM + BiomedBERT)
        0.0      = still NULL (no imputation possible)
        (0, cap] = imputed (from KG neighbor propagation)
                   confidence decreases with each propagation round
    """

    def __init__(
        self,
        kg_dir: str,
        max_neighbors: int = 20,
        min_informed: int = 2,
        confidence_cap: float = 0.8,
        num_rounds: int = 3,
        evidence_weights: Optional[Dict[str, float]] = None,
    ):
        self.kg_dir = Path(kg_dir)
        self.max_neighbors = max_neighbors
        self.min_informed = min_informed
        self.confidence_cap = confidence_cap
        self.num_rounds = num_rounds
        self.ev_weights = evidence_weights or {
            "experiments": 1.0,
            "database": 0.8,
            "experiments_transferred": 0.5,
            "textmining": 0.3,
            "textmining_transferred": 0.2,
        }

        # Load KG sources
        print("[KGFacetImputer] Loading KG data...")
        self.string = self._load_json("string.json")
        self.bioplex = self._load_json("bioplex.json")

        # Build co-annotation neighbors from GO and Reactome
        self.go_neighbors = self._build_co_annotation_neighbors("go.json")
        self.reactome_neighbors = self._build_co_pathway_neighbors("reactome.json")

        print(
            f"[KGFacetImputer] STRING: {len(self.string)} genes, "
            f"BioPlex: {len(self.bioplex)} genes, "
            f"GO co-annotation: {len(self.go_neighbors)} genes, "
            f"Reactome co-pathway: {len(self.reactome_neighbors)} genes"
        )

    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file from kg_dir."""
        path = self.kg_dir / filename
        if not path.exists():
            print(f"[KGFacetImputer] Warning: {path} not found.")
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _build_co_annotation_neighbors(self, filename: str) -> Dict[str, List[str]]:
        """Build gene-gene neighbors from shared GO annotations.

        go.json is a list of 3 dicts:
          [0]: gene -> [[GO_id, relation], ...]
          [1]: GO_id -> [[gene, relation], ...]  (reverse index)
          [2]: GO ontology structure (not needed)

        Two genes are co-annotation neighbors if they share at least
        2 GO terms (to filter noise from very broad terms).
        """
        raw = self._load_json(filename)
        if not isinstance(raw, list) or len(raw) < 2:
            return {}

        gene_to_go = raw[0]  # gene -> [[GO_id, relation], ...]
        go_to_genes = raw[1]  # GO_id -> [[gene, relation], ...]

        # Count shared GO terms between gene pairs
        # For efficiency, only consider GO terms with <= 200 genes (skip broad terms)
        gene_pair_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for go_term, gene_list in go_to_genes.items():
            if not isinstance(gene_list, list):
                continue
            genes = [entry[0] for entry in gene_list if isinstance(entry, list)]
            if len(genes) > 200 or len(genes) < 2:
                continue
            for i, g1 in enumerate(genes):
                for g2 in genes[i + 1:]:
                    gene_pair_count[g1][g2] += 1
                    gene_pair_count[g2][g1] += 1

        # Build neighbor lists: keep pairs with >= 2 shared GO terms
        neighbors: Dict[str, List[str]] = {}
        for gene, partners in gene_pair_count.items():
            strong = [p for p, count in partners.items() if count >= 2]
            if strong:
                # Sort by shared count descending, limit to top neighbors
                strong.sort(key=lambda p: -partners[p])
                neighbors[gene] = strong[:50]

        return neighbors

    def _build_co_pathway_neighbors(self, filename: str) -> Dict[str, List[str]]:
        """Build gene-gene neighbors from shared Reactome pathways.

        reactome.json is a list of 2 dicts:
          [0]: gene -> [[pathway, location], ...]
          [1]: pathway -> [[gene, location], ...]  (reverse index)
        """
        raw = self._load_json(filename)
        if not isinstance(raw, list) or len(raw) < 2:
            return {}

        pathway_to_genes = raw[1]  # pathway -> [[gene, location], ...]

        gene_pair_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for pathway, gene_list in pathway_to_genes.items():
            if not isinstance(gene_list, list):
                continue
            genes = [entry[0] for entry in gene_list if isinstance(entry, list)]
            if len(genes) > 100 or len(genes) < 2:
                continue
            for i, g1 in enumerate(genes):
                for g2 in genes[i + 1:]:
                    gene_pair_count[g1][g2] += 1
                    gene_pair_count[g2][g1] += 1

        neighbors: Dict[str, List[str]] = {}
        for gene, partners in gene_pair_count.items():
            # Reactome co-pathway with >= 1 shared pathway
            strong = [p for p, count in partners.items() if count >= 1]
            if strong:
                strong.sort(key=lambda p: -partners[p])
                neighbors[gene] = strong[:50]

        return neighbors

    def _get_weighted_neighbors(self, gene: str) -> List[Tuple[str, float]]:
        """Get neighbors from ALL KG sources with quality-based weights.

        Priority: STRING/BioPlex (PPI) > Reactome co-pathway > GO co-annotation
        """
        neighbors: Dict[str, float] = {}

        # STRING neighbors (highest quality)
        if gene in self.string:
            for entry in self.string[gene]:
                partner, evidences = entry[0], entry[1]
                w = max(
                    (self.ev_weights.get(e, 0.1) for e in evidences),
                    default=0.1,
                )
                neighbors[partner] = max(neighbors.get(partner, 0.0), w)

        # BioPlex neighbors (experimental PPI)
        if gene in self.bioplex:
            for entry in self.bioplex[gene]:
                partner = entry[0]
                neighbors[partner] = max(neighbors.get(partner, 0.0), 0.9)

        # Reactome co-pathway neighbors (medium quality)
        if gene in self.reactome_neighbors:
            for partner in self.reactome_neighbors[gene]:
                neighbors[partner] = max(neighbors.get(partner, 0.0), 0.5)

        # GO co-annotation neighbors (lower quality, broad)
        if gene in self.go_neighbors:
            for partner in self.go_neighbors[gene]:
                neighbors[partner] = max(neighbors.get(partner, 0.0), 0.3)

        return sorted(neighbors.items(), key=lambda x: -x[1])

    def impute(
        self,
        tensor: torch.Tensor,
        gene_to_idx: Dict[str, int],
        facet_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Impute NULL facet embeddings using multi-round KG propagation.

        Round 1: uses only native (LLM-derived) embeddings as sources.
        Round 2+: also uses embeddings imputed in previous rounds,
                  with decaying confidence per round.
        """
        G, K, D = tensor.shape
        imputed = tensor.clone()
        confidence = torch.ones(G, K)

        # Identify original zero (NULL) facets
        original_zero = tensor.abs().sum(dim=-1) == 0  # (G, K)
        confidence[original_zero] = 0.0

        # Compute native embedding norm
        non_zero_norms = tensor[~original_zero].norm(dim=-1)
        if non_zero_norms.numel() > 0:
            native_norm = non_zero_norms.mean().item()
        else:
            native_norm = 1.0

        idx_to_gene = {v: k for k, v in gene_to_idx.items()}

        # Pre-compute all neighbor lists
        print("[KGFacetImputer] Pre-computing neighbor lists...")
        neighbor_cache: Dict[str, List[Tuple[str, float]]] = {}
        for gene in gene_to_idx:
            neighbor_cache[gene] = self._get_weighted_neighbors(gene)

        total_null = original_zero.sum().item()
        print(
            f"[KGFacetImputer] Total NULL facets: {int(total_null)} "
            f"across {G} genes (native norm={native_norm:.2f})"
        )

        total_imputed = 0

        for round_idx in range(self.num_rounds):
            # Current zero mask (updates each round as we fill values)
            current_zero = imputed.abs().sum(dim=-1) == 0  # (G, K)
            remaining = current_zero.sum().item()

            if remaining == 0:
                print(f"  Round {round_idx + 1}: all facets filled, stopping early.")
                break

            # Confidence decay: later rounds get lower max confidence
            round_conf_cap = self.confidence_cap * (0.7 ** round_idx)
            # In round 0, require min_informed neighbors
            # In later rounds, allow single neighbor (min_informed=1)
            round_min_informed = self.min_informed if round_idx == 0 else 1

            round_imputed = 0

            for gi, gene in tqdm(
                sorted(idx_to_gene.items()),
                desc=f"Round {round_idx + 1}/{self.num_rounds}",
                leave=False,
            ):

                if not current_zero[gi].any():
                    continue

                neighbors = neighbor_cache.get(gene, [])
                if not neighbors:
                    continue

                for ki in range(K):
                    if not current_zero[gi, ki]:
                        continue

                    informed_embs = []
                    informed_weights = []
                    informed_confs = []

                    for nb_gene, w in neighbors[: self.max_neighbors]:
                        nb_idx = gene_to_idx.get(nb_gene, -1)
                        if nb_idx < 0:
                            continue
                        # Check if neighbor has a non-zero value for this facet
                        if not current_zero[nb_idx, ki]:
                            nb_conf = confidence[nb_idx, ki].item()
                            # Weight by both edge weight and source confidence
                            effective_w = w * nb_conf
                            informed_embs.append(imputed[nb_idx, ki])
                            informed_weights.append(effective_w)
                            informed_confs.append(nb_conf)

                    if len(informed_embs) >= round_min_informed:
                        w_t = torch.tensor(informed_weights, dtype=torch.float32)
                        w_t = w_t / w_t.sum()
                        stacked = torch.stack(informed_embs)
                        imputed_emb = (stacked * w_t.unsqueeze(-1)).sum(dim=0)

                        # L2-normalize to match native norm
                        emb_norm = imputed_emb.norm()
                        if emb_norm > 0:
                            imputed_emb = (
                                F.normalize(imputed_emb.unsqueeze(0), dim=-1).squeeze(0)
                                * native_norm
                            )

                        imputed[gi, ki] = imputed_emb

                        # Confidence: based on neighbor count, quality, and round
                        avg_src_conf = sum(informed_confs) / len(informed_confs)
                        conf = min(
                            round_conf_cap,
                            len(informed_embs) / 5.0 * avg_src_conf,
                        )
                        confidence[gi, ki] = max(conf, 0.05)  # floor at 0.05
                        round_imputed += 1

            total_imputed += round_imputed
            new_remaining = (imputed.abs().sum(dim=-1) == 0).sum().item()
            print(
                f"  Round {round_idx + 1}: imputed {round_imputed} facets "
                f"(conf_cap={round_conf_cap:.2f}, min_nb={round_min_informed}), "
                f"remaining NULL: {int(new_remaining)}"
            )

            if round_imputed == 0:
                print(f"  No new imputations in round {round_idx + 1}, stopping.")
                break

        # Final statistics
        final_zero = imputed.abs().sum(dim=-1) == 0
        remaining_null = final_zero.sum(dim=0)
        print(f"\n[KGFacetImputer] Imputation complete:")
        print(f"  Original NULL facets: {int(total_null)}")
        print(f"  Total imputed: {total_imputed}")
        print(f"  Remaining NULL: {int(final_zero.sum().item())}")
        print(f"  Remaining NULL per facet:")
        for ki, name in enumerate(facet_names):
            orig = original_zero[:, ki].sum().item()
            remain = remaining_null[ki].item()
            pct = 100 * remain / G
            print(
                f"    {name}: {int(orig)} -> {int(remain)} "
                f"({int(orig - remain)} imputed, {pct:.1f}% still NULL)"
            )

        # Confidence distribution
        imputed_mask = (confidence > 0) & (confidence < 1.0)
        if imputed_mask.any():
            imp_confs = confidence[imputed_mask]
            print(f"  Imputed confidence: min={imp_confs.min():.3f}, "
                  f"mean={imp_confs.mean():.3f}, max={imp_confs.max():.3f}")

        return imputed, confidence
