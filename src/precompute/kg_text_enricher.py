"""Enrich gene texts with Knowledge Graph neighbor context.

Uses STRING PPI, GO annotations, Reactome pathways, and BioPlex PPI
to augment gene descriptions before LLM facet decomposition. This
is especially important for poorly-annotated genes that would
otherwise produce all-NULL facets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class KGTextEnricher:
    """Loads KG sources and enriches gene text with neighbor context.

    KG Data Formats (as stored in data/kg/):
        string.json:   {gene: [[partner, [evidence_types]], ...]}
        bioplex.json:  {gene: [[partner, cell_line], ...]}
        go.json:       list of dicts, each {gene: [[GO_id, relation], ...]}
        reactome.json: list of dicts, each {gene: [[pathway_name, location], ...]}
        go_dict.json:  {GO_id: term_name}
    """

    def __init__(
        self,
        kg_dir: str,
        string_evidence_filter: Optional[List[str]] = None,
        max_neighbors: int = 5,
    ):
        self.kg_dir = Path(kg_dir)
        self.evidence_filter = string_evidence_filter or ["experiments", "database"]
        self.max_neighbors = max_neighbors

        # Load KG sources
        print("[KGTextEnricher] Loading KG data...")
        self.string = self._load_json("string.json", default={})
        self.bioplex = self._load_json("bioplex.json", default={})
        self.go = self._load_list_json("go.json")
        self.reactome = self._load_list_json("reactome.json")
        self.go_dict = self._load_json("go_dict.json", default={})

        print(
            f"[KGTextEnricher] Loaded: STRING ({len(self.string)} genes), "
            f"BioPlex ({len(self.bioplex)} genes), "
            f"GO ({len(self.go)} genes), "
            f"Reactome ({len(self.reactome)} genes), "
            f"GO terms ({len(self.go_dict)} terms)"
        )

    def _load_json(self, filename: str, default=None):
        """Load a JSON file, returning default if not found."""
        path = self.kg_dir / filename
        if not path.exists():
            print(f"[KGTextEnricher] Warning: {path} not found, skipping.")
            return default if default is not None else {}
        with open(path, "r") as f:
            return json.load(f)

    def _load_list_json(self, filename: str) -> Dict:
        """Load a JSON file that is a list of dicts, merge into one dict."""
        raw = self._load_json(filename, default=[])
        if isinstance(raw, list):
            merged = {}
            for d in raw:
                if isinstance(d, dict):
                    merged.update(d)
            return merged
        elif isinstance(raw, dict):
            return raw
        return {}

    def enrich(
        self,
        gene: str,
        base_text: str,
        corpus: Dict[str, str],
    ) -> str:
        """Enrich a gene's text with KG neighbor context.

        Args:
            gene: Gene symbol.
            base_text: Original gene description text.
            corpus: Full corpus dict {gene_symbol: text} for looking up
                    neighbor descriptions.

        Returns:
            Enriched text string with KG context appended.
        """
        sections = [base_text]

        # --- STRING PPI: top partners with experimental/database evidence ---
        string_section = self._get_string_context(gene, corpus)
        if string_section:
            sections.append(string_section)

        # --- GO annotations ---
        go_section = self._get_go_context(gene)
        if go_section:
            sections.append(go_section)

        # --- Reactome pathways ---
        reactome_section = self._get_reactome_context(gene)
        if reactome_section:
            sections.append(reactome_section)

        # --- BioPlex PPI (supplementary) ---
        bioplex_section = self._get_bioplex_context(gene, corpus)
        if bioplex_section:
            sections.append(bioplex_section)

        return " ".join(sections)

    def _get_string_context(self, gene: str, corpus: Dict[str, str]) -> Optional[str]:
        """Get STRING PPI neighbor descriptions filtered by evidence quality."""
        if gene not in self.string:
            return None

        partners = self.string[gene]
        # Filter by evidence type
        filtered = []
        for entry in partners:
            partner, evidences = entry[0], entry[1]
            if any(ev in self.evidence_filter for ev in evidences):
                filtered.append(partner)

        if not filtered:
            return None

        # Get short descriptions of top neighbors from corpus
        descs = []
        for nb in filtered[: self.max_neighbors]:
            if nb in corpus and "No detailed" not in corpus[nb]:
                # Truncate neighbor description to 200 chars
                desc = corpus[nb][:200].rstrip()
                descs.append(f"{nb}: {desc}")

        if not descs:
            # At least list the partner names
            names = filtered[: self.max_neighbors * 2]
            return f"Known protein interaction partners (STRING): {', '.join(names)}."

        return (
            f"Known protein interaction partners (STRING, experimental evidence): "
            f"{'; '.join(descs)}."
        )

    def _get_go_context(self, gene: str) -> Optional[str]:
        """Get GO term annotations as readable text."""
        if gene not in self.go:
            return None

        annotations = self.go[gene]
        term_names = []
        for entry in annotations[:15]:
            go_id = entry[0]
            term_name = self.go_dict.get(go_id)
            if term_name and "obsolete" not in term_name.lower():
                term_names.append(term_name)

        if not term_names:
            return None

        return f"Gene Ontology annotations: {', '.join(term_names)}."

    def _get_reactome_context(self, gene: str) -> Optional[str]:
        """Get Reactome pathway memberships."""
        if gene not in self.reactome:
            return None

        pathways = self.reactome[gene]
        pathway_names = []
        seen = set()
        for entry in pathways[:10]:
            name = entry[0]
            if name not in seen:
                pathway_names.append(name)
                seen.add(name)

        if not pathway_names:
            return None

        return f"Reactome pathways: {', '.join(pathway_names)}."

    def _get_bioplex_context(
        self, gene: str, corpus: Dict[str, str]
    ) -> Optional[str]:
        """Get BioPlex PPI partners (supplementary to STRING)."""
        if gene not in self.bioplex:
            return None

        partners = self.bioplex[gene]
        # Only add if STRING didn't already provide partners
        if gene in self.string:
            return None

        names = list(set(entry[0] for entry in partners[: self.max_neighbors]))
        if not names:
            return None

        return f"BioPlex interaction partners: {', '.join(names)}."

    def enrich_corpus(
        self, corpus: Dict[str, str]
    ) -> Dict[str, str]:
        """Enrich all genes in a corpus.

        Args:
            corpus: {gene_symbol: original_text}

        Returns:
            {gene_symbol: enriched_text}
        """
        enriched = {}
        enriched_count = 0

        for gene, text in corpus.items():
            new_text = self.enrich(gene, text, corpus)
            enriched[gene] = new_text
            if len(new_text) > len(text):
                enriched_count += 1

        print(
            f"[KGTextEnricher] Enriched {enriched_count}/{len(corpus)} genes "
            f"with KG context."
        )
        return enriched
