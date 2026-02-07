"""Assemble gene text corpus from the gene info TSV for facet decomposition."""

import pandas as pd
from typing import Dict, List, Optional


class GeneTextBuilder:
    """Reads the gene annotation TSV and assembles text blocks for each gene.

    The assembled text concatenates summary, UniProt function, GO terms, and
    pathway information into a single paragraph suitable for LLM facet
    decomposition.
    """

    # Columns used to assemble the gene text
    TEXT_COLUMNS = [
        "summary",
        "uniprot_function",
        "GO_Biological_Process",
        "GO_Molecular_Function",
        "Reactome_Pathways",
    ]

    def __init__(self, tsv_path: str):
        self.df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        # No longer filter by protein-coding to allow all gene types
        # Index by gene symbol for fast lookup; keep first occurrence if duplicated
        self.df = self.df.drop_duplicates(subset="Symbol", keep="first")
        self.df = self.df.set_index("Symbol", drop=False)

    def get_gene_text(self, gene_symbol: str) -> Optional[str]:
        """Assemble a comprehensive text block for a single gene.

        Returns None if the gene is not found in the database.
        """
        if gene_symbol not in self.df.index:
            return None

        row = self.df.loc[gene_symbol]
        parts = []

        # Full name / description (always present)
        desc = row.get("description", "")
        if pd.notna(desc) and str(desc).strip() and str(desc).strip() != "-":
            parts.append(f"Gene: {gene_symbol} ({desc}).")

        # Official full name
        full_name = row.get("Full_name_from_nomenclature_authority", "")
        if pd.notna(full_name) and str(full_name).strip() and str(full_name).strip() != "-":
            parts.append(f"Full name: {full_name}.")

        # Other designations
        other_desig = row.get("Other_designations", "")
        if pd.notna(other_desig) and str(other_desig).strip() and str(other_desig).strip() != "-":
            parts.append(f"Other designations: {str(other_desig).strip()[:500]}")

        # Summary (main functional description)
        summary = row.get("summary", "")
        if pd.notna(summary) and str(summary).strip() and str(summary).strip() != "-":
            parts.append(str(summary).strip())

        # UniProt function
        uniprot = row.get("uniprot_function", "")
        if pd.notna(uniprot) and str(uniprot).strip() and str(uniprot).strip() != "-":
            text = str(uniprot).strip()[:1000]
            parts.append(f"UniProt function: {text}")

        # UniProt keywords
        uniprot_kw = row.get("uniprot_keywords", "")
        if pd.notna(uniprot_kw) and str(uniprot_kw).strip() and str(uniprot_kw).strip() != "-":
            parts.append(f"UniProt keywords: {uniprot_kw}")

        # GO Biological Process
        go_bp = row.get("GO_Biological_Process", "")
        if pd.notna(go_bp) and str(go_bp).strip():
            parts.append(f"GO Biological Process: {go_bp}")

        # GO Molecular Function
        go_mf = row.get("GO_Molecular_Function", "")
        if pd.notna(go_mf) and str(go_mf).strip():
            parts.append(f"GO Molecular Function: {go_mf}")

        # GO Cellular Component
        go_cc = row.get("GO_Cellular_Component", "")
        if pd.notna(go_cc) and str(go_cc).strip():
            parts.append(f"GO Cellular Component: {go_cc}")

        # Reactome Pathways
        pathways = row.get("Reactome_Pathways", "")
        if pd.notna(pathways) and str(pathways).strip():
            parts.append(f"Pathways: {pathways}")

        if not parts:
            return None

        return " ".join(parts)

    def build_corpus(self, gene_list: List[str]) -> Dict[str, str]:
        """Build text corpus for a list of gene symbols.

        Args:
            gene_list: Gene symbols from the perturbation dataset.

        Returns:
            Dict mapping gene_symbol -> assembled_text_block.
            Genes not found in the TSV are included with a minimal
            placeholder description.
        """
        corpus = {}
        missing = []

        for gene in gene_list:
            text = self.get_gene_text(gene)
            if text:
                corpus[gene] = text
            else:
                # Minimal fallback for genes not in the database
                corpus[gene] = f"Gene: {gene}. No detailed functional annotation available."
                missing.append(gene)

        if missing:
            print(
                f"[GeneTextBuilder] {len(missing)}/{len(gene_list)} genes "
                f"had no annotation. Examples: {missing[:5]}"
            )

        return corpus
