"""Targeted LLM regeneration for NULL facets using KG neighbor context.

Instead of imputing NULL facets at the embedding level (weighted average),
this module feeds KG neighbor information into a specialized prompt and
asks the LLM to infer the gene's function in each NULL facet based on:
  1. The gene's own description
  2. Its PPI neighbors' descriptions for the same facet
  3. Shared GO terms and Reactome pathways

This produces high-quality text descriptions that are then encoded
by BiomedBERT, yielding semantically richer embeddings than
neighbor-averaging.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Tuple

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.precompute.llm_utils import extract_json_from_llm_response


NULL_TOKEN = "<NULL>"

SYSTEM_PROMPT = """You are a molecular biology expert specializing in gene function annotation.

You will be given:
1. A target gene with its known description
2. Several NULL biological facets where this gene's function is unknown
3. For each NULL facet: descriptions of the gene's interaction partners that DO have known functions in that facet

Your task: Based on the gene's known interactions and its partners' functions, infer whether the target gene likely participates in each NULL facet. Use biological reasoning (e.g., proteins that physically interact often participate in the same pathways).

RULES:
- Output a JSON object with facet names as keys
- For each facet, write a 2-4 sentence description of the gene's INFERRED role, clearly stating it is inferred from interaction partners
- If there is genuinely insufficient evidence to infer a function, output "<NULL>"
- Do NOT fabricate functions without supporting evidence from the provided interaction data
- Be conservative: only infer functions when the interaction evidence is compelling"""


class NullFacetRegenerator:
    """Re-generates NULL facets using KG-informed LLM prompts.

    For each gene with NULL facets, constructs a prompt that includes
    neighbor genes' non-NULL descriptions for those specific facets,
    enabling the LLM to infer the target gene's role.
    """

    def __init__(self, cfg):
        self.client = AsyncOpenAI(
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key,
        )
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        self.max_tokens = cfg.llm.max_tokens
        self.batch_size = cfg.llm.batch_size
        self.retry_max = cfg.llm.retry_max
        self.retry_delay = cfg.llm.retry_delay
        self.facet_names: List[str] = list(cfg.facets.names)

    def find_null_facets(
        self,
        facets_path: str,
    ) -> Dict[str, Dict[str, str]]:
        """Load facets JSONL and identify genes with NULL facets.

        Returns:
            {gene: {facet_name: text_or_NULL}} for genes that have at least
            one NULL facet but not all-NULL (all-NULL genes need Phase 0 retry first).
        """
        genes_with_nulls = {}
        with open(facets_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                gene = obj["gene"]
                facets = obj["facets"]
                null_count = sum(1 for v in facets.values() if v == NULL_TOKEN)
                # Skip genes with no NULLs or all NULLs
                if 0 < null_count < len(facets):
                    genes_with_nulls[gene] = facets

        return genes_with_nulls

    def build_neighbor_context(
        self,
        gene: str,
        null_facet_names: List[str],
        all_facets: Dict[str, Dict[str, str]],
        neighbors: List[str],
        max_neighbors_per_facet: int = 3,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """For each NULL facet, collect neighbor genes' non-NULL descriptions.

        Returns:
            {facet_name: [(neighbor_gene, neighbor_description), ...]}
        """
        context = {}
        for facet_name in null_facet_names:
            facet_examples = []
            for nb in neighbors:
                if nb in all_facets:
                    nb_desc = all_facets[nb].get(facet_name, NULL_TOKEN)
                    if nb_desc != NULL_TOKEN and nb_desc.strip():
                        facet_examples.append((nb, nb_desc))
                        if len(facet_examples) >= max_neighbors_per_facet:
                            break
            context[facet_name] = facet_examples

        return context

    def build_prompt(
        self,
        gene: str,
        gene_text: str,
        null_facets: List[str],
        neighbor_context: Dict[str, List[Tuple[str, str]]],
    ) -> str:
        """Build the user prompt for NULL facet regeneration."""
        sections = []
        sections.append(f"TARGET GENE: {gene}")
        sections.append(f"Gene description: {gene_text[:1000]}")
        sections.append("")

        sections.append(f"NULL FACETS TO INFER ({len(null_facets)}):")
        for facet_name in null_facets:
            sections.append(f"\n--- {facet_name} ---")
            examples = neighbor_context.get(facet_name, [])
            if examples:
                sections.append(
                    f"Interaction partners with known '{facet_name}' functions:"
                )
                for nb_gene, nb_desc in examples:
                    sections.append(f"  - {nb_gene}: {nb_desc[:300]}")
            else:
                sections.append("  No interaction partner data available for this facet.")

        sections.append("")
        sections.append(
            "Based on the interaction evidence above, infer the target gene's "
            "role in each facet. Output a JSON object with only the NULL facet "
            "names as keys. Use \"<NULL>\" if evidence is insufficient."
        )

        return "\n".join(sections)

    async def regenerate_single(
        self,
        gene: str,
        gene_text: str,
        null_facets: List[str],
        neighbor_context: Dict[str, List[Tuple[str, str]]],
    ) -> Tuple[str, Dict[str, str]]:
        """Call LLM to infer NULL facets for one gene."""

        # Skip if no neighbor context at all
        total_examples = sum(len(v) for v in neighbor_context.values())
        if total_examples == 0:
            return gene, {f: NULL_TOKEN for f in null_facets}

        user_prompt = self.build_prompt(
            gene, gene_text, null_facets, neighbor_context
        )

        for attempt in range(self.retry_max):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                raw = response.choices[0].message.content.strip()
                result = extract_json_from_llm_response(raw)

                # Validate: only return results for requested null facets
                cleaned = {}
                for facet_name in null_facets:
                    val = result.get(facet_name, NULL_TOKEN)
                    if not isinstance(val, str) or not val.strip():
                        val = NULL_TOKEN
                    cleaned[facet_name] = val.strip()

                return gene, cleaned

            except Exception as e:
                if attempt < self.retry_max - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait)
                else:
                    return gene, {f: NULL_TOKEN for f in null_facets}

    async def regenerate_batch(
        self,
        genes_with_nulls: Dict[str, Dict[str, str]],
        corpus: Dict[str, str],
        all_facets: Dict[str, Dict[str, str]],
        neighbor_map: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, str]]:
        """Regenerate NULL facets for all genes with LLM."""

        semaphore = asyncio.Semaphore(self.batch_size)

        async def limited_regen(gene, gene_text, null_facets, ctx):
            async with semaphore:
                return await self.regenerate_single(gene, gene_text, null_facets, ctx)

        tasks = []
        for gene, facets in genes_with_nulls.items():
            null_facets = [f for f, v in facets.items() if v == NULL_TOKEN]
            if not null_facets:
                continue

            gene_text = corpus.get(gene, "")
            neighbors = neighbor_map.get(gene, [])
            ctx = self.build_neighbor_context(
                gene, null_facets, all_facets, neighbors
            )

            tasks.append(limited_regen(gene, gene_text, null_facets, ctx))

        results = {}
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            gene, new_facets = await coro
            results[gene] = new_facets

        return results

    def run(
        self,
        genes_with_nulls: Dict[str, Dict[str, str]],
        corpus: Dict[str, str],
        all_facets: Dict[str, Dict[str, str]],
        neighbor_map: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, str]]:
        """Synchronous entry point."""
        return asyncio.run(
            self.regenerate_batch(
                genes_with_nulls, corpus, all_facets, neighbor_map
            )
        )
