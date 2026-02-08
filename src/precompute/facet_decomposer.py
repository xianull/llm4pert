"""LLM-based decomposition of gene descriptions into K biological facets."""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Set

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.precompute.llm_utils import extract_json_from_llm_response

# Facet names (must match config)
DEFAULT_FACET_NAMES = [
    "Molecular Function & Protein Interactions",
    "Biological Pathways & Signaling",
    "Regulatory Network & Expression Dependencies",
    "Essentiality & Cellular Stress Response",
]

SYSTEM_PROMPT_TEMPLATE = """You are a molecular biology expert specializing in CRISPRi \
perturbation experiments on essential genes. Given a gene description, decompose its \
biological role into exactly {K} facets. The goal is to predict which other genes' \
expression levels change when this gene is knocked down via CRISPRi.

For each facet, write a concise paragraph (2-4 sentences). If the gene has NO known \
function in a facet, output exactly "<NULL>".

The {K} facets are:
{facet_list}

GUIDELINES for each facet:

1. Molecular Function & Protein Interactions:
   Describe the gene product's core molecular activity, enzymatic function, direct \
physical binding partners, and protein complex membership. Prioritize experimentally \
validated interactions.

2. Biological Pathways & Signaling:
   Which biological pathways does this gene participate in? What is its position in \
signaling cascades (upstream regulator, core component, or downstream effector)?

3. Regulatory Network & Expression Dependencies:
   What upstream transcription factors, signaling pathways, or cellular conditions \
control this gene's expression? When this gene's activity is reduced, through what \
indirect routes might other genes' expression be affected? Consider feedback loops, \
compensatory pathways, and downstream transcriptional programs.

4. Essentiality & Cellular Stress Response:
   Why is this gene essential for cell survival or proliferation? What cellular process \
fails without it? What stress response or compensatory mechanisms are activated when \
this gene is partially depleted? Does its essentiality differ between cancer cells \
and normal cells (e.g., synthetic lethality)?

OUTPUT FORMAT:
- Valid JSON object with the {K} facet names as keys.
- Values: description strings or "<NULL>" if genuinely unknown.
- Do NOT fabricate functions. Use "<NULL>" if evidence is insufficient.
- Keep each description focused and factual (2-4 sentences max)."""


class FacetDecomposer:
    """Calls LLM API to decompose gene descriptions into K biological facets.

    Supports async concurrent requests with semaphore-limited parallelism.
    Writes results incrementally to JSONL for crash recovery.
    """

    def __init__(self, cfg):
        self.client = AsyncOpenAI(
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key,
        )
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        self.max_tokens = cfg.llm.max_tokens
        self.batch_size = cfg.llm.batch_size  # concurrent request limit
        self.retry_max = cfg.llm.retry_max
        self.retry_delay = cfg.llm.retry_delay
        self.facet_names: List[str] = list(cfg.facets.names)
        self.num_facets = cfg.facets.num_facets

        # Build system prompt
        facet_list = "\n".join(
            f"  {i+1}. {name}" for i, name in enumerate(self.facet_names)
        )
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            K=self.num_facets, facet_list=facet_list
        )

    async def decompose_single(
        self, gene_symbol: str, gene_text: str
    ) -> Tuple[str, Dict[str, str]]:
        """Decompose one gene's text into K facets via LLM API.

        Returns (gene_symbol, {facet_name: facet_text_or_NULL}).
        Implements retry with exponential backoff.
        """
        user_prompt = (
            f"Gene: {gene_symbol}\n\n"
            f"Description:\n{gene_text}\n\n"
            f"Decompose into {self.num_facets} facets as JSON:"
        )

        for attempt in range(self.retry_max):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                raw = response.choices[0].message.content.strip()
                facets = extract_json_from_llm_response(raw)

                # Validate keys
                result = {}
                for name in self.facet_names:
                    val = facets.get(name, "<NULL>")
                    if not isinstance(val, str) or not val.strip():
                        val = "<NULL>"
                    result[name] = val.strip()

                return gene_symbol, result

            except Exception as e:
                if attempt < self.retry_max - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(
                        f"[FacetDecomposer] Retry {attempt+1}/{self.retry_max} "
                        f"for {gene_symbol}: {e}. Waiting {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    print(
                        f"[FacetDecomposer] FAILED for {gene_symbol} "
                        f"after {self.retry_max} attempts: {e}"
                    )
                    # Return all NULL facets on failure
                    return gene_symbol, {
                        name: "<NULL>" for name in self.facet_names
                    }

    def load_existing(self, output_path: str) -> Set[str]:
        """Load already-processed gene symbols from existing JSONL."""
        processed = set()
        path = Path(output_path)
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            processed.add(obj["gene"])
                        except (json.JSONDecodeError, KeyError):
                            continue
        return processed

    async def decompose_batch(
        self, gene_corpus: Dict[str, str], output_path: str
    ) -> None:
        """Decompose all genes with concurrent API calls.

        Writes results incrementally to a JSONL file for crash recovery.
        """
        # Load already-processed genes for resumption
        processed = self.load_existing(output_path)
        remaining = {g: t for g, t in gene_corpus.items() if g not in processed}

        if not remaining:
            print("[FacetDecomposer] All genes already processed.")
            return

        print(
            f"[FacetDecomposer] Processing {len(remaining)} genes "
            f"({len(processed)} already done)."
        )

        semaphore = asyncio.Semaphore(self.batch_size)

        async def limited_decompose(gene: str, text: str):
            async with semaphore:
                return await self.decompose_single(gene, text)

        # Create tasks
        tasks = [
            limited_decompose(gene, text)
            for gene, text in remaining.items()
        ]

        # Run with progress bar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                gene_symbol, facets = await coro
                record = {"gene": gene_symbol, "facets": facets}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def run(self, gene_corpus: Dict[str, str], output_path: str) -> None:
        """Synchronous entry point for facet decomposition."""
        asyncio.run(self.decompose_batch(gene_corpus, output_path))
