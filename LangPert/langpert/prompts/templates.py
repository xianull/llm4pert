"""
Prompt templates for different biological contexts.
"""

# Default general-purpose prompt
DEFAULT_PROMPT = r'''
Instruction: Analyze the gene {gene} and identify {k_range} most similar genes from the provided list. Rank them by similarity (most similar first).

Consider similarity based on:
- Shared biological pathways
- Co-regulation patterns
- Similar protein-protein interactions
- Similar effects when knocked out

Available genes: {list_of_genes}

Format your response as JSON with two parts:
1. "reasoning": Explain your analysis, discussing potential connections between {gene} and relevant genes
2. "kNN": List the most similar genes in order of similarity

Example response format:
{{
  "reasoning": "Gene X is involved in pathway Y which directly interacts with gene Z...",
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}

DO NOT provide multiple JSON objects or alternative analyses. Provide ONLY ONE response.
'''

# Minimal prompt
MINIMAL_PROMPT = r'''
Instruction: From the provided list, identify the {k_range} genes most similar to gene {gene}.

Available genes: {list_of_genes}

Format your response as JSON:
{{
  "reasoning": "Brief explanation",
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}
'''

# No reasoning version
NO_REASONING_PROMPT = r'''
Instruction: Identify the {k_range} genes most similar to gene {gene} from the provided list.

Available genes: {list_of_genes}

Format your response as JSON containing only the ranked list:
{{
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}

Do not include explanations, reasoning, or any other fields. Provide only the JSON object with the "kNN" field.
'''

# Cell-type specific prompt (K562)
K562_PROMPT = r'''
Instruction: Analyze the gene {gene} and identify {k_range} most similar genes from the provided list. Rank them by similarity (most similar first).

Consider similarity based on:
- Shared biological pathways
- Co-regulation patterns
- Similar protein-protein interactions
- Similar effects when knocked out

Context: Analysis should focus on the K562 cell line (chronic myeloid leukemia model). Consider cancer-relevant pathways including ribosome biogenesis, transcriptional regulation, mitochondrial function, and stress responses.

Available genes: {list_of_genes}

Format your response as JSON with two parts:
1. "reasoning": Explain your analysis, discussing potential connections between {gene} and relevant genes
2. "kNN": List the most similar genes in order of similarity

Example response format:
{{
  "reasoning": "Gene X is involved in pathway Y which directly interacts with gene Z...",
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}

DO NOT provide multiple JSON objects or alternative analyses. Provide ONLY ONE response.
'''

# Paper prompt (Appendix A.1) — exact prompt from LangPert MLGenX 2025 paper
PAPER_PROMPT = r'''Given a gene of interest {gene}, choose around {k_range} genes from the list that are most similar to gene {gene} based on shared involvement in specific biological pathways, co-regulation, or protein-protein interactions. These genes should be relevant for perturbation prediction, meaning their knockout effect is likely to result in similar changes in gene expression as the knockout of gene {gene}. Rank the genes in order of decreasing similarity, with the most similar gene first. Consider data from relevant databases or literature to assess the similarity between genes.
Focus on the context of the K562 cell line, a model for chronic myeloid leukemia. Consider the role of genes in pathways relevant to cancer biology, including, but not limited to, ribosome biogenesis, transcriptional regulation, mitochondrial function, and stress responses.
Here is the list of genes available to choose from: {list_of_genes}
Provide your response as LIST:
Note: You may choose NO genes if NOT CONFIDENT in the similarity of others. Equally, when there are many genes involved in the same pathway, feel free to include more relevant genes in the list.
OUTPUT JSON FORMAT'''


# Template registry
PROMPT_TEMPLATES = {
    "default": DEFAULT_PROMPT,
    "minimal": MINIMAL_PROMPT,
    "no_reasoning": NO_REASONING_PROMPT,
    "k562": K562_PROMPT,
    "paper": PAPER_PROMPT,
}