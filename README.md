# DyGenePT

**Dynamic Gene Perturbation Transformer** — Predicting post-perturbation gene expression by integrating LLM-derived biological facets, scGPT cell encodings, and knowledge graph imputation.

## Overview

DyGenePT predicts how gene perturbations (knockouts, knockdowns, overexpressions) alter cellular transcriptomic profiles. The core idea is to represent each gene as **8 biological facets** decomposed by an LLM, then use a **cross-attention mechanism with Sparsemax** to dynamically weight these facets conditioned on cell state, and finally predict expression changes via **latent space arithmetic**.

**Supported datasets**: Norman, Adamson, Dixit, Replogle K562 Essential, Replogle RPE1 Essential (via [GEARS](https://github.com/snap-stanford/GEARS)).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DyGenePT                                │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │ Gene Encoder │    │ Cell Encoder │                           │
│  │  (BiomedBERT │    │   (scGPT)    │                           │
│  │   frozen +   │    │  partial     │                           │
│  │   adapter)   │    │  fine-tune)  │                           │
│  └──────┬───────┘    └──────┬───────┘                           │
│         │ (B,P,K=8,768)     │ (B,768)                           │
│         ▼                   ▼                                   │
│  ┌──────────────────────────────────┐                           │
│  │      Cross-Attention (Sparsemax) │                           │
│  │  cell_query × gene_facets → dynamic_emb                     │
│  └──────────────┬───────────────────┘                           │
│                 │ (B,P,768)                                     │
│                 ▼                                               │
│  ┌──────────────────────────────────┐                           │
│  │   Perturbation Decoder           │                           │
│  │   z_pred = z_ctrl + shift        │                           │
│  │   + interaction (combo perts)    │                           │
│  │   + residual gate                │                           │
│  └──────────────┬───────────────────┘                           │
│                 │ (B, num_genes)                                 │
│                 ▼                                               │
│         predicted expression                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Four Modules

| Module | File | Description |
|--------|------|-------------|
| **Gene Encoder** | `src/model/gene_encoder.py` | Serves precomputed 8-facet gene embeddings (frozen BiomedBERT + learnable adapter). Supports confidence masks from KG imputation. |
| **Cell Encoder** | `src/model/cell_encoder.py` | Encodes control cell expression via frozen scGPT (last N layers fine-tuned), projects to cross-attention space. |
| **Cross-Attention** | `src/model/cross_attention.py` | Multi-head cross-attention with Sparsemax over 8 facets per gene, conditioned on cell state. Confidence-based attention bias. |
| **Decoder** | `src/model/decoder.py` | Latent arithmetic decoder: `z_pred = z_ctrl + shift`. Supports combinatorial perturbations via interaction MLP. Residual gating on control expression. |

### 8 Biological Facets

Each gene is decomposed into 8 facets by an LLM:

1. Transcriptional Regulation
2. Cell Cycle & Proliferation
3. Cell Death & Survival
4. Metabolic Processes
5. Immune Response
6. Signal Transduction
7. Cell Motility & Adhesion
8. Transport & Localization

## Project Structure

```
LLM4Pert/
├── configs/
│   ├── default.yaml              # Base configuration
│   ├── large.yaml                # Larger decoder (12 heads, [1024,512,256])
│   ├── replogle_k562.yaml        # K562 5-fold CV config
│   └── replogle_rpe1.yaml        # RPE1 5-fold CV config
├── scripts/
│   ├── run_precompute.sh         # Precompute pipeline
│   ├── run_train.sh              # Training script
│   └── run_cv.sh                 # Cross-validation script
├── src/
│   ├── model/
│   │   ├── dygenept.py           # Main model (orchestrates all modules)
│   │   ├── gene_encoder.py       # Module 1: Gene facet embeddings
│   │   ├── cell_encoder.py       # Module 2: scGPT cell encoder
│   │   ├── cross_attention.py    # Module 3: Sparsemax cross-attention
│   │   └── decoder.py            # Module 4: Latent arithmetic decoder
│   ├── precompute/
│   │   ├── run_precompute.py     # Pipeline orchestrator
│   │   ├── gene_text_builder.py  # Step 1: Gene text assembly
│   │   ├── kg_text_enricher.py   # Step 1.5: KG-enriched text
│   │   ├── facet_decomposer.py   # Step 2: LLM facet decomposition
│   │   ├── null_facet_regenerator.py  # Step 2.5: Regenerate NULL facets
│   │   ├── facet_embedder.py     # Step 3: BiomedBERT encoding
│   │   ├── kg_facet_imputer.py   # Step 4: KG-based facet imputation
│   │   └── llm_utils.py          # LLM response JSON parsing
│   ├── data/
│   │   ├── dataset.py            # PerturbationDataset (GEARS wrapper)
│   │   └── collator.py           # Batch collation with padding
│   ├── train.py                  # Training loop (DDP support)
│   ├── train_cv.py               # K-fold cross-validation
│   ├── evaluate.py               # Evaluation metrics
│   ├── config.py                 # OmegaConf config loader
│   └── utils.py                  # Seed, device, logging utilities
├── data/
│   ├── kg/                       # Knowledge graphs (STRING, BioPlex, GO, Reactome)
│   └── perturb_data/             # GEARS datasets (auto-downloaded)
├── checkpoints/
│   └── scgpt/whole_human/        # Pre-trained scGPT weights
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd LLM4Pert

# Create environment
conda create -n dygenept python=3.10 -y
conda activate dygenept

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

- **scGPT checkpoint**: Download the `whole_human` pretrained model and place it under `checkpoints/scgpt/whole_human/` (should contain `best_model.pt`, `vocab.json`, `args.json`).
- **Knowledge graphs**: Place KG JSON files (`string.json`, `bioplex.json`, `go.json`, `reactome.json`, etc.) under `data/kg/`.
- **LLM API key**: Set the environment variable `LLM_API_KEY` for the facet decomposition step.

## Usage

### 1. Precompute Gene Facet Embeddings

The precomputation pipeline converts raw gene annotations into the facet embedding tensor used by the model.

```bash
# Run all steps
bash scripts/run_precompute.sh

# Or run individual steps
bash scripts/run_precompute.sh --step 1    # Gene text assembly
bash scripts/run_precompute.sh --step 1.5  # KG text enrichment
bash scripts/run_precompute.sh --step 2    # LLM facet decomposition
bash scripts/run_precompute.sh --step 2.5  # NULL facet regeneration
bash scripts/run_precompute.sh --step 3    # BiomedBERT encoding
bash scripts/run_precompute.sh --step 4    # KG-based imputation
```

**Pipeline**:

```
Gene TSV ──→ Gene Corpus ──→ KG-Enriched Corpus ──→ LLM Facets (8 per gene)
                                                          │
                                                          ▼
                                              BiomedBERT Embeddings
                                                   (G, 8, 768)
                                                          │
                                                          ▼
                                              KG Imputed Tensor
                                             + Confidence Mask
```

### 2. Train

```bash
# Single GPU
python -m src.train --config configs/default.yaml

# Multi-GPU (DDP)
torchrun --nproc_per_node=8 -m src.train --config configs/default.yaml

# Custom dataset
python -m src.train --config configs/default.yaml training.dataset=replogle_k562_essential
```

### 3. Cross-Validation

For LangPert-comparable evaluation on Replogle datasets:

```bash
bash scripts/run_cv.sh k562   # 5-fold CV on K562
bash scripts/run_cv.sh rpe1   # 5-fold CV on RPE1
```

## Configuration

All hyperparameters are managed via YAML configs with [OmegaConf](https://github.com/omry/omegaconf). Key settings:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `cell_encoder` | `freeze_layers` | -2 | Number of scGPT layers to fine-tune (negative = from end) |
| `cross_attention` | `num_heads` | 8 | Multi-head attention heads |
| `decoder` | `hidden_dims` | [512, 256] | Decoder MLP layers |
| `training` | `lr` | 1e-4 | Learning rate |
| `training` | `epochs` | 20 | Max training epochs |
| `training.loss` | `mse_weight` | 1.0 | Global MSE loss weight |
| `training.loss` | `de_mse_weight` | 0.5 | Top-DE-gene focused MSE weight |
| `training.loss` | `direction_weight` | 0.1 | Directional correctness loss weight |
| `imputation` | `num_rounds` | 3 | KG imputation propagation rounds |
| `imputation` | `confidence_cap` | 0.8 | Max confidence for imputed facets |

Override any parameter via CLI:

```bash
python -m src.train --config configs/default.yaml training.lr=5e-5 decoder.hidden_dims=[1024,512,256]
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error on all genes |
| MAE | Mean absolute error on top-20 DE genes (delta) |
| Pearson (top20) | Pearson correlation on top-20 DE genes (absolute expression) |
| Pearson Delta (top20) | Pearson correlation on top-20 DE gene deltas |
| Direction Accuracy | Fraction of top-20 DE genes with correct up/down direction |

## Key Design Decisions

- **Sparsemax over Softmax**: Produces sparse, interpretable attention weights over facets — each gene uses only the relevant facets for a given cell context.
- **Frozen + Adapter**: Gene facet embeddings from BiomedBERT are frozen with a learnable adapter, preserving pretrained biological semantics while allowing task-specific adaptation.
- **Latent Arithmetic**: Following the paradigm `z_perturbed = z_control + shift`, enabling additive reasoning about perturbation effects.
- **KG Imputation with Confidence Tracking**: Missing facets are filled from knowledge graph neighbors with decaying confidence, and the confidence scores bias the cross-attention.
- **Multi-Source KG**: STRING PPI, BioPlex PPI, Gene Ontology, and Reactome pathways provide complementary biological signals with configurable evidence weights.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU (recommended)
- LLM API access (for precomputation step 2)
