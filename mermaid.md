# DyGenePT Architecture Overview

## 1. Full System Pipeline

```mermaid
graph TB
    subgraph INPUT["Raw Data Sources"]
        TSV["Gene Annotation TSV<br/>(NCBI, UniProt, GO, Reactome)"]
        KG["Knowledge Graphs<br/>(STRING, BioPlex, GO, Reactome)"]
        H5AD["Perturbation Data<br/>(GEARS h5ad)"]
        SCGPT["scGPT Pretrained<br/>(whole_human checkpoint)"]
    end

    subgraph PRECOMPUTE["Precompute Pipeline"]
        direction TB
        S1["Step 1: Gene Text Builder<br/>TSV → gene_corpus.json"]
        S15["Step 1.5: KG Text Enricher<br/>+ KG context → gene_corpus_enriched.json"]
        S2["Step 2: LLM Facet Decomposer<br/>Gemini API → gene_facets.jsonl"]
        S25["Step 2.5: NULL Facet Regenerator<br/>LLM + KG neighbors → updated facets"]
        S3["Step 3: Facet Embedder<br/>BiomedBERT → gene_facet_embeddings.pt"]
        S4["Step 4: KG Facet Imputer<br/>Neighbor propagation → imputed tensor + confidence"]

        S1 --> S15
        S15 --> S2
        S2 --> S25
        S25 --> S3
        S3 --> S4
    end

    subgraph MODEL["DyGenePT Model"]
        direction TB
        M1["Module 1: GeneEncoder<br/>Frozen facets + Adapter"]
        M2["Module 2: CellEncoder<br/>scGPT + AttentionPooling"]
        M3["Module 3: FacetCrossAttention<br/>Sparsemax"]
        M35["Module 3.5: PerturbationInteraction<br/>Self-Attention over P"]
        M4["Module 4: PerturbationDecoder<br/>Latent Arithmetic + Gating"]
    end

    subgraph OUTPUT["Output"]
        PRED["Predicted Expression<br/>(B, G)"]
        ATTN["Attention Weights<br/>(B, P, H, 8) — Interpretability"]
    end

    TSV --> S1
    KG --> S15
    KG --> S25
    KG --> S4

    S4 -->|"tensor (G,8,768)<br/>+ confidence"| M1
    H5AD --> M2
    SCGPT --> M2

    M1 --> M3
    M2 --> M3
    M3 --> M35
    M35 --> M4
    M2 -->|cell_query| M4

    M4 --> PRED
    M3 --> ATTN
```

## 2. Precompute Pipeline Detail

```mermaid
flowchart LR
    subgraph Step1["Step 1"]
        A1[Gene Annotation TSV] -->|parse| A2[gene_corpus.json<br/>5000 genes text]
    end

    subgraph Step15["Step 1.5"]
        A2 --> B1{KG Sources}
        B1 -->|STRING PPI| B2[Interaction partners]
        B1 -->|BioPlex| B3[Co-complex partners]
        B1 -->|GO| B4[Co-annotated genes]
        B1 -->|Reactome| B5[Co-pathway genes]
        B2 & B3 & B4 & B5 --> B6[gene_corpus_enriched.json]
    end

    subgraph Step2["Step 2"]
        B6 -->|async 64 concurrent| C1[LLM API<br/>Gemini 3 Pro]
        C1 -->|8 facets per gene| C2[gene_facets.jsonl]
    end

    subgraph Step25["Step 2.5"]
        C2 -->|genes with NULL facets| D1[KG Neighbor Context]
        D1 -->|targeted LLM prompts| D2[Regenerated facets]
        D2 --> D3[gene_facets.jsonl<br/>updated]
    end

    subgraph Step3["Step 3"]
        D3 -->|text → embedding| E1[BiomedBERT /<br/>BioLORD-2023-M]
        E1 --> E2["gene_facet_embeddings.pt<br/>(G × 8 × 768)"]
    end

    subgraph Step4["Step 4"]
        E2 -->|NULL facets| F1[Multi-round KG<br/>Neighbor Propagation]
        F1 --> F2["Imputed Tensor<br/>+ Confidence Mask<br/>(native=1.0, imputed≤0.8)"]
    end
```

## 3. Model Architecture Detail

```mermaid
flowchart TB
    subgraph Inputs
        CTRL["ctrl_expression<br/>(B, G)"]
        PERT["pert_gene_names"]
    end

    subgraph M2["Module 2: CellEncoder"]
        CE1["Top-K genes by expression<br/>→ scGPT tokens"]
        CE2["Frozen scGPT<br/>(unfreeze last N layers)"]
        CE3["CLS + AttentionPooling"]
        CE4["Projection 512 → 768"]
        CE1 --> CE2 --> CE3 --> CE4
    end

    subgraph M1["Module 1: GeneEncoder"]
        GE1["Lookup facet tensor<br/>(frozen, from precompute)"]
        GE2["Adapter: Linear → GELU → LayerNorm"]
        GE3["confidence scores"]
        GE1 --> GE2
        GE1 --> GE3
    end

    subgraph M3["Module 3: FacetCrossAttention"]
        CA1["Q = project(cell_query)<br/>(B, P, H, 1, d)"]
        CA2["K,V = project(gene_facets)<br/>(B, P, H, 8, d)"]
        CA3["scores = Q·Kᵀ/√d + log(conf)"]
        CA4["α = sparsemax(scores)<br/>→ sparse weights over 8 facets"]
        CA5["out = α · V"]
        CA1 --> CA3
        CA2 --> CA3
        CA3 --> CA4 --> CA5
    end

    subgraph M35["Module 3.5: PerturbInteraction"]
        PI1["Self-Attention over P<br/>(combo perturbation modeling)"]
    end

    subgraph M4["Module 4: PerturbationDecoder"]
        DE1["shift = MLP(dynamic_emb)<br/>sum across P perturbations"]
        DE2["z_ctrl = project(cell_query)"]
        DE3["z_pred = z_ctrl + shift"]
        DE4["delta = MLP(z_pred ‖ z_ctrl)"]
        DE5["gate = σ(MLP(z_pred ‖ ctrl_enc))"]
        DE6["pred = ctrl + gate ⊙ delta"]
        DE1 --> DE3
        DE2 --> DE3
        DE3 --> DE4
        DE3 --> DE5
        DE4 --> DE6
        DE5 --> DE6
    end

    CTRL --> M2
    PERT --> M1

    CE4 -->|"cell_query (B, 768)"| M3
    GE2 -->|"facets (B, P, 8, 768)"| M3
    GE3 -->|"conf (B, P, 8)"| M3

    CA5 -->|"dynamic_emb (B, P, 768)"| M35
    M35 --> M4
    CE4 -->|cell_query| M4
    CTRL -->|ctrl_expression| M4

    DE6 -->|"pred_expression (B, G)"| RESULT["Output"]
    CA4 -->|"attn_weights (B, P, H, 8)"| INTERP["Interpretability"]
```

## 4. 8 Biological Facets

```mermaid
graph LR
    GENE["Gene X"] --> F1["F1: Transcriptional Regulation"]
    GENE --> F2["F2: Cell Cycle & Proliferation"]
    GENE --> F3["F3: Cell Death & Survival"]
    GENE --> F4["F4: Metabolic Processes"]
    GENE --> F5["F5: Immune Response"]
    GENE --> F6["F6: Signal Transduction"]
    GENE --> F7["F7: Cell Motility & Adhesion"]
    GENE --> F8["F8: Transport & Localization"]

    style F1 fill:#e1f5fe
    style F2 fill:#f3e5f5
    style F3 fill:#fce4ec
    style F4 fill:#e8f5e9
    style F5 fill:#fff3e0
    style F6 fill:#f1f8e9
    style F7 fill:#e0f2f1
    style F8 fill:#fafafa
```

## 5. Training & Loss

```mermaid
flowchart LR
    subgraph Loss["Combined Loss Function"]
        L1["MSE (all genes)<br/>weight: 1.0"]
        L2["DE MSE (top-20 DE genes)<br/>weight: 0.5"]
        L3["Direction BCE<br/>weight: 0.1"]
        L1 & L2 & L3 --> TOTAL["total_loss"]
    end

    subgraph Optimizer
        OPT["AdamW<br/>lr=1e-4, wd=0.01"]
        SCHED["Warmup + Cosine Annealing"]
    end

    subgraph Eval["Evaluation Metrics"]
        E1["MSE / MAE"]
        E2["Pearson r (top-20 DE)"]
        E3["Pearson r (delta)"]
        E4["Direction Accuracy"]
    end

    TOTAL --> OPT
    OPT --> SCHED

    SCHED -->|"early stop on<br/>pearson_delta_top20"| Eval
```

## 6. Knowledge Graph Imputation (Step 4)

```mermaid
flowchart TB
    NULL["Gene with NULL facet<br/>(zero vector)"]

    subgraph KG["KG Neighbor Sources (weighted)"]
        S1["STRING PPI<br/>w=1.0 (experiments)<br/>w=0.8 (database)"]
        S2["BioPlex PPI<br/>w=0.7"]
        S3["Reactome co-pathway<br/>w=0.5"]
        S4["GO co-annotation<br/>w=0.3"]
    end

    PROP["Multi-round Propagation<br/>(3 rounds default)"]

    RESULT["Imputed Embedding<br/>+ Confidence Score ≤ 0.8"]

    NULL --> KG
    KG --> PROP
    PROP -->|"L2-normalized to<br/>match native norms"| RESULT
    PROP -->|"Round N uses<br/>Round N-1 results"| PROP
```

## 7. File Dependency Graph

```mermaid
graph TD
    TSV["Homo_sapiens_gene_info_<br/>with_go_and_pathways.tsv"]
    STRING["kg/string.json"]
    BIOPLEX["kg/bioplex.json"]
    GO["kg/go.json"]
    REACTOME["kg/reactome.json"]
    GODICT["kg/go_dict.json"]
    SCGPT_CK["checkpoints/scgpt/<br/>whole_human/"]
    H5AD["perturb_data/<br/>*.h5ad"]

    CORPUS["gene_corpus.json"]
    ENRICHED["gene_corpus_enriched.json"]
    FACETS["gene_facets.jsonl"]
    EMBED["gene_facet_embeddings.pt"]

    MODEL_CK["checkpoints/<br/>best_model.pt"]
    METRICS["test_metrics.json"]

    TSV --> CORPUS
    CORPUS --> ENRICHED
    STRING & BIOPLEX & GO & REACTOME & GODICT --> ENRICHED
    ENRICHED --> FACETS
    FACETS --> EMBED
    STRING & BIOPLEX & GO & REACTOME --> EMBED

    EMBED & H5AD & SCGPT_CK --> MODEL_CK
    MODEL_CK --> METRICS
```
