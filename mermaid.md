# DyGenePT 系统架构总览

DyGenePT（Dynamic Gene Perturbation Transformer）是一个用于预测基因扰动后细胞转录组变化的深度学习框架。其核心思想是：同一个基因在不同细胞状态下可能发挥不同的功能（基因多效性），因此不应该用一个静态向量来表示基因。我们将每个基因通过 LLM 分解为 8 个生物学功能维度（facet），再由模型根据当前细胞状态动态选择相关维度，从而实现上下文感知的扰动预测。

## 1. 系统全局流程

系统分为三个阶段：**预计算**、**模型训练**、**推理评估**。预计算阶段将原始基因注释和知识图谱转化为固定的 facet 嵌入张量；模型阶段接收扰动数据和 facet 张量，通过 scGPT 编码细胞状态、Sparsemax 交叉注意力动态加权 facet、最终用潜空间算术预测表达谱变化。

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

## 2. 预计算流水线

预计算流水线共 6 步，目标是将每个基因转化为一个 `(8, 768)` 的 facet 嵌入矩阵。Step 1 从 NCBI/UniProt 注释构建基因文本；Step 1.5 用 STRING、BioPlex 等知识图谱补充交互上下文，对注释稀疏的基因尤为关键；Step 2 调用 LLM（Gemini 3 Pro）将文本分解为 8 个 facet 描述，无法识别的标记为 `<NULL>`；Step 2.5 针对 NULL facet，利用知识图谱邻居基因的已知 facet 构造定向 prompt，再次调用 LLM 尝试推断；Step 3 用 BiomedBERT 将文本编码为 768 维向量；Step 4 对仍然为空的 facet，通过多轮知识图谱邻居传播进行向量级填补，并标注置信度。

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

## 3. 模型架构

模型由四个模块串联组成。**GeneEncoder**（Module 1）从预计算的冻结张量中查表取出扰动基因的 8 个 facet 嵌入，经过可学习的 Adapter 层微调表征。**CellEncoder**（Module 2）将对照组表达谱输入预训练的 scGPT，选取表达量最高的基因作为 token 序列，通过 CLS + Attention Pooling 得到一个 768 维的细胞状态向量。**FacetCrossAttention**（Module 3）是核心创新：以细胞状态为 Query、8 个 facet 为 Key/Value，使用 Sparsemax 产生稀疏注意力权重，使模型在不同细胞上下文中只激活相关的功能维度，同时用 `log(confidence)` 偏置惩罚低置信度的填补 facet。对于组合扰动，Module 3.5 通过自注意力建模多个扰动基因之间的交互。**PerturbationDecoder**（Module 4）采用潜空间算术范式：将动态基因嵌入编码为 shift 向量，加到细胞潜表征上，再通过门控残差机制 `pred = ctrl + gate * delta` 输出最终预测。

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

## 4. 八维生物学 Facet

我们将基因功能分解为 8 个正交的生物学维度：转录调控、细胞周期与增殖、细胞死亡与存活、代谢过程、免疫应答、信号转导、细胞运动与黏附、转运与定位。这 8 个维度覆盖了细胞生物学的主要功能轴，由 LLM 根据基因注释文本为每个维度生成 2-4 句描述。Sparsemax 注意力机制会在推理时自动选择与当前细胞状态最相关的维度——例如在免疫细胞中，同一个激酶的"免疫应答"和"信号转导"维度可能被激活，而在上皮细胞中则可能切换到"细胞运动与黏附"维度。

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

## 5. 训练与损失函数

训练使用三项联合损失：全基因 MSE 保证整体表达谱的重建精度；Top-20 差异表达基因 MSE 聚焦于扰动效应最显著的基因，这也是领域内通用的评估重点；方向性 BCE 损失确保模型预测的上调/下调方向与真实值一致。优化器为 AdamW，配合 Warmup + Cosine Annealing 学习率调度。早停策略基于验证集的 `pearson_delta_top20` 指标。训练支持 DDP 多卡并行，以及 Replogle 数据集上的 5-fold 交叉验证。

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

## 6. 知识图谱 Facet 填补（Step 4）

经过 LLM 分解和二次推断后，仍有部分基因的某些 facet 为空（注释不足或 LLM 无法推断）。Step 4 利用蛋白质交互网络和功能注释图谱进行向量级填补：从 STRING PPI（实验验证权重最高）、BioPlex、Reactome 共通路、GO 共注释四个来源收集邻居基因的非空 facet 嵌入，按证据强度加权平均得到填补向量。采用多轮迭代传播（默认 3 轮），第 N 轮可以利用第 N-1 轮的填补结果。填补后的向量经 L2 归一化对齐到原生嵌入的模长，并标注置信度（原生=1.0，填补上限=0.8），供交叉注意力模块作为偏置使用。

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

## 7. 文件依赖关系

下图展示了整个系统中所有输入数据和中间产物之间的依赖关系。上游是原始的基因注释 TSV 和四套知识图谱 JSON，经过预计算流水线逐步产出 `gene_corpus.json` → `gene_corpus_enriched.json` → `gene_facets.jsonl` → `gene_facet_embeddings.pt`。训练阶段需要三个输入汇聚：预计算的 facet 嵌入张量、GEARS 格式的扰动数据（h5ad）、以及 scGPT 预训练权重。最终输出模型 checkpoint 和测试指标。

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
