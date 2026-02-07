# Graph-guided Prompt Gradient (GGPG) 技术方案

## 1. 核心动机

| 方法 | 优化信号来源 | 问题 |
|------|-------------|------|
| TextGrad | LLM自由改写 | 无结构化指导，容易漂移 |
| **GGPG (Ours)** | 图结构分析 | 基于错误样本的图模式，有据可依 |

**核心洞察**：错误分类的基因在知识图谱上往往呈现**结构化的模式**（共同邻居、缺失的通路信息等），这些模式可以转化为prompt优化的**方向性指导**。

---

## 2. 问题形式化

**输入：**
- 知识图谱 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，节点为基因/蛋白质/通路/疾病等
- 下游任务数据集 $\mathcal{D} = \{(g_i, y_i)\}_{i=1}^N$
- 初始prompt $p_0$

**目标：**
$$p^* = \arg\max_p \mathcal{L}_{task}(f_\theta(\text{Embed}(\text{LLM}(p, \mathcal{G}, g))), y)$$

**关键创新：** 设计 Graph Gradient $\nabla_{\mathcal{G}} p$ 来指导prompt更新

---

## 3. 方法框架

```
┌─────────────────────────────────────────────────────────────────┐
│                    GGPG Framework                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Knowledge│───▶│   Gene   │───▶│Embedding │───▶│Downstream│  │
│  │  Graph   │    │  Agent   │    │  Model   │    │   Task   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │              ▲                               │          │
│       │              │                               │          │
│       ▼              │                               ▼          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐      │
│  │  Error   │───▶│  Graph   │───▶│   Prompt Updater     │      │
│  │ Subgraph │    │ Gradient │    │ (Structured Update)  │      │
│  │ Mining   │    │Generator │    └──────────────────────┘      │
│  └──────────┘    └──────────┘                                   │
│       ▲                                                         │
│       │                                                         │
│       └─────────── Error Samples ◀──────────────────────────────│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心技术组件

### 4.1 Error Subgraph Mining（错误子图挖掘）

**目标：** 找到错误样本在图上的共同结构模式

```python
# 伪代码
def mine_error_patterns(G, error_genes, correct_genes):
    # Step 1: 提取错误样本的k-hop子图
    error_subgraphs = [extract_subgraph(G, g, k=2) for g in error_genes]

    # Step 2: 频繁子图挖掘
    frequent_patterns = gSpan(error_subgraphs, min_support=0.3)

    # Step 3: 对比分析 - 找到错误样本特有的模式
    correct_subgraphs = [extract_subgraph(G, g, k=2) for g in correct_genes]
    discriminative_patterns = filter_discriminative(
        frequent_patterns,
        error_subgraphs,
        correct_subgraphs
    )

    return discriminative_patterns
```

**输出示例：**
- "错误样本普遍连接到 `DNA repair pathway`，但描述中未提及"
- "错误样本的 `protein-protein interaction` 邻居数量 > 20，信息过载"

### 4.2 Graph Attention Feature Importance

**目标：** 学习图中哪些特征对分类最重要，哪些被当前prompt忽略

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}h_i \| \mathbf{W}h_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}h_i \| \mathbf{W}h_k]))}$$

**关键改进：** 对比正确/错误样本的attention分布

```python
def compute_feature_gap(G, error_genes, correct_genes):
    # 训练GAT分类器
    gat_model = train_GAT(G, labels)

    # 提取attention weights
    error_attention = get_attention(gat_model, error_genes)
    correct_attention = get_attention(gat_model, correct_genes)

    # 计算attention差异 - 哪些边/节点类型被忽略？
    attention_gap = compute_divergence(error_attention, correct_attention)

    # 映射到语义特征
    missing_features = map_to_semantic(attention_gap, G.edge_types)

    return missing_features  # e.g., ["pathway", "protein_domain", "disease_association"]
```

### 4.3 Graph Gradient Generator

**核心创新：** 将图分析结果转化为结构化的"梯度信号"

$$\nabla_{\mathcal{G}} p = \text{Aggregate}(\text{PatternSignal}, \text{AttentionGap}, \text{TopologySignal})$$

**三类梯度信号：**

| 信号类型 | 来源 | 示例 |
|----------|------|------|
| **Pattern Signal** | 错误子图挖掘 | "强调 DNA repair pathway 的功能" |
| **Attention Gap** | GAT特征重要性 | "增加 protein domain 的描述权重" |
| **Topology Signal** | 图结构统计 | "对高度节点(degree>20)进行信息筛选" |

```python
def generate_graph_gradient(G, error_analysis):
    gradient_signals = []

    # Signal 1: Pattern-based
    for pattern in error_analysis.discriminative_patterns:
        signal = f"EMPHASIZE: {pattern.semantic_meaning}"
        gradient_signals.append(signal)

    # Signal 2: Attention-based
    for feature in error_analysis.missing_features:
        signal = f"ADD_FOCUS: {feature} information is underrepresented"
        gradient_signals.append(signal)

    # Signal 3: Topology-based
    if error_analysis.avg_degree > threshold:
        signal = "FILTER: Prioritize most relevant neighbors, avoid information overload"
        gradient_signals.append(signal)

    return GraphGradient(signals=gradient_signals)
```

### 4.4 Structured Prompt Updater

**关键设计：** Prompt模板化 + 分区更新

```
┌────────────────────────────────────────────────────┐
│              Prompt Template Structure             │
├────────────────────────────────────────────────────┤
│ [TASK_CONTEXT]     ← 任务描述，较少更新            │
│ [FOCUS_AREAS]      ← 重点关注的图特征，频繁更新    │
│ [FILTERING_RULES]  ← 信息筛选规则，按需更新        │
│ [OUTPUT_FORMAT]    ← 输出格式，固定                │
└────────────────────────────────────────────────────┘
```

**更新算法：**

```python
def update_prompt(current_prompt, graph_gradient):
    prompt_sections = parse_prompt(current_prompt)

    for signal in graph_gradient.signals:
        if signal.type == "EMPHASIZE":
            prompt_sections["FOCUS_AREAS"].add(signal.content)
        elif signal.type == "ADD_FOCUS":
            prompt_sections["FOCUS_AREAS"].add(signal.content)
        elif signal.type == "FILTER":
            prompt_sections["FILTERING_RULES"].add(signal.content)

    # 使用LLM进行语言润色，但结构已确定
    new_prompt = llm_polish(prompt_sections)

    return new_prompt
```

---

## 5. 完整算法

```
Algorithm: Graph-guided Prompt Gradient (GGPG)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Knowledge Graph G, Dataset D, Initial Prompt p₀, Iterations T
Output: Optimized Prompt p*

1:  Initialize: p ← p₀, best_score ← 0
2:
3:  for t = 1 to T do
4:      // Stage 1: Generate & Evaluate
5:      for each gene g in D do
6:          context ← RetrieveContext(G, g)
7:          description ← LLM(p, context)
8:          embedding ← Embed(description)
9:      end for
10:
11:     predictions ← Classifier(embeddings)
12:     score ← Evaluate(predictions, labels)
13:
14:     // Stage 2: Error Analysis
15:     E_error ← {g : prediction(g) ≠ label(g)}
16:     E_correct ← {g : prediction(g) = label(g)}
17:
18:     // Stage 3: Graph Gradient Computation
19:     patterns ← MineErrorPatterns(G, E_error, E_correct)
20:     attention_gap ← ComputeFeatureGap(G, E_error, E_correct)
21:     topology_stats ← AnalyzeTopology(G, E_error)
22:
23:     ∇_G p ← GenerateGraphGradient(patterns, attention_gap, topology_stats)
24:
25:     // Stage 4: Structured Update
26:     p ← StructuredPromptUpdate(p, ∇_G p)
27:
28:     if score > best_score then
29:         best_score ← score
30:         p* ← p
31:     end if
32: end for
33:
34: return p*
```

---

## 6. 理论分析

**Proposition 1 (信息增益)**：Graph Gradient提供的优化方向比纯LLM优化具有更低的方差。

**Proposition 2 (收敛性)**：在一定条件下，GGPG算法收敛到局部最优。

**与TextGrad的对比：**

| 维度 | TextGrad | GGPG |
|------|----------|------|
| 梯度来源 | LLM隐式推理 | 图结构显式分析 |
| 可解释性 | 低 | 高（可追溯到图模式） |
| 稳定性 | 依赖LLM | 有结构约束 |
| 领域适应 | 通用 | 图数据特化 |

---

## 7. 实验设计

### 7.1 数据集与任务

| 任务名称 | 数据规模 | 类型 | 来源 |
|---------|---------|------|------|
| Dosage Sensitivity | ~1000 genes | 二分类 | ClinGen |
| Gene-Gene Interaction | ~2000 pairs | 二分类 | STRING |
| Gene Type | ~4000 genes | 多分类(5) | Ensembl |
| Perturbation Response | ~1500 genes | 二分类 | DepMap |
| Methylation State | ~2000 genes | 二分类 | ENCODE |
| Marker Gene | ~2000 genes | 二分类 | PanglaoDB |
| Gene Range | ~2500 genes | 多分类(3) | NCBI |

### 7.2 Baseline方法

| 方法 | 描述 |
|------|------|
| GenePT | 静态prompt，GPT-4生成描述 |
| GenePT + RAG | GenePT + 知识图谱检索增强 |
| TextGrad | LLM-based prompt优化 |
| OPRO | Meta-optimization for prompts |
| GNN-Embed | 纯图神经网络嵌入（无LLM） |
| Hybrid | GNN + GenePT嵌入拼接 |

### 7.3 评估指标

- **二分类任务**: ROC-AUC, PR-AUC, F1-Score
- **多分类任务**: Macro-F1, Micro-F1, Accuracy
- **收敛性**: 达到90%最优性能所需迭代次数
- **稳定性**: 多次运行的标准差

### 7.4 消融实验

| 变体 | 移除的组件 |
|------|----------|
| w/o Subgraph Mining | 移除错误子图挖掘 |
| w/o Attention Analysis | 移除GAT特征分析 |
| w/o Topology Signals | 移除拓扑统计信号 |
| w/o Structured Update | 使用自由形式更新 |
| Random Gradient | 随机梯度信号 |

### 7.5 实现细节

```python
# 配置参数
CONFIG = {
    "llm": {
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 500
    },
    "embedding": {
        "model": "text-embedding-3-large",
        "dimension": 3072
    },
    "graph": {
        "k_hops": 2,
        "max_neighbors": 20
    },
    "mining": {
        "min_support": 0.3,
        "discriminative_threshold": 0.1
    },
    "gat": {
        "layers": 2,
        "heads": 8,
        "hidden_dim": 64
    },
    "optimization": {
        "iterations": 5,
        "classifier": "LogisticRegression",
        "cv_folds": 5
    }
}
```

---

## 8. 预期结果

### 8.1 主要结果预期

| 方法 | Dosage | GGI | Type | Avg. |
|------|--------|-----|------|------|
| GenePT | ~78% | ~75% | ~62% | ~72% |
| TextGrad | ~83% | ~80% | ~67% | ~77% |
| **GGPG** | **~87%** | **~84%** | **~72%** | **~81%** |

### 8.2 关键发现预期

1. **性能提升**: 比最强baseline提升4-5%
2. **收敛加速**: 比TextGrad快2-3倍达到收敛
3. **稳定性**: 运行方差降低50%以上
4. **可解释性**: 每次prompt修改都有明确的图结构依据

---

## 9. 项目时间线

| 阶段 | 任务 | 产出 |
|------|------|------|
| Week 1-2 | 实现GGPG核心框架 | 代码框架 |
| Week 3-4 | 运行7个任务实验 | 实验数据 |
| Week 5 | 消融实验+分析 | 分析结果 |
| Week 6 | 论文撰写+修改 | 完整论文 |

---

## 10. 潜在问题与解决方案

| 问题 | 风险 | 解决方案 |
|------|------|----------|
| 子图挖掘计算量大 | 中 | 使用近似算法，限制子图大小 |
| GAT训练不稳定 | 低 | 多次运行取平均，早停策略 |
| LLM API成本高 | 中 | 缓存机制，批量处理 |
| 图谱不完整 | 中 | 多源整合，处理缺失值 |
