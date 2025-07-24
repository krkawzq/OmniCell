# CellFM: a large-scale foundation model pretrained on transcriptomics of 100 million human cells


## Abstract

单细胞测序(single-cell sequencing, scRNA-seq)提供单细胞分辨率的转录组分析，以前所未有的精确度揭示细胞的多样性。

然而，当前的单细胞数据分析受到固有的**数据噪音(inherent data noise)**、**批量效应(batch effects)**和**稀疏性(sparsity)**的影响，我们非常需要一个统一模型来表示细胞状态(cellular state)。

- 数据噪声：测序手段可能引入的误差、生物变异等
- 批量效应：不同批次测序的细胞可能存在系统性差异
- 稀疏性：很多基因在大多数细胞中不表达

目前的很多模型，通过**大数据集**训练单细胞基础模型(single-cell foundation models, scFMs)

但是目前人类细胞的scRNA-seq数据集规模较小，人类基础模型(human foundation model, HFM)的**大小**和**训练数据**都受限制。

收集了包含100m人类细胞的多样性数据集，并训练了800m参数的CellFM-800m

使用了RetNet修改后的ERetNet作为骨干网络。

大量实验表明，CellFM在细胞注释(Cell Annotation)、扰动预测(Perturbation Prediction)、基因功能预测(Gene Function Prediction)和基因-基因关系捕获(Gene-Gene Relationship Capture)方面优于现有模型。


### scRNA-seq 单细胞RNA表达测序

**什么是 scRNA-seq？它测什么？怎么测？**

scRNA-seq（single-cell RNA sequencing）是对单个细胞的** RNA 表达情况进行测序**的技术。

主要目标是：测出一个细胞中有**哪些基因被表达了**，每个基因**表达多少**。

> 它不是用 RNA 来“测 DNA 序列”
> scRNA-seq 不关注 DNA 序列（那是 genome sequencing 的事）
> 它也 不会通过 RNA 推出 DNA，因为 RNA 已经反映了基因“在用什么”而不是“有什么”。

**原理简述：**
1. 提取单细胞 → 每个细胞独立分析
2. 提取 mRNA（细胞中转录活跃的 RNA）
3. 反转录成 cDNA（这是你提到的“RNA → DNA”，但目的只是为了能测序）
4. 高通量测序 + 比对到参考基因组
5. 统计每个基因的 mRNA 数量 → 得到表达谱


### expression profile 表达谱

表达谱（expression profile） 就是一个细胞中所有基因表达量的向量。可以理解为这个细胞“在干什么”的快照。

`[1.0, 0.0, 3.5, ...]`这样的向量，向量长度是所有测序基因的数量。


### Cell Annotation 细胞注释

细胞注释，就是得到细胞的基因表达谱，然后根据基因表达谱，得到细胞的类型。

模型的输入是表达谱向量，输出的是细胞类型


### Perturbation Prediction 扰动预测

扰动预测，就是给定一个**表达谱**和一个**扰动**，预测扰动后的表达谱。

扰动有很多种，如：
- 药物扰动
- 环境扰动
- 基因扰动
- 反事实扰动（敲掉一个基因）

扰动的嵌入方式和表示方式也有很多，如使用onehot标签，或者带剂量的onehot标签。药物联动可以表示为多个onehot标签的组合。


### Gene Function Prediction 基因功能预测

给定一个基因（或多个基因）及其相关信息（如序列、表达、互作等），**预测这个基因在生物体内的功能，例如：是否参与细胞周期、凋亡、DNA 修复、信号转导通路等。**

人类基因组中仍有大量未注释或功能未知的基因，基因功能预测可以用于：
- 疾病机制研究
- 生物通路构建
- 药物靶点发现
- 合成生物学回路设计

**任务形式**：
输入：
- 基因序列（dna/mrna）
- 蛋白质序列（fasta）
- 表达谱（不同组织或扰动下的功能预测）
- 基因互作网络（PPI）
- 转录因子调控网络（GRN）
- Gene Ontology（GO）结构信息
- 文本描述

输出具体的功能标签：
- GO term（如GO:0005515）
- pathway menbership（如：Wnt signaling pathway）
- 自定义功能类（分类任务）


### Gene-Gene Relationship Capture 基因-基因关系捕获

指的是在模型或分析任务中，**显式或隐式地建模、学习和利用基因之间的功能/调控/表达等方面的相互关系，以增强对生物系统的理解或提升任务性能。**

> Gene-Gene Relationship Capture 是指模型理解或利用 基因之间的“联系”而不是只看每个基因单独的表达值或特征。


如何捕捉基因间关系：

| 方法                        | 思路                          | 示例                   |
| ------------------------- | --------------------------- | -------------------- |
| **协表达网络**                 | 基因间表达相关性                    | WGCNA                |
| **调控网络**                  | TF → Target edge            | GRN, DoRothEA        |
| **蛋白互作网络（PPI）**           | 蛋白水平直接相互作用                  | STRING               |
| **基因嵌入向量学习**              | 学习每个基因的 latent vector，捕捉相似性 | Gene2Vec             |
| **图神经网络（GNN）**            | 显式建图，学习局部结构                 | GraphSAGE, GAT       |
| **Transformer attention** | 注意力权重自动学习基因间依赖              | scFormer, Geneformer |


### Summary - Abstract

scRNA-seq技术带来前景。

数据集具有：
- 数据噪声
- 批量效应
- 稀疏性

现有模型专注于单一任务、单一数据分布。

人类数据集规模小，人类单细胞分析困难。



## Introduction

scRNA-seq 技术能够显著提高基因表达分析准确率，同时积累了大量的数据集，但这些数据集具有固有的**数据噪声、批量效应**和**稀疏性**等问题。

现有的模型，虽然能够应对这些问题，但是**在数据分布不一致的情况下表现变差**，且**不能适应日益增大的数据集**。

且现有的方法，**没有充分利用数据集中的嵌入信息**，我们需要一种新的方法。

现有的一些方法：
- Cell2Sentences：将表达谱按照表达量进行排序，将名称排序得到句子用来微调GPT2
- GenePT：根据细胞文本描述和metadata微调GPT3.5，用于生成嵌入

这些方法虽然利用了LLM的amazing效果，但是实际上没有充分利用细胞数据集本身的信息。

从头分析构建模型，现有的模型分为三类：
- ordering
- value categorization
- value projection

### odering

将单细胞表达谱中的基因按表达值排序，构成“伪序列”，然后将其作为token 序列输入 LLM（如 GPT/BERT），训练模型理解细胞状态中的基因序列语义。

**为什么 Ordering 有意义？**
- 用排序**消除了表达值 scale 的问题，避免不同测序平台的偏差**
- 排序隐含了细胞功能特征：表达靠前的基因一般是活跃核心调控因子
- 模仿自然语言建模中**词序的重要性**（如 attention）

相关模型：
- iSEEEK(10m cells)： 输入排序后的gene名称，喂给BERT，进行掩码重建任务，得到cell embedding可以用于后续任务
- tGPT(22.3m single cell)：输入表达排序序列，让gpt自回归预测下一个基因名称，可以得到cell embedding 和 gene embedding（每个token是gene embed，全部的是cell embed）
- Geneformer(30m single cell):引入 rank-aware embedding + gene-level 对比学习

#### Genefermer

你Geneformer 是一个**Transformer-based 单细胞 foundation model**，用来同时建模 gene 和 cell 的语义空间，**捕捉表达谱中的 gene-gene 和 gene-cell 关系**。不使用LLM，而是完全重新预训练。

**输入：**

Geneformer 输入的是一个“**gene-cell pair 的序列**”，而不是单一的 gene name 序列：

```
[ (Gene1, Cell1), (Gene2, Cell1), ..., (GeneK, Cell1) ]
```

每个 token 是 `(gene, cell)` 的组合嵌入，类似双 token。

**Embedding 设计：**

| 组成部分           | 描述                                      |
| -------------- | --------------------------------------- |
| Gene embedding | 每个基因的唯一 ID 向量（初始化为 trainable embedding） |
| Cell embedding | 每个细胞有一个独立的 ID 向量                        |
| Rank embedding | 基因在表达谱中的 rank（如前10% / 中等 / 后10%）        |
| Type embedding | 基因是扰动目标还是背景 gene？是否为 marker？            |

→ 这些向量会拼接/相加形成最终输入 token embedding

**输出：**

- 每个 gene 的 contextual embedding
- 每个 cell 的整体 embedding（如 average over token）

**训练方式：**

Geneformer 不做显式监督，而是用**无监督 rank 预测任务（rank contrastive pretraining）**，如下：

1. Rank Prediction：

- 给定打乱的 gene-cell pair 序列
- 训练模型预测每个 gene 在表达谱中的“真实位置”（高、中、低 rank）
- 类似排序学习或回归学习

2. 对比学习：

- 构建正负样本对：
  - 正例：同一个基因在不同细胞中的 rank 一致
  - 负例：rank 明显差异的情况
- 用对比损失（如 InfoNCE）拉近正样本、推远负样本


### value categorization

表达谱中的基因表达量是连续的值，但是我们可以将连续的值划分入不同的桶中，**对连续值进行离散化**，这样就可以**将分类任务应用到基因表达量预测中**。

**相关模型：**
| 模型         | 输入表达               | 建模方式           | 任务目标         |
| ---------- | ------------------ | -------------- | ------------ |
| **scBERT** | 表达 → 离散 bucket     | BERT，MLM       | 分类任务，表达预测    |
| **scGPT**(33m single cell)  | 表达 → bucket + mask | GPT，自回归 + mask | 预测表达模式，生成表达谱 |
| **UCE**(650m param, 36m cells)    | 表达 + 蛋白 + 多物种      | 多模态 Masked LM  | 表达存在性预测，泛化建模 |



### value projection

不是直接将表达量离散化，而是将**表达量映射到向量**上来**保留数据全部精度**。

基因表达向量 xᵢ 被表示为两个部分之和：
- 表达值的投影（projection）
- 位置或基因嵌入（positional/gene embedding）

**相关模型：**
| 模型               | 类型                       | 核心方法                  | 特点                  |
| ---------------- | ------------------------ | --------------------- | ------------------- |
| **scFoundation** | Value projection         | MAE，重建表达值             | 保留连续表达，泛用性强         |
| **GeneCompass**  | Value projection + prior | 加入4类生物先验              | 更强生物机制建模能力          |
| **scELMo**       | Multi-modal              | GPT-3.5生成元数据嵌入 + 表达拼接 | 支持 zero-shot，LLM融合型 |

- scFoundation(50 million human cells, 100m param): 使用MAE框架进行token重建任务。
- GeneCompass(50 million human & 50 million mouse cells, 100m param): 基于 Transformer 的 cross-species foundation model，结合了基因表达、物种信息和四类生物先验（融合 gene ID、rank、表达值和 GRN/promoter/family/co-expression 嵌入），采用自监督方式训练。
- scELMo(100 million human cells, 100m param): GPT-3.5将metadata（文本元数据）转化成embedding，并和表达谱token拼接作为输入，进行自回归预测。支持zero-shot和ft。


当前还没有大规模（100M+）人类单细胞模型被充分训练或测试，单物种单细胞基础模型的上限尚未清晰。

single-species数据规模有限，尤其是人类数据集，并且由于不同的测序平台，数据分布不统一，且难以收集。

作者**清洗并格式化统一、标准化处理了100m的人类单细胞数据集**。并训练了800m的CellFM，是之前最大单细胞模型的**8倍**大小。模型采用了线性复杂度的ERetNet，平衡效率与性能。（实际上由于数据集较小，训练transformer可能欠拟合）

CellFM是**value-projection的单细胞基础模型**，训练目标是**通过基因表达值的线性投影来恢复被 mask 基因的向量嵌入。**（MAE式任务）


## Results

### overview

#### 数据集

单细胞数据集分布在多个公开数据库中，主要包括：
- NCBI GEO
- ENA
- GSA
- ImmPort

**数据清洗流程：**
1. 使用厂家提供的原始分析工具，将FASTQ原始数据处理为**基因表达矩阵**
2. 将表达矩阵使用Singleron Biotech 提供的 SynEcoSys 平台进行标准化流程处理
   - 过滤低质量细胞与基因
   - 基因名称标准化，依据HGNC指南
   - 数据格式统一，转化为稀疏矩阵格式

**数据分布：**
- 整合了 **19,914 个样本**，共计 **1.02 亿个人类细胞**，来自多个器官与测序平台。
- 4630 万细胞来自健康供体
- 710 万为病毒感染
- 350 万为肺癌患者细胞
- 细胞类型：
  - T细胞 1920万
  - 单核吞噬细胞 701万
  - 神经元 629万
  - 成纤维细胞 300万

大部分数据来自 **10x Genomics 平台**，共计**6670万细胞**。

在整理后的数据集中，约有 **7000 万细胞具有细胞类型注释信息**。

#### 模型

模型核心由三部分组成：
- 嵌入模块（embedding module）
- 多层堆叠的 ERetNet 层
- LoRA 模块（微调新数据集时可减少训练参数两）

CellFM 首先通过嵌入模块将**标量表达值转化为高维嵌入表示**。

这些基因嵌入接着被送入 L 层 ERetNet 层，用以**建模基于表达谱的精细 gene-gene 关系。**

每个 ERetNet 层由若干核心模块组成：
- Gated 多头注意力（MHA）
- 简化门控线性单元（SGLU）
- 层归一化（LayerNorm）

这些模块共同赋予 ERetNet 层以下能力：并行训练、高效推理、卓越性能（见图1c）。

模型训练后可被用于下游任务：
- gene function prediction
- cell type annotation
- perturbation effect prediction
- gene network analysis


### gene function prediction.

基因功能预测是揭示基因在**不同条件**下**作用**和**属性**的核心基础任务。

人类基因组大约包含 **2万个**编码蛋白质的基因，其中**很大一部分尚无功能注释**，因此准确预测基因功能对于深入理解其在生物系统中的作用至关重要。

我们通过三种不同类别的基因任务评估了 CellFM 在基因功能预测方面的表现，分别为：（gene级别）
1. 计量敏感（T1）：判断一个基因是否为 dosage sensitive（对 copy number 变化敏感）。CellFM表现明显区分能力。
2. 双价甲基化 vs 非甲基化（T2）：判断某个基因是否处于“双价甲基化”状态（同时带有激活和抑制标记，如 H3K4me3 + H3K27me3）。
3. 双价甲基化 vs Lys4单甲基化（T3）：进一步区分双价甲基化基因和只具有 H3K4me1/3（激活性）修饰的基因。

这些分类任务是标准的**二分类问题**，其中模型预测是根据实际的基因功能标签进行评估的。

由于这三类任务涉及的基因数量较少（通常少于1000个），因此对现有基础模型进行**微调（fine-tuning）**具有较大挑战。

为了公平比较，我们在基因功能预测任务中采用了 zero-shot（零样本）学习策略。

为了评估 CellFM 在多类别任务中的表现，我们进一步使用 Gene Ontology（GO）数据库的数据进行基因功能预测。该数据集包含三大类标签：（多酚类，只选择前10类）
- 生物过程（BP）
- 细胞组分（CC）
- 分子功能（MF）

考虑到预测全部功能类别的复杂性（BP: 1578类，CC: 253类，MF: 299类），我们聚焦评估每类中出现频率最高的前10个功能。在保持现实性的同时，也保证了可控的比较难度。

为确保不同方法之间比较公平，我们对各基础模型的基因集进行了交集处理，并保持训练集、验证集和测试集一致。

我们未将 scELMo 纳入对比，因为该模型在训练阶段使用了如 GPT-3.5 等大型语言模型**生成元数据嵌入**，其中**已包含了部分基因功能信息**。（可能泄露信息）

### predicting perturbation responses

随着测序和基因编辑技术的发展，已经可以进行**大规模实验扰动模拟**，用于研究基因表达和细胞行为的变化。


模型通过对**基因维度应用自注意力机制（self-attention）**，能够捕捉复杂的基因间相互作用，并**准确预测未见扰动下的表达变化**。

扰动建模能力在 AI 驱动的药物发现中尤为关键：
- 可以预测药物/基因对细胞过程的影响
- 发现新靶点
- 探索药物重定位（repurposing）

**基因扰动**：

为了评估 CellFM 在扰动预测中的表现，我们使用了两个 Perturb-seq 数据集：
1. Adamson 数据集：87 个单基因扰动，每个扰动约 100 个细胞，7000 个对照细胞
2. Norman 数据集：包含 131 个双基因和 105 个单基因扰动

我们使用 Pearson 相关系数（PCC）和均方误差（MSE）来评估模型性能。

我们将所有单细胞基础模型与**经典扰动预测模型 GEARS 相结合**，在扰动预测任务中进行评估。GEARS 利用 **基因调控网络**作为先验知识，提升对单基因与多基因扰动的建模能力。

我们用 CellFM 生成的 **gene embedding 替代 GEARS 中原有的基因表示**。（GEARS是评估框架）

图 3d 展示的 两个实际扰动案例的可视化结果，进一步表明 CellFM 能**准确预测扰动趋势（表达上调或下调方向）**。

**药物扰动**：

我们进一步在 药物扰动数据上验证 CellFM 的表现，并与经典药物扰动模型 CellOT 结合。

CellFM 与 CellOT、scGEN 以及 Identity 基线模型 进行了比较。

**可视化**：

在已有扰动实验有限（如 Norman 数据集覆盖率仅 5%）的情况下，利用 CellFM 模型对所有未实验的基因扰动组合进行大规模预测，并用可视化手段进行验证和解释，探索模型的泛化能力和生物学可解释性。

1. in silico 扩展预测
用 CellFM 预测 原实验中未涵盖的所有基因组合扰动后的表达谱

在计算上模拟实验结果（in silico），即 “虚拟实验”

2. 剔除扰动基因表达
假设我们预测的是 SET + CLDN6 的扰动组合

我们将 SET 和 CLDN6 的表达值剔除，只保留其余基因的表达谱输入到 UMAP

原因：被扰动的 gene 本身表达量往往变化剧烈，容易主导模型分布，干扰聚类结构

3. UMAP 降维 + 聚类观察
用 UMAP 把预测的表达谱（不含扰动基因）降维到 2D 可视化

得到一张“扰动效果图”：每一个点是一个预测的扰动组合结果

聚成一簇的表示它们表达模式类似，远离表示差异大

4. 识别“主导基因”主导的聚类
观察聚类是否以某个基因为核心形成（如所有包含 SET 的组合聚在一起）

如果 SET + CLDN6、SET + MIDN 等都在同一类 → SET 是“dominant gene”

表明模型能捕捉基因调控主导性的表达结构

### Reverse perturbation prediction

即不是预测某个扰动的结果，而是 给定一个目标表达状态，预测哪种扰动组合最可能实现该表达状态（例如从疾病状态恢复到健康状态）

除了预测基因扰动的结果，**准确预测可促使细胞从疾病状态恢复的 CRISPR 靶基因组合**同样重要。

在此，我们遵循 **scGPT16 的研究方法**，使用 Norman 数据集 开展了 in silico 反向扰动预测实验。

具体而言，我们从 Norman 数据集中选择了 20 个扰动基因，构建微调与测试用的扰动组合。这 20 个基因的所有单基因与双基因组合共计 **210 种扰动情况。**这 20 个基因的所有单基因与双基因组合共计 210 种扰动情况。由于 scGPT 并未指定随机种子，我们使用了默认随机种子进行划分。

划分后的数据集包含 47 个已知扰动（约 22%），其中：
- 33 个训练
- 3 个验证
- 11 个测试
- 其余 163 个为 未见扰动组合。

我们使用该新划分的数据集对 scGPT 与 CellFM 进行了评估，因为 scGPT 原文未公开划分方式，无法完全复现其原始实验。CellFM 表现优异，能够准确预测导致目标细胞状态的扰动组合。

**反向扰动预测流程：**
| 步骤                | 操作说明                              |
| ----------------- | --------------------------------- |
| ① 定义目标表达谱         | 来自某个“健康”或“理想”细胞状态                 |
| ② 遍历所有候选扰动组合      | 如：20 个基因 → 枚举所有 210 个 1/2-gene 组合 |
| ③ 使用 CellFM 做正向预测 | 对每个扰动组合，预测其扰动后表达谱（in silico）      |
| ④ 计算差异度指标         | 与目标表达谱做比较，常用指标如：L2 距离、PCC（相关系数）   |
| ⑤ 结果排序            | 选出使预测谱最接近目标谱的 top-k 扰动组合          |

只涉及了基因扰动没有药物扰动。

### cell annotation

**常规数据集：**

采用scEval框架进行评估

遵循 scGPT 和 scEval 等先前研究的做法，我们在数据集内评估中采用随机划分训练集与测试集的方式，以在一致的实验条件下评估模型表现。

除了数据集内评估，我们还进行了跨数据集测试，这种方式更能反映现实场景中由于批次效应（batch effect）带来的挑战。（数据分布迁移，批次效应指的是不同批次的数据分布不同）

在跨数据集评估中，我们按照数据批次或病人 ID 对数据进行划分，迭代地将每一个批次作为测试集，其他批次作为训练集。（每个批次数据分布不一致，选取其中一个作为验证）

对于 hPancreas 数据集，我们直接采用了 scGPT 论文中设定的训练-测试划分方式，确保训练集和测试集来自不同批次。

在所有的标注任务以及迁移批次标注任务中，都冻结模型参数，并训练分类头分类。CellFM-80m表现出色，且比CelFM-800m表现更好。可能是因为，细胞注释数据集较小，无法充分训练。

我们还使用scIB指标得分评估了所有单细胞基础模型在跨数据集上的嵌入（embedding）质量。
（注：scIB是一套用于评估单细胞数据整合效果的指标体系。）

此外，我们还结合了scIB指标分数来评估CellFM中MLP层的影响，并发现将层从一层调整到三层时性能差异很小。（也就是说MLP的层数对分类性能影响不大）

**细胞亚型数据集：**

为了进一步评估 CellFM 在区分特定细胞亚型（例如耗竭型和活化型CD8+ T细胞）方面的能力，我们在基底细胞癌（BCC，数据集编号GSE123813）和肝细胞癌（LIHC，数据集编号GSE140228）两个数据集上进行了评估。这两个数据集均来自数据库文章36，因为它们具有h5ad格式以及详细的细胞类型注释信息。



大模型在分类任务上finetune效果更好：

- 分类任务的**优化路径稳定**，是因为其具有明确的监督目标（如交叉熵损失），模型只需简单地沿着梯度方向拟合标签信息。这种明确的信号，使模型能快速收敛，训练难度较低，数据量需求也相对较小。

- 而自监督任务（如掩码自编码 MAE 或对比学习 Contrastive Learning），本质上是为了学习泛化能力强的通用特征表示。这类任务并非简单的标签拟合，而是要求模型**逐步探索潜在表征空间**（latent space），捕获数据中隐含但未明确标注的语义结构或关联关系。

具体地说，自监督任务缺少显式监督信号，没有特定明确的优化方向，而是通过对比或重构损失在高维空间中逐渐寻找有意义的表示结构，训练过程更接近于“探索”，而非“简单拟合”。

与自监督任务（如MAE、mask prediction）相比，分类任务的**训练目标明确、优化路径稳定**，更有利于大模型在有限数据中充分释放表达能力。

**LORA：**

应用LoRA后，同样微调，LoRA微调性能变化很小。但是显著减少了微调时间。

而使用LoRA进行微调，引入潜在的瓶颈，增加训练轮数，反而可能提升性能。

**IncRNA：**

由于**长链非编码RNA**（lncRNAs）被包含在CellFM的训练数据中，我们利用**注意力得分**（attention scores）来评估CellFM识别特定细胞类型相关lncRNA的能力。

实验发现引入IncRNA能够提高识别准确率。

**归一化方法评估：**

我们测试了scTransform方法（文献38），该方法能够纠正基因表达数据的方差-均值偏差。使用scTransform归一化方法的表现略低于CellFM原先使用的log1p归一化方法。这些结果表明，尽管scTransform能够明确校正方差-均值偏差，但在我们评估的细胞类型注释任务中，它并未明显提升CellFM的表现。(单纯纠正方差-均值偏差，不一定提升效果)

**ERetNet：**

测试了ERetNet和RetNet的效率。

们进一步测试了对RetNet的两个关键改动：一是将传统前馈网络替换为门控双线性网络；二是将标准的预层LayerNorm替换为DeepNorm层归一化技术。

如补充图S21(b)所示，移除Simple Gated Linear Unit（简单门控线性单元）和DeepNorm分别导致性能下降了0.8%和0.9%。

此外，去除L_cls损失项（分类头损失）也使性能略微下降了0.4%。（gene对齐之外，还对齐cls）

此外，CellFM中采用的门控多头注意力（Gated Multi-head Attention，MHA）结构将计算复杂度从O(l²_max d)降低到O(l_max d²/h)，其中d设置为1536，注意力头数h为48。

因此，CellFM的实际计算复杂度为O(2048×1536²/48)，明显小于O(2048²×1536)。具体的公式推导可参见补充说明1。

**批次效应整合性能的评估：**

为进一步评估CellFM在整合存在批次效应的数据集中的性能，我们对比了三个单细胞基础模型：scELMo、scGPT和UCE。

如补充图S19所示，CellFM在这些数据集中达到了最高的平均AvgBio分数，比第二名的UCE模型高出2.1%。

由于 CellFM 使用的深度学习框架 MindSpore 不支持梯度反转层（Gradient Reversal Layer, GRL）技术。
- GRL 是一种常用于领域对抗训练的技巧，常用于消除批次效应。


### deciphering gene relationship

我们检验了 CellFM 是否能够通过其基因**嵌入向量（embeddings）**和**注意力图（attention maps）**有效编码这些基因关系。

为了评估预训练的 CellFM 对基因关系的捕获能力，我们分别使用来自免疫数据集的32484个免疫细胞以及人脑数据集的约20万个非免疫细胞，对 CellFM 模型进行了微调（fine-tuning）。

如图5a和补充图S22所示，我们展示了三个基因关系图：
- 图5(a)：预训练的 CellFM
- 图S22(b)：在免疫细胞上微调后的 CellFM
- 图S22(c)：在非免疫细胞（人脑细胞）上微调后的 CellFM

综上所述，CellFM 能有效地保持生物学相关的免疫基因之间的关系。

为了进一步验证识别的基因模块，我们参照 scGPT 方法，在由 K-最近邻（KNN）构建的基因相似性图上使用 Leiden 聚类算法，提取包含5个或以上基因的基因模块。随后，我们利用 KEGG（京都基因与基因组百科全书）对这些基因模块进行了全面的通路富集分析（Pathway enrichment analysis）。如图5b所示，我们将 CellFM 得到的结果与共表达网络分析得到的结果进行了对比。

CellFM 在所有聚类分辨率（resolution）下揭示了更多显著富集的生物通路，仅在分辨率为10时例外。

为了进一步验证 CellFM 所识别通路的有效性，我们在聚类分辨率为40的条件下，对 CellFM 和共表达网络识别的通路进行了比较分析。两种方法均识别出25个共同通路。CellFM 额外独立识别出59个通路，其中7个通路与免疫系统过程相关。相反，共表达网络独立识别出32个通路，其中只有2个通路与免疫功能有关。

**这里的通路识别具体方法为：**

1. 因嵌入和注意力分析：
   - 首先，利用 CellFM 模型获得每个基因的嵌入表示（gene embeddings），同时分析 CellFM 的注意力图（attention maps）。
   - 具体而言，模型中通常使用一个特殊的分类标记（CLS token），通过分析 CLS token 和各基因之间的注意力权重，可以反映基因对分类决策的重要性及基因之间的相互关联强度。

2. 构建基因相似性图（gene similarity graph）：
   - 基于基因的嵌入表示或注意力权重（如 CLS-gene attention 分数），利用K-最近邻 (KNN) 算法，构建基因之间的相似性网络。

   - 在这个图中，每个节点代表一个基因，节点之间的连接强度代表基因之间的相似性或功能相关性。

3. 聚类分析（Leiden clustering）：
   - 对构建好的基因相似性网络，使用Leiden 聚类算法进行聚类，划分出多个基因模块（gene programs）。

   - 每个模块代表一组功能相近或具有潜在生物学联系的基因。

4. 基因模块的通路富集分析（Pathway enrichment analysis）：
   - 对上述聚类所得的基因模块，利用 Kyoto Encyclopedia of Genes and Genomes (KEGG) 数据库进行富集分析。

   - 具体而言，对于每个基因模块，分析模块中包含的基因是否显著富集于特定的已知生物通路。

   - 通常通过统计检验（如超几何检验或 Fisher's 精确检验），判定模块中基因在特定生物通路中的富集程度是否达到显著性阈值。

5. 比较分析与验证：
   - 将上述 CellFM 分析结果与传统的共表达网络分析（co-expression network analysis）结果进行比较，以确定 CellFM 在揭示通路方面的优势与差异。

   - 本文中提到，CellFM 在大多数情况下识别出了更多富集的生物通路，尤其是在免疫相关通路识别方面表现更佳。


### identified genes most affected by perturbations

在这一部分，我们通过注意力图（attention map，如图5c所示）分析了扰动实验中被扰动的基因及其最显著受影响的基因。

具体而言，我们通过向模型输入未受扰动的对照细胞表达谱，并明确指出扰动的基因，来模拟该基因扰动后的影响。 

CellFM能够通过注意力得分准确识别受影响的基因。

**通路分析：**
- **通路**（Pathway）被建模为具有生物功能逻辑的信号网络
  - 若某个基因（或其产物：蛋白质）在这个网络中发挥催化、激活、抑制等作用，它就被认为“参与该通路”。
  - 这种参与不需要高表达，也不必和其他基因共表达。它只要在机制上发挥作用（比如被激活、被磷酸化），就算参与。
- **共表达模式（co-expression）**：多个基因在同一类细胞中一起高表达，代表可能在同一功能模块中活跃。

- **通路富集分析**：如果你筛出的前n个高变、差异、扰动响应基因中，有多个出现在同一通路 → 认为该通路被激活/抑制。

| 分析对象                              | 所需先验               | 分析目标                   |
| --------------------------------- | ------------------ | ---------------------- |
| **通路富集分析（如 KEGG、GO）**             | ✅ 已知通路基因集合（不含拓扑）   | 判断某通路中基因是否在你的筛选结果中显著富集 |
| **通路激活分析（如 GSEA, SPIA, PROGENy）** | ✅ 基因集合 + ❗可能需要通路拓扑 | 推测某通路是否处于激活或抑制状态       |


## Discussion

Firstly, the attention map in CellFM was limited in capturing gene relationships related to static or global biological knowledge.
首先，CellFM 的注意力机制尚不擅长捕捉与静态或全局生物学知识相关的基因关系。

它擅长挖掘 数据中局部、样本相关的表达共现关系，但难以自动学习那些不直接体现在表达数据中的、跨样本一致的全局知识，比如：
- Gene SPI1 是一个已知的转录因子；
- Gene CENPF 在大量文献中被证实和 SPI1 调控相关；
- 但是，如果在训练数据中，SPI1 和 CENPF 表达相关性弱（比如只在某些类型中相关），
- 那么 CellFM 的 attention 很可能 无法建立起这两个基因之间的显著连接。
- 也 就是说，attention map 只能挖局部表达结构，难学全局生物通路知识。


## Method

### Data collection

本研究所使用的所有训练数据均来自可靠的公共数据库。具体而言，从 2021 年 4 月至 2023 年 8 月，我们使用关键词“单细胞 RNA 测序”、“单细胞转录组”以及“单细胞 RNA”来检索数据集。我们在 NCBI GEO、ENA、GSA、ImmPort 等数据库中使用这些关键词进行检索。在筛选过程中，我们严格挑选数据集，仅保留与本研究相关的人类单细胞数据集。这些数据集的格式多种多样，包括原始的 FASTQ 文件、表达矩阵、Seurat 或 Scanpy 对象等。首先，我们使用测序仪厂商提供的基础分析软件将原始 FASTQ 数据转换为表达矩阵。随后，所有获得并转换后的表达矩阵都通过了 Singleron Biotechnologies 公司提供的 SynEcoSys® 单细胞数据库的标准化预处理流程。

1. 质量控制：每个细胞至少表达 200 个基因，低于该阈值的细胞会被过滤。这是典型的质量控制策略，旨在剔除死亡细胞、空滴或低质量细胞。

2. 基因名称标准化：遵循 HUGO 基因命名委员会（HGNC）指南，将每个数据集中的基因别名统一为 HGNC 批准的标准符号，以消除命名差异，例如“CD14”和“Monocyte Differentiation Antigen CD14”都会被统一为“CD14”。

此步骤确保所有数据集中基因名称的唯一性与一致性，便于后续跨样本的分析与模型输入对齐。

3. 最后，每个样本的表达矩阵被统一转换为稀疏矩阵格式，以供后续模型训练使用。稀疏矩阵有利于节省内存，加快大规模训练，且统一格式便于批量处理。


### CellFM architecture

CellFM 模型包含三个核心模块：嵌入模块（embedding module）、ERetNet 模块和 LoRA 模块（见图1）。
- 嵌入模块：将表达值映射为向量；
- ERetNet 模块：用于建模表达特征间的关系（主干网络）；
- LoRA 模块：低秩适配器，用于高效微调。

#### embedding

为提高 CellFM 的训练效率，我们将模型的输入基因数上限设置为 2048。

对于每个细胞，如果其表达的基因数超过最大值，我们会从高表达基因中随机选取 2048 个基因。

相反，如果细胞表达的基因数少于 2048，则补零（padding）补齐 ID，并将对应表达值设为 0。

这些填充值不会参与模型训练计算，因此不会影响模型性能。

我们使用 MLP 将标量的基因表达值映射为 ERetNet 所需的向量嵌入，具体如下：
$$
\begin{align*}
X_1 & = \text{LeakyReLU}(X_0 W_0) \\
X_2 & = X_1 W_1 + \alpha X_1 \quad (\text{残差项}) \\
X_3 & = \text{Softmax}(X_2) W_2
\end{align*}

\text{where} \quad 
W_0 \in \mathbb{R}^{1 \times b}, \quad 
W_1 \in \mathbb{R}^{b \times b}, \quad 
W_2 \in \mathbb{R}^{b \times d}
$$
其中 $\alpha$ 是可学习的残差系数，b = 256, d = 1536

**训练目标：**
随机遮掩20%的基因表达值，并利用未遮掩的基因表达值来预测被遮掩的基因表达值。被遮罩的基因仅从真实表达（非填充）的基因中选取。我们用一个可学习的、初始化为零的向量$X_M$替代被遮罩的表达嵌入。

$$
X_{\text{tmp}} = M \odot X_3 + (1 - M) \odot X_M
$$
> 该公式就是把遮掩的换成X_M然后拼接起来
最终的遮罩后嵌入为 $X_{\text{tmp}}$，用掩码选择原始表达还是替代向量。

**Gene ID embedding：**

为了学习每个基因独有的特征，我们初始化了一个大小为 24079 × 1536 的 gene ID 嵌入矩阵。

$$
X_{\text{emb}} = \text{concat}(E_{G_{g_1}}, \ldots, E_{G_{g_{\text{max}}}}) + X_{\text{tmp}}
$$

最后拼接一个可学习的cls token

#### ERetNet

RetNet（Recurrent Token Network）是一种线性复杂度、保留注意力结构本质的高效序列建模架构。它兼顾了 Transformer 的并行性与 RNN 的递推性，核心目标是在不牺牲建模能力的前提下，将计算复杂度从 $O(L^2)$ 降至 $O(L)$，适用于长序列建模。

在 RetNet 中，递推公式为：$h_t = \alpha \cdot h_{t-1} + \beta \cdot x_t$ ，我们可以将这个公式展开到 $x_0$，得到：

$$
h_t = \alpha^t \cdot h_0 + \beta \cdot \sum_{k=0}^{t-1} \alpha^k \cdot x_{t-k}
$$

展开后的计算方法可以表示为矩阵乘法 $X_{\text{seq}} \cdot W^T$，其中 $W^T$ 是一个下三角矩阵，具体形式为：

$$
W^T = 
\begin{bmatrix}
\beta & 0 & 0 & \cdots & 0 \\
\alpha \beta & \beta & 0 & \cdots & 0 \\
\alpha^2 \beta & \alpha \beta & \beta & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\alpha^{t-1} \beta & \alpha^{t-2} \beta & \alpha^{t-3} \beta & \cdots & \beta
\end{bmatrix}
$$

因此，$h_t$ 可以通过以下矩阵形式从 $x$ 计算得到：

$$
\begin{bmatrix}
h_1 \\
h_2 \\
h_3 \\
\vdots \\
h_t
\end{bmatrix}
=
\begin{bmatrix}
\beta & 0 & 0 & \cdots & 0 \\
\alpha \beta & \beta & 0 & \cdots & 0 \\
\alpha^2 \beta & \alpha \beta & \beta & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
\alpha^{t-1} \beta & \alpha^{t-2} \beta & \alpha^{t-3} \beta & \cdots & \beta
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_t
\end{bmatrix}
+
\begin{bmatrix}
\alpha \\
\alpha^2 \\
\alpha^3 \\
\vdots \\
\alpha^t
\end{bmatrix}
h_0
$$

分析展开后的隐含计算公式，可以看出每个 $h_t$ 是通过对所有之前的输入 $x_k$ 进行加权求和得到的，其中权重由 $\alpha$ 和 $\beta$ 控制，体现了递推关系的累积效应。

**ERetNet：**

第一，我们用 门控双线性网络（gated bilinear network） 替代了 RetNet 中的传统前馈神经网络，从而提升了模型性能并使训练过程更平稳。

第二，我们将 RetNet 中的 前层归一化（pre-LN） 替换为 DeepNorm 归一化技术，以进一步增强模型的训练稳定性和性能。
- DeepNorm 是为训练深层 Transformer 提出的归一化方法，通过 缩放残差连接权重 来提升训练稳定性。适合参数量巨大的 CellFM。

综合以上改进，我们构建了 ERetNet 模块，包含 **门控多头注意力机制（Gated MHA）**、**门控线性单元（SGLU）** 和 **层归一化（LN）**，这些组件共同增强了模型在基因表达分析中的稳定性与效果。

##### Gated multi-head attention (MHA)

Gated MHA 模块用于学习基因之间的依赖关系，它是 RetNet 中 Retention 机制的一种变体。

为了解决 RetNet 中指数级注意力计算 $O(l_{\text{max}}^2 \cdot d)$ 的低效问题，我们采用了 Shen 等人提出的方法。

该方法通过先计算 K 和 V，再计算 Q，从而将计算复杂度降为 $O(l_{\text{max}} \cdot \frac{d^2}{h})$，大大降低了计算开销。

此外，为了支持半精度训练，我们对 Q、K、V 进行了如下缩放：
$$
Q = \frac{\text{ReLU}(XW_Q)}{d}, \quad K = \frac{\text{ReLU}(XW_K)}{d}, \quad V = \frac{XW_V}{d}
$$

##### Simple Gated Linear Unit (SGLU)

为了提升模型性能并使训练过程更平滑，我们用一个门控双线性网络替代了 RetNet 中传统的前馈网络（FFN）。

LU（Gated Linear Unit）引入了**乘性门控机制**，它明确地表示模型对每一个特征维度的“记忆程度”，从而使训练过程更平滑，同时促进了不同通道之间的特征融合。

考虑到门控机制本身已经引入了非线性关系，为了进一步加速模型的计算，本文参考文献 [61]，采用了 SGLU —— 它基于 GLU 的公式，但省略了其中的 Swish 激活函数。

$$
\text{SGLU}(X) = X (W_u \odot W_v) W_o 
$$

RetNet 使用了 pre-norm（前归一化）策略，使训练过程更稳定，但这可能会牺牲一部分模型性能。
$$
f(LayerNorm(x)) + x
$$

CellFM 使用了一种新的 post-norm 方法 —— DeepNorm。

$$
Y^{(l)} = \text{LN}(\text{MHA}(X^{(l)}) + \lambda X^{(l)}), \quad
X^{(l+1)} = \text{LN}(\text{SGLU}(Y^{(l)}) + \lambda Y^{(l)})
$$
其中lamba是超参数，与深度有关


#### Loss

均方误差（MSE） 是 CellFM 的核心优化目标，用来衡量模型预测的基因表达值与真实值之间的差异，尤其在遮蔽基因（masked genes）上进行。MSE 尤其适用于该任务，因为它对**大误差惩罚更重**，有助于更准确地恢复被遮蔽基因的表达表示。

模型使用 一个**前馈 MLP **结合 ERetNet 模块，来预测被遮蔽的 $M$ 个基因的表达值。

目标优化在遮蔽位置处使用 MSE 损失，记作 $M_{\text{mask}}$，其具体定义如下：

$$
\hat{y}_i = \text{MLP}(x_i^{(L)})
$$

$$
L_{\text{MSE}} = \frac{1}{|M_{\text{mask}}|} \sum_{i, M_{\text{mask}}(i) = 1} (\hat{y}_i - y_i)^2
$$

cls也进行损失计算

最终损失为：
$$
L = L_{\text{cls}} + L_{\text{MSE}}
$$

#### 细节

CellFM 包含 40 层堆叠的 ERetNet 模块，每一层拥有 48 个注意力头（multi-head attention）。

使用 Adam 优化器，初始学习率为 1e-7，训练了 2 个 epoch。

总 batch size 为 128，平均分配在 **4 台 Altas800 服务器（每台 8 个 Ascend910 NPU）**上。

将训练限制在 2 个 epoch，是参考了大规模模型的通用训练经验 —— 大模型往往在前几个 epoch 就已经收敛。



