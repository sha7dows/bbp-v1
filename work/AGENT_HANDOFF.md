<!-- 用途：向后续agent交接BeHM论文起点、稳定BBP基线、已完成探索、证据位置及后续研究边界；不启动任何新训练。 -->
# BeHM → BBP → 类HSL探索：Agent交接报告

整理日期：2026-11-10（Asia/Shanghai）。工作目录：`/root/TRY/work`。本报告基于服务器原始PDF、当前实现及结果JSON核对，适用于接手研究与代码任务的agent。

## 0. 接手时先明确的结论

1. **BeHM论文是原始方法起点；BBP是用户指定的当前稳定基线。** 最新关联反馈模型只是探索分支，未替代稳定BBP。
2. 用户当前研究重点是**类HSL的结构—表示协同优化**。用户已经明确将“大超边规模处理”暂时搁置，不要自行把规模预算、成员精简或超边分解混回主线。
3. 本阶段依次尝试了整边反馈重评分、受约束成员删除、冻结/分类型上下文反馈、STC分解，以及连续关联加权。**尚未得到可靠的优于BBP的证据。**
4. 最新关联反馈已完成标准CV5和用户授权的最终测试：CV正类F1 `0.669186 ± 0.035959`；最终194会话五模型集成F1 `0.697674`。历史BBP对应值分别为 `0.672858 ± 0.041571` 和 `0.697674`。
5. 没有同环境、严格匹配初始化/随机流的BBP/raw/context完整CV5对照，因此不能将历史结果差异直接归因为反馈机制；也不能由相同F1推出逐样本决策完全一致。
6. 最终测试集**已经被查看过**，包括历史BBP评估及后续用户授权的关联模型评估。不要再把它描述为从未使用的新测试集，更不能用它筛选结构方案或阈值。
7. 本次交接整理没有启动新训练。核查时未发现上述实验训练入口仍在运行。

建议先读本报告第2～5节建立方法与协议边界，再读第6节的尝试记录，最后按第10节开展后续工作。

## 1. 执行约定与服务器

| 项目 | 当前约定 |
|---|---|
| 远端项目 | `/root/TRY` |
| SSH | `ssh root@hn01-ssh.gpuhome.cc -p 30963` |
| 本地项目 | `D:\20909\Desktop\TRY`，与远端不自动同步 |
| Python | `/opt/conda/bin/python`，训练记录为Python 3.12.13 |
| GPU | RTX 4090，约24GB，使用 `cuda:0` |
| 项目协作规范 | `/root/TRY/AGENTS.md` |
| 版本管理 | 核查时远端没有 `.git`；使用备份、源码快照与SHA256清单追溯 |

**用户明确要求：项目代码的编写、修改、检查、训练和真实数据诊断都在服务器完成。本地只作SSH通道与结果阅读入口。** 不要在本地运行项目代码，不要假定本地同名文件就是当前服务器版本。

新增文件开头说明用途；代码注释使用简体中文，运行日志使用英文。密码不得写入文档、脚本或配置；使用交互式认证或另行配置的密钥。本报告不记录密码。

最新标准CV5保存的环境：

| 组件 | 版本 |
|---|---|
| PyTorch | `2.12.1+cu130` |
| CUDA运行时 | `13.0` |
| torch-geometric | `2.8.1` |
| NumPy | `2.4.4` |
| scikit-learn | `1.9.1` |
| NetworkX | `3.6.1` |
| python-louvain | `0.16` |
| NVIDIA驱动 | `595.91.07` |

原始记录：`/root/TRY/outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/environment.json`。
本次仅为核对PDF补装了 `pypdf 7.0.0`，没有重训模型或修改训练依赖参数；安装记录在 `/root/TRY/work/pdf_reader_setup.log`。

## 2. 原始起点：BeHM论文实际提出了什么

### 2.1 原文与问题

原始PDF：

`/root/TRY/docs/Paper3008#Behavior-driven Hypergraph Modeling for Accurate Cyberbullying Detection in Social Media.pdf`

标题：**Behavior-driven Hypergraph Modeling for Accurate Cyberbullying Detection in Social Media**。
作者：Xiaoyu Sean Lu、Zihao Ding、Xugong Qin、Di Wu、Jia Duan。
当前提供的PDF共6页；本次没有核实正式发表场所和年份，不要根据文件名“Paper3008”自行补造出版信息。

页码索引：

| PDF页 | 内容 |
|---|---|
| 1～2 | 群体协同行为动机、会话级二分类问题、节点输入 |
| 3 | Fig.2整体结构，STC/User/Mention三类行为超边，Eq.2～4 |
| 4 | IBG、GAT、Louvain、多维融合、Eq.5～9与实验设置 |
| 5 | Table I性能、Table II超边分布及消融 |
| 6 | 后续讨论与参考文献 |

辅助提取文本在 `/root/TRY/work/behm_paper_extracted.txt`，数学符号可能有提取误差，引用仍应回看PDF。原PDF及其哈希在证据清单中登记。

任务是对**整个媒体会话**进行欺凌/非欺凌分类：帖子和评论组成内容节点，用户、时间、提及等构成行为关系。没有逐评论、逐关联的欺凌真值；会话标签不能直接当成每个节点的标签。

### 2.2 论文的三阶段结构

```text
会话内容节点 + 用户/时间/提及信息
  → PLM节点特征
  → 三类行为超图
  → HGNN学习节点与超边表示
  → 超边作为IBG中的“行为节点”
  → Jaccard行为关联 + GAT传播 + Louvain社区
  → 节点/行为/社区多维注意力融合
  → 会话级分类
```

论文的重点是**把超边作为有语义的群体行为，再建模行为与行为之间的关系**，不是预算选择或HSL。

三类超边：

- **STC：语义—时间协同行为。** 以内容节点为中心，根据语义相似与时间邻近构造。
- **User：用户一致性。** 同一用户的内容节点组成一条边。
- **Mention：提及交互。** 论文Eq.4是“提及用户u的节点”与“用户u发出的节点”的无序并集。

STC融合强度的原文形式为：

\[
F_{ij}=s_{ij}\left(1+\beta\,\mathbf1[s_{ij}>\eta]\exp(-T_{ij}/\tau)\right).
\]

IBG的每个节点对应一条行为超边，边权为成员集合Jaccard：

\[
J(e,f)=\frac{|V_e\cap V_f|}{|V_e\cup V_f|}.
\]

论文Fig.2及第4页明确使用 **GAT** 进行行为传播，再做加权图上的Louvain和社区均值池化。多维融合用社区全局表示作为查询，汇聚节点与超边局部信息。

分类端采用 `logits = f(C_final) - δ log π` 和会话标注置信度加权交叉熵；其中类频率不应来自验证集或测试集。

### 2.3 原论文结果不是本项目CV5基线数字

论文第4页实验设置：80/20划分、40轮、学习率 `1e-4`、dropout `0.2`，结果为五次运行平均；不是本项目的“固定留出+开发集五折”。

论文Table I的Vine BeHM结果为：

| Accuracy | Precision | Recall | F1 | AUC |
|---:|---:|---:|---:|---:|
| 0.816 | 0.702 | 0.738 | 0.716 | 0.845 |

这些是论文报告值，**不是本次在服务器复现得到的值**。不要将论文F1 0.716、BBP开发CV均值和194会话集成测试结果混到同一排行榜。

论文Table II中Vine原始STC/User/Mention数量为63389/5704/23。当前稳定配置会先做STC子集去重，因此当前候选数量不能直接照抄论文分布。

## 3. 当前稳定基线：BBP + BeHM主干

### 3.1 稳定基线的身份

- 模型类：`/root/TRY/bbp_behm/model.py` 中的 `BBPBeHM`。
- 选择器：`/root/TRY/bbp_behm/selector.py` 中的 `EdgeSelector`。
- 主干：`/root/TRY/bbp_behm/backbone.py` 中的 `BeHMBackbone`。
- 标准入口：`/root/TRY/scripts/train_vine_cv5.py`。
- 稳定架构开关：`--architecture original --mode bbp`。

**`--mode baseline` 只是关闭BBP选择，不等于论文严格复现。** 它仍使用当前配置的去重、GCN、融合实现等。

稳定数据流：

```text
原始特征X及会话元数据
 → 固定规则构造候选H0（STC/User/Mention，STC subset去重）
 → 投影特征P = Dropout(ReLU(Linear(X)))
 → BBP整边评分与稳健自适应选择
 → 二值前向/直通反向掩码m，H = m ⊙ H0
 → 两层HGNN与超边池化
 → 在保留边上建IBG、GCN传播、Louvain社区池化
 → 社区引导的局部融合与分类损失
 → 通过直通掩码更新评分器
```

原BBP本来就有 `分类损失→掩码→评分器` 的端到端梯度。额外多做一次前向，并不能自动证明新增了有效的“双向学习”。

### 3.2 数据与超图对象

当前已验证的数据：

- 文件：`/root/TRY/data/vine/KbertConfidence/processed/KbertConfidence_data_bert.pt`。
- SHA256：`C50F3263CA28D156953414D9E4AD1CCF48FF0045A7867880636F3A2E99339548`。
- 970会话、68662节点、768维离线BERT特征。
- 训练中不加载或微调PLM。不要把节点投影器称为端到端BERT训练。
- 当前交接中的新实验均为Vine；存在其他数据文件，不代表其他数据集已经完成验证。

`HyperedgeBatch.incidence` 的方向为 **E×N**，与一些论文使用的N×E相反。
其他字段包括 `edge_to_session`、`type_ids`、四维结构统计及来源记录。

`HyperedgeRecord.members` 存的是**全局节点ID**，不是批内列号。
四维统计为：
`[log1p(size), construction_strength, log1p(time_span_days), log1p(unique_user_count)]`。

当前候选构建参数：
`tau=0.1, threshold=0.60, semantic_gate=0.55, time_weight=0.5, time_window_seconds=86400`。
同类型完全相同成员集合合并，少于两个成员的候选不保留。稳定配置另对STC执行 `dedup_mode=subset`，去掉被其他STC严格包含的候选；不是跨类型去重。

### 3.3 BBP评分、覆盖和梯度

评分器输入包括投影节点的边内均值、16维行为类型嵌入、归一化四维统计；再由共享MLP与带会话上下文的MLP相加。稳定配置未开启IBG中心性特征。

覆盖口径使用固定的同类型代表性矩阵：

\[
R(e,f)=\tfrac13J(e,f)
+\tfrac13\frac{\cos(\bar X_e,\bar X_f)+1}{2}
+\tfrac13\exp\left(-\mathrm{mean}|\mathrm{LN}(a_e)-\mathrm{LN}(a_f)|\right).
\]

语义核使用原始特征的边均值作为稳定锚，不是随训练更新的关联加权表示；对角线设为1。

三种预算模式必须区分：

| 模式 | 行为 |
|---|---|
| `fixed` | 每会话保留 `max(1,ceil(ρE))` 条边；不自动附带稳健尾部覆盖保证 |
| `adaptive` | 按任务分数排序取前缀，达到按类型平均覆盖阈值时停止 |
| `robust_adaptive` | 任务效用与覆盖进度增益联合贪心，按最差行为类型的尾部覆盖停止 |

稳定配置使用 `robust_adaptive`：
每个原候选边由同类型保留边中的最大R值代表；每种类型取最低 `max(1,ceil(q·n_type))` 个覆盖值的平均；所有存在类型的最小值须达到 `γ`。

- `q=0.20`，`γ=0.85`，任务权重 `η=0.25`。
- 贪心效用为归一化任务分数与归一化覆盖进度增益的组合。
- 覆盖进度按类型平均并在达标后截断，停止判据仍为最差类型。
- 保留比例边界为0～1；实际实现有非空候选至少保留一条的兜底。
- 空类型不等同于有一类零覆盖边，不要据此伪造固定类型配额。

**正确的直通公式：**

```python
# 用途：说明当前选择器真实使用的直通掩码方向。
straight_through = soft_mask + (hard_mask - soft_mask).detach()
```

它的前向值是hard，梯度通过soft。旧架构说明中的 `hard + (soft-hard).detach()` 方向写反，不能照抄。对应实现见 `selector.py` 的固定、自适应和稳健选择分支。

预算数、停止条件、覆盖核和离散选择不直接求导；稳健模式的代理分数保留任务项梯度。`redundancy_lambda>0` 时可另加kept-set冗余正则，但稳定归档及当前关联CV5均为0。
配置里有 `lambda_fidelity`、`lambda_comprehensiveness` 等历史字段，**当前主训练并没有因此启用蒸馏/充分性/必要性损失**；要核对实际loss代码，不能根据字段名推断方法组成。

### 3.4 与论文的已知差异

| 部分 | 原论文 | 当前稳定实现 |
|---|---|---|
| 节点语言特征 | 定义PLM编码M(n) | 使用离线768维快照，训练图无PLM参数 |
| Mention | 提及u的节点与u所有内容的并集 | 顺序扫描；用户被提及后，其后续内容才加入对应边 |
| STC冗余 | 论文未加入BBP式构建去重 | 类型内部完全重复合并、稳定配置subset去重 |
| 行为关系传播 | GAT | `GCNConv`，以Jaccard作为edge_weight |
| 融合 | Eq.6～7的社区条件注意力与残差 | Q/K/V线性投影，实际为对 `C_G+Attended` 再做输出投影、LN和Dropout |
| 损失 | 置信度加权CE、logit调整 | 另外使用正类权重1.5 |
| 行为选择 | 全部构建的行为超边 | 先经BBP选择保留行为 |
| 实验 | 40轮、dropout0.2、80/20五次运行平均 | 20轮、dropout0.35、固定留出+开发CV5 |

不要把上述工程差异全部归为“只替换一行GAT即可严格复现”。

## 4. 稳定基线与标准评估协议

### 4.1 划分与训练设置

固定seed42外层分层划分：开发776会话，最终194会话；最终集为133负类、61正类。
开发集为 `StratifiedKFold(5, shuffle=True, random_state=42)`。

第一折620训练/156验证；其余折621训练/155验证。各折同一训练种子42，**不是五个训练种子，也不是五次随机划分重复实验**。
每个开发样本只在自己的验证折产生一条OOF预测。

主配置：

| 参数 | 值 |
|---|---|
| epoch / batch | 20 / 64 |
| hidden / dropout / selector dropout | 128 / 0.35 / 0.35 |
| Adam lr / weight decay | 0.0001 / 0.005 |
| 正类权重 / 置信度 | 1.5 / raw |
| HGNN residual / edge pooling | false / sum |
| IBG weight mode | jaccard |
| 预算 | robust_adaptive，q=0.2、γ=0.85、η=0.25 |
| 训练/验证采样 | 训练启用Gumbel；验证禁用 |
| curriculum / checkpoint soup / feature noise | 关闭 / 关闭 / 0 |

类别先验、置信度统计只来自各折训练集。
各折完成20轮后，按同一epoch汇总五折验证正类F1，优先均值高、再标准差小、再epoch早，选**一个共同epoch**。
最终评估是该epoch五模型正类概率平均，固定阈值0.5，而不是选五个各自最佳epoch，也不是挑测试上最好的某一折。

CV选epoch后的验证均值不是嵌套CV的无偏测试估计；五折样本标准差也不是独立种子训练方差或置信区间。

### 4.2 稳定BBP记录

基线归档：
`/root/TRY/outputs/cv5/vine_dedup_cv5_8020_seed42_pw1.5`。

|指标|开发CV5共同epoch|最终194会话五模型集成|
|---|---:|---:|
|正类F1|0.672858 ± 0.041571|0.697674|
|Macro-F1|0.757215|0.773548|
|Precision|0.650394|0.661765|
|Recall|0.707823|0.737705|
|AUROC|0.835392|0.851966|

共同epoch为17，五个 `fold_XX_epoch_017.pt` 检查点存在。
基线数据快照与新关联模型一致，开发/最终索引及五折清单已逐项核对一致。
但历史基线训练的软件环境与新模型没有做匹配重训，不能仅凭同seed宣称严格因果对照。

原归档INI：
`/root/TRY/configs/vine_dedup_cv5_8020_seed42_pw1.5.ini`，
内含旧数据路径 `/home/dingzihao/BBP/...` 和 `cuda:2`，不要直接在当前机器照跑。

可运行服务器配置：
`/root/TRY/configs/vine_hsl_server.ini`。
其核心超参数与稳定归档对齐，路径和设备改为当前实例；显式 `--architecture original --mode bbp` 才是稳定BBP分支。

## 5. 类HSL研究范围：借鉴什么，不混入什么

用户明确指的HSL为 **Hypergraph Structure Learning for Hypergraph Neural Networks**，IJCAI 2022：
https://www.ijcai.org/proceedings/2022/0267.pdf 。

原文第3.3～3.5节包含：
粗粒度超边采样、基于节点—超边表示的缺失关联发现与关联采样、原/学习结构之间的同节点跨视图对比学习，并与任务目标联合优化。

“双向增强”的实质是表示指导结构、学习结构反过来改善表示，不要求先堆叠多轮循环。原论文是节点分类，其中超边采样有逐边可训练参数；本项目是跨会话图分类，不能把训练会话的自由边参数当作能泛化到新会话的结构学习器。

当前各分支只是借鉴，不是HSL完整复现：
- 整边反馈版本没有补关联和对比损失。
- 成员编辑版本只能删已有STC成员，且固定编辑数量。
- 最新关联版本只做连续权重，支持集合没有改变。
- 当前没有验证“完整HSL机制在本任务上无效”，更未证明论文新颖性。

## 6. 本阶段实际做过的尝试

### 6.1 A：整边反馈重评分与受约束成员精简

代码：`/root/TRY/bbp_behm/hsl.py`。
正式结果：`/root/TRY/outputs/hsl_pilot/fold1_seed42_fixed70_6ep_v2`。

协议：仅开发第一折，620/156，seed42，batch8，6轮，每会话固定 `ceil(0.7E)`，额外Gumbel关闭。**不是标准稳健预算CV5。**

| 模式 | 实现 |
|---|---|
| bbp | 原选择器 |
| control | 同样试探传播，但适配器只用原投影节点，控制新增容量 |
| feedback | 初选硬支持→共享第一层HGNN→残差适配器→对全部原候选重评分 |
| feedback_edit | 反馈重评分后，对满足规模门槛的STC执行受约束成员删除 |

初选支持detach，但feedback试探HGNN参数可以收到最终分类梯度；使用 `fork_rng` 避免初次试探评分消耗最终评分的随机流。

成员精简：
训练STC规模p75且至少16；每条边最多删除原成员10%，保护所选User/Mention共享成员，避免删除节点最后关联，每条保留边至少两成员。不补关联、不改User/Mention。

**关键设计局限：删除数量预先由floor(0.1·size)等约束决定。能学删除谁，却不能根据任务选择完全不删。** 该局限不能被解释为HSL关联学习的普遍失败。

|第一折固定第6轮|正类F1|AUROC|平均训练秒/轮|
|---|---:|---:|---:|
|bbp|0.590909|0.809079|12.15|
|control|0.574713|0.809079|13.62|
|feedback|0.547619|0.802975|13.43|
|feedback_edit|0.564706|0.802022|51.42|

成员编辑组末轮删除6820个关联，约为所选关联的3.97%；平均训练时间约51.42秒/轮，相比同轮BBP约12.15秒/轮更慢，未获得F1收益。该时间来自当次缓存图实验，不可跨硬件或跨并发设置泛化。

原候选锚覆盖没有把成员删除失真纳入同一口径，所以另报真实成员Jaccard覆盖；不能以原锚覆盖解释编辑后的结构仍满足原始保证。

### 6.2 B：冻结节点反馈与分类型IBG上下文反馈

代码：`/root/TRY/bbp_behm/hsl_context.py`。
结果：`/root/TRY/outputs/hsl_context_20261109`。

实现：
- `node_control`：原投影节点作为适配器上下文。
- `node_detach`：冻结第一层HGNN试探输出，取消试探分支额外编码器梯度。
- `edge_control`：按类型的全局保留边均值作为控制。
- `edge_context`：同会话内，以候选边Jaccard邻接及初选支持构造STC/User/Mention三类邻域和全局上下文，残差重评分。
- 整边候选集合不变，不执行成员编辑。边上下文残差为有界 `0.25*tanh(MLP)`，末层零初始化。

协议：
第一折seed42五组各8轮筛查；按预定末三轮F1、平局AUROC规则将node_detach带到第二折；第二折seed42/43，bbp/node_control/node_detach各12轮。固定70%、batch8、关闭额外Gumbel。
这是开发折探索及复核，第二折也不是全新独立测试。

第一折末三轮平均F1：bbp、node_detach、edge_control、edge_context均约0.544809，node_control约0.539410。按平局规则复核候选，不等于认定其已优于BBP。

|第二折/训练种子|方法|第12轮F1|末三轮平均F1|第12轮AUROC|
|---|---|---:|---:|---:|
|42|bbp|0.638298|0.644917|0.846184|
|42|node_control|0.659574|0.652009|0.848131|
|42|node_detach|0.631579|0.642677|0.846379|
|43|bbp|0.659574|0.634456|0.854945|
|43|node_control|0.659574|0.625195|0.854361|
|43|node_detach|0.659574|0.627510|0.854751|

node_detach相对bbp的双种子末轮平均F1差 `-0.003359`；
4000次按会话、标签分层的配对bootstrap区间约 `[-0.010984, 0]`。
相对node_control差约 `-0.013998`。
这些区间只条件于已训练模型和当前验证样本，不估计完整训练种子方差。

第二折node_detach每次前向初选→重选的支持变化约为保留数的0.17%/0.21%，不是与另一个模型的掩码差异。
训练集24例同预算同类型替换诊断中，平均概率变化约0.031，说明结构并非绝对不起作用，但该反馈形式实际改变支持很少。
结果不支持稳定增益；原因是待验证问题，不能武断地写成“梯度断了”。

### 6.3 C：大STC诊断与受约束分解——已完成、方向搁置

代码：`/root/TRY/bbp_behm/hsl_split.py`。
报告：`/root/TRY/outputs/stc_split_20261109/REPORT.md`。

这条线曾因用户讨论“大超边会影响很多成员”而尝试，后来用户明确要求先聚焦类HSL，**不再继续推进规模方向**。实验状态是complete，不是被取消而未产出结果；保留 `scope_change.json` 记录范围变更。

方法：
- 训练折STC规模p75（第一折63、第二折62）和训练大边语义分散度中位数作为门槛；分解后分散度至少降低0.01。
- 分散度为 `n/(n-1)·(1-||mean(normalize(x))||²)`，只用原始特征，无任务标签。
- BBP在原始父边上选；仅传播时对符合条件父边做平衡二分。
- 最多复制4个关键User/Mention共享锚点，额外关联不超过父边10%；其余成员分到一个子边，子边并集恢复原父边。
- User/Mention不改，保留父边来源；原父边不是同时保留作为另一条传播边。
- 随机组使用同一门控、同一锚点、同一子边数与子边规模，检验分区质量；不是去除全部语义先验的随机方法。

第一折训练诊断：25574条STC，3136父边/138会话触发分解；额外候选关联12303/1162136，约1.06%。
大边分散度中位数0.3633低于全体STC中位数0.3780，**不支持“大边天然更混杂”的前提**。

24个受影响训练会话、保留全部候选父边的静态HGNN算子诊断，相对变化均值语义/随机约10.42%/6.82%；这不是训练后分类收益。
约39.17%语义子边被其他原始STC候选包含，随机约11.91%；其他候选未必会被选中，不能据此直接证明选中图冗余。

协议：第一折seed42三组8轮；第二折seed42/43三组各12轮，固定70%、batch8，最终194会话未用于此支线。
第二折两个种子固定末轮均值：

|第二折双种子均值|第12轮F1|AUROC|额外关联|
|---|---:|---:|---:|
|none|0.659574|0.850857|0.000%|
|random|0.652368|0.848715|1.170%|
|semantic|0.648936|0.849299|1.168%|

语义组相对不分解F1差约 `-0.010638`，相对随机差约 `-0.003432`，未见收益。
不同训练模型的父边选择会分化，因此“逐父边条件匹配”不等于最终实际传播成本严格相同。
不同折/种子进程共享GPU，耗时不能视为严格独占GPU基准。

### 6.4 D：最新关联加权反馈——标准CV5与最终测试均已完成

模型：`/root/TRY/bbp_behm/hsl_association.py`。
配置：`/root/TRY/configs/vine_association_cv5_server.ini`。
主结果目录：
`/root/TRY/outputs/cv5/vine_association_context_cv5_8020_seed42_20261109`。

这是当前最新、最接近用户重新聚焦后设计的分支，但仍是实验模型。

#### D1. 实际架构

设P为原投影特征，H0为原候选E×N关联：

\[
Z=\mathrm{LN}\left(\mathrm{ReLU}(\mathrm{HGNNFirst}_{\theta}(P,H_0))\right),\quad
z_e=\mathrm{LN}\left(\frac{\sum_v H_{0,ev}Z_v}{|e|}\right).
\]

仅对真实STC成员对计算：

\[
a_{ev}=\mathrm{MLP}_{\phi}[Z_v,z_e,Z_v\odot z_e,|Z_v-z_e|],
\quad s_{ev}=0.2\,\tanh(a_{ev}),
\quad w_{ev}=|e|\frac{\exp(s_{ev})}{\sum_{u\in e}\exp(s_{eu})}.
\]

随后：

\[
H_1=(m_{\mathrm{BBP}}\odot H_0)\odot W.
\]

- 共享原HGNN第一层参数，不增加独立teacher；第一次上下文传播用全部原候选，不局限于初选支持。
- MLP为 `4h→h→1`，末层零初始化，初始所有权重严格为1；只在已有STC成员上分块计算，避免E×N×h全配对张量。
- 每条边的成员支持和权重总和保持不变；非STC权重为1；没有成员增删、补关联或大边门槛。
- H1进入完整两层HGNN和超边聚合。
- IBG的Jaccard和Louvain拓扑只读取 `m⊙H0` 的二值前向值；原BBP直通梯度路径仍保留。
- **不能把连续加权矩阵直接传入当前集合Jaccard公式。** 两个矩阵入口已显式分开。
- 正式CV5是联合训练BBP、主干和关联MLP，没有冻结父边掩码；冻结掩码仅是此前提出的隔离诊断方案，不是本轮正式协议。
- 损失仍是原分类损失，冗余正则为0；没有对比、一致性、结构重构或蒸馏损失。

该分支是**连续关联贡献学习**，支持集合没有改变，不能称为已经实现完整离散超图结构优化。

模式开关：
`--association-mode none|raw|context`。
raw使用原投影节点替代HGNN上下文；none关闭加权。**本次仅context完成标准CV5；raw的同环境完整CV5尚未做。**

#### D2. 完整CV5结果

原标准协议全部执行：5折×20轮、batch64、稳健自适应预算、训练Gumbel。
五折独立进程并发，同一份校验过元信息的原始图缓存，最后统一聚合选epoch；没有按中间结果调参或选某一折。

共同epoch：**17**。

|指标|五折均值|样本标准差|
|---|---:|---:|
|正类F1|0.669186|0.035959|
|Macro-F1|0.754408|0.016014|
|Precision|0.646918|0.030071|
|Recall|0.703741|0.106956|
|AUROC|0.835581|0.023783|

|折|正类F1|AUROC|
|---|---:|---:|
|1|0.702703|0.838451|
|2|0.707965|0.871301|
|3|0.622222|0.820483|
|4|0.659574|0.808433|
|5|0.653465|0.839238|

OOF由776个开发会话各自产生一次折外预测，合并正类F1 `0.671906`、AUROC `0.830457`。
它与五折指标均值不同，不要混称“五模型集成开发集结果”。
固定第20轮的五折F1均值 `0.653375`，没有隐藏末轮结果。

#### D3. 最终194会话测试

用户在CV结束后明确授权最终评估。锁定第17轮五模型，平均正类概率，阈值0.5，没有训练、调阈值或按测试挑模型。

结果目录：
`/root/TRY/outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/final_test_evaluation`。

|指标|结果|
|---|---:|
|正类F1|0.697674|
|Macro-F1|0.773548|
|Precision|0.661765|
|Recall|0.737705|
|AUROC|0.852459|
|Accuracy|0.798969|

混淆矩阵：TN=110、FP=23、FN=16、TP=45。五模型分别的测试指标也保留，但没有挑单个最佳模型来代替预定集成。

与历史稳定BBP比较：
- CV F1均值：`0.669186 vs 0.672858`，数值差约 `-0.003672`。
- 最终集F1：均为 `0.697674`。
- 最终集AUROC：`0.852459 vs 0.851966`，差约 `+0.000493`。
- 不构成稳定增益证据。环境和新增参数初始化造成的随机流没有严格匹配；相同seed不等于公平控制了所有因素。
- CV阶段 `result.json` 和 `summary.json` 仍保留当时 `final_test evaluated=false`；**最终评估以单独的 `final_test_evaluation/result.json` 为准**，不是评估没做，也不是模型被重新训练。

#### D4. 机制与工程验证

- 零初始化时，与原BBP的验证logits最大差为0。
- 缓存与直接构图在核查批上逐张量、逐记录一致。
- 支持集合、非STC关联、逐边权重总和、空图、无STC图、二值IBG及真实batch64训练检查通过。
- 第一轮独立fold worker与顺序预跑的预测概率最大差约 `1.79e-7`。
- 共同epoch五个模型重载后，776条OOF概率全部重放，最大差约 `1.94e-7`。
- 五折从实际分类损失到第一次上下文HGNN输出的梯度L1约 `0.03175/0.04284/0.04067/0.04193/0.00857`，均非零有限。因此不能简单将无增益解释为“反馈分支完全断梯度”。
- 平均 `|w-1|` 约 `0.001390`，即均匀权重1的约0.139%；共同epoch权重大致在0.978～1.012之间，实际学习较保守。
- 分段训练记录最大行质量误差约 `3.1e-4`，属于当前float32求和误差量级；不能用浮点逐位相等判定失败。
- 单进程batch64预检峰值约1228MiB；五进程共享GPU一次观测总占用约12544MiB（约12.25GiB），不是独占GPU速度基准。
- 正式训练期间源码哈希未变；各checkpoint及训练源码在最终评估前后也未变。

证据：
`preflight_checks.json`、`artifact_checks.json`、`trained_replay_audit.json`、`historical_reference.json` 及 `final_test_evaluation/result.json`。

## 7. 代码地图与本阶段工程改动

以下路径都在服务器 `/root/TRY` 下。

| 文件 | 职责/状态 |
|---|---|
| `/root/TRY/bbp_behm/data.py` | 数据校验、会话切片、元数据、分层划分与组批 |
| `/root/TRY/bbp_behm/hypergraph.py` | 三类候选构建、STC去重及来源记录 |
| `/root/TRY/bbp_behm/selector.py` | 稳定BBP评分、预算、代表核、稳健贪心和ST |
| `/root/TRY/bbp_behm/model.py` | 默认BBP组装；增加显式恒等扩展点 |
| `/root/TRY/bbp_behm/backbone.py` | HGNN/IBG/GCN/Louvain/融合；增加独立拓扑输入 |
| `/root/TRY/bbp_behm/training.py` | 标准训练/评估；新增可选逐样本预测与关联观测、关联模型有限梯度核查 |
| `/root/TRY/bbp_behm/cached_hypergraph.py` | 无标签原图缓存；正确重编号批内记录 |
| `/root/TRY/bbp_behm/hsl.py` | 第一轮整边反馈及固定删除数量的成员精简 |
| `/root/TRY/bbp_behm/hsl_context.py` | 第二轮节点/行为上下文重评分与控制 |
| `/root/TRY/bbp_behm/hsl_split.py` | 已搁置的大STC分解 |
| `/root/TRY/bbp_behm/hsl_association.py` | 最新连续关联加权模型 |
| `/root/TRY/scripts/train_vine_cv5.py` | 标准CV5；默认original，支持显式association、独立worker与统一聚合 |
| `/root/TRY/scripts/run_association_cv5.py` | 本次关联CV5编排/快照/审计；输出ROOT写死且拒绝覆盖 |
| `/root/TRY/scripts/train_hsl_pilot.py` | 短程单折实验，不能当标准CV5入口 |
| `/root/TRY/scripts/run_hsl_context_study.py` | 第二轮筛查与预定第二折复核 |
| `/root/TRY/scripts/run_stc_split_study.py` | 分解支线编排，不应自行重新开启 |
| `/root/TRY/scripts/check_hsl_association.py` | 新关联架构约束与batch64预检 |
| `/root/TRY/scripts/check_hsl_invariants.py` | 原成员精简检查 |
| `/root/TRY/scripts/check_hsl_context.py` | 原上下文分支检查 |

默认 `original` 不会自动启用关联加权、成员删除或规模分解。不要根据“存在新类”就认为稳定基线被替换。

核心工程改动：

1. `model.py` 的选择、关联编辑、关联权重和传播准备采用显式扩展点，默认恒等。
2. `backbone.forward_projected` 接受独立 `topology_incidence`；默认仍兼容旧调用。
3. Louvain建图原本逐边GPU标量转CPU，改为一次性转移边索引和权重，再在CPU建图；20个合成图输出验证一致。这是性能优化，不是新学习机制。
4. 通用原图缓存只存确定性拓扑，不缓存训练表示或梯度。较早 `train_hsl_pilot.CachedBuilder` 中records未正确重排批内编号；旧分类计算主要用张量，不据此否定其分类结果，但**不要拿旧缓存记录直接做跨会话解释导出**。新通用缓存已修正并核对。
5. 新标准入口保存100份逐轮预测、100份epoch检查点、5份共同epoch模型和 `ensemble_model.pt`，另有776条OOF。
6. 独立worker只完成自己的折，`worker_complete.json` 不代表整个CV5完成；要看主目录 `run_status.json`、`result.json` 和审计。
7. 可选最终测试分支已修复为独立构图，不复用仅含开发会话的缓存。

## 8. 证据位置、异常和版本追溯

### 8.1 主要产物

| 证据 | 绝对路径 |
|---|---|
| BeHM论文 | `/root/TRY/docs/Paper3008#Behavior-driven Hypergraph Modeling for Accurate Cyberbullying Detection in Social Media.pdf` |
| 稳定BBP结果/模型 | `/root/TRY/outputs/cv5/vine_dedup_cv5_8020_seed42_pw1.5` |
| 第一轮正式短实验 | `/root/TRY/outputs/hsl_pilot/fold1_seed42_fixed70_6ep_v2` |
| 第二轮上下文报告/总结 | `/root/TRY/outputs/hsl_context_20261109/REPORT.md`、`summary.json` |
| 分解支线报告/总结 | `/root/TRY/outputs/stc_split_20261109/REPORT.md`、`summary.json` |
| 最新关联完整CV5 | `/root/TRY/outputs/cv5/vine_association_context_cv5_8020_seed42_20261109` |
| 最新最终集评估 | `/root/TRY/outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/final_test_evaluation` |
| HSL原文核查与范围重聚焦 | `/root/TRY/docs/hsl_structure_refocus.md` |
| 关联实现协议 | `/root/TRY/docs/hsl_association_cv5.md` |
| 早期文献笔记 | `/root/TRY/docs/bbp_hsl_research_notes_20261109.md` |

本报告中的实验数字取自上述JSON，不依赖旧聊天总结。
`/root/TRY/work/verified_experiment_facts.json` 保留本次提取的原始字段，`/root/TRY/work/evidence_manifest.json` 记录文件存在性和哈希。

### 8.2 不要混入正式结果的目录

- `/root/TRY/outputs/hsl_pilot/preflight`：早期小样本，随机流修正前，只作预检。
- `/root/TRY/outputs/hsl_pilot/fold1_seed42_fixed70_6ep`：性能优化前的预跑，不是v2正式对照。
- `/root/TRY/outputs/hsl_context_20261109/confirm_fold2_seed42_numeric_check_failed`：训练前严格逐位logits检查失败，约1.19e-7差异；已保留记录，改为硬掩码严格一致、logits明确1e-6容差。不是隐藏失败后重挑结果。
- `/root/TRY/outputs/cv5/vine_association_context_cv5_8020_seed42_20261109_sequential_preflight`：顺序预跑，后改独立折并发并修复可选最终集缓存兼容；未混入正式CV5。
- `/root/TRY/outputs/cv5/association_worker_smoke_20261109`：单折一轮的执行方式等价性检查。
- `/root/TRY/outputs/association_checks_20261109`：约束与batch64预检，不是分类效果报告。

### 8.3 快照与恢复原则

备份：
`/root/TRY/.backups/pre_hsl_20261109.tgz`、
`/root/TRY/.backups/pre_context_20261109.tgz`、
`/root/TRY/.backups/pre_split_20261109.tgz`、
`/root/TRY/.backups/pre_association_cv5_20261109.tgz`。

正式关联目录另有 `source_snapshot.tgz`、`source_hashes.json`、`environment.json`、`command.json`。
当前核心代码与关联训练记录的源码哈希已核对一致。本次报告在work中生成，没有修改训练实现或模型。

先只读检查tar内容和哈希；需要旧版本时提取到新的隔离目录，不要直接覆盖当前服务器工程。路径上的 `20261109` 是历史产物名称，不要因今天日期变更就重命名这些证据目录。

## 9. 接手操作示例（本报告没有执行新训练）

所有命令都在服务器执行；先查看结果，再决定新任务。

### 9.1 只读查看已完成结果

```bash
cd /root/TRY
cat outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/REPORT.md
cat outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/final_test_evaluation/REPORT.md
cat outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/artifact_checks.json
cat outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/trained_replay_audit.json
```

### 9.2 若后续任务明确要求当前环境重跑稳定BBP

```bash
cd /root/TRY
PYTHONPATH=. /opt/conda/bin/python scripts/train_vine_cv5.py \
  --config configs/vine_hsl_server.ini \
  --mode bbp --architecture original --seed 42 \
  --cache-graphs --torch-threads 4 \
  --run-name handoff_bbp_cv5_new
```

使用全新run_name，保留原20轮、batch64、稳健预算和训练采样规则。不要将 `--mode baseline` 当作稳定BBP。

### 9.3 若后续任务明确要求新的关联CV5

```bash
cd /root/TRY
PYTHONPATH=. /opt/conda/bin/python scripts/train_vine_cv5.py \
  --config configs/vine_association_cv5_server.ini \
  --mode bbp --architecture association \
  --association-mode context --association-alpha 0.2 --seed 42 \
  --cache-graphs --torch-threads 4 \
  --run-name handoff_association_context_cv5_new
```

以上直接入口顺序完成五折，方法协议与并发相同。raw控制组须使用新目录并改为 `--association-mode raw`，不能覆盖context结果。

`run_association_cv5.py` 是本次硬编码ROOT的编排器，原样运行会拒绝覆盖现有结果。若确需并发复现，应先明确参数化新输出目录并保留源码快照，不要删除已有结果绕过防覆盖检查。

### 9.4 加载已有模型与最终评估

共同epoch模型在：
`/root/TRY/outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/fold_checkpoints`。

关联检查点与ensemble记录有 `model_options`。加载时使用与之相符的 `AssociationBBPBeHM` 或 `scripts.train_vine_cv5.build_fold_model(..., **model_options)`，严格加载state_dict，并按训练折重建先验/置信度统计。仅用 `BBPBeHM` 加载关联检查点会不匹配。

只想评估已保存模型，**不要运行训练入口附加 `--evaluate-final-test` 来重新训练**。
本次冻结评估程序在：
`/root/TRY/outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/evaluate_final_frozen.py`。
它固定当前run和第17轮，且拒绝覆盖已有最终评估；读取结果即可。后续如果用户明确要求另一冻结run的评估，需要显式选择那个run、保持预定阈值并另存评估，不据测试挑模型。

## 10. 后续任务建议与未解决问题

以下是建议，不是已完成的实验，也不是本报告新增的自动执行任务。

### 优先级1：补齐可归因的对照

先在同一环境、同一划分与标准协议下做稳定BBP / raw关联 / context关联对照。
共享主干初始state，控制新增MLP初始化对后续Dropout随机流的影响；相同seed本身不够。
若先做短诊断，可固定同一父边掩码隔离关联作用，再恢复完整联合训练。
先看是否优于匹配容量的raw组，而不是仅比历史记录高一点就归因于结构上下文。

### 优先级2：解释“梯度非零但更新很小”

已证实：真实分类梯度能到第一次上下文输出，支持和质量约束成立，但平均权重变化仅0.139%。
未证实的候选原因包括：
- HGNN上下文在同一超边内过度相似，经过边内归一化后共同分量抵消；
- 主干或其他分支已能拟合任务，分类目标没有足够推动关联偏离均匀；
- gate、温度/幅度、优化速度和任务监督强度不匹配；
- BBP已去掉部分冗余，使剩余关联调整的收益较小。

应在训练/开发数据上做受控诊断，例如固定预算/掩码、分析边内score方差与权重熵、分开控制HGNN传播加权和边池化加权、做同成本关联扰动。
**结构改动量本身不是优化目标；不要为了“看起来改动多”强制加大α或固定删除比例。**

### 优先级3：分阶段扩展HSL机制

只有当前关联机制得到可重复证据，再分别考虑：
- 原/学习结构之间同评论的跨视图一致性或对比目标；
- 可学习的离散保留/新增关联，允许“不修改”，不强制每条边删除一定比例；
- 同会话、行为类型合法的候选关联，显式User/Mention事实不应由单纯语义相似替代。

每次只增加一个机制，并保留匹配容量/计算路径的对照。不要把同边所有评论都当成相同标签或无条件正样本。原HSL机制、图注意力式加权等都有既有工作，新颖性需要另行核查。

### 当前不推进的方向

大超边规模预算、分解、固定比例成员裁剪。用户已要求暂置；报告保留其结果是为了避免重复，不代表建议下一步继续做。

### 评估边界

最终194会话已用于历史基线和当前冻结模型评估。后续超参数、阈值、结构规则、模型选择应继续只用开发数据。
如果需要新的论文级独立终评，应明确讨论额外独立数据或预定评估方案，不能通过反复查看既有测试集获得“提升”后仍声称未见测试。
不要将一次CV5、两个短实验种子或条件bootstrap包装为充分的泛化和统计显著性证明。

## 11. 已知实现/文档陷阱

1. `docs/architecture_bbp_behm.md` 可作早期导览，但ST公式方向、折样本数概括、缓存/运行状态等存在过时或错误表述；以本报告对应代码与JSON为准。
2. 根目录 `task_plan.md`、`progress.md`、`findings.md` 保留了历史“进行中”“未评估”等语句。它们是时间线，不是当前状态表；后续complete和最终测试记录覆盖早期状态解释。
3. 被搁置的分解实验实际已经完成，不要只看旧未勾选清单就重新跑。
4. 整边数、成员数、关联总量、IBG关系数不同；当前稠密HGNN仍在含零行的矩阵上计算。保留比例降低不等于全链路耗时按同比例下降。
5. 原始HGNN已有边度数与节点度数归一化，不能以“大边未经归一化”解释问题。
6. 新关联模型中IBG必须用二值前向结构；当前集合Jaccard不适合直接吃连续权重。
7. 历史脚本如阈值、full-graph验证、解释导出等可能依赖旧产物格式；未逐项在最新架构下重测，先读加载路径和构造器，不要假定全部即插即用。
8. 不要悄悄替换稳定GCN为GAT、改Mention并集或换评估协议，再把结果当作仅改变HSL模块的对照。
9. 类别加权、logit adjustment、置信度加权共同影响决策；测试后换阈值或改先验会改变比较口径。
10. 源码、checkpoint、配置和划分需要成套追溯；本地旧草稿、同名远端目录与历史tar不能互相代替。

## 12. 交接完成条件与证据优先级

已完成：原文核对、稳定基线身份确认、各分支结果归档核对、最新标准CV5和授权最终评估状态确认，以及文件路径/关键指标核验。
本次只新增交接材料及工作记录，没有修改训练实现、模型权重或数据集。

证据优先级：
**原论文与当前运行源码 → 对应run的配置/划分/JSON/快照 → 对应run报告 → 历史架构说明和聊天摘要。**

后续agent应首先读取：
1. 本报告；
2. `/root/TRY/AGENTS.md`；
3. 稳定BBP的 `result.json` 与服务器配置；
4. 最新关联模型源码、CV5 `summary.json` 与最终评估 `result.json`；
5. 本目录 `evidence_manifest.json` 及 `verified_experiment_facts.json`。

本次交接未启动新实验，下一步工作由用户后续任务确定。
