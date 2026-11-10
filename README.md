<!-- 用途：说明远端 BBP-BeHM 项目的基线、代码入口、复现条件和交接材料。 -->
# BBP-BeHM

以 **Behavior-driven Hypergraph Modeling for Accurate Cyberbullying Detection in Social Media（BeHM）** 为论文起点，在行为超图上进行预算约束的超边选择（BBP），并探索类 HSL 的结构反馈。

**BBP 是当前稳定基线。类 HSL 代码是实验分支，当前结果未显示稳定优于 BBP。**

## 从哪里开始

- [Agent 交接报告](work/AGENT_HANDOFF.md)：论文与实现差异、稳定基线、历次实验、结果、限制和后续建议。
- [证据清单](work/evidence_manifest.json)与[实验事实](work/verified_experiment_facts.json)。
- [服务器运行环境](docs/remote_runtime.md)。
- [本次上传说明](work/GITHUB_UPLOAD.md)。
- [协作约定](AGENTS.md)。

交接报告比早期架构草稿更新；阅读旧文档时以交接报告中的纠正和实际源码为准。

## 代码组织

| 路径 | 内容 |
| --- | --- |
| `bbp_behm/` | 超图构建、BBP 选择器、HGNN/IBG 主干、训练与结构学习模块 |
| `scripts/train_vine_cv5.py` | 标准开发集 CV5 训练入口，默认 `--architecture original` |
| `scripts/run_association_cv5.py` | 最近关联反馈实验的五折任务编排，含固定路径，执行前阅读 |
| `configs/vine_hsl_server.ini` | 当前服务器可用的稳定基线配置 |
| `configs/vine_association_cv5_server.ini` | 关联反馈配置 |
| `docs/` | 原论文、方法说明、实验协议及架构图 |
| `outputs/` | 历史实验与本次保留的汇总、划分和诊断统计 |
| `work/` | 交接报告、证据及原论文提取文本 |

稳定路径：原始候选超图 → 预算约束 BBP → HGNN → 超边行为图（IBG）→ 社区与全局注意力 → 会话分类。

关联反馈路径：共享 HGNN 首层获得上下文 → 对 STC 的节点—超边关联进行归一化加权 → 在 BBP 选择后用于主干传播；保持原支持集和超边规模，未引入大超边拆分。

## 运行条件与示例

项目代码修改、检查、训练和真实数据诊断均在服务器 `/root/TRY` 完成，本机仅作为 SSH 和结果传输入口。

现有环境为 Python 3.12.13、PyTorch 2.12.1+cu130、torch-geometric 2.8.1、NumPy 2.4.4、scikit-learn 1.9.1、NetworkX 3.6.1、python-louvain 0.16。此处记录已运行的环境，不代表其他版本已通过验证。数据使用已冻结的 BERT 特征。

数据文件未上传：`/root/TRY/data/vine/KbertConfidence/processed/KbertConfidence_data_bert.pt`。SHA256：
`c50f3263ca28d156953414d9e4ad1ccf48ff0045a7867880636f3a2e99339548`。

在服务器准备数据后，可使用新的运行名称启动稳定 BBP 的开发集 CV5：

```bash
cd /root/TRY
python scripts/train_vine_cv5.py \
  --config configs/vine_hsl_server.ini \
  --mode bbp --architecture original --seed 42 --epochs 20 \
  --run-name bbp_dev_cv5_new_run
```

示例会启动训练，本次 GitHub 上传没有执行它。不要覆盖已有实验目录。归档配置 `vine_dedup_cv5_8020_seed42_pw1.5.ini` 保留历史路径；运行前使用服务器配置。固定 80/20 划分，开发集内部五折，共同按五折平均验证 F1 选一个全局 epoch；此处 CV5 不是五个随机种子。

最终测试集已经评估，不应用于下一步模型选型。关联反馈的已完成最终评估见 [最终评估报告](outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/final_test_evaluation/REPORT.md)。根目录旧 JSON 中的历史 `evaluated=false` 不能替代此报告；不要为了单独评估再次运行完整训练入口。

## 已核实结果

| 方法 | 选定 epoch | CV5 验证 F1（均值 ± 折间标准差） | 最终测试 F1 | 最终测试 AUROC |
| --- | ---: | ---: | ---: | ---: |
| 稳定 BBP（历史归档） | 17 | 0.672858 ± 0.041571 | 0.697674 | 0.851966 |
| 关联上下文反馈 | 17 | 0.669186 ± 0.035959 | 0.697674 | 0.852459 |

最终集 194 个会话，五折模型概率平均、阈值 0.5。当前没有同环境重新训练的完整 BBP/raw/context 对照，不能把小幅 AUROC 差异认定为结构学习带来的提升。前期反馈、上下文重评分和大超边拆分实验均保留在交接报告中；大超边方向按用户要求暂置。

## 上传范围

本分支在已有私有仓库历史上追加服务器代码、配置、文档及实验汇总。未新增上传数据集、模型权重、候选图缓存、备份包或逐样本预测。仓库此前已经跟踪的历史权重继续保留，未删除或重写历史。

未提供新的开源许可；仓库保持私有。训练数据、现成模型和第三方论文的使用约束应分别遵循各自来源。
