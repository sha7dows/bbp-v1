<!-- 用途：记录远端类 HSL 实验约定，确保后续代码修改与运行仅发生在本服务器。 -->
# 服务器实验约定
用户指定：本项目代码编写、修改、检查、训练与数据诊断均在 /root/TRY 完成；本地只作为 SSH 通道。不要将本地草稿视为远端当前版本。
代码注释使用简体中文，新增文件开头说明用途；运行日志使用英文。
原始模型与配置备份：/root/TRY/.backups/pre_hsl_20261109.tgz。
归档 INI 保持不变；服务器配置为 configs/vine_hsl_server.ini。
类 HSL 实验为探索性单折，不是原 HSL 完整复现；不访问最终测试集进行模型选型。
代码源快照、哈希、参数、划分和运行结果保存在对应 outputs/hsl_pilot 子目录。

2026-11-09新增关联反馈标准CV5：方法见docs/hsl_association_cv5.md，模型bbp_behm/hsl_association.py，标准入口scripts/train_vine_cv5.py增加显式architecture开关；默认original不变。已完成输出outputs/cv5/vine_association_context_cv5_8020_seed42_20261109/REPORT.md。新工作先核查此结果，勿将前期固定70%短实验或规模分解结果混作关联反馈CV5。

用户后续已授权并完成上述关联CV5的最终集评估：final_test_evaluation/REPORT.md及result.json。锁定epoch17五模型概率平均、阈值0.5、194会话，无再训练；原CV result.json未改写，勿据其历史evaluated=false误判最终评估尚未完成。

2026-11-10交接入口：请优先阅读/root/TRY/work/AGENT_HANDOFF.md及work/evidence_manifest.json。报告以BeHM论文为原始起点、BBP为当前稳定基线，区分所有探索与完整结果；用户当前聚焦类HSL结构学习，大超边规模方向暂置。交接报告已纠正旧架构文档和历史状态中的易误读信息。
