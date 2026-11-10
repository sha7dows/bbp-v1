"""用途：汇总服务器类 HSL 单折试验，验证代码快照和同预算条件，输出带局限的报告。"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    args=parser.parse_args()
    folder=args.folder
    result=json.loads((folder/"result.json").read_text())
    modes=result["modes"]
    epochs=result["arguments"]["epochs"]
    assert set(modes)=={"bbp","control","feedback","feedback_edit"}
    assert all(len(value["history"])==epochs for value in modes.values())
    signatures=json.loads((folder/"source_manifest.json").read_text())
    source_matches={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()==value for name,value in signatures.items()}
    assert all(source_matches.values()), "Source changed since launch"
    for offset in range(epochs):
        counts=[value["history"][offset]["validation"]["kept"] for value in modes.values()]
        assert len(set(counts))==1, "Different edge budgets"
    rows=[]
    for mode,value in modes.items():
        best=next(item for item in value["history"] if item["epoch"]==value["best_epoch"])
        final=value["history"][-1]
        v=final["validation"]
        rows.append({"mode":mode,"epoch6_positive_f1":v["positive_f1"],"epoch6_recall":v["recall"],"epoch6_auroc":v["auroc"],"best_epoch":best["epoch"],"best_validation_positive_f1":best["validation"]["positive_f1"],"best_validation_auroc":best["validation"]["auroc"],"edge_retention":v["edge_retention"],"member_deleted_fraction":1-v["member_retention_within_selected"],"member_edits":v["edits"],"actual_structural_coverage":v["actual_structural_coverage"],"anchor_coverage":v["anchor_coverage"],"average_cached_train_seconds":mean(h["train_seconds"] for h in value["history"]),"peak_memory_mb":value["peak_memory_mb"]})
    summary={"protocol":result["protocol"],"source_hashes_match":source_matches,"matched_kept_edges":True,"rows":rows}
    (folder/"summary.json").write_text(json.dumps(summary,indent=2))
    report=["<!-- 用途：记录类 HSL 服务器首轮实际结果、源版本和解释边界。 -->","# 类 HSL 单折试验报告","","训练/验证：620/156；排除最终测试 194 会话。seed=42，6 epoch，batch=8，逐会话 ceil(0.7 E) 条边；额外 Gumbel 关闭。所有源文件哈希与启动快照一致，所有组实际保留边数相同。","","## 固定第 6 轮比较","","| 版本 | 正类 F1 | Recall | AUROC | 选中关联删除率 | 平均训练秒/轮 |","|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append("| {mode} | {epoch6_positive_f1:.4f} | {epoch6_recall:.4f} | {epoch6_auroc:.4f} | {member_deleted_fraction:.2%} | {average_cached_train_seconds:.2f} |".format(**r))
    report+=["","## 各组验证 F1 选出的轮次","","| 版本 | 最佳轮次 | 验证正类 F1 | 同轮 AUROC |","|---|---:|---:|---:|"]
    for r in rows:
        report.append("| {mode} | {best_epoch} | {best_validation_positive_f1:.4f} | {best_validation_auroc:.4f} |".format(**r))
    report+=["","## 解释限制","","- 这是单折单种子、6 轮筛查，不是原归档 20 轮五折结果，也不是完整 HSL 复现。","- 本次固定 K 机制试验未强制类型尾部覆盖；不能直接替代原 robust_adaptive 主协议。","- 最佳轮次由同一验证集选出，不能把其验证指标当成独立测试结果。","- control 控制额外容量与试探前向；其试探结果不参与反向，因此不是严格同 FLOPs 比较。","- 成员编辑量上限是规则设定，学习的是删除哪些关联。若成员精简有收益，还需随机/冻结评分的同规模删除控制。","- 时间来自缓存候选图的执行，含选择、传播、社区和检查；不代表完整原始数据管线的端到端加速。","- 编辑只保护所选 STC—User/Mention 的共享成员与节点最后关联，并不保证全部 IBG 权重和社区不变。","","完整记录：result.json、split.json、source_manifest.json、source.tgz、console.log、各组 checkpoint 和验证预测。"]
    (folder/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":
    main()
