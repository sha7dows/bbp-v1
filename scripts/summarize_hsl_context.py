"""用途：汇总上下文筛查与第二折复核，核验配对条件并对固定轮次F1差值做会话级分层bootstrap。"""

import json
from pathlib import Path
from statistics import mean
import numpy as np

ROOT=Path("/root/TRY/outputs/hsl_context_20261109")

def tail_metrics(value):
    history=value["history"]
    final=history[-1]
    tail=history[-3:]
    return {"final_f1":final["validation"]["positive_f1"],
            "tail3_f1":mean(h["validation"]["positive_f1"] for h in tail),
            "final_auroc":final["validation"]["auroc"],
            "tail3_auroc":mean(h["validation"]["auroc"] for h in tail),
            "recall":final["validation"]["recall"],
            "seconds_per_epoch":mean(h["train_seconds"] for h in history),
            "support_changed_fraction":final["validation"]["changed"]/max(1,2*final["validation"]["kept"]),
            "edge_retention":final["validation"]["edge_retention"]}

def f1(y,pred):
    tp=(pred & y).sum(-1)
    fp=(pred & ~y).sum(-1)
    fn=(~pred & y).sum(-1)
    return 2*tp/np.maximum(2*tp+fp+fn,1)

def main():
    study=json.loads((ROOT/"study.json").read_text())
    candidate,control=study["promotion"],study["control"]
    screen={m:tail_metrics(v) for m,v in study["screen"]["modes"].items()}
    confirmations={}
    paired={}
    expected=None
    labels=None
    for seed in ["42","43"]:
        run=study["confirmations"][seed]
        confirmations[seed]={m:tail_metrics(v) for m,v in run["modes"].items()}
        folder=ROOT/f"confirm_fold2_seed{seed}"
        predictions={}
        for mode in ["bbp",control,candidate]:
            records=json.loads((folder/f"{mode}_epoch_12_predictions.json").read_text())
            identity=[(r["index"],r["label"]) for r in records]
            if expected is None:
                expected=identity
                labels=np.array([r["label"] for r in records],dtype=bool)
            assert identity==expected,"Unpaired validation cases"
            predictions[mode]=np.array([r["positive_probability"]>=0.5 for r in records])
            actual=float(f1(labels,predictions[mode]))
            prediction=predictions[mode]
            confirmations[seed][mode]["confusion"]={"tp":int((prediction&labels).sum()),"fp":int((prediction&~labels).sum()),"fn":int((~prediction&labels).sum()),"tn":int((~prediction&~labels).sum())}
            assert abs(actual-confirmations[seed][mode]["final_f1"])<1e-10
        for mode in [control,candidate]:
            confirmations[seed][mode]["decision_disagreements_vs_bbp"]=int((predictions[mode]!=predictions["bbp"]).sum())
        paired[seed]=predictions
    rng=np.random.default_rng(20261109)
    positive=np.flatnonzero(labels); negative=np.flatnonzero(~labels)
    draws=np.concatenate([rng.choice(positive,(4000,len(positive))),rng.choice(negative,(4000,len(negative)))],axis=1)
    intervals={}
    for reference in ["bbp",control]:
        samples=[]
        for seed in ["42","43"]:
            a=f1(labels[draws],paired[seed][candidate][draws])
            b=f1(labels[draws],paired[seed][reference][draws])
            samples.append(a-b)
        delta=np.mean(samples,axis=0)
        point=mean(confirmations[s][candidate]["final_f1"]-confirmations[s][reference]["final_f1"] for s in ["42","43"])
        intervals[reference]={"mean_delta_f1":point,"case_bootstrap_95_percentile_ci":np.quantile(delta,[0.025,0.975]).tolist(),"resamples":4000}
    result={"candidate":candidate,"control":control,"screen":screen,"confirmation":confirmations,"paired_case_bootstrap":intervals,"interpretation":"Exploratory development-set analysis; intervals condition on these trained models and do not measure full training-seed uncertainty"}
    (ROOT/"summary.json").write_text(json.dumps(result,indent=2))
    report=["<!-- 用途：记录服务器上下文改进的完整筛查、复核结果及不确定性，不作为最终测试结论。 -->",
            "# 类 HSL 上下文改进实验",
            "",
            "五组第一折8轮筛查；按预记录末三轮平均正类F1（AUROC打破平局）在两种新机制中选候选，第二折两个训练种子分别训练12轮。最终194会话始终排除。各方法使用相同逐会话ceil(0.7E)预算、初始化及数据顺序，无成员编辑、无额外Gumbel。",
            "",
            "## 第一折筛查",
            "",
            "| 方法 | 第8轮F1 | 末3轮平均F1 | 第8轮AUROC | 秒/轮 |",
            "|---|---:|---:|---:|---:|"]
    for mode,value in screen.items():
        report.append(f"| {mode} | {value['final_f1']:.4f} | {value['tail3_f1']:.4f} | {value['final_auroc']:.4f} | {value['seconds_per_epoch']:.2f} |")
    report+=["",f"按既定规则复核候选：{candidate}；匹配容量控制：{control}。",
             "","## 第二折复核","","| 种子 | 方法 | 第12轮F1 | 末3轮平均F1 | 第12轮AUROC | Recall | 秒/轮 |","|---|---|---:|---:|---:|---:|---:|"]
    for seed,modes in confirmations.items():
        for mode,value in modes.items():
            report.append(f"| {seed} | {mode} | {value['final_f1']:.4f} | {value['tail3_f1']:.4f} | {value['final_auroc']:.4f} | {value['recall']:.4f} | {value['seconds_per_epoch']:.2f} |")
    report+=["","## 配对差值","","固定第12轮，候选减去参照的两个训练种子平均F1差值。对相同验证会话按标签分层重采样4000次，两个种子共享同一抽样索引。"]
    for reference,value in intervals.items():
        lo,hi=value["case_bootstrap_95_percentile_ci"]
        report.append(f"- 相对 {reference}：平均差值 {value['mean_delta_f1']:+.4f}，会话级95%区间 [{lo:+.4f}, {hi:+.4f}]。")
    report+=["","## 机制与边界",
             "",
             "- node_control/node_detach 使用同样的节点适配器；后者使用冻结的试探传播输出。",
             "- edge_control/edge_context 使用同样的5h上下文评分器。控制组使用所选行为的类型全局均值；实验组使用同类型槽位中的候选特定Jaccard邻域聚合。邻域上下文冻结，原边表示和评分器仍接受任务梯度。",
             "- 行为残差为0.25*tanh(MLP)，零初始化；对全部原候选重选，覆盖几何保持原始特征口径。",
             "- 对照控制额外评分容量。控制组和机制组的无梯度邻域均被计算，但原基线不执行额外上下文路径；实际开销分别报告。",
             "- 固定最终划分种子42，训练种子42/43独立变化；Louvain随机种子固定42。第二折621/155会话，两个种子使用相同数据划分。",
             "- 第一折与第二折是同一开发集的轮换，第二折验证会话曾出现在第一折训练中；属于开发阶段复核，不是严格独立测试或嵌套CV。",
             "- bootstrap区间仅描述给定模型下的验证会话抽样不确定性；两个训练种子不足以估计完整训练方差，也没有消除开发期反复探索的选择偏差。",
             "- 本次固定K探索不等于原robust_adaptive五折20轮协议；不报告最终测试性能，不据单一最佳轮次宣称收益。",
             "",
             "源码快照、参数、每轮预测、last/best checkpoint、划分清单及原始日志均保留于各阶段目录。"]
    reliable=all(v["mean_delta_f1"]>0 and v["case_bootstrap_95_percentile_ci"][0]>0 for v in intervals.values())
    report+=["","## 本轮判断",""]
    if reliable:
        report.append("给定本轮模型和会话抽样，候选对基线与容量控制出现正向信号；仍需更多训练种子及原鲁棒预算协议验证，不能等同于最终泛化结论。")
    else:
        report.append("本轮尚无同时优于基线与匹配容量控制的可靠证据。应保留负结果，不将额外容量、种子波动或开发集选优写成已证实的结构优化增益。")
    (ROOT/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps(result))
if __name__=="__main__":
    main()
