"""用途：在服务器按预定折次和种子完成大STC三组消融，保存源码快照并生成配对结果报告。"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path("outputs/stc_split_20261109")
RUNS = [dict(name="screen_fold1_seed42", fold=1, seed=42, epochs=8),
        dict(name="confirm_fold2_seed42", fold=2, seed=42, epochs=12),
        dict(name="confirm_fold2_seed43", fold=2, seed=43, epochs=12)]


def dump(path, value):
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def source_hashes():
    files = []
    for folder in ["bbp_behm","scripts","configs","docs"]:
        files.extend(p for p in Path(folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix in [".py",".ini",".md"])
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def summary():
    modes = ["none","random","semantic"]
    results = {}
    table = ["|开发折 / 种子|方法|末轮F1|AUROC|最后3轮F1均值|传播边/父边|额外关联|训练秒/轮|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for run in RUNS:
        data = json.loads((ROOT/run["name"]/"result.json").read_text())
        results[run["name"]] = data
        for mode in modes:
            history = data["modes"][mode]["history"]
            v = history[-1]["validation"]
            table.append(f'|{run["fold"]} / {run["seed"]}|{mode}|{v["positive_f1"]:.6f}|{v["auroc"]:.6f}|'
                f'{np.mean([h["validation"]["positive_f1"] for h in history[-3:]]):.6f}|'
                f'{v["propagation_edges"]:.0f}/{v["kept_parents"]:.0f}|{v["extra_incidence_fraction"]:.3%}|'
                f'{np.mean([h["train_seconds"] for h in history]):.2f}|')
    confirm = RUNS[1:]
    comparisons = {}
    for control in ["none","random"]:
        pairs = []
        for run in confirm:
            predictions = []
            for mode in ["semantic",control]:
                p = json.loads((ROOT/run["name"]/f'{mode}_epoch_12_predictions.json').read_text())
                predictions.append(sorted(p,key=lambda x:x["index"]))
            a,b = predictions
            assert [(p["index"],p["label"]) for p in a] == [(p["index"],p["label"]) for p in b]
            pairs.append((np.array([p["label"] for p in a]),np.array([p["positive_probability"]>=.5 for p in a]),
                          np.array([p["positive_probability"]>=.5 for p in b])))
        assert np.array_equal(pairs[0][0],pairs[1][0])
        y = pairs[0][0]
        observed = np.mean([f1_score(y,a)-f1_score(y,b) for y,a,b in pairs])
        rng = np.random.default_rng(20261109)
        strata = [np.flatnonzero(y==c) for c in [0,1]]
        bootstrap = []
        for _ in range(4000):
            sample = np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in strata])
            bootstrap.append(np.mean([f1_score(y[sample],a[sample],zero_division=0)-f1_score(y[sample],b[sample],zero_division=0)
                                      for y,a,b in pairs]))
        comparisons[control] = dict(semantic_minus_control_mean_f1=float(observed),
            conditional_stratified_bootstrap_95_ci=np.quantile(bootstrap,[.025,.975]).tolist(),
            changed_decisions=[int(np.sum(a!=b)) for y,a,b in pairs])
    means = {mode:{metric:float(np.mean([results[r["name"]]["modes"][mode]["history"][-1]["validation"][metric]
                                       for r in confirm])) for metric in ["positive_f1","auroc","extra_incidence_fraction",
                                           "actual_structural_coverage","parent_size_mean","propagation_size_mean"]}
             for mode in modes}
    dump(ROOT/"summary.json",dict(confirm_means=means,comparisons=comparisons,
        interval_note="Conditional on fitted models; paired validation resampling shares indices across seeds; not full training uncertainty",
        final_test_evaluated=False))
    diagnostics = json.loads((ROOT/RUNS[0]["name"]/"diagnostic.json").read_text())
    report = "\n".join(["<!-- 用途：保存服务器STC分解消融的全部固定轮次结果和解释边界。 -->",
        "# 大STC受约束分解结果","", "已完成第一折8轮与第二折两个种子各12轮，最终194会话未评估。",
        "阈值仅由各折训练数据拟合，源码哈希在运行前后逐文件核对。", "", *table,"",
        "## 第二折两个种子平均","", "~~~json",json.dumps(means,indent=2),"~~~","",
        "## 固定末轮配对比较","", "~~~json",json.dumps(comparisons,indent=2),"~~~","",
        "区间只描述给定模型的验证样本不确定性，两个种子不足以估计完整训练方差。第二折已参与先前开发，不是独立最终测试。",
        "三个独立折/种子进程共享同一GPU并发运行，耗时受资源竞争影响，仅为运行记录，不能作为严格性能基准。",
        "随机组与语义组在同一父边下严格匹配子边规模与关联预算；各自训练后父边选择可不同，实际成本见表。",
        "候选门控对两种分区均使用相同的语义可分解条件，所以本对照检验分区质量，而非去除全部语义先验。",
        "", "## 第一折训练诊断", "", "~~~json",json.dumps(diagnostics["summaries"]["train"],indent=2),"~~~","",
        "分散度降低是构造机制的结果，不能替代分类效果证据。默认未启用分解，需结合语义组相对随机组与原模型的稳定提升决定是否升级。",
        "", "文件：每组每轮预测、末轮checkpoint、split.json、diagnostic.json、provenance.json、result.json、training.log。"])
    (ROOT/"REPORT.md").write_text(report,encoding="utf-8")


def main():
    ROOT.mkdir(exist_ok=True,parents=True)
    assert not (ROOT/"source_hashes.json").exists(), "Existing study must not be overwritten"
    hashes = source_hashes()
    dump(ROOT/"source_hashes.json",hashes)
    with tarfile.open(ROOT/"source_snapshot.tgz","w:gz") as archive:
        for name in hashes:
            archive.add(name)
    dump(ROOT/"plan.json",dict(runs=RUNS,modes=["none","random","semantic"],parent_retention=.7,
        final_test_evaluated=False,selection="Fixed final epoch; last-three mean supplementary; no validation promotion",
        split_seed=42,minimum_semantic_gain=.01))
    def execute(run):
        out = ROOT/run["name"]
        out.mkdir(exist_ok=True)
        assert not (out/"result.json").exists(), "Existing training results must not be overwritten"
        with (out/"console.log").open("w") as console:
            process = subprocess.run(["python","scripts/train_stc_split.py","--output",str(out),"--fold",str(run["fold"]),
                "--seed",str(run["seed"]),"--epochs",str(run["epochs"])],
                stdout=console,stderr=subprocess.STDOUT,env={**os.environ,"PYTHONPATH":"."})
        if process.returncode:
            raise RuntimeError(f"Study failed: {run}; returncode={process.returncode}")
        assert hashes == source_hashes(), "Source changed during experiment"
        return run["name"]

    # 独立进程并发不同折/种子；随机状态隔离，耗时仅作为共享GPU下的运行记录。
    dump(ROOT/"status.json",dict(phase="training",runs=RUNS,concurrent_processes=3,completed=[],
        started_at=datetime.now(timezone.utc).isoformat()))
    completed = []
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(execute,run) for run in RUNS]
            for future in as_completed(futures):
                completed.append(future.result())
                dump(ROOT/"status.json",dict(phase="training",runs=RUNS,concurrent_processes=3,completed=completed))
    except Exception as error:
        dump(ROOT/"status.json",dict(phase="failed",completed=completed,error=str(error)))
        raise
    dump(ROOT/"status.json",dict(phase="summarizing",final_test_evaluated=False))
    summary()
    assert hashes == source_hashes()
    dump(ROOT/"artifact_checks.json",dict(source_hashes_unchanged=True,runs_complete=len(RUNS),final_test_evaluated=False))
    dump(ROOT/"status.json",dict(phase="complete",final_test_evaluated=False,completed_at=datetime.now(timezone.utc).isoformat()))
    print("STC_SPLIT_STUDY_COMPLETE")


if __name__ == "__main__":
    main()
