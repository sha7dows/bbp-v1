"""用途：按原标准协议运行一次关联反馈CV5，冻结源码并审计五折预测、检查点和最终留出集隔离。"""

from datetime import datetime,timezone
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tarfile
import time

import torch
from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.cached_hypergraph import CachedHyperedgeBuilder

ROOT=Path("outputs/cv5/vine_association_context_cv5_8020_seed42_20261109")


def dump(path,value):
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def main():
    run_started=time.time()
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/"source_hashes.json").exists(),"Existing run must not be overwritten"
    paths=[]
    for folder in ["bbp_behm","scripts","configs","docs"]:
        paths.extend(p for p in Path(folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py",".ini",".md"})
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    dump(ROOT/"source_hashes.json",hashes)
    with tarfile.open(ROOT/"source_snapshot.tgz","w:gz") as archive:
        for name in hashes:archive.add(name)
    packages={name:importlib.metadata.version(name) for name in ["torch","torch-geometric","numpy","scikit-learn","networkx","python-louvain"]}
    dump(ROOT/"environment.json",dict(python=platform.python_version(),packages=packages,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip()))
    shutil.copyfile("outputs/association_checks_20261109/checks.json",ROOT/"preflight_checks.json")
    dump(ROOT/"run_status.json",dict(phase="preparing_graph_cache",final_test_evaluated=False))
    config=load_experiment_config(Path("configs/vine_association_cv5_server.ini"))
    torch.set_num_threads(4)
    dataset=PreprocessedSessionDataset(config.data.processed_path,config.data.processed_sha256,
        config.data.expected_sessions,config.data.expected_nodes,config.data.expected_features)
    split=dataset.make_split(42,config.split.test_fraction,config.split.validation_fraction)
    indices=sorted((*split.train,*split.validation))
    cache_path=ROOT/"prepared_graphs.pt"
    cache=CachedHyperedgeBuilder.prepare(dataset,indices,config.hypergraph,torch.device("cuda:0"))
    torch.save(dict(cache=cache.cache,indices=indices,processed_sha256=config.data.processed_sha256,
        hypergraph_config=asdict(config.hypergraph)),cache_path)
    del cache,dataset
    torch.cuda.empty_cache()
    common=["python","scripts/train_vine_cv5.py","--config","configs/vine_association_cv5_server.ini",
        "--mode","bbp","--architecture","association","--association-mode","context","--association-alpha","0.2",
        "--torch-threads","4","--seed","42"]
    commands=[common+["--cache-graphs","--prepared-cache",str(cache_path),"--worker-fold",str(fold),
        "--run-name",ROOT.name+f"/workers/fold_{fold:02d}"] for fold in range(1,6)]
    aggregate_command=common+["--aggregate-workers","--run-name",ROOT.name,"--total-started-at",str(run_started)]
    dump(ROOT/"command.json",dict(worker_commands=commands,aggregate_command=aggregate_command,
        concurrent_fold_processes=5,cwd=str(Path.cwd()),started_at=datetime.now(timezone.utc).isoformat()))
    dump(ROOT/"run_status.json",dict(phase="training",completed_folds=[],final_test_evaluated=False))
    def execute(fold,command):
        worker=ROOT/"workers"/f"fold_{fold:02d}"
        worker.mkdir(parents=True,exist_ok=True)
        with (worker/"console.log").open("w") as console:
            run=subprocess.run(command,stdout=console,stderr=subprocess.STDOUT,env={**os.environ,"PYTHONPATH":"."})
        if run.returncode:
            raise RuntimeError(f"CV5 worker {fold} failed; preserve logs before retry")
        return fold
    completed=[]
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(execute,fold,command) for fold,command in enumerate(commands,1)]
            for future in as_completed(futures):
                completed.append(future.result())
                dump(ROOT/"run_status.json",dict(phase="training",completed_folds=sorted(completed),final_test_evaluated=False))
        dump(ROOT/"run_status.json",dict(phase="aggregating",completed_folds=sorted(completed),final_test_evaluated=False))
        with (ROOT/"aggregate.log").open("w") as console:
            run=subprocess.run(aggregate_command,stdout=console,stderr=subprocess.STDOUT,env={**os.environ,"PYTHONPATH":"."})
        if run.returncode:
            raise RuntimeError("CV5 aggregation failed; preserve worker results")
    except Exception as error:
        dump(ROOT/"run_status.json",dict(phase="failed",completed_folds=sorted(completed),error=str(error)))
        raise
    result=json.loads((ROOT/"result.json").read_text())
    manifest=json.loads((ROOT/"split.json").read_text())
    history=json.loads((ROOT/"cv_history.json").read_text())
    assert len(history["fold_histories"])==5 and all(len(h)==20 for h in history["fold_histories"])
    assert len(manifest["final_test_indices"])==194 and len(manifest["development_indices"])==776
    assert result["final_test"]["evaluated"] is False
    oof=json.loads((ROOT/"oof_predictions.json").read_text())
    assert len(oof)==776 and len({p["index"] for p in oof})==776
    assert not {p["index"] for p in oof}.intersection(manifest["final_test_indices"])
    assert len(list((ROOT/"validation_predictions").glob("*.json")))==100
    assert len(list((ROOT/"epoch_checkpoints").glob("*.pt")))==100
    assert len(list((ROOT/"fold_checkpoints").glob("*.pt")))==5
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    metrics=["positive_f1","macro_f1","precision","recall","auroc"]
    selected=result["cv"]["per_fold_validation_at_global_epoch"]
    summary={k:dict(mean=statistics.fmean(v[k] for v in selected),sample_std=statistics.stdev(v[k] for v in selected)) for k in metrics}
    final=[h[-1]["validation"] for h in history["fold_histories"]]
    fixed20={k:dict(mean=statistics.fmean(v[k] for v in final),sample_std=statistics.stdev(v[k] for v in final)) for k in metrics}
    instrument={k:[v.get(k) for v in selected] for k in ["association_mean_abs_weight_delta","association_weight_min",
        "association_weight_max","association_row_mass_max_error","tail_coverage_min","edge_retention"]}
    dump(ROOT/"summary.json",dict(selected_global_epoch=result["cv"]["global_epoch"],selected_cv=summary,
        fixed_epoch20_cv=fixed20,oof=result["oof_metrics_at_global_epoch"],instrumentation=instrument,final_test_evaluated=False))
    lines=["<!-- 用途：报告关联加权架构的标准CV5结果、结构观测及验证范围。 -->",
        "# 关联加权反馈标准CV5","",
        f'五折全部完成20轮，batch64，robust_adaptive，训练Gumbel启用；共同epoch={result["cv"]["global_epoch"]}。',
        "194最终会话未评估。按开发五折验证均值选共同epoch，未进行嵌套CV，不能把选优后验证均值当作独立测试表现。","",
        "|指标|共同epoch五折均值|样本标准差|第20轮五折均值|","|---|---:|---:|---:|"]
    for k in metrics:lines.append(f'|{k}|{summary[k]["mean"]:.6f}|{summary[k]["sample_std"]:.6f}|{fixed20[k]["mean"]:.6f}|')
    lines+=["","## 各折共同epoch","", "|折|F1|AUROC|平均关联权重变化|保留比例|","|---|---:|---:|---:|---:|"]
    for fold,v in enumerate(selected,1):
        lines.append(f'|{fold}|{v["positive_f1"]:.6f}|{v["auroc"]:.6f}|{v.get("association_mean_abs_weight_delta",0):.8f}|{v["edge_retention"]:.6f}|')
    lines+=["","## 合并OOF","", "~~~json",json.dumps(result["oof_metrics_at_global_epoch"],indent=2),"~~~","",
        "## 机制观测","", "~~~json",json.dumps(instrument,indent=2),"~~~","",
        "本次仅完成context关联模型一次CV5，尚未完成同环境原BBP和raw匹配容量CV5，不能单独据此归因双向反馈的增益。",
        "既有关联支持、逐边总质量及User/Mention权重保持不变，权重只用于HGNN传播/超边池化；IBG使用二值拓扑。",
        "无大边分解、固定比例删成员、补关联或额外对比损失。架构与重建参数见model_options和configs/vine_association_cv5_server.ini。",
        "五折以独立进程并发共享同一GPU，种子和训练状态隔离；时间仅作运行记录，不是独占GPU速度基准。",
        "所有逐轮验证预测与checkpoint保留；共同epoch的五折模型和ensemble_model.pt均已保存。"]
    (ROOT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    dump(ROOT/"artifact_checks.json",dict(folds=5,epochs_per_fold=20,prediction_files=100,
        epoch_checkpoints=100,selected_fold_checkpoints=5,oof_sessions=776,source_hashes_unchanged=True,
        final_test_evaluated=False))
    dump(ROOT/"run_status.json",dict(phase="complete",completed_at=datetime.now(timezone.utc).isoformat(),final_test_evaluated=False))
    print("ASSOCIATION_STANDARD_CV5_COMPLETE")


if __name__=="__main__":
    main()
