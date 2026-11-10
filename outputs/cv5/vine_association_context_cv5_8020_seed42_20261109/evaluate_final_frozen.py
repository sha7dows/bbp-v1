"""用途：仅加载CV5共同epoch的五个冻结模型，在固定194会话最终集上做一次概率平均评估，不训练或调阈值。"""

from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score,confusion_matrix

from bbp_behm.cached_hypergraph import CachedHyperedgeBuilder
from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset
from bbp_behm.training import make_loader,set_seed
from scripts.train_vine_cv5 import build_fold_model,classification_metrics,json_ready

ROOT=Path("outputs/cv5/vine_association_context_cv5_8020_seed42_20261109")
OUTPUT=ROOT/"final_test_evaluation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=True)
    assert not (OUTPUT/"result.json").exists(),"Refusing to overwrite a final-test evaluation"
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(),logging.FileHandler(OUTPUT/"evaluation.log",encoding="utf-8")])
    torch.set_num_threads(4)
    started=time.perf_counter()
    config=load_experiment_config(Path("configs/vine_association_cv5_server.ini"))
    cv=json.loads((ROOT/"result.json").read_text())
    manifest=json.loads((ROOT/"split.json").read_text())
    assert json_ready(asdict(config))==cv["resolved_config"],"Configuration changed after CV"
    hashes=json.loads((ROOT/"source_hashes.json").read_text())
    assert all(sha(Path(name))==digest for name,digest in hashes.items()),"Training source changed"
    epoch=int(cv["cv"]["global_epoch"])
    assert epoch==17 and cv["model_options"]==dict(architecture="association",association_mode="context",association_alpha=.2)
    dataset=PreprocessedSessionDataset(config.data.processed_path,config.data.processed_sha256,
        config.data.expected_sessions,config.data.expected_nodes,config.data.expected_features)
    split=dataset.make_split(42,config.split.test_fraction,config.split.validation_fraction)
    indices=list(manifest["final_test_indices"])
    assert indices==list(split.test) and len(indices)==194 and len(set(indices))==194
    assert not set(indices).intersection(manifest["development_indices"])
    device=torch.device("cuda:0")
    builder=CachedHyperedgeBuilder.prepare(dataset,indices,config.hypergraph,device)
    loader=make_loader(dataset,indices,64,False,42)
    all_probabilities=[]
    per_fold=[]
    checkpoint_hashes={}
    for fold in manifest["folds_detail"]:
        fold_id=int(fold["fold"])
        assert not set(indices).intersection(fold["train"]+fold["validation"])
        checkpoint=ROOT/"fold_checkpoints"/f"fold_{fold_id:02d}_epoch_{epoch:03d}.pt"
        checkpoint_hashes[str(checkpoint)]=sha(checkpoint)
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        assert saved["global_epoch"]==epoch and saved["model_options"]==cv["model_options"]
        set_seed(42)
        model=build_fold_model(config,dataset,tuple(fold["train"]),device,**saved["model_options"])
        model.load_state_dict(saved["model"],strict=True)
        model.eval()
        probabilities,labels,observed_indices=[],[],[]
        with torch.no_grad():
            for cpu_batch in loader:
                batch=cpu_batch.to(device)
                graph=builder.build_batch(batch)
                output=model(batch,graph,stochastic_selection=False)
                probabilities.extend(output.logits.softmax(1)[:,1].cpu().tolist())
                labels.extend(batch.labels.cpu().tolist())
                observed_indices.extend(batch.session_indices.cpu().tolist())
        assert observed_indices==indices
        assert labels==dataset.labels[indices].tolist()
        assert len(probabilities)==194 and np.isfinite(probabilities).all()
        all_probabilities.append(probabilities)
        metrics=classification_metrics(labels,probabilities)
        per_fold.append(dict(fold=fold_id,epoch=epoch,metrics=metrics,checkpoint=str(checkpoint)))
        (OUTPUT/f"fold_{fold_id:02d}_predictions.json").write_text(json.dumps([
            dict(index=i,label=y,positive_probability=p) for i,y,p in zip(indices,labels,probabilities)],indent=2),encoding="utf-8")
        logging.info("FINAL_EVAL_FOLD_COMPLETE fold=%d epoch=%d f1=%.6f auroc=%.6f",fold_id,epoch,
            metrics["positive_f1"],metrics["auroc"])
        del model,output
        torch.cuda.empty_cache()
    assert len(all_probabilities)==5
    ensemble=np.mean(np.asarray(all_probabilities,dtype=np.float64),axis=0)
    decisions=ensemble>=.5
    metrics=classification_metrics(labels,ensemble.tolist())
    metrics["accuracy"]=float(accuracy_score(labels,decisions))
    tn,fp,fn,tp=confusion_matrix(labels,decisions,labels=[0,1]).ravel().tolist()
    confusion=dict(tn=tn,fp=fp,fn=fn,tp=tp)
    records=[dict(index=i,label=y,positive_probability=float(p),prediction=int(p>=.5))
             for i,y,p in zip(indices,labels,ensemble)]
    (OUTPUT/"ensemble_predictions.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    assert all(sha(Path(name))==digest for name,digest in checkpoint_hashes.items())
    assert all(sha(Path(name))==digest for name,digest in hashes.items())
    result=dict(purpose="User-authorized frozen-model final holdout evaluation; no fitting or threshold selection",
        created_at=datetime.now(timezone.utc).isoformat(),global_epoch=epoch,model_options=cv["model_options"],
        sessions=194,class_counts=dict(negative=int(sum(y==0 for y in labels)),positive=int(sum(y==1 for y in labels))),
        ensemble="mean_positive_probability",models=5,threshold=.5,metrics=metrics,confusion=confusion,
        per_fold=per_fold,checkpoint_sha256=checkpoint_hashes,training_source_hashes_unchanged=True,
        no_training_performed=True,elapsed_seconds=time.perf_counter()-started,
        evaluation_source_sha256=sha(Path(__file__)),processed_sha256=config.data.processed_sha256)
    (OUTPUT/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines=["<!-- 用途：报告锁定CV5模型在194会话最终测试集上的一次评估。 -->",
        "# 固定最终测试集评估","",
        f"加载共同第{epoch}轮的五个模型，平均正类概率，固定阈值0.5；没有训练或调参。","",
        "|指标|五模型概率平均|","|---|---:|"]
    for name,value in metrics.items():lines.append(f"|{name}|{value:.6f}|")
    lines+=["",f"混淆矩阵：TN={tn}，FP={fp}，FN={fn}，TP={tp}。","",
        "逐模型结果仅作记录，不根据测试表现挑选某个模型。CV阶段原result.json保持原始训练记录；本目录result.json记录本次最终评估。",
        "模型checkpoint及训练源码哈希在评估前后均保持不变。"]
    (OUTPUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    with (ROOT/"REPORT.md").open("a",encoding="utf-8") as file:
        file.write("\n\n## 用户授权后的最终测试评估\n\n已锁定第17轮五模型，以概率平均和固定0.5阈值评估194会话，未重新训练或调参。结果见final_test_evaluation/REPORT.md及result.json；CV阶段原始result.json保持不变。\n")
    print(json.dumps(dict(metrics=metrics,confusion=confusion,per_fold=per_fold,elapsed_seconds=result["elapsed_seconds"]),indent=2))
    print("FROZEN_FINAL_TEST_EVALUATION_COMPLETE")


if __name__=="__main__":
    main()
