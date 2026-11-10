"""用途：重载CV5共同epoch模型，重放全部OOF预测，并核验真实分类损失到上下文传播的梯度路径。"""

import hashlib
import json
from pathlib import Path

import torch

from bbp_behm.cached_hypergraph import CachedHyperedgeBuilder
from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset,collate_sessions
from bbp_behm.training import evaluate,make_loader
from scripts.train_vine_cv5 import build_fold_model

ROOT=Path("outputs/cv5/vine_association_context_cv5_8020_seed42_20261109")


def main():
    torch.set_num_threads(4)
    status=json.loads((ROOT/"run_status.json").read_text())
    assert status["phase"]=="complete","Training and aggregation must finish before replay"
    result=json.loads((ROOT/"result.json").read_text())
    split=json.loads((ROOT/"split.json").read_text())
    config=load_experiment_config(Path("configs/vine_association_cv5_server.ini"))
    dataset=PreprocessedSessionDataset(config.data.processed_path,config.data.processed_sha256,
        config.data.expected_sessions,config.data.expected_nodes,config.data.expected_features)
    saved=torch.load(ROOT/"prepared_graphs.pt",map_location="cpu",weights_only=False)
    builder=CachedHyperedgeBuilder(saved["cache"])
    epoch=result["cv"]["global_epoch"]
    device=torch.device("cuda:0")
    details=[]
    for fold in split["folds_detail"]:
        number=fold["fold"]
        checkpoint=ROOT/"fold_checkpoints"/f"fold_{number:02d}_epoch_{epoch:03d}.pt"
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        model=build_fold_model(config,dataset,tuple(fold["train"]),device,**payload["model_options"])
        model.load_state_dict(payload["model"],strict=True)
        model.eval()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        predictions=[]
        metrics=evaluate(model,make_loader(dataset,fold["validation"],64,False,42),builder,device,
            retention_ratio=config.pruning.retention_ratio,pruning_enabled=True,predictions_out=predictions)
        expected=json.loads((ROOT/"validation_predictions"/f"fold_{number:02d}_epoch_{epoch:03d}.json").read_text())
        assert [(p["index"],p["label"]) for p in predictions]==[(p["index"],p["label"]) for p in expected]
        maximum=max(abs(a["positive_probability"]-b["positive_probability"]) for a,b in zip(predictions,expected))
        assert maximum<1e-5,maximum
        ids=[]
        for label in [0,1]:
            ids.extend([i for i in fold["train"] if int(dataset.labels[i])==label and bool((builder.cache[i].type_ids==0).any())][:2])
        batch=collate_sessions([dataset[i] for i in ids]).to(device)
        graph=builder.build_batch(batch)
        captured=[]
        handle=model.backbone.hgnn.first.register_forward_hook(lambda module,args,output:captured.append(output))
        output=model(batch,graph,stochastic_selection=False)
        handle.remove()
        assert len(captured)==2
        context_grad=torch.autograd.grad(output.total_loss,captured[0])[0]
        norm=float(context_grad.abs().sum())
        assert norm>0 and bool(torch.isfinite(context_grad).all())
        original=graph.incidence*output.selection.hard_mask[:,None]
        actual=output.masked_incidence.detach()
        assert torch.equal(actual>0,original>0)
        assert torch.equal(actual[graph.type_ids!=0],original[graph.type_ids!=0])
        torch.testing.assert_close(actual.sum(1),original.sum(1),atol=1e-3,rtol=1e-6)
        details.append(dict(fold=number,replayed_validation_sessions=len(predictions),
            replay_probability_max_difference=maximum,trained_context_gradient_l1=norm,
            support_and_non_stc_unchanged=True,row_mass_max_error=float((actual.sum(1)-original.sum(1)).abs().max()),
            replay_positive_f1=metrics["positive_f1"]))
        del model,output,captured
        torch.cuda.empty_cache()
    hashes=json.loads((ROOT/"source_hashes.json").read_text())
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    report=dict(all_776_oof_predictions_replayed=True,trained_feedback_gradient_verified=True,
        source_hashes_unchanged=True,details=details,final_test_evaluated=False,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/"trained_replay_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (ROOT/"REPORT.md").open("a",encoding="utf-8") as file:
        file.write("\n\n## 检查点重放与已训练反馈梯度\n\n全部776条OOF概率已重载模型重放，最大容差1e-5；五折均从实际分类损失核验到第一次上下文HGNN输出的非零有限梯度。具体数值见trained_replay_audit.json；本审计不使用最终194会话。\n")
    print(json.dumps(report,indent=2))
    print("TRAINED_CV5_REPLAY_AUDIT_PASS")


if __name__=="__main__":
    main()
