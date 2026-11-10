"""用途：在服务器核验关联加权的原模型等价、质量守恒、二值IBG、上下文梯度及标准batch64可运行性。"""

from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from bbp_behm.cached_hypergraph import CachedHyperedgeBuilder
from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset,collate_sessions
from bbp_behm.hsl_association import AssociationBBPBeHM
from bbp_behm.hypergraph import HyperedgeBatch,HyperedgeBuilder
from bbp_behm.model import BBPBeHM
from bbp_behm.training import class_priors,confidence_statistics,evaluate,make_loader,set_seed


def main():
    torch.set_num_threads(4)
    device=torch.device("cuda:0")
    config=load_experiment_config(Path("configs/vine_hsl_server.ini"))
    dataset=PreprocessedSessionDataset(config.data.processed_path,config.data.processed_sha256,
        config.data.expected_sessions,config.data.expected_nodes,config.data.expected_features)
    prepared=torch.load("outputs/stc_split_20261109/screen_fold1_seed42/prepared.pt",weights_only=False)
    train=prepared["train"]
    builder=CachedHyperedgeBuilder(prepared["cache"])
    indices=train[:4]
    sample=collate_sessions([dataset[i] for i in indices]).to(device)
    graph=builder.build_batch(sample)
    direct=HyperedgeBuilder(config.hypergraph).build_batch(sample)
    for field in ["incidence","edge_to_session","type_ids","structural_stats"]:
        torch.testing.assert_close(getattr(graph,field),getattr(direct,field),atol=0,rtol=0)
    assert graph.records==direct.records
    priors=class_priors(dataset,train)
    means,overall=confidence_statistics(dataset,train)
    set_seed(42)
    baseline=BBPBeHM(config,priors,means,overall).to(device).eval()
    model=AssociationBBPBeHM(config,priors,means,overall).to(device).eval()
    model.load_state_dict(baseline.state_dict(),strict=False)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        a=baseline(sample,graph,stochastic_selection=False)
        b=model(sample,graph,stochastic_selection=False)
    torch.testing.assert_close(a.selection.hard_mask,b.selection.hard_mask,atol=0,rtol=0)
    torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-6)
    equality=float((a.logits-b.logits).abs().max())
    # 非均匀权重下必须保持关联支持和逐边总权重，且IBG仍只看到0/1。
    with torch.no_grad():
        model.association_scorer[-1].weight.normal_(std=.1)
    original_build=model.backbone.build_ibg
    calls=[]
    def audited_build(h,*args,**kwargs):
        assert bool(((h==0)|(h==1)).all()),"Weighted association leaked into binary IBG"
        calls.append(len(h))
        return original_build(h,*args,**kwargs)
    model.backbone.build_ibg=audited_build
    out=model(sample,graph,hard_mask_override=torch.ones(graph.num_edges,device=device))
    h=out.masked_incidence
    assert torch.equal(h>0,graph.incidence>0)
    torch.testing.assert_close(h.sum(1),graph.incidence.sum(1),atol=1e-3,rtol=1e-6)
    assert torch.equal(h[graph.type_ids!=0],graph.incidence[graph.type_ids!=0])
    assert float((h-graph.incidence).detach().abs().sum())>0 and len(calls)==sample.num_graphs
    # 单独从关联矩阵取梯度，排除最终HGNN主分支自身带来的梯度。
    objective=(h*torch.randn_like(h)).sum()
    feedback_gradient=torch.autograd.grad(objective,model.backbone.hgnn.first.linear.weight,retain_graph=True)[0]
    assert float(feedback_gradient.abs().sum())>0 and bool(torch.isfinite(feedback_gradient).all())
    empty=HyperedgeBatch(graph.incidence[:0],graph.edge_to_session[:0],graph.type_ids[:0],graph.structural_stats[:0],[])
    with torch.no_grad():
        assert bool(torch.isfinite(model(sample,empty,stochastic_selection=False).logits).all())
        non_stc=replace(graph,incidence=graph.incidence[graph.type_ids!=0],
            edge_to_session=graph.edge_to_session[graph.type_ids!=0],type_ids=graph.type_ids[graph.type_ids!=0],
            structural_stats=graph.structural_stats[graph.type_ids!=0],records=[r for r in graph.records if int(r.behavior_type)!=0])
        no_stc=model(sample,non_stc,pruning_enabled=False)
        torch.testing.assert_close(no_stc.masked_incidence,non_stc.incidence,atol=0,rtol=0)
    # 恢复初始条件，按正式batch64和训练Gumbel完成两个真实优化步骤。
    model.backbone.build_ibg=original_build
    model.load_state_dict(initial)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.training.learning_rate,weight_decay=config.training.weight_decay)
    loader=make_loader(dataset,train,64,True,42)
    model.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    updates=[]
    for step,cpu_batch in enumerate(loader):
        batch=cpu_batch.to(device);g=builder.build_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        result=model(batch,g,stochastic_selection=True)
        assert bool(torch.isfinite(result.total_loss))
        result.total_loss.backward()
        assert bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
        scorer_gradient=sum(float(p.grad.abs().sum()) for p in model.association_scorer.parameters() if p.grad is not None)
        assert scorer_gradient>0
        optimizer.step()
        updates.append(dict(step=step+1,loss=float(result.total_loss.detach()),scorer_gradient=scorer_gradient))
        if step==1:break
    prediction_sink=[]
    metrics=evaluate(model,make_loader(dataset,indices,4,False,42),builder,device,
        retention_ratio=config.pruning.retention_ratio,pruning_enabled=True,predictions_out=prediction_sink)
    assert len(prediction_sink)==4 and metrics["association_mean_abs_weight_delta"]>0
    report=dict(default_logit_max_difference=equality,cache_equals_direct=True,
        support_unchanged=True,row_mass_preserved=True,non_stc_unchanged=True,binary_ibg=True,
        empty_graph=True,no_stc=True,context_gradient_l1=float(feedback_gradient.abs().sum()),
        batch64_updates=updates,batch64_check_seconds=time.perf_counter()-start,
        peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2,after_update_metrics=metrics)
    output=Path("outputs/association_checks_20261109")
    output.mkdir(exist_ok=True)
    (output/"checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("ASSOCIATION_CHECKS_PASS")


if __name__=="__main__":
    main()
