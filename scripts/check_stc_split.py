"""用途：在服务器检查STC分解的成员守恒、共享锚点、随机匹配、默认等价及分类梯度。"""

from dataclasses import replace
import json
from pathlib import Path

import torch

from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset, SessionSample, collate_sessions
from bbp_behm.hypergraph import BehaviorType, HyperedgeBatch, HyperedgeRecord
from bbp_behm.hsl_split import SplitBBPBeHM, SplitCachedBuilder, dispersion, eligible, propose_split
from bbp_behm.model import BBPBeHM
from bbp_behm.training import class_priors, confidence_statistics, set_seed
from scripts.train_stc_split import check_output


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    # 合成两簇与完全一致特征，验证混杂判据能区分可分解结构。
    features = torch.cat([torch.tensor([[1.,0.,0.,0.]]).repeat(10,1),
                          torch.tensor([[0.,1.,0.,0.]]).repeat(10,1)])
    sample = SessionSample(0,"synthetic",features,torch.arange(20),[],0,1.)
    record = HyperedgeRecord(0,0,"synthetic",BehaviorType.STC,tuple(range(20)),("center:0",),1.,0.,2)
    p = propose_split(record,sample,{0:2,10:2})
    q = propose_split(record,sample,{0:2,10:2})
    assert p == q and p["gain"] > .2
    flat = replace(sample,x=torch.ones_like(features))
    f = propose_split(record,flat,{0:2,10:2})
    assert dispersion(flat.x) < 1e-12 and not eligible(f,dict(size=16,dispersion=0.,minimum_gain=.01))
    for groups in [p["semantic"],p["random"]]:
        assert set(groups[0]) | set(groups[1]) == set(range(20))
        assert set(groups[0]) & set(groups[1]) == {0,10}
    config = load_experiment_config(Path("configs/vine_hsl_server.ini"))
    config = replace(config,pruning=replace(config.pruning,budget_mode="fixed"))
    dataset = PreprocessedSessionDataset(config.data.processed_path,config.data.processed_sha256,
        config.data.expected_sessions,config.data.expected_nodes,config.data.expected_features)
    bundle = torch.load("outputs/stc_split_20261109/screen_fold1_seed42/prepared.pt",weights_only=False)
    indices = [i for i in bundle["train"] if bundle["plans"][i]][:3]
    indices += [i for i in bundle["train"] if not bundle["plans"][i]][:1]
    cache = {i:bundle["cache"][i] for i in indices}
    plans = {i:bundle["plans"][i] for i in indices}
    builder = SplitCachedBuilder(cache,plans,dataset)
    priors = class_priors(dataset,bundle["train"])
    means, overall = confidence_statistics(dataset,bundle["train"])
    set_seed(42)
    baseline = BBPBeHM(config,priors,means,overall).to(device).eval()
    state = baseline.state_dict()
    batch = collate_sessions([dataset[i] for i in indices]).to(device)
    original = builder.build_batch(batch)
    with torch.no_grad():
        reference = baseline(batch,original,retention_ratio=.7,stochastic_selection=False)
    details, shapes = {}, {}
    for mode in ["none","random","semantic"]:
        builder.mode = mode
        graph = builder.build_batch(batch)
        model = SplitBBPBeHM(config,priors,means,overall,mode=mode).to(device).eval()
        model.load_state_dict(state)
        out = model(batch,graph,retention_ratio=.7,stochastic_selection=False)
        check_output(model,graph,out)
        torch.testing.assert_close(out.selection.hard_mask,reference.selection.hard_mask,atol=0,rtol=0)
        if mode == "none":
            torch.testing.assert_close(out.logits,reference.logits,atol=1e-6,rtol=1e-6)
        out.total_loss.backward()
        grads = [p.grad for p in model.selector.parameters() if p.grad is not None]
        grad = sum(float(g.abs().sum()) for g in grads)
        assert grad > 0 and all(bool(torch.isfinite(g).all()) for g in grads)
        shapes[mode] = out.masked_incidence.detach().sum(1).tolist()
        if mode != "none":
            # 逐子边核对真实记录，防止把批内编号误当全局节点号。
            for r,row in zip(model.last_propagation_graph.records, model.last_propagation_graph.incidence):
                actual = set(batch.global_node_indices[torch.nonzero(row).flatten()].tolist())
                assert actual == set(r.members)
        # 用同一个父边掩码比较三组，排除父边选择变化对匹配检查的干扰。
        with torch.no_grad():
            override = model(batch,graph,hard_mask_override=reference.selection.hard_mask)
            torch.testing.assert_close(override.logits,out.logits,atol=1e-6,rtol=1e-6)
        details[mode] = dict(selector_gradient_l1=grad,parents=int(out.selection.hard_mask.sum()),
            propagation_edges=int((out.masked_incidence.sum(1)>0).sum()),
            incidence=float(out.masked_incidence.detach().sum()),logit_max_difference=float((out.logits-reference.logits).detach().abs().max()))
    assert shapes["random"] == shapes["semantic"], "Per-child size budgets must match under common parent mask"
    # 空图与无可分解边必须仍走合法的主干路径。
    empty = HyperedgeBatch(torch.zeros((0,batch.x.size(0)),device=device),
        torch.empty(0,dtype=torch.long,device=device),torch.empty(0,dtype=torch.long,device=device),
        torch.zeros((0,4),device=device),[])
    empty.propagation, empty.parent_rows, empty.split_mode = empty,torch.empty(0,dtype=torch.long,device=device),"semantic"
    with torch.no_grad():
        out = model(batch,empty,stochastic_selection=False)
        assert torch.isfinite(out.logits).all()
    output = dict(synthetic_checks=True,homogeneous_not_split=True,empty_graph=True,
        exact_common_parent_budget=True,per_child_size_match=True,details=details)
    path = Path("outputs/stc_split_20261109/invariant_checks.json")
    path.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))
    print("STC_SPLIT_INVARIANTS_PASS")


if __name__ == "__main__":
    main()
