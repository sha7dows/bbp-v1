"""用途：在服务器验证类 HSL 的预算、桥接保护、节点可达性、关闭路径和评分梯度。"""

from dataclasses import replace
from pathlib import Path
import torch
from bbp_behm.config import load_experiment_config
from bbp_behm.hsl import HSLBBPBeHM
from bbp_behm.hypergraph import HyperedgeBatch
from scripts.train_hsl_pilot import check_invariants

def main():
    config=load_experiment_config(Path("configs/vine_hsl_server.ini"))
    config=replace(config,data=replace(config.data,expected_features=4),model=replace(config.model,hidden_size=8,dropout=0.0,selector_dropout=0.0),pruning=replace(config.pruning,budget_mode="fixed"))
    torch.manual_seed(42)
    model=HSLBBPBeHM(config,mode="feedback_edit",large_edge_threshold=8).cuda()
    incidence=torch.ones((3,12),device="cuda")
    incidence[2,2:]=0
    graph=HyperedgeBatch(incidence,torch.zeros(3,dtype=torch.long,device="cuda"),torch.tensor([0,0,1],device="cuda"),torch.ones((3,4),device="cuda"),[])
    nodes=torch.randn((12,8),device="cuda",requires_grad=True)
    hard=torch.ones(3,device="cuda")
    model._pilot_nodes=nodes
    model.last_hsl_stats={}
    actual=model.refine_incidence(nodes,graph,incidence,hard)
    assert torch.equal(actual[2],incidence[2])
    assert torch.equal(actual[:2,:2],incidence[:2,:2])
    assert torch.all(actual.sum(0)>0)
    assert torch.all(incidence.sum(1)-actual.sum(1)<=torch.floor(incidence.sum(1)*0.1))
    assert model.last_hsl_stats["edits"]==2
    (actual*torch.arange(12,device="cuda")).sum().backward()
    gradient=sum(float(p.grad.abs().sum()) for p in model.member_scorer.parameters() if p.grad is not None)
    assert gradient>0 and torch.isfinite(torch.tensor(gradient))
    model.edit_fraction=0
    assert torch.equal(model.refine_incidence(nodes,graph,incidence,hard),incidence)
    model.edit_fraction=0.1
    model.large_edge_threshold=20
    assert torch.equal(model.refine_incidence(nodes,graph,incidence,hard),incidence)
    model.eval()
    kwargs=dict(projected_nodes=nodes,batch_x=torch.randn(12,4,device="cuda"),graph_ids=torch.zeros(12,dtype=torch.long,device="cuda"),num_sessions=1,hypergraph=graph,retention_ratio=0.7,gumbel_temperature=1.0,stochastic=False,budget_mode="fixed")
    selected=model.select_hyperedges(**kwargs)
    assert int(selected.hard_mask.sum())==3
    torch.testing.assert_close(model.feedback_adapter(torch.cat([nodes,nodes],1)),torch.zeros_like(nodes),atol=0,rtol=0)
    empty=HyperedgeBatch(incidence[:0],graph.edge_to_session[:0],graph.type_ids[:0],graph.structural_stats[:0],[])
    result=model.select_hyperedges(**{**kwargs,"hypergraph":empty})
    assert result.hard_mask.numel()==0
    print("HSL_INVARIANTS_PASS member_gradient_l1=",gradient)

if __name__=="__main__":
    main()
