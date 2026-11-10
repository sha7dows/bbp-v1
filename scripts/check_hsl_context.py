"""用途：验证节点冻结反馈、分类型行为反馈、梯度隔离、零初始化及空候选行为。"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import torch
from bbp_behm.config import load_experiment_config
from bbp_behm.hsl_context import ContextHSLBBPBeHM
from bbp_behm.hypergraph import HyperedgeBatch

def main():
    config=load_experiment_config(Path("configs/vine_hsl_server.ini"))
    config=replace(config,data=replace(config.data,expected_features=4),model=replace(config.model,hidden_size=8,dropout=0.0,selector_dropout=0.0),pruning=replace(config.pruning,budget_mode="fixed"))
    incidence=torch.zeros((5,12),device="cuda")
    incidence[0,:6]=1; incidence[1,3:9]=1; incidence[2,6:]=1
    incidence[3,[1,4,7,10]]=1; incidence[4,:]=1
    graph=HyperedgeBatch(incidence,torch.zeros(5,dtype=torch.long,device="cuda"),torch.tensor([0,0,1,2,0],device="cuda"),torch.randn(5,4,device="cuda"),[])
    for mode in ["bbp","node_control","node_detach","edge_control","edge_context"]:
        torch.manual_seed(42)
        model=ContextHSLBBPBeHM(config,mode=mode).cuda().eval()
        nodes=torch.randn(12,8,device="cuda",requires_grad=True)
        kwargs=dict(projected_nodes=nodes,batch_x=torch.randn(12,4,device="cuda"),graph_ids=torch.zeros(12,dtype=torch.long,device="cuda"),num_sessions=1,hypergraph=graph,retention_ratio=0.7,gumbel_temperature=1.0,stochastic=False,budget_mode="fixed")
        base=model.selector(**kwargs)
        selected=model.select_hyperedges(**kwargs)
        torch.testing.assert_close(base.scores,selected.scores,rtol=0,atol=0)
        assert int(selected.hard_mask.sum())==4
        assert model.selector.context_features is None
        loss=(selected.straight_through_mask*torch.arange(5,device="cuda")).sum()
        loss.backward()
        if mode=="node_detach":
            assert all(p.grad is None for p in model.backbone.hgnn.first.parameters())
            assert model.feedback_adapter[-1].weight.grad.abs().sum()>0
        if mode=="edge_context":
            assert model.selector.context_scorer[-1].weight.grad.abs().sum()>0
            all_kept=SimpleNamespace(hard_mask=torch.ones(5,device="cuda"))
            features=model._edge_context(nodes,graph,all_kept,1)
            gradient=torch.autograd.grad(features[:,8:].sum(),nodes)[0]
            assert torch.equal(gradient,torch.zeros_like(gradient))
            # 第一条边与用户边不重叠，该类型关系上下文必须为零。
            torch.testing.assert_close(features[0,16:24],torch.zeros(8,device="cuda"),rtol=0,atol=0)
        empty=HyperedgeBatch(incidence[:0],graph.edge_to_session[:0],graph.type_ids[:0],graph.structural_stats[:0],[])
        assert model.select_hyperedges(**{**kwargs,"hypergraph":empty}).hard_mask.numel()==0
        print("CONTEXT_CHECK_PASS",mode)
if __name__=="__main__":
    main()
