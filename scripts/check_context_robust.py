"""用途：在真实训练会话上验证新上下文模式与鲁棒尾部覆盖预算的兼容性，不作为性能试验。"""
import json
from pathlib import Path
from dataclasses import replace
import torch
from bbp_behm.config import load_experiment_config
from bbp_behm.data import PreprocessedSessionDataset,collate_sessions
from bbp_behm.hypergraph import HyperedgeBuilder
from bbp_behm.hsl_context import ContextHSLBBPBeHM
from scripts.train_hsl_pilot import check_invariants

def main():
    torch.set_num_threads(4)
    root=Path("outputs/hsl_context_20261109")
    folder=root/"screen_fold1_seed42"
    config=load_experiment_config(Path("configs/vine_hsl_server.ini"))
    config=replace(config,pruning=replace(config.pruning,budget_mode="robust_adaptive"))
    dataset=PreprocessedSessionDataset(config.data.processed_path,expected_sessions=970,expected_nodes=68662,expected_features=768,verify_hash=False)
    split=json.loads((folder/"split.json").read_text())
    ids=split["train"][:4]
    assert not set(ids)&set(split["final_test_excluded"])
    batch=collate_sessions([dataset[i] for i in ids]).to(torch.device("cuda"))
    graph=HyperedgeBuilder(config.hypergraph).build_batch(batch)
    rows=[]
    for mode in ["bbp","node_control","node_detach","edge_control","edge_context"]:
        torch.manual_seed(42)
        checkpoint=torch.load(folder/(mode+"_last.pt"),map_location="cpu",weights_only=False)
        model=ContextHSLBBPBeHM(config,mode=mode).cuda().train()
        model.load_state_dict(checkpoint["model"])
        output=model(batch,graph,stochastic_selection=False)
        check_invariants(model,batch,graph,output)
        output.total_loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        coverage=min(output.selection.tail_coverage_per_session.values())
        assert coverage+1e-6>=config.pruning.coverage_tail_gamma
        rows.append({"mode":mode,"min_tail_coverage":coverage,"kept":output.selection.kept_per_session})
    (root/"robust_context_compatibility.json").write_text(json.dumps({"purpose":"Functional compatibility, not robust-budget efficacy","training_indices":ids,"checks":rows},indent=2))
    print("CONTEXT_ROBUST_CHECK_PASS",json.dumps(rows))
if __name__=="__main__":
    main()
